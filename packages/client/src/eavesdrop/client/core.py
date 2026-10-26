"""Live audio capture and timed transcription over a persistent WebSocket."""

import asyncio
import secrets
from collections.abc import Callable
from types import TracebackType
from typing import Literal

from structlog.stdlib import BoundLogger

from eavesdrop.client.audio import SAMPLE_RATE, AudioCapture
from eavesdrop.client.connection import WebSocketConnection
from eavesdrop.common import get_logger
from eavesdrop.wire import (
  TranscriptionMessage,
  UserTranscriptionOptions,
)

RECONNECT_DELAY_SECONDS = 10.0


class EavesdropClient:
  """Capture live audio and push timed transcript messages to one caller."""

  def __init__(
    self,
    host: str = "localhost",
    port: int = 9090,
    audio_device: str | None = None,
    transcription_options: UserTranscriptionOptions | None = None,
    on_capture: Callable[[bytes], None] | None = None,
    on_transcription: Callable[[TranscriptionMessage], None] | None = None,
    on_connected: Callable[[str], None] | None = None,
    on_disconnected: Callable[[str, str | None], None] | None = None,
    on_reconnecting: Callable[[str, int, float], None] | None = None,
    on_reconnected: Callable[[str], None] | None = None,
  ) -> None:
    """Initialize EavesdropClient.

    :param host: Server hostname
    :param port: Server port
    :param audio_device: Audio device to capture from
    :param transcription_options: Transcription configuration options
    :param on_capture: Receives audio sent to the server.
    :param on_transcription: Receives accepted wire transcript messages in arrival order.
    :param on_connected: Receives the session identifier after initial connection.
    :param on_disconnected: Receives the session identifier and terminal reason.
    :param on_reconnecting: Receives the session identifier, attempt, and retry delay.
    :param on_reconnected: Receives the session identifier after reconnection.
    """
    self._host: str = host
    self._port: int = port
    self._audio_device: str | None = audio_device
    self._transcription_options: UserTranscriptionOptions = (
      transcription_options or UserTranscriptionOptions()
    )
    self._on_capture: Callable[[bytes], None] | None = on_capture
    self.on_transcription: Callable[[TranscriptionMessage], None] | None = on_transcription
    self.on_connected: Callable[[str], None] | None = on_connected
    self.on_disconnected: Callable[[str, str | None], None] | None = on_disconnected
    self.on_reconnecting: Callable[[str, int, float], None] | None = on_reconnecting
    self.on_reconnected: Callable[[str], None] | None = on_reconnected

    self._session_id: str = f"t{secrets.token_hex(2)}"
    self._connection: WebSocketConnection | None = None
    self._audio_capture: AudioCapture | None = None
    self._connected: bool = False
    self._streaming: bool = False
    self._message_task: asyncio.Task[None] | None = None
    self._audio_loop_task: asyncio.Task[None] | None = None
    self._reconnect_task: asyncio.Task[None] | None = None
    self._recording_transition_lock: asyncio.Lock = asyncio.Lock()
    self._disconnect_requested: bool = False
    self._reconnect_enabled: bool = False
    self._live_setup_options: UserTranscriptionOptions | None = None
    self._current_recording_id: str | None = None
    self._logger: BoundLogger = get_logger("client/core", host=host, port=port)
    self._flush_future: asyncio.Future[TranscriptionMessage] | None = None
    self._disconnect_reason: str | None = None

    self._validate_configuration()

  def _validate_configuration(self) -> None:
    """Validate client configuration parameters."""

    if not self._audio_device:
      self._logger.error("audio device is required")
      raise ValueError("Live transcription requires audio device specification")

    if self._port <= 0 or self._port > 65535:
      self._logger.error("invalid server port", port=self._port)
      raise ValueError(f"Invalid port number: {self._port}")

  @classmethod
  def transcriber(
    cls,
    host: str = "localhost",
    port: int = 9090,
    audio_device: str = "default",
    on_capture: Callable[[bytes], None] | None = None,
    send_last_n_segments: int | None = None,
    language: Literal["en"] | None = None,
    on_transcription: Callable[[TranscriptionMessage], None] | None = None,
    on_connected: Callable[[str], None] | None = None,
    on_disconnected: Callable[[str, str | None], None] | None = None,
    on_reconnecting: Callable[[str, int, float], None] | None = None,
    on_reconnected: Callable[[str], None] | None = None,
  ) -> "EavesdropClient":
    """Create a client for English-only Orukeet transcription.

    :param host: Server hostname.
    :param port: Server port.
    :param audio_device: Audio device to capture from.
    :param on_capture: Optional callback receiving each captured audio chunk.
    :param send_last_n_segments: Number of recent segments to include.
    :param language: English-only compatibility hint; None also selects English.
    :param on_transcription: Receives each accepted wire transcript message.
    :param on_connected: Receives the session identifier after initial connection.
    :param on_disconnected: Receives the session identifier and disconnect reason.
    :param on_reconnecting: Receives the session identifier, attempt, and retry delay.
    :param on_reconnected: Receives the session identifier after reconnection.
    :returns: Configured EavesdropClient in transcriber mode.
    :raises pydantic.ValidationError: If transcription options are invalid.
    """
    transcription_options = UserTranscriptionOptions(
      send_last_n_segments=send_last_n_segments,
      language=language,
    )

    return cls(
      host=host,
      port=port,
      audio_device=audio_device,
      transcription_options=transcription_options,
      on_capture=on_capture,
      on_transcription=on_transcription,
      on_connected=on_connected,
      on_disconnected=on_disconnected,
      on_reconnecting=on_reconnecting,
      on_reconnected=on_reconnected,
    )

  async def _await_prior_audio_loop(self) -> None:
    """Wait for any previous live audio-loop task to finish before restarting."""
    if self._audio_loop_task is None or self._audio_loop_task.done():
      return
    await self._audio_loop_task

  def _build_connection(self) -> WebSocketConnection:
    """Create a websocket connection bound to this client's stable stream."""
    return WebSocketConnection(
      host=self._host,
      port=self._port,
      stream_name=self._session_id,
      on_error=self._on_error,
      on_transcription=self._accept_transcription,
      on_disconnect=self._on_disconnect,
    )

  def _start_message_task(self) -> None:
    """Start websocket message handling for the active connection."""
    if not self._connection:
      raise RuntimeError("Connection not initialized")

    self._message_task = asyncio.create_task(self._connection.handle_messages())

  async def connect(self, setup_options: UserTranscriptionOptions | None = None) -> None:
    """Establish WebSocket connection to server.

    :param setup_options: Optional setup message override for this connection.
    :type setup_options: UserTranscriptionOptions | None
    """
    if self._connected:
      return

    self._disconnect_requested = False

    try:
      self._live_setup_options = setup_options or self._transcription_options
      self._reconnect_enabled = True
      await self._connect_transcriber(setup_options)
    except Exception:
      self._logger.exception("initial client connection failed")
      self._reconnect_enabled = False
      raise

    self._connected = True
    self._disconnect_reason = None
    if self.on_connected is not None:
      self.on_connected(self._session_id)

  async def _connect_transcriber(
    self, setup_options: UserTranscriptionOptions | None = None
  ) -> None:
    """Connect in transcriber mode."""
    # Initialize audio capture for transcriber mode
    if self._audio_capture is None:
      self._audio_capture = AudioCapture(on_error=self._on_error, audio_device=self._audio_device)

    self._connection = self._build_connection()

    options = setup_options or self._transcription_options
    await self._connection.connect(options)
    self._start_message_task()

  async def disconnect(self) -> None:
    """Close WebSocket connection and cleanup resources."""
    if not self._connected and self._connection is None and self._reconnect_task is None:
      return

    self._disconnect_requested = True
    self._reconnect_enabled = False
    self._connected = False
    self._streaming = False
    self._fail_pending_flush("flush() failed because the client disconnected")

    if self._audio_capture:
      self._audio_capture.stop_recording()
      self._audio_capture = None

    if self._audio_loop_task and not self._audio_loop_task.done():
      _ = self._audio_loop_task.cancel()
      _ = await asyncio.gather(self._audio_loop_task, return_exceptions=True)
    self._audio_loop_task = None

    if self._reconnect_task and not self._reconnect_task.done():
      _ = self._reconnect_task.cancel()
      _ = await asyncio.gather(self._reconnect_task, return_exceptions=True)
    self._reconnect_task = None

    if self._connection:
      await self._connection.disconnect()
      self._connection = None
    if self._message_task is not None:
      _ = self._message_task.cancel()
      _ = await asyncio.gather(self._message_task, return_exceptions=True)
    self._message_task = None
    self._current_recording_id = None
    self._disconnect_reason = None

  async def start_streaming(self, recording_id: str) -> None:
    """Start capture after any preceding recording's flush has completed."""
    async with self._recording_transition_lock:
      await self._start_streaming(recording_id)

  async def _start_streaming(self, recording_id: str) -> None:
    if not self._connected:
      self._logger.error("recording start requires a connection", recording_id=recording_id)
      raise RuntimeError("Must connect() before start_streaming()")

    if self._streaming:
      return

    if not self._audio_capture:
      self._logger.error("recording start requires capture", recording_id=recording_id)
      raise RuntimeError("Audio capture not initialized")

    if not self._connection:
      self._logger.error("recording start requires transport", recording_id=recording_id)
      raise RuntimeError("Connection not initialized")

    await self._await_prior_audio_loop()
    self._current_recording_id = recording_id
    self._audio_capture.clear_queue()
    await self._connection.send_recording_started(recording_id, SAMPLE_RATE)

    self._audio_capture.start_recording()
    self._streaming = True

    self._audio_loop_task = asyncio.create_task(self._audio_streaming_loop())

  async def stop_streaming(self) -> None:
    """Stop audio streaming and wait for the active audio loop to exit."""
    if not self._streaming:
      await self._await_prior_audio_loop()
      return

    if self._audio_capture:
      self._audio_capture.stop_recording()

    # Capture callbacks already scheduled on this loop must reach the queue before draining.
    await asyncio.sleep(0)
    self._streaming = False
    await self._await_prior_audio_loop()

  async def cancel_utterance(self, recording_id: str) -> None:
    """Cancel the current live utterance without closing the websocket session.

    :raises RuntimeError: If cancel is unsupported for the current mode/session.
    """
    if (
      not self._connected
      or not self._connection
      or not self._connection.is_connected()
      or self._message_task is None
    ):
      self._logger.error("cancellation requires a live connection", recording_id=recording_id)
      raise RuntimeError("cancel_utterance() requires an active live transcriber connection")

    if self._current_recording_id != recording_id:
      self._logger.error("cancellation requires active recording", recording_id=recording_id)
      raise RuntimeError("cancel_utterance() requires the active recording id")

    await self.stop_streaming()
    await self._connection.send_utterance_cancelled(recording_id)
    self._current_recording_id = None
    self._fail_pending_flush("flush() failed because the recording was cancelled")

  async def _reconnect_loop(self) -> None:
    """Reconnect a live transcriber session on a fixed retry cadence."""
    attempt = 1

    try:
      if self._connection is not None:
        await self._connection.disconnect()
      if self._message_task is not None:
        await self._message_task
        self._message_task = None
      while self._reconnect_enabled and not self._disconnect_requested:
        if self.on_reconnecting is not None:
          self.on_reconnecting(self._session_id, attempt, RECONNECT_DELAY_SECONDS)
        self._logger.warning(
          "live transcriber reconnect scheduled",
          attempt=attempt,
          retry_delay_s=RECONNECT_DELAY_SECONDS,
          reason=self._disconnect_reason,
        )
        await asyncio.sleep(RECONNECT_DELAY_SECONDS)

        if self._disconnect_requested or not self._reconnect_enabled:
          return

        try:
          self._connection = self._build_connection()
          await self._connection.connect(self._live_setup_options or self._transcription_options)
        except asyncio.CancelledError:
          raise
        except Exception:
          self._logger.exception("live transcriber reconnect attempt failed", attempt=attempt)
          attempt += 1
          continue

        self._connected = True
        self._disconnect_reason = None
        self._start_message_task()
        if self.on_reconnected is not None:
          self.on_reconnected(self._session_id)
        self._logger.info("live transcriber reconnected", stream=self._session_id)
        return
    finally:
      self._reconnect_task = None

  async def flush(self, recording_id: str, *, force_complete: bool = True) -> TranscriptionMessage:
    """Request a live transcription flush and await the terminal flush response.

    A pending flush retains its recording identity until its terminal response arrives.
    Starting the next recording waits for that response. Transcript messages are pushed
    directly to the configured callback before the flush completes.

    :param force_complete: Whether the server should force-complete the current tentative tail.
    :type force_complete: bool
    :returns: The flush-satisfying transcription update from the server.
    :rtype: TranscriptionMessage
    :raises RuntimeError: If flush is unsupported for the current mode/session, another local
      flush is already pending, the server rejects the request, or the live connection closes
      before the flush completes.
    """
    if self._flush_future is not None:
      self._logger.warning("rejecting concurrent flush", recording_id=recording_id)
      raise RuntimeError("flush() already in progress")
    future: asyncio.Future[TranscriptionMessage] = asyncio.get_running_loop().create_future()
    self._flush_future = future
    try:
      async with self._recording_transition_lock:
        return await self._flush_recording(
          recording_id, force_complete=force_complete, future=future
        )
    finally:
      self._flush_future = None
      if not future.done():
        _ = future.cancel()
      elif not future.cancelled():
        _ = future.exception()

  async def _flush_recording(
    self,
    recording_id: str,
    *,
    force_complete: bool,
    future: asyncio.Future[TranscriptionMessage],
  ) -> TranscriptionMessage:
    if (
      not self._connected
      or not self._connection
      or not self._connection.is_connected()
      or self._message_task is None
    ):
      self._logger.error("flush requires a live connection", recording_id=recording_id)
      raise RuntimeError("flush() requires an active live transcriber connection")

    if self._current_recording_id != recording_id:
      self._logger.error("flush requires active recording", recording_id=recording_id)
      raise RuntimeError("flush() requires the active recording id")

    await self._connection.send_flush_control(recording_id, force_complete=force_complete)
    return await future

  async def _audio_streaming_loop(self) -> None:
    """Background task to stream audio data to server."""
    if not self._connection or not self._audio_capture:
      return

    try:
      while self._connected:
        audio_data = await self._audio_capture.get_audio_data(timeout=0.1)
        if audio_data is None and not self._streaming:
          break
        if audio_data:
          await self._connection.send_audio_data(audio_data)
          if self._on_capture is not None:
            self._on_capture(audio_data)
    except asyncio.CancelledError:
      raise
    except Exception:
      self._logger.exception("audio streaming loop failed")
    finally:
      self._streaming = False

  def is_connected(self) -> bool:
    """Check if client is connected to server."""
    return self._connected

  def is_streaming(self) -> bool:
    """Check if client is actively streaming live audio."""
    return self._streaming

  async def __aenter__(self) -> "EavesdropClient":
    """Enter async context manager."""
    await self.connect()
    return self

  async def __aexit__(
    self,
    _exc_type: type[BaseException] | None,
    _exc_val: BaseException | None,
    _exc_tb: TracebackType | None,
  ) -> None:
    """Exit async context manager."""
    await self.disconnect()

  def _on_disconnect(self, reason: str | None) -> None:
    """Handle terminal disconnect signaling from the server."""
    self._disconnect_reason = reason
    self._fail_pending_flush(
      reason or "flush() failed because the live connection closed before flush completion"
    )

    if not self._reconnect_enabled:
      return

    if self._disconnect_requested:
      return

    if self._audio_capture:
      self._audio_capture.stop_recording()

    self._connected = False
    self._streaming = False
    if self.on_disconnected is not None:
      self.on_disconnected(self._session_id, reason)
    self._logger.warning("live transcriber disconnected", stream=self._session_id, reason=reason)

    if self._reconnect_task is None or self._reconnect_task.done():
      self._reconnect_task = asyncio.create_task(self._reconnect_loop())

  def _on_error(self, error: str) -> None:
    """Handle error callback."""
    self._fail_pending_flush(error)
    self._logger.error("client error", error=error)

  def _fail_pending_flush(self, error: str) -> None:
    """Fail an outstanding request without poisoning a later recording."""
    if self._flush_future is not None and not self._flush_future.done():
      self._flush_future.set_exception(RuntimeError(error))

  def _accept_transcription(self, message: TranscriptionMessage) -> None:
    """Push accepted messages and settle only the active recording's pending flush."""
    if message.recording_id != self._current_recording_id:
      self._logger.debug(
        "ignoring transcription for an inactive recording",
        recording_id=message.recording_id,
        active_recording_id=self._current_recording_id,
      )
      return

    if self.on_transcription is not None:
      self.on_transcription(message)

    if message.flush_complete and self._flush_future is not None and not self._flush_future.done():
      self._flush_future.set_result(message)
