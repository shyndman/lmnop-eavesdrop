"""
EavesdropClient: Programmatic interface for live eavesdrop transcription services.

Provides async iterator and context manager protocols for streaming live audio
to the transcription server and receiving transcription events.
"""

import asyncio
import secrets
from collections.abc import AsyncIterator, Callable
from types import TracebackType
from typing import Literal

from structlog.stdlib import BoundLogger

from eavesdrop.client.audio import SAMPLE_RATE, AudioCapture
from eavesdrop.client.connection import WebSocketConnection
from eavesdrop.client.events import (
  ConnectedEvent,
  DisconnectedEvent,
  LiveClientEvent,
  ReconnectedEvent,
  ReconnectingEvent,
  TranscriptionEvent,
)
from eavesdrop.common import get_logger
from eavesdrop.wire import (
  ClientType,
  TranscriptionMessage,
  UserTranscriptionOptions,
)

RECONNECT_DELAY_SECONDS = 10.0


class EavesdropClient:
  """Live microphone transcription client for eavesdrop services."""

  def __init__(
    self,
    client_type: ClientType = ClientType.TRANSCRIBER,
    host: str = "localhost",
    port: int = 9090,
    audio_device: str | None = None,
    transcription_options: UserTranscriptionOptions | None = None,
    on_capture: Callable[[bytes], None] | None = None,
  ) -> None:
    """Initialize EavesdropClient.

    :param client_type: Type of client (TRANSCRIBER).
    :type client_type: ClientType
    :param host: Server hostname.
    :type host: str
    :param port: Server port.
    :type port: int
    :param audio_device: Audio device to use.
    :type audio_device: str | None
    :param transcription_options: Transcription configuration options.
    :type transcription_options: UserTranscriptionOptions | None
    :param on_capture: Optional callback receiving each captured audio chunk.
    :type on_capture: Callable[[bytes], None] | None
    """
    self._host: str = host
    self._port: int = port
    self._client_type: ClientType = client_type
    self._audio_device: str | None = audio_device
    self._transcription_options: UserTranscriptionOptions = (
      transcription_options or UserTranscriptionOptions()
    )
    self._on_capture: Callable[[bytes], None] | None = on_capture

    # Internal state
    self._stream_name: str | None = None
    self._connection: WebSocketConnection | None = None
    self._audio_capture: AudioCapture | None = None
    self._connected: bool = False
    self._streaming: bool = False
    self._message_queue: asyncio.Queue[TranscriptionMessage] = asyncio.Queue()
    self._event_queue: asyncio.Queue[LiveClientEvent] = asyncio.Queue()
    self._background_tasks: set[asyncio.Task[None]] = set()
    self._message_task: asyncio.Task[None] | None = None
    self._audio_loop_task: asyncio.Task[None] | None = None
    self._reconnect_task: asyncio.Task[None] | None = None
    self._disconnect_event: asyncio.Event = asyncio.Event()
    self._disconnect_reason: str | None = None
    self._flush_waiting: bool = False
    self._flush_error: str | None = None
    self._disconnect_requested: bool = False
    self._event_stream_open: bool = False
    self._reconnect_enabled: bool = False
    self._live_setup_options: UserTranscriptionOptions | None = None
    self._current_recording_id: str | None = None
    self._logger: BoundLogger = get_logger(
      "client/core",
      client_type=client_type.value,
      host=host,
      port=port,
    )

    self._validate_configuration()

  def _validate_configuration(self) -> None:
    """Validate client configuration parameters."""
    if self._client_type != ClientType.TRANSCRIBER:
      raise ValueError(f"Unsupported client type: {self._client_type}")

    if not self._audio_device:
      raise ValueError("Transcriber mode requires audio device specification")

    if self._port <= 0 or self._port > 65535:
      raise ValueError(f"Invalid port number: {self._port}")

  @classmethod
  def transcriber(
    cls,
    host: str = "localhost",
    port: int = 9090,
    audio_device: str = "default",
    on_capture: Callable[[bytes], None] | None = None,
    word_timestamps: bool = True,
    send_last_n_segments: int | None = None,
    language: Literal["en"] | None = None,
  ) -> "EavesdropClient":
    """Create a client for English-only Orukeet transcription.

    :param host: Server hostname.
    :type host: str
    :param port: Server port.
    :type port: int
    :param audio_device: Audio device to capture from.
    :type audio_device: str
    :param on_capture: Optional callback receiving each captured audio chunk.
    :type on_capture: Callable[[bytes], None] | None
    :param word_timestamps: Include native word timestamps, enabled by default.
    :type word_timestamps: bool
    :param send_last_n_segments: Number of recent segments to include.
    :type send_last_n_segments: int | None
    :param language: English-only compatibility hint; None also selects English.
    :type language: Literal["en"] | None
    :returns: Configured EavesdropClient in transcriber mode.
    :rtype: EavesdropClient
    """
    transcription_options = UserTranscriptionOptions(
      word_timestamps=word_timestamps,
      send_last_n_segments=send_last_n_segments,
      language=language,
    )

    return cls(
      client_type=ClientType.TRANSCRIBER,
      host=host,
      port=port,
      audio_device=audio_device,
      transcription_options=transcription_options,
      on_capture=on_capture,
    )

  def _ensure_stream_name(self) -> str:
    """Return the stable protocol stream identifier for this client instance."""
    if self._stream_name is None:
      self._stream_name = f"t{secrets.token_hex(2)}"
    return self._stream_name

  def _track_background_task(self, task: asyncio.Task[None]) -> asyncio.Task[None]:
    """Track a background task for coordinated client cleanup."""
    self._background_tasks.add(task)
    task.add_done_callback(self._background_tasks.discard)
    return task

  def _emit_event(self, event: LiveClientEvent) -> None:
    """Publish a live-client event to async iterator consumers."""
    if self._event_stream_open:
      self._event_queue.put_nowait(event)

  def _set_audio_loop_task(self, task: asyncio.Task[None] | None) -> None:
    """Store the dedicated live audio-loop task and clear it on completion."""
    self._audio_loop_task = task
    if task is None:
      return

    def _clear_if_current(completed: asyncio.Task[None]) -> None:
      if self._audio_loop_task is completed:
        self._audio_loop_task = None

    task.add_done_callback(_clear_if_current)

  async def _await_prior_audio_loop(self) -> None:
    """Wait for any previous live audio-loop task to finish before restarting."""
    if self._audio_loop_task is None or self._audio_loop_task.done():
      return
    await self._audio_loop_task

  def _build_connection(self) -> WebSocketConnection:
    """Create a websocket connection bound to this client's stable stream."""
    stream_name = self._ensure_stream_name()
    return WebSocketConnection(
      host=self._host,
      port=self._port,
      stream_name=stream_name,
      on_ready=self._on_ready,
      on_transcription=self._on_transcription_text,
      on_error=self._on_error,
      client_type=ClientType.TRANSCRIBER,
      on_transcription_message=self._on_transcription_message,
      on_disconnect=self._on_disconnect,
    )

  def _start_message_task(self) -> None:
    """Start websocket message handling for the active connection."""
    if not self._connection:
      raise RuntimeError("Connection not initialized")

    task = asyncio.create_task(self._connection.handle_messages())
    self._message_task = self._track_background_task(task)

  async def connect(self, setup_options: UserTranscriptionOptions | None = None) -> None:
    """Establish WebSocket connection to server.

    :param setup_options: Optional setup message override for this connection.
    :type setup_options: UserTranscriptionOptions | None
    """
    if self._connected:
      return

    self._disconnect_requested = False
    self._disconnect_event.clear()
    self._disconnect_reason = None
    self._event_stream_open = True

    try:
      options = setup_options or self._transcription_options
      self._live_setup_options = options
      self._reconnect_enabled = True
      await self._connect_transcriber(options)
    except Exception:
      self._event_stream_open = False
      self._reconnect_enabled = False
      raise

    self._connected = True
    self._emit_event(ConnectedEvent(stream=self._ensure_stream_name()))

  async def _connect_transcriber(self, setup_options: UserTranscriptionOptions) -> None:
    """Connect in transcriber mode."""
    if self._audio_capture is None:
      self._audio_capture = AudioCapture(on_error=self._on_error, audio_device=self._audio_device)

    self._connection = self._build_connection()
    await self._connection.connect(setup_options)
    self._start_message_task()

  async def disconnect(self) -> None:
    """Close WebSocket connection and cleanup resources."""
    if (
      not self._connected
      and self._connection is None
      and self._reconnect_task is None
      and not self._event_stream_open
    ):
      return

    self._disconnect_requested = True
    self._reconnect_enabled = False
    self._connected = False
    self._streaming = False
    self._event_stream_open = False

    if self._audio_capture:
      self._audio_capture.stop_recording()
      self._audio_capture = None

    if self._audio_loop_task and not self._audio_loop_task.done():
      _ = self._audio_loop_task.cancel()
      _ = await asyncio.gather(self._audio_loop_task, return_exceptions=True)
    self._set_audio_loop_task(None)

    if self._reconnect_task and not self._reconnect_task.done():
      _ = self._reconnect_task.cancel()
      _ = await asyncio.gather(self._reconnect_task, return_exceptions=True)
    self._reconnect_task = None

    if self._connection:
      await self._connection.disconnect()
      self._connection = None
    self._message_task = None

    for task in list(self._background_tasks):
      _ = task.cancel()

    if self._background_tasks:
      _ = await asyncio.gather(*self._background_tasks, return_exceptions=True)
    self._background_tasks.clear()
    self._disconnect_event.clear()
    self._disconnect_reason = None
    self._flush_error = None
    self._current_recording_id = None
    self._clear_event_queue()

  async def start_streaming(self, recording_id: str) -> None:
    """Start audio streaming."""
    if not self._connected:
      raise RuntimeError("Must connect() before start_streaming()")

    if self._streaming:
      return

    if not self._audio_capture:
      raise RuntimeError("Audio capture not initialized")

    if not self._connection:
      raise RuntimeError("Connection not initialized")

    await self._await_prior_audio_loop()
    self._current_recording_id = recording_id
    self._audio_capture.clear_queue()
    self._clear_message_queue()
    self._clear_event_queue()
    await self._connection.send_recording_started(recording_id, SAMPLE_RATE)

    self._audio_capture.start_recording()
    self._streaming = True

    task = asyncio.create_task(self._audio_streaming_loop())
    self._set_audio_loop_task(self._track_background_task(task))

  async def stop_streaming(self) -> None:
    """Stop audio streaming and wait for the active audio loop to exit."""
    if not self._streaming:
      await self._await_prior_audio_loop()
      return

    self._streaming = False

    if self._audio_capture:
      self._audio_capture.stop_recording()

    await self._await_prior_audio_loop()

  async def cancel_utterance(self, recording_id: str) -> None:
    """Cancel the current live utterance without closing the websocket session.

    :param recording_id: Active live recording identifier.
    :type recording_id: str
    :raises RuntimeError: If the live connection is inactive or recording id mismatches.
    """
    if (
      not self._connected
      or not self._connection
      or not self._connection.is_connected()
      or self._message_task is None
    ):
      raise RuntimeError("cancel_utterance() requires an active live transcriber connection")

    if self._current_recording_id != recording_id:
      raise RuntimeError("cancel_utterance() requires the active recording id")

    await self.stop_streaming()
    await self._connection.send_utterance_cancelled(recording_id)
    self._current_recording_id = None
    self._clear_message_queue()
    self._clear_event_queue()

  async def _reconnect_loop(self) -> None:
    """Reconnect a live transcriber session on a fixed retry cadence."""
    attempt = 1

    try:
      while self._reconnect_enabled and not self._disconnect_requested:
        self._emit_event(
          ReconnectingEvent(
            stream=self._ensure_stream_name(),
            attempt=attempt,
            retry_delay_s=RECONNECT_DELAY_SECONDS,
          )
        )
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
        self._disconnect_event.clear()
        self._disconnect_reason = None
        self._start_message_task()
        self._emit_event(ReconnectedEvent(stream=self._ensure_stream_name()))
        self._logger.info("live transcriber reconnected", stream=self._ensure_stream_name())
        return
    finally:
      self._reconnect_task = None

  async def flush(self, recording_id: str, *, force_complete: bool = True) -> TranscriptionMessage:
    """Request a live transcription flush and await the terminal flush response.

    :param recording_id: Active live recording identifier.
    :type recording_id: str
    :param force_complete: Whether the server should force-complete the current tentative tail.
    :type force_complete: bool
    :returns: The flush-satisfying transcription update from the server.
    :rtype: TranscriptionMessage
    :raises RuntimeError: If no live connection is active, another flush is already pending,
      the server rejects the request, or the connection closes before completion.
    """
    if (
      not self._connected
      or not self._connection
      or not self._connection.is_connected()
      or self._message_task is None
    ):
      raise RuntimeError("flush() requires an active live transcriber connection")

    if self._flush_waiting:
      raise RuntimeError("flush() already in progress")

    if self._current_recording_id != recording_id:
      raise RuntimeError("flush() requires the active recording id")

    self._flush_waiting = True
    self._flush_error = None

    try:
      self._clear_message_queue()
      await self._connection.send_flush_control(recording_id, force_complete=force_complete)

      while True:
        if self._flush_error is not None:
          raise RuntimeError(self._flush_error)

        if self._message_task.done() and self._message_queue.empty():
          raise RuntimeError(self._flush_disconnect_message())

        try:
          message = await asyncio.wait_for(self._message_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
          if self._disconnect_event.is_set():
            raise RuntimeError(self._flush_disconnect_message())
          continue

        if message.recording_id == recording_id and message.flush_complete is True:
          return message
    finally:
      self._flush_waiting = False
      self._flush_error = None

  def _clear_message_queue(self) -> None:
    """Drain stale buffered messages before a new operation or recording epoch."""
    while not self._message_queue.empty():
      _ = self._message_queue.get_nowait()

  def _clear_event_queue(self) -> None:
    """Drain stale transcription iterator events while preserving connection events."""
    preserved_events: list[LiveClientEvent] = []
    while not self._event_queue.empty():
      event = self._event_queue.get_nowait()
      if isinstance(event, TranscriptionEvent):
        continue
      preserved_events.append(event)

    for event in preserved_events:
      self._event_queue.put_nowait(event)

  def _flush_disconnect_message(self) -> str:
    """Build a truthful flush failure message from current disconnect state."""
    if self._disconnect_reason:
      return self._disconnect_reason
    return "flush() failed because the live connection closed before flush completion"

  async def _audio_streaming_loop(self) -> None:
    """Background task to stream audio data to server."""
    if not self._connection or not self._audio_capture:
      return

    try:
      while self._streaming and self._connected:
        audio_data = await self._audio_capture.get_audio_data(timeout=0.1)
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
    """Check if client is actively streaming."""
    return self._streaming

  def __aiter__(self) -> AsyncIterator[LiveClientEvent]:
    """Return async iterator for ordered live-client events."""
    return self

  async def __anext__(self) -> LiveClientEvent:
    """Get the next live-client event in temporal order."""
    while self._event_stream_open or not self._event_queue.empty():
      try:
        return await asyncio.wait_for(self._event_queue.get(), timeout=0.1)
      except asyncio.TimeoutError:
        continue
    raise StopAsyncIteration

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

  def _on_ready(self, _backend: str) -> None:
    """Handle server ready callback."""
    pass

  def _on_transcription_message(self, message: TranscriptionMessage) -> None:
    """Handle full TranscriptionMessage from connection."""
    if not self._should_accept_transcription_message(message):
      self._logger.debug(
        "dropping stale transcription message",
        stream=message.stream,
        recording_id=message.recording_id,
        active_recording_id=self._current_recording_id,
      )
      return

    if self._flush_waiting:
      self._message_queue.put_nowait(message)
    self._emit_event(TranscriptionEvent(stream=message.stream, message=message))

  def _should_accept_transcription_message(self, message: TranscriptionMessage) -> bool:
    """Return whether an incoming transcription belongs to the current client epoch."""
    current_recording_id = self._current_recording_id
    if current_recording_id is None:
      return True

    return message.recording_id == current_recording_id

  def _on_transcription_text(self, _text: str) -> None:
    """Handle legacy transcription text callback from existing connection."""
    pass

  def _on_disconnect(self, reason: str | None) -> None:
    """Handle terminal disconnect signaling from the server."""
    self._disconnect_reason = reason
    self._disconnect_event.set()

    if self._flush_waiting and self._flush_error is None:
      self._flush_error = self._flush_disconnect_message()

    if not self._reconnect_enabled or self._disconnect_requested:
      return

    if self._audio_capture:
      self._audio_capture.stop_recording()

    self._connected = False
    self._streaming = False
    self._emit_event(DisconnectedEvent(stream=self._ensure_stream_name(), reason=reason))
    self._logger.warning(
      "live transcriber disconnected", stream=self._ensure_stream_name(), reason=reason
    )

    if self._reconnect_task is None or self._reconnect_task.done():
      self._reconnect_task = self._track_background_task(
        asyncio.create_task(self._reconnect_loop())
      )

  def _on_error(self, error: str) -> None:
    """Handle error callback."""
    if self._flush_waiting and self._flush_error is None:
      self._flush_error = error
    self._logger.error("client error", error=error)
