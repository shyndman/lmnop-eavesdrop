"""
EavesdropClient: Unified programmatic interface for eavesdrop transcription services.

Provides factory methods for creating mode-specific clients with async iterator
and context manager protocols for streaming transcription results.
"""

import asyncio
import secrets
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from structlog.stdlib import BoundLogger

from eavesdrop.client.audio import AudioCapture
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
  Segment,
  TranscriptionMessage,
  TranscriptionSourceMode,
  UserTranscriptionOptions,
)

RECONNECT_DELAY_SECONDS = 10.0


@dataclass(frozen=True)
class FileTranscriptionResult:
  """Deterministic result envelope returned by ``EavesdropClient.transcribe_file``."""

  segments: list[Segment]
  text: str
  language: str | None
  warnings: list[str] = field(default_factory=list)


class EavesdropClient:
  """
  Unified client for eavesdrop transcription services.

  Supports both transcriber mode (sending audio for transcription) and
  subscriber mode (receiving transcriptions from RTSP streams).
  """

  def __init__(
    self,
    client_type: ClientType,
    host: str = "localhost",
    port: int = 9090,
    stream_names: list[str] | None = None,
    audio_device: str | None = None,
    transcription_options: UserTranscriptionOptions | None = None,
  ):
    """Initialize EavesdropClient.

    :param client_type: Type of client (TRANSCRIBER or RTSP_SUBSCRIBER)
    :param host: Server hostname
    :param port: Server port
    :param stream_names: For subscriber mode, list of stream names to subscribe to
    :param audio_device: For transcriber mode, audio device to use
    :param transcription_options: Transcription configuration options
    """
    self._host = host
    self._port = port
    self._client_type = client_type
    self._stream_names = stream_names or []
    self._audio_device = audio_device
    self._transcription_options: UserTranscriptionOptions = (
      transcription_options or UserTranscriptionOptions()
    )

    # Internal state
    self._stream_name: str | None = None
    self._connection: WebSocketConnection | None = None
    self._audio_capture: AudioCapture | None = None
    self._connected = False
    self._streaming = False
    self._message_queue: asyncio.Queue[TranscriptionMessage] = asyncio.Queue()
    self._event_queue: asyncio.Queue[LiveClientEvent] = asyncio.Queue()
    self._background_tasks: set[asyncio.Task[None]] = set()
    self._message_task: asyncio.Task[None] | None = None
    self._audio_loop_task: asyncio.Task[None] | None = None
    self._reconnect_task: asyncio.Task[None] | None = None
    self._operation_lock = asyncio.Lock()
    self._disconnect_event = asyncio.Event()
    self._disconnect_reason: str | None = None
    self._flush_waiting = False
    self._flush_error: str | None = None
    self._disconnect_requested = False
    self._event_stream_open = False
    self._reconnect_enabled = False
    self._live_setup_options: UserTranscriptionOptions | None = None
    self._logger: BoundLogger = get_logger(
      "client/core",
      client_type=client_type.value,
      host=host,
      port=port,
    )

    # Validate configuration
    self._validate_configuration()

  def _validate_configuration(self) -> None:
    """Validate client configuration parameters."""
    if self._client_type == ClientType.RTSP_SUBSCRIBER:
      if not self._stream_names:
        raise ValueError("Subscriber mode requires at least one stream name")
    elif self._client_type == ClientType.TRANSCRIBER:
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
    word_timestamps: bool | None = None,
    initial_prompt: str | None = None,
    hotwords: list[str] | None = None,
    send_last_n_segments: int | None = None,
    beam_size: int | None = None,
    model: str | None = None,
  ) -> "EavesdropClient":
    """Create a transcriber client for sending audio for transcription.

    :param host: Server hostname
    :param port: Server port
    :param audio_device: Audio device to capture from
    :param word_timestamps: Enable word-level timestamps
    :param initial_prompt: Initial prompt for transcription context
    :param hotwords: Hotwords for improved recognition
    :param send_last_n_segments: Number of recent segments to include
    :param beam_size: Override beam search width for deterministic decoding
    :param model: Whisper model alias override for this client session
    :returns: Configured EavesdropClient in transcriber mode
    """
    transcription_options = UserTranscriptionOptions(
      word_timestamps=word_timestamps,
      initial_prompt=initial_prompt,
      hotwords=hotwords,
      send_last_n_segments=send_last_n_segments,
      beam_size=beam_size,
      model=model,
    )

    return cls(
      client_type=ClientType.TRANSCRIBER,
      host=host,
      port=port,
      audio_device=audio_device,
      transcription_options=transcription_options,
    )

  @classmethod
  def subscriber(
    cls,
    stream_names: list[str],
    host: str = "localhost",
    port: int = 9090,
  ) -> "EavesdropClient":
    """Create a subscriber client for receiving RTSP stream transcriptions.

    :param stream_names: List of RTSP stream names to subscribe to
    :param host: Server hostname
    :param port: Server port
    :returns: Configured EavesdropClient in subscriber mode
    """
    if not stream_names:
      raise ValueError("stream_names cannot be empty")

    return cls(
      client_type=ClientType.RTSP_SUBSCRIBER,
      host=host,
      port=port,
      stream_names=stream_names,
    )

  def _ensure_stream_name(self) -> str:
    """Return the stable protocol stream identifier for this client instance."""
    if self._stream_name is None:
      prefix = "t" if self._client_type == ClientType.TRANSCRIBER else "s"
      self._stream_name = f"{prefix}{secrets.token_hex(2)}"
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
    if self._client_type == ClientType.TRANSCRIBER:
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

    return WebSocketConnection(
      host=self._host,
      port=self._port,
      stream_name=stream_name,
      on_ready=self._on_ready,
      on_transcription=self._on_transcription_text,
      on_error=self._on_error,
      client_type=ClientType.RTSP_SUBSCRIBER,
      stream_names=self._stream_names,
      on_transcription_message=self._on_transcription_message,
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
      # Create connection based on client type
      if self._client_type == ClientType.TRANSCRIBER:
        options = setup_options or self._transcription_options
        self._live_setup_options = options
        self._reconnect_enabled = options.source_mode != TranscriptionSourceMode.FILE
        await self._connect_transcriber(setup_options)
      else:
        self._reconnect_enabled = False
        await self._connect_subscriber()
    except Exception:
      self._event_stream_open = False
      self._reconnect_enabled = False
      raise

    self._connected = True
    self._emit_event(ConnectedEvent(stream=self._ensure_stream_name()))

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

  async def _connect_subscriber(self) -> None:
    """Connect in subscriber mode."""
    self._connection = self._build_connection()

    await self._connection.connect()
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

    # Stop audio capture if active
    if self._audio_capture:
      self._audio_capture.stop_recording()
      self._audio_capture = None

    if self._audio_loop_task and not self._audio_loop_task.done():
      self._audio_loop_task.cancel()
      await asyncio.gather(self._audio_loop_task, return_exceptions=True)
    self._set_audio_loop_task(None)

    if self._reconnect_task and not self._reconnect_task.done():
      self._reconnect_task.cancel()
      await asyncio.gather(self._reconnect_task, return_exceptions=True)
    self._reconnect_task = None

    # Close WebSocket connection
    if self._connection:
      await self._connection.disconnect()
      self._connection = None
    self._message_task = None

    # Cancel background tasks
    for task in list(self._background_tasks):
      task.cancel()

    if self._background_tasks:
      await asyncio.gather(*self._background_tasks, return_exceptions=True)
    self._background_tasks.clear()
    self._disconnect_event.clear()
    self._disconnect_reason = None
    self._flush_error = None
    self._clear_event_queue()

  async def start_streaming(self) -> None:
    """Start audio streaming (transcriber mode only)."""
    if self._client_type != ClientType.TRANSCRIBER:
      raise RuntimeError("start_streaming() only supported in transcriber mode")

    if not self._connected:
      raise RuntimeError("Must connect() before start_streaming()")

    if self._streaming:
      return

    if not self._audio_capture:
      raise RuntimeError("Audio capture not initialized")

    await self._await_prior_audio_loop()

    # Start audio capture
    self._audio_capture.start_recording()
    self._streaming = True

    # Start audio streaming task
    task = asyncio.create_task(self._audio_streaming_loop())
    self._set_audio_loop_task(self._track_background_task(task))

  async def stop_streaming(self) -> None:
    """Stop audio streaming and wait for the active audio loop to exit."""
    if self._client_type != ClientType.TRANSCRIBER:
      raise RuntimeError("stop_streaming() only supported in transcriber mode")

    if not self._streaming:
      await self._await_prior_audio_loop()
      return

    self._streaming = False

    if self._audio_capture:
      self._audio_capture.stop_recording()

    await self._await_prior_audio_loop()

  async def cancel_utterance(self) -> None:
    """Cancel the current live utterance without closing the websocket session.

    :raises RuntimeError: If cancel is unsupported for the current mode/session.
    """
    if self._client_type != ClientType.TRANSCRIBER:
      raise RuntimeError("cancel_utterance() only supported in transcriber mode")

    if self._operation_lock.locked():
      raise RuntimeError("cancel_utterance() unavailable during transcribe_file() operation")

    if (
      not self._connected
      or not self._connection
      or not self._connection.is_connected()
      or self._message_task is None
    ):
      raise RuntimeError("cancel_utterance() requires an active live transcriber connection")

    await self.stop_streaming()
    await self._connection.send_utterance_cancelled()

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

  async def flush(self, *, force_complete: bool = True) -> TranscriptionMessage:
    """Request a live transcription flush and await the terminal flush response.

    This API is only valid for active transcriber websocket sessions. The client drains
    stale buffered transcription updates before sending the flush command, then waits for
    the next ``TranscriptionMessage`` whose ``flush_complete`` field is ``True``. Ordinary
    transcription updates received while waiting are ignored.

    :param force_complete: Whether the server should force-complete the current tentative tail.
    :type force_complete: bool
    :returns: The flush-satisfying transcription update from the server.
    :rtype: TranscriptionMessage
    :raises RuntimeError: If flush is unsupported for the current mode/session, another local
      flush is already pending, the server rejects the request, or the live connection closes
      before the flush completes.
    """
    if self._client_type != ClientType.TRANSCRIBER:
      raise RuntimeError("flush() only supported in transcriber mode")

    if self._operation_lock.locked():
      raise RuntimeError("flush() unavailable during transcribe_file() operation")

    if (
      not self._connected
      or not self._connection
      or not self._connection.is_connected()
      or self._message_task is None
    ):
      raise RuntimeError("flush() requires an active live transcriber connection")

    if self._flush_waiting:
      raise RuntimeError("flush() already in progress")

    self._flush_waiting = True
    self._flush_error = None

    try:
      self._clear_message_queue()
      await self._connection.send_flush_control(force_complete=force_complete)

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

        if message.flush_complete is True:
          return message
    finally:
      self._flush_waiting = False
      self._flush_error = None

  async def transcribe_file(
    self,
    file_path: str,
    timeout_s: float | None = None,
  ) -> FileTranscriptionResult:
    """Transcribe a finite local audio file in one non-reentrant operation.

    :param file_path: Path to the local audio file to upload.
    :type file_path: str
    :param timeout_s: Optional per-call timeout in seconds.
    :type timeout_s: float | None
    :returns: Final reduced transcription result for the file.
    :rtype: FileTranscriptionResult
    :raises RuntimeError: If called in non-transcriber mode or another operation is active.
    :raises TimeoutError: If the operation exceeds ``timeout_s``.
    """
    if self._client_type != ClientType.TRANSCRIBER:
      raise RuntimeError("transcribe_file() only supported in transcriber mode")

    if self._streaming or self._connected:
      raise RuntimeError("transcribe_file() requires an idle client instance")

    if self._operation_lock.locked():
      raise RuntimeError("transcribe_file() operation already in progress")

    target_path = Path(file_path)
    file_bytes = target_path.read_bytes()

    async with self._operation_lock:
      operation = self._transcribe_file_operation(file_bytes=file_bytes)

      try:
        if timeout_s is None:
          return await operation
        return await asyncio.wait_for(operation, timeout=timeout_s)
      except asyncio.TimeoutError as exc:
        raise TimeoutError(f"transcribe_file() timed out after {timeout_s}s") from exc
      except asyncio.CancelledError:
        raise
      except Exception:
        self._logger.exception("transcribe_file failed", file_path=file_path)
        raise
      finally:
        await self.disconnect()

  async def _transcribe_file_operation(self, file_bytes: bytes) -> FileTranscriptionResult:
    """Execute connect/upload/reduce/finalize steps for one finite-file transcription."""
    self._clear_message_queue()
    self._disconnect_event.clear()
    self._disconnect_reason = None

    setup_options = self._transcription_options.model_copy(
      update={"source_mode": TranscriptionSourceMode.FILE}
    )
    await self.connect(setup_options=setup_options)

    if not self._connection:
      raise RuntimeError("Connection not initialized for transcribe_file operation")

    await self._connection.send_file_bytes(file_bytes)
    await self._connection.send_end_of_audio()

    return await self._collect_file_result()

  async def _collect_file_result(self) -> FileTranscriptionResult:
    """Collect transcription windows until disconnect or socket-close terminal state."""
    reduced_segments: list[Segment] = []
    warnings: list[str] = []
    language: str | None = None
    last_committed_id: int | None = None

    while True:
      if self._disconnect_event.is_set():
        break

      if self._message_task and self._message_task.done() and self._message_queue.empty():
        break

      try:
        message = await asyncio.wait_for(self._message_queue.get(), timeout=0.1)
      except asyncio.TimeoutError:
        continue

      if message.language is not None:
        language = message.language

      last_committed_id = self._reduce_windowed_segments(
        message=message,
        reduced_segments=reduced_segments,
        last_committed_id=last_committed_id,
        warnings=warnings,
      )

    text = " ".join(segment.text.strip() for segment in reduced_segments if segment.text.strip())
    return FileTranscriptionResult(
      segments=reduced_segments,
      text=text,
      language=language,
      warnings=warnings,
    )

  def _reduce_windowed_segments(
    self,
    message: TranscriptionMessage,
    reduced_segments: list[Segment],
    last_committed_id: int | None,
    warnings: list[str],
  ) -> int | None:
    """Reduce completed-window emissions to new committed segments only."""
    if not message.segments:
      return last_committed_id

    completed_segments = message.segments[:-1]
    if not completed_segments:
      return last_committed_id

    if last_committed_id is None:
      new_segments = completed_segments
    else:
      sentinel_index: int | None = None
      for idx in range(len(completed_segments) - 1, -1, -1):
        if completed_segments[idx].id == last_committed_id:
          sentinel_index = idx
          break

      if sentinel_index is None:
        warning = (
          f"Reducer sentinel missing for stream={message.stream}: "
          f"last_committed_id={last_committed_id}"
        )
        warnings.append(warning)
        self._logger.warning(
          "reducer sentinel missing",
          stream=message.stream,
          last_committed_id=last_committed_id,
        )
        new_segments = completed_segments
      else:
        new_segments = completed_segments[sentinel_index + 1 :]

    if not new_segments:
      return last_committed_id

    reduced_segments.extend(new_segments)
    return new_segments[-1].id

  def _clear_message_queue(self) -> None:
    """Drain stale buffered messages before starting a one-shot file operation."""
    while not self._message_queue.empty():
      self._message_queue.get_nowait()

  def _clear_event_queue(self) -> None:
    """Drain buffered iterator events when the client is explicitly closed."""
    while not self._event_queue.empty():
      self._event_queue.get_nowait()

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
    """Check if client is actively streaming (transcriber mode only)."""
    if self._client_type != ClientType.TRANSCRIBER:
      return False
    return self._streaming

  # Async iterator protocol
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

  # Async context manager protocol
  async def __aenter__(self) -> "EavesdropClient":
    """Enter async context manager."""
    await self.connect()
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    """Exit async context manager."""
    await self.disconnect()

  # Callback handlers
  def _on_ready(self, backend: str) -> None:
    """Handle server ready callback."""
    # Server is ready for transcription
    pass

  def _on_transcription_message(self, message: TranscriptionMessage) -> None:
    """Handle full TranscriptionMessage from connection."""
    self._message_queue.put_nowait(message)
    self._emit_event(TranscriptionEvent(stream=message.stream, message=message))

  def _on_transcription_text(self, text: str) -> None:
    """Handle legacy transcription text callback from existing connection."""
    # This is primarily for backward compatibility
    # The main message flow now uses _on_transcription_message
    pass

  def _on_disconnect(self, reason: str | None) -> None:
    """Handle terminal disconnect signaling from the server."""
    self._disconnect_reason = reason
    self._disconnect_event.set()

    if self._flush_waiting and self._flush_error is None:
      self._flush_error = self._flush_disconnect_message()

    if self._client_type != ClientType.TRANSCRIBER or not self._reconnect_enabled:
      return

    if self._disconnect_requested:
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
