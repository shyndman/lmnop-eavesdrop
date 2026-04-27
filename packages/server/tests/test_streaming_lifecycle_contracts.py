"""Deterministic lifecycle contract tests for ``WebSocketStreamingClient``.

These tests pin orchestration guarantees around completion handling:
- EOF ingestion drives completion and teardown.
- Pending task cancellation happens before stop orchestration.
- Stop processing/disconnect occurs before source shutdown.
- Active-task teardown remains bounded.
"""

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Protocol, cast
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray
from websockets.asyncio.server import ServerConnection
from websockets.datastructures import Headers

from eavesdrop.server.config import BufferConfig, TranscriptionConfig
from eavesdrop.server.connection_handler import (
  PRE_SETUP_FLUSH_REJECTION,
  PRE_SETUP_UTTERANCE_CANCEL_REJECTION,
  WebSocketConnectionHandler,
)
from eavesdrop.server.server import (
  RTSP_FLUSH_REJECTION,
  RTSP_UTTERANCE_CANCEL_REJECTION,
  TranscriptionServer,
)
from eavesdrop.server.streaming.audio_flow import WebSocketAudioSource
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.client import (
  LIVE_FLUSH_ALREADY_PENDING_MESSAGE,
  LIVE_FLUSH_FILE_MODE_MESSAGE,
  LIVE_UTTERANCE_CANCEL_FILE_MODE_MESSAGE,
  WebSocketStreamingClient,
)
from eavesdrop.server.streaming.flush_state import LiveSessionFlushState
from eavesdrop.server.streaming.interfaces import TranscriptionResult
from eavesdrop.server.streaming.processor import AudioChunk, StreamingTranscriptionProcessor
from eavesdrop.server.transcription.session import create_session
from eavesdrop.wire import (
  FlushControlMessage,
  TranscriptionSetupMessage,
  UserTranscriptionOptions,
  UtteranceCancelledMessage,
  deserialize_message,
  serialize_message,
)

Float32Audio = NDArray[np.float32]


class _AsyncMockAssertions(Protocol):
  await_count: int

  def __call__(self, *args: object, **kwargs: object) -> Awaitable[None]: ...

  def assert_awaited_once(self) -> None: ...

  def assert_awaited_once_with(self, *args: object, **kwargs: object) -> None: ...

  def assert_not_awaited(self) -> None: ...


class _MagicMockAssertions(Protocol):
  def __call__(self, *args: object, **kwargs: object) -> object: ...

  def assert_called_once(self) -> None: ...


def _async_mock(*, side_effect: object = None, return_value: object = None) -> _AsyncMockAssertions:
  return cast(
    _AsyncMockAssertions,
    AsyncMock(side_effect=side_effect, return_value=return_value),
  )


def _magic_mock(*, side_effect: object = None, return_value: object = None) -> _MagicMockAssertions:
  return cast(
    _MagicMockAssertions,
    MagicMock(side_effect=side_effect, return_value=return_value),
  )


@dataclass
class _InfoCall:
  args: tuple[str, ...]


@dataclass
class _LoggerRecorder:
  call_args_list: list[_InfoCall] = field(default_factory=list)

  def __call__(self, message: str, **_kwargs: object) -> None:
    self.call_args_list.append(_InfoCall(args=(message,)))


@dataclass
class _LoggerDouble:
  debug: _LoggerRecorder = field(default_factory=_LoggerRecorder)
  info: _LoggerRecorder = field(default_factory=_LoggerRecorder)


class _FakeRequest:
  """Minimal request object exposing websocket headers for routing tests."""

  def __init__(self, headers: dict[str, str] | None = None) -> None:
    self.headers: Headers = Headers(headers or {})


class _SequentialRecvWebSocket:
  """Minimal websocket double with ordered recv results and send capture."""

  def __init__(self, responses: list[str | bytes], headers: dict[str, str] | None = None) -> None:
    self._responses: list[str | bytes] = list(responses)
    self.request: _FakeRequest = _FakeRequest(headers)
    self.sent_payloads: list[str] = []
    self.close: AsyncMock = AsyncMock()

  async def recv(self, decode: bool = False) -> str | bytes:
    if not self._responses:
      raise RuntimeError("No more queued websocket responses")
    response = self._responses.pop(0)
    if decode and isinstance(response, bytes):
      return response.decode("utf-8")
    return response

  async def send(self, payload: str) -> None:
    self.sent_payloads.append(payload)


@dataclass
class _SinkDouble:
  disconnect: _AsyncMockAssertions
  send_error: _AsyncMockAssertions


@dataclass
class _AudioSourceDouble:
  close: _MagicMockAssertions


@dataclass
class _ProcessorDouble:
  initialize: _AsyncMockAssertions = field(default_factory=_async_mock)
  stop_processing: _AsyncMockAssertions = field(default_factory=_async_mock)
  start_processing: Callable[[], Awaitable[None]] = field(default=lambda: _completed())


async def _completed() -> None:
  return


class _NoopSink:
  """Minimal sink double for processor wait-interruption tests."""

  async def send_result(self, result: TranscriptionResult) -> None:
    del result
    return

  async def send_error(self, error: str) -> None:
    del error
    return

  async def send_language_detection(self, language: str, probability: float) -> None:
    del language, probability
    return

  async def send_server_ready(self, backend: str) -> None:
    del backend
    return

  async def disconnect(self) -> None:
    return


async def _run_until_cancelled(cancelled_event: asyncio.Event) -> None:
  """Block forever until cancellation, then signal deterministic cancellation observation."""
  try:
    await asyncio.Future()
  except asyncio.CancelledError:
    cancelled_event.set()
    raise


def _as_server_connection(websocket: _SequentialRecvWebSocket) -> ServerConnection:
  return cast(ServerConnection, cast(object, websocket))


def _assert_awaited_once(mock: object) -> None:
  cast(_AsyncMockAssertions, mock).assert_awaited_once()


def _assert_awaited_once_with(mock: object, *args: object) -> None:
  cast(_AsyncMockAssertions, mock).assert_awaited_once_with(*args)


def _assert_not_awaited(mock: object) -> None:
  cast(_AsyncMockAssertions, mock).assert_not_awaited()


def _assert_called_once(mock: object) -> None:
  cast(_MagicMockAssertions, mock).assert_called_once()


def _set_attr(target: object, name: str, value: object) -> None:
  setattr(target, name, value)


def _get_bool_attr(target: object, name: str) -> bool:
  return cast(bool, getattr(target, name))


def _get_task_attr(target: object, name: str) -> asyncio.Task[None]:
  return cast(asyncio.Task[None], getattr(target, name))


def _get_flush_state(client: WebSocketStreamingClient) -> LiveSessionFlushState:
  return cast(LiveSessionFlushState, getattr(client, "_flush_state"))


def _build_client(*, processor: object, audio_source: object) -> WebSocketStreamingClient:
  """Create a minimally wired client instance for lifecycle orchestration tests."""
  client = WebSocketStreamingClient.__new__(WebSocketStreamingClient)
  client.websocket = MagicMock()
  client.stream_name = "stream-1"
  client.logger = MagicMock()
  client.session = MagicMock()
  client.buffer = MagicMock()
  client.transcription_sink = MagicMock()
  client.processor = cast(StreamingTranscriptionProcessor, processor)
  client.audio_source = cast(WebSocketAudioSource, audio_source)
  _set_attr(client, "_flush_state", LiveSessionFlushState())
  _set_attr(client, "_processing_task", None)
  _set_attr(client, "_audio_task", None)
  _set_attr(client, "_completion_task", None)
  _set_attr(client, "_exit", False)
  _set_attr(client, "_stopped", False)
  return client


def _build_processor(
  *,
  buffer: AudioStreamBuffer,
  flush_state: LiveSessionFlushState,
) -> StreamingTranscriptionProcessor:
  """Create a processor wired only with in-memory collaborators for wait tests."""
  return StreamingTranscriptionProcessor(
    buffer=buffer,
    sink=_NoopSink(),
    config=TranscriptionConfig(silence_completion_threshold=0.8),
    session=create_session("stream-1"),
    stream_name="stream-1",
    flush_state=flush_state,
  )


def _add_frames(buffer: AudioStreamBuffer, frames: Float32Audio) -> None:
  cast(Callable[[Float32Audio], None], buffer.add_frames)(frames)


def _get_chunk(buffer: AudioStreamBuffer) -> tuple[Float32Audio, float, float]:
  return buffer.get_chunk_for_processing()


def _set_processor_wait_for_flush_wakeup(
  processor: StreamingTranscriptionProcessor,
  mock: AsyncMock,
) -> None:
  _set_attr(processor, "_wait_for_flush_wakeup", mock)


def _processor_get_next_audio_chunk(
  processor: StreamingTranscriptionProcessor,
) -> Coroutine[object, object, AudioChunk | None]:
  return cast(
    Callable[[], Coroutine[object, object, AudioChunk | None]],
    getattr(processor, "_get_next_audio_chunk"),
  )()


def _processor_wait_for_next_interval(
  processor: StreamingTranscriptionProcessor,
  start_time: float,
) -> Coroutine[object, object, None]:
  wait_for_next_interval = cast(
    Callable[[float], Coroutine[object, object, None]],
    getattr(processor, "_wait_for_next_interval"),
  )
  return wait_for_next_interval(start_time)


def _client_wait_for_completion(client: WebSocketStreamingClient) -> Awaitable[None]:
  return cast(
    Callable[[], Coroutine[object, object, None]], getattr(client, "_wait_for_completion")
  )()


def _client_handle_live_text_frame(
  client: WebSocketStreamingClient, payload: str
) -> Awaitable[None]:
  return cast(
    Callable[[str], Coroutine[object, object, None]],
    getattr(client, "_handle_live_text_frame"),
  )(payload)


def _client_handle_file_text_frame(
  client: WebSocketStreamingClient, payload: str
) -> Awaitable[None]:
  return cast(
    Callable[[str], Coroutine[object, object, None]],
    getattr(client, "_handle_file_text_frame"),
  )(payload)


def _server_reject_subscriber_control_frame(
  server: TranscriptionServer,
  websocket: _SequentialRecvWebSocket,
  payload: str,
) -> Coroutine[object, object, None]:
  reject_frame = cast(
    Callable[[ServerConnection, str], Coroutine[object, object, None]],
    getattr(server, "_reject_subscriber_control_frame"),
  )
  return reject_frame(_as_server_connection(websocket), payload)


async def test_ingestion_eof_completes_and_tears_down_cleanly() -> None:
  """EOF from audio ingestion must complete orchestration and stop the client exactly once."""
  processing_cancelled = asyncio.Event()

  processor = _ProcessorDouble()

  async def _start_processing() -> None:
    await _run_until_cancelled(processing_cancelled)

  processor.start_processing = _start_processing
  audio_source = _AudioSourceDouble(close=MagicMock())

  client = _build_client(processor=processor, audio_source=audio_source)
  client.websocket.recv = AsyncMock(return_value=b"END_OF_AUDIO")

  completion_task = await client.start()
  await asyncio.wait_for(completion_task, timeout=0.5)

  assert processing_cancelled.is_set()
  _assert_awaited_once(processor.initialize)
  _assert_awaited_once(processor.stop_processing)
  _assert_called_once(audio_source.close)
  assert _get_bool_attr(client, "_exit") is True


async def test_completion_cancels_pending_task_before_stop() -> None:
  """Completion wait must cancel the still-running task before invoking stop orchestration."""
  pending_cancelled = asyncio.Event()

  processor = _ProcessorDouble()
  audio_source = _AudioSourceDouble(close=MagicMock())

  client = _build_client(processor=processor, audio_source=audio_source)
  _set_attr(client, "_audio_task", asyncio.create_task(asyncio.sleep(0)))
  _set_attr(
    client, "_processing_task", asyncio.create_task(_run_until_cancelled(pending_cancelled))
  )

  async def _stop_guard() -> None:
    assert pending_cancelled.is_set()

  client.stop = AsyncMock(side_effect=_stop_guard)

  await asyncio.wait_for(_client_wait_for_completion(client), timeout=0.5)

  assert _get_task_attr(client, "_processing_task").cancelled()
  _assert_awaited_once(client.stop)


async def test_completion_path_runs_disconnect_before_source_close() -> None:
  """Completion-triggered stop must disconnect processor output before closing audio source."""
  events: list[str] = []

  async def _disconnect() -> None:
    events.append("disconnect")

  sink = _SinkDouble(disconnect=_async_mock(side_effect=_disconnect), send_error=_async_mock())
  processor = _ProcessorDouble()

  async def _stop_processing() -> None:
    events.append("stop_processing")
    await sink.disconnect()

  processor.stop_processing = _async_mock(side_effect=_stop_processing)
  audio_source = _AudioSourceDouble(
    close=_magic_mock(side_effect=lambda: events.append("audio_close"))
  )

  client = _build_client(processor=processor, audio_source=audio_source)
  _set_attr(client, "transcription_sink", sink)
  _set_attr(client, "_audio_task", asyncio.create_task(asyncio.sleep(0)))
  _set_attr(client, "_processing_task", asyncio.create_task(asyncio.sleep(3600)))

  await asyncio.wait_for(_client_wait_for_completion(client), timeout=0.5)

  assert events == ["stop_processing", "disconnect", "audio_close"]
  _assert_awaited_once(processor.stop_processing)
  _assert_called_once(audio_source.close)


async def test_stop_during_active_processing_is_bounded() -> None:
  """Stopping during active tasks must complete quickly while cancelling both running loops."""
  processing_cancelled = asyncio.Event()
  audio_cancelled = asyncio.Event()

  processor = _ProcessorDouble()
  audio_source = _AudioSourceDouble(close=MagicMock())

  client = _build_client(processor=processor, audio_source=audio_source)
  _set_attr(
    client, "_processing_task", asyncio.create_task(_run_until_cancelled(processing_cancelled))
  )
  _set_attr(client, "_audio_task", asyncio.create_task(_run_until_cancelled(audio_cancelled)))

  await asyncio.sleep(0)
  await asyncio.wait_for(client.stop(), timeout=0.5)

  assert processing_cancelled.is_set()
  assert audio_cancelled.is_set()
  assert _get_task_attr(client, "_processing_task").cancelled()
  assert _get_task_attr(client, "_audio_task").cancelled()
  _assert_awaited_once(processor.stop_processing)
  _assert_called_once(audio_source.close)


@pytest.mark.asyncio
async def test_pending_flush_interrupts_minimum_chunk_wait() -> None:
  """Accepted flushes must wake the minimum-chunk wait so partial buffered audio can run."""
  flush_state = LiveSessionFlushState()
  buffer = AudioStreamBuffer(
    BufferConfig(sample_rate=16000, min_chunk_duration=0.5, transcription_interval=0.5)
  )
  _add_frames(buffer, np.zeros(1600, dtype=np.float32))
  processor = _build_processor(buffer=buffer, flush_state=flush_state)

  wait_task = asyncio.create_task(_processor_get_next_audio_chunk(processor))
  await asyncio.sleep(0)

  accepted = flush_state.accept(
    boundary_sample=buffer.get_buffer_end_sample(),
    force_complete=True,
  )

  chunk = await asyncio.wait_for(wait_task, timeout=0.1)

  assert accepted is not None
  assert chunk is not None
  assert abs(chunk.duration - 0.1) < 1e-9


@pytest.mark.asyncio
async def test_minimum_chunk_wait_logs_once_per_processor() -> None:
  """Repeated below-threshold polling must not spam the minimum-wait info log."""
  buffer = AudioStreamBuffer(
    BufferConfig(sample_rate=16000, min_chunk_duration=0.5, transcription_interval=0.5)
  )
  _add_frames(buffer, np.zeros(1600, dtype=np.float32))
  processor = _build_processor(buffer=buffer, flush_state=LiveSessionFlushState())
  logger = _LoggerDouble()
  _set_attr(processor, "logger", logger)
  _set_processor_wait_for_flush_wakeup(processor, AsyncMock(return_value=True))

  _ = await _processor_get_next_audio_chunk(processor)
  _ = await _processor_get_next_audio_chunk(processor)

  minimum_wait_logs = [
    call
    for call in logger.info.call_args_list
    if call.args and call.args[0] == "Transcription loop minimum chunk wait"
  ]

  assert len(minimum_wait_logs) == 1


@pytest.mark.asyncio
async def test_pending_flush_interrupts_interval_wait() -> None:
  """Accepted flushes must wake the interval wait so boundary coverage can continue immediately."""
  flush_state = LiveSessionFlushState()
  buffer = AudioStreamBuffer(
    BufferConfig(sample_rate=16000, min_chunk_duration=0.1, transcription_interval=0.5)
  )
  processor = _build_processor(buffer=buffer, flush_state=flush_state)

  wait_task = asyncio.create_task(_processor_wait_for_next_interval(processor, 0.0))
  await asyncio.sleep(0)

  accepted = flush_state.accept(boundary_sample=0, force_complete=False)
  await asyncio.wait_for(wait_task, timeout=0.1)

  assert accepted is not None


@pytest.mark.asyncio
async def test_second_live_flush_is_rejected_and_original_flush_stays_pending() -> None:
  """A second live flush must fail without replacing the first pending boundary."""
  client = WebSocketStreamingClient.__new__(WebSocketStreamingClient)
  client.logger = MagicMock()
  client.buffer = MagicMock()
  client.buffer.get_buffer_end_sample = MagicMock(side_effect=[3200, 6400])
  _set_attr(
    client,
    "transcription_sink",
    _SinkDouble(disconnect=AsyncMock(), send_error=AsyncMock()),
  )
  _set_attr(client, "_flush_state", LiveSessionFlushState())

  first_flush = serialize_message(FlushControlMessage(stream="stream-1", force_complete=False))
  second_flush = serialize_message(FlushControlMessage(stream="stream-1", force_complete=True))

  await _client_handle_live_text_frame(client, first_flush)
  pending_flush = _get_flush_state(client).pending()
  await _client_handle_live_text_frame(client, second_flush)

  assert pending_flush is not None
  assert _get_flush_state(client).pending() == pending_flush
  _assert_awaited_once_with(
    client.transcription_sink.send_error, LIVE_FLUSH_ALREADY_PENDING_MESSAGE
  )


@pytest.mark.asyncio
async def test_live_utterance_cancel_discards_tail_without_merging_future_audio() -> None:
  """Accepted cancel control must drop the tail and preserve the monotonic processed cursor."""
  client = WebSocketStreamingClient.__new__(WebSocketStreamingClient)
  client.logger = MagicMock()
  client.buffer = AudioStreamBuffer(BufferConfig(sample_rate=10, min_chunk_duration=1.0))
  _set_attr(
    client,
    "transcription_sink",
    _SinkDouble(disconnect=AsyncMock(), send_error=AsyncMock()),
  )
  _set_attr(client, "_flush_state", LiveSessionFlushState())

  _add_frames(client.buffer, np.ones(8, dtype=np.float32))
  client.buffer.advance_processed_boundary(0.3)

  await _client_handle_live_text_frame(
    client,
    serialize_message(UtteranceCancelledMessage(stream="stream-1")),
  )

  assert abs(client.buffer.processed_up_to_time - 0.3) < 1e-9
  assert abs(client.buffer.buffer_start_time - 0.3) < 1e-9
  assert client.buffer.available_duration == 0.0
  assert client.buffer.total_duration == 0.0

  _add_frames(client.buffer, np.full(4, 0.5, dtype=np.float32))
  chunk, duration, start_time = _get_chunk(client.buffer)

  assert chunk.shape[0] == 4
  assert abs(duration - 0.4) < 1e-9
  assert abs(start_time - 0.3) < 1e-9
  _assert_not_awaited(client.transcription_sink.send_error)


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ("message", "expected_rejection"),
  [
    (FlushControlMessage(stream="stream-1"), PRE_SETUP_FLUSH_REJECTION),
    (
      UtteranceCancelledMessage(stream="stream-1"),
      PRE_SETUP_UTTERANCE_CANCEL_REJECTION,
    ),
  ],
)
async def test_pre_setup_live_controls_are_rejected_without_closing_connection(
  message: FlushControlMessage | UtteranceCancelledMessage,
  expected_rejection: str,
) -> None:
  """Live control sent before transcriber setup must be rejected while keeping the socket usable."""
  websocket = _SequentialRecvWebSocket(
    [
      serialize_message(message),
      serialize_message(
        TranscriptionSetupMessage(
          stream="stream-1",
          options=UserTranscriptionOptions(),
        )
      ),
    ]
  )
  initialized_client = MagicMock()
  handler = WebSocketConnectionHandler(
    client_initializer=AsyncMock(return_value=initialized_client)
  )

  result = await handler.handle_connection(_as_server_connection(websocket))

  assert result is not None
  assert websocket.close.await_count == 0
  assert len(websocket.sent_payloads) == 1
  rejection_message = deserialize_message(websocket.sent_payloads[0])
  assert rejection_message.type == "error"
  assert rejection_message.message == expected_rejection


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ("message", "expected_rejection"),
  [
    (FlushControlMessage(stream="stream-1"), LIVE_FLUSH_FILE_MODE_MESSAGE),
    (
      UtteranceCancelledMessage(stream="stream-1"),
      LIVE_UTTERANCE_CANCEL_FILE_MODE_MESSAGE,
    ),
  ],
)
async def test_file_mode_live_controls_are_rejected_without_tearing_down_session(
  message: FlushControlMessage | UtteranceCancelledMessage,
  expected_rejection: str,
) -> None:
  """File-mode uploads must reject live control frames and keep ingest alive."""
  client = WebSocketStreamingClient.__new__(WebSocketStreamingClient)
  _set_attr(
    client,
    "transcription_sink",
    _SinkDouble(disconnect=AsyncMock(), send_error=AsyncMock()),
  )

  await _client_handle_file_text_frame(client, serialize_message(message))

  _assert_awaited_once_with(client.transcription_sink.send_error, expected_rejection)


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ("message", "expected_rejection"),
  [
    (FlushControlMessage(stream="cam-a"), RTSP_FLUSH_REJECTION),
    (
      UtteranceCancelledMessage(stream="cam-a"),
      RTSP_UTTERANCE_CANCEL_REJECTION,
    ),
  ],
)
async def test_subscriber_live_controls_are_rejected_without_closing_connection(
  message: FlushControlMessage | UtteranceCancelledMessage,
  expected_rejection: str,
) -> None:
  """RTSP subscriber sessions must reject live controls and keep the socket open."""
  websocket = _SequentialRecvWebSocket([])
  server = TranscriptionServer()

  await _server_reject_subscriber_control_frame(server, websocket, serialize_message(message))

  assert websocket.close.await_count == 0
  assert len(websocket.sent_payloads) == 1
  rejection_message = deserialize_message(websocket.sent_payloads[0])
  assert rejection_message.type == "error"
  assert rejection_message.message == expected_rejection
