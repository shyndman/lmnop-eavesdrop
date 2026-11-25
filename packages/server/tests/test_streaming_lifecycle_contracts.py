"""Deterministic lifecycle contract tests for ``WebSocketStreamingClient``.

These tests pin orchestration guarantees around completion handling:
- EOF ingestion drives completion and teardown.
- Pending task cancellation happens before stop orchestration.
- Stop processing/disconnect occurs before source shutdown.
- Active-task teardown remains bounded.
- Live recording boundaries reset buffer/session zero, making subsequent
  buffer offsets recording-relative for the active recording.
"""

import asyncio
from collections.abc import Awaitable, Callable
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
  PRE_SETUP_RECORDING_START_REJECTION,
  PRE_SETUP_UTTERANCE_CANCEL_REJECTION,
  WebSocketConnectionHandler,
)
from eavesdrop.server.streaming.audio_flow import WebSocketTranscriptionSink
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.client import (
  LIVE_FLUSH_ALREADY_PENDING_MESSAGE,
  LIVE_RECORDING_ID_MISMATCH_MESSAGE,
  LIVE_RECORDING_START_REQUIRED_MESSAGE,
  StreamingProcessorLike,
  WebSocketStreamingClient,
)
from eavesdrop.server.streaming.flush_state import LiveSessionFlushState
from eavesdrop.server.streaming.interfaces import TranscriptionResult
from eavesdrop.server.streaming.processor import StreamingTranscriptionProcessor
from eavesdrop.server.transcription.session import create_session
from eavesdrop.wire import (
  FlushControlMessage,
  RecordingStartedMessage,
  Segment,
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


def _async_mock(*, side_effect: object = None, return_value: object = None) -> _AsyncMockAssertions:
  return cast(
    _AsyncMockAssertions,
    AsyncMock(side_effect=side_effect, return_value=return_value),
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
class _ProcessorDouble:
  initialize_calls: int = 0
  stop_calls: int = 0
  on_start: Callable[[], Awaitable[None]] = field(default=lambda: _completed())
  on_stop: Callable[[], Awaitable[None]] | None = None
  reset_calls: list[tuple[str | None, int, bool]] = field(default_factory=list)
  language: str | None = "en"

  async def initialize(self) -> None:
    self.initialize_calls += 1

  async def start_processing(self) -> None:
    await self.on_start()

  async def stop_processing(self) -> None:
    self.stop_calls += 1
    if self.on_stop is not None:
      await self.on_stop()

  def add_audio_frames(self, frames: Float32Audio) -> None:
    del frames

  def reset_live_recording(
    self,
    *,
    recording_id: str | None,
    generation: int,
    preserve_language: bool = True,
  ) -> None:
    self.reset_calls.append((recording_id, generation, preserve_language))


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


def _assert_awaited_once_with(mock: object, *args: object) -> None:
  cast(_AsyncMockAssertions, mock).assert_awaited_once_with(*args)


def _assert_not_awaited(mock: object) -> None:
  cast(_AsyncMockAssertions, mock).assert_not_awaited()


def _build_client(*, processor: StreamingProcessorLike) -> WebSocketStreamingClient:
  """Create a minimally wired client instance for lifecycle orchestration tests."""
  client = WebSocketStreamingClient.__new__(WebSocketStreamingClient)
  client.websocket = MagicMock()
  client.stream_name = "stream-1"
  client.logger = MagicMock()
  client.session = MagicMock()
  client.buffer = MagicMock()
  client.transcription_sink = cast(WebSocketTranscriptionSink, MagicMock())
  client.processor = processor
  client._flush_state = LiveSessionFlushState()
  client._processing_task = None
  client._audio_task = None
  client._completion_task = None
  client._exit = False
  client._stopped = False
  client._current_recording_id = None
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


async def test_ingestion_eof_completes_and_tears_down_cleanly() -> None:
  """EOF from audio ingestion must complete orchestration and stop the client exactly once."""
  processing_cancelled = asyncio.Event()

  async def _start_processing() -> None:
    await _run_until_cancelled(processing_cancelled)

  processor = _ProcessorDouble(on_start=_start_processing)

  client = _build_client(processor=processor)
  client.websocket.recv = AsyncMock(return_value=b"END_OF_AUDIO")

  completion_task = await client.start()
  await asyncio.wait_for(completion_task, timeout=0.5)

  assert processing_cancelled.is_set()
  assert processor.initialize_calls == 1
  assert processor.stop_calls == 1
  assert client._exit is True


async def test_completion_cancels_pending_task_before_stop(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Completion wait must cancel the still-running task before invoking stop orchestration."""
  pending_cancelled = asyncio.Event()

  processor = _ProcessorDouble()

  client = _build_client(processor=processor)
  client._audio_task = asyncio.create_task(asyncio.sleep(0))
  client._processing_task = asyncio.create_task(_run_until_cancelled(pending_cancelled))

  async def _stop_guard() -> None:
    assert pending_cancelled.is_set()

  stop_mock = AsyncMock(side_effect=_stop_guard)
  monkeypatch.setattr(client, "stop", stop_mock)

  await asyncio.wait_for(client._wait_for_completion(), timeout=0.5)

  assert client._processing_task.cancelled()
  stop_mock.assert_awaited_once()


async def test_completion_path_runs_stop_processing_and_disconnect() -> None:
  """Completion-triggered stop must call stop_processing which disconnects the sink."""
  events: list[str] = []

  async def _disconnect() -> None:
    events.append("disconnect")

  sink = _SinkDouble(disconnect=_async_mock(side_effect=_disconnect), send_error=_async_mock())

  async def _stop_processing() -> None:
    events.append("stop_processing")
    await sink.disconnect()

  processor = _ProcessorDouble(on_stop=_stop_processing)

  client = _build_client(processor=processor)
  client.transcription_sink = cast(WebSocketTranscriptionSink, cast(object, sink))
  client._audio_task = asyncio.create_task(asyncio.sleep(0))
  client._processing_task = asyncio.create_task(asyncio.sleep(3600))

  await asyncio.wait_for(client._wait_for_completion(), timeout=0.5)

  assert events == ["stop_processing", "disconnect"]
  assert processor.stop_calls == 1


async def test_stop_during_active_processing_is_bounded() -> None:
  """Stopping during active tasks must complete quickly while cancelling both running loops."""
  processing_cancelled = asyncio.Event()
  audio_cancelled = asyncio.Event()

  processor = _ProcessorDouble()

  client = _build_client(processor=processor)
  client._processing_task = asyncio.create_task(_run_until_cancelled(processing_cancelled))
  client._audio_task = asyncio.create_task(_run_until_cancelled(audio_cancelled))

  await asyncio.sleep(0)
  await asyncio.wait_for(client.stop(), timeout=0.5)

  assert processing_cancelled.is_set()
  assert audio_cancelled.is_set()
  assert client._processing_task.cancelled()
  assert client._audio_task.cancelled()
  assert processor.stop_calls == 1


@pytest.mark.asyncio
async def test_pending_flush_interrupts_minimum_chunk_wait() -> None:
  """Accepted flushes must wake the minimum-chunk wait so partial buffered audio can run."""
  flush_state = LiveSessionFlushState()
  buffer = AudioStreamBuffer(
    BufferConfig(sample_rate=16000, min_chunk_duration=0.5, transcription_interval=0.5)
  )
  buffer.add_frames(np.zeros(1600, dtype=np.float32))
  processor = _build_processor(buffer=buffer, flush_state=flush_state)

  wait_task = asyncio.create_task(processor._get_next_audio_chunk())
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
async def test_minimum_chunk_wait_logs_once_per_processor(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Repeated below-threshold polling must not spam the minimum-wait info log."""
  buffer = AudioStreamBuffer(
    BufferConfig(sample_rate=16000, min_chunk_duration=0.5, transcription_interval=0.5)
  )
  buffer.add_frames(np.zeros(1600, dtype=np.float32))
  processor = _build_processor(buffer=buffer, flush_state=LiveSessionFlushState())
  logger = _LoggerDouble()
  monkeypatch.setattr(processor, "logger", logger)
  monkeypatch.setattr(processor, "_wait_for_live_control_wakeup", AsyncMock(return_value=True))

  _ = await processor._get_next_audio_chunk()
  _ = await processor._get_next_audio_chunk()

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

  wait_task = asyncio.create_task(processor._wait_for_next_interval(0.0))
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
  sink = _SinkDouble(disconnect=AsyncMock(), send_error=AsyncMock())
  client.transcription_sink = cast(WebSocketTranscriptionSink, cast(object, sink))
  client._flush_state = LiveSessionFlushState()
  client._current_recording_id = "rec-1"

  first_flush = serialize_message(
    FlushControlMessage(stream="stream-1", recording_id="rec-1", force_complete=False)
  )
  second_flush = serialize_message(
    FlushControlMessage(stream="stream-1", recording_id="rec-1", force_complete=True)
  )

  await client._handle_live_text_frame(first_flush)
  pending_flush = client._flush_state.pending()
  await client._handle_live_text_frame(second_flush)

  assert pending_flush is not None
  assert client._flush_state.pending() == pending_flush
  _assert_awaited_once_with(sink.send_error, LIVE_FLUSH_ALREADY_PENDING_MESSAGE)


@pytest.mark.asyncio
async def test_recording_boundary_resets_live_buffer_session_and_flush_state() -> None:
  """A new live boundary must reset server state to recording-relative zero."""
  client = WebSocketStreamingClient.__new__(WebSocketStreamingClient)
  client.logger = MagicMock()
  client.buffer = AudioStreamBuffer(BufferConfig(sample_rate=10, min_chunk_duration=1.0))
  client.session = create_session("stream-1")
  client.session.update_audio_context(3.0, 0.4)
  sink = _SinkDouble(disconnect=AsyncMock(), send_error=AsyncMock())
  client.transcription_sink = cast(WebSocketTranscriptionSink, cast(object, sink))
  client._flush_state = LiveSessionFlushState()
  processor_double = _ProcessorDouble()
  client.processor = processor_double
  client._current_recording_id = None

  client.session.add_completed_segment(
    Segment(
      id=1,
      start=0.0,
      end=0.2,
      text="alpha",
      words=None,
      completed=True,
    )
  )
  client.buffer.add_frames(np.ones(8, dtype=np.float32))
  client.buffer.advance_processed_boundary(0.3)
  pending_flush = client._flush_state.accept(
    boundary_sample=client.buffer.get_buffer_end_sample(),
    force_complete=True,
  )

  assert pending_flush is not None

  await client._handle_live_text_frame(
    serialize_message(
      RecordingStartedMessage(
        stream="stream-1",
        recording_id="rec-2",
        sample_rate_hz=10,
      )
    ),
  )

  assert client._current_recording_id == "rec-2"
  assert client.buffer.frames_np is None
  assert client.buffer.buffer_start_time == 0.0
  assert client.buffer.processed_up_to_time == 0.0
  assert client.session.completed_segments == []
  assert client.session.audio_start_offset == 0.0
  assert client.session.audio_duration == 0.0
  assert client._flush_state.pending() is None
  assert client._flush_state.current_generation() == 1
  assert processor_double.reset_calls == [("rec-2", 1, False)]


@pytest.mark.asyncio
async def test_live_controls_require_recording_boundary_before_epoch_exists() -> None:
  """Flush and cancel controls must be rejected until a recording boundary is established."""
  client = WebSocketStreamingClient.__new__(WebSocketStreamingClient)
  sink = _SinkDouble(disconnect=AsyncMock(), send_error=AsyncMock())
  client.transcription_sink = cast(WebSocketTranscriptionSink, cast(object, sink))
  client._flush_state = LiveSessionFlushState()
  client._current_recording_id = None

  await client._handle_live_text_frame(
    serialize_message(FlushControlMessage(stream="stream-1", recording_id="rec-1")),
  )
  await client._handle_live_text_frame(
    serialize_message(UtteranceCancelledMessage(stream="stream-1", recording_id="rec-1")),
  )

  assert sink.send_error.await_count == 2
  cast(AsyncMock, sink.send_error).assert_any_await(LIVE_RECORDING_START_REQUIRED_MESSAGE)


@pytest.mark.asyncio
async def test_live_controls_reject_mismatched_recording_id() -> None:
  """Live controls must stay scoped to the active recording epoch."""
  client = WebSocketStreamingClient.__new__(WebSocketStreamingClient)
  sink = _SinkDouble(disconnect=AsyncMock(), send_error=AsyncMock())
  client.transcription_sink = cast(WebSocketTranscriptionSink, cast(object, sink))
  client._flush_state = LiveSessionFlushState()
  client._current_recording_id = "rec-active"

  await client._handle_live_text_frame(
    serialize_message(FlushControlMessage(stream="stream-1", recording_id="rec-stale")),
  )

  _assert_awaited_once_with(sink.send_error, LIVE_RECORDING_ID_MISMATCH_MESSAGE)


@pytest.mark.asyncio
async def test_live_utterance_cancel_discards_tail_without_merging_future_audio() -> None:
  """Accepted cancel control must drop the tail and preserve the monotonic processed cursor."""
  client = WebSocketStreamingClient.__new__(WebSocketStreamingClient)
  client.logger = MagicMock()
  client.buffer = AudioStreamBuffer(BufferConfig(sample_rate=10, min_chunk_duration=1.0))
  client.session = create_session("stream-1")
  client.session.add_completed_segment(
    Segment(
      id=1,
      start=0.0,
      end=0.2,
      text="alpha",
      words=None,
      completed=True,
    )
  )
  sink = _SinkDouble(disconnect=AsyncMock(), send_error=AsyncMock())
  client.transcription_sink = cast(WebSocketTranscriptionSink, cast(object, sink))
  client._flush_state = LiveSessionFlushState()
  client._current_recording_id = "rec-1"
  processor_double = _ProcessorDouble()
  client.processor = processor_double

  client.buffer.add_frames(np.ones(8, dtype=np.float32))
  client.buffer.advance_processed_boundary(0.3)
  pending_flush = client._flush_state.accept(
    boundary_sample=client.buffer.get_buffer_end_sample(),
    force_complete=True,
  )

  assert pending_flush is not None

  await client._handle_live_text_frame(
    serialize_message(UtteranceCancelledMessage(stream="stream-1", recording_id="rec-1")),
  )

  assert abs(client.buffer.processed_up_to_time - 0.3) < 1e-9
  assert abs(client.buffer.buffer_start_time - 0.3) < 1e-9
  assert client.buffer.available_duration == 0.0
  assert client.buffer.total_duration == 0.0
  assert client.session.completed_segments == []
  assert client._flush_state.pending() is None
  assert client._flush_state.current_generation() == 1
  assert processor_double.reset_calls == [("rec-1", 1, False)]

  client.buffer.add_frames(np.full(4, 0.5, dtype=np.float32))
  chunk, duration, start_time = client.buffer.get_chunk_for_processing()

  assert chunk.shape[0] == 4
  assert abs(duration - 0.4) < 1e-9
  assert abs(start_time - 0.3) < 1e-9
  _assert_not_awaited(sink.send_error)


@pytest.mark.asyncio
@pytest.mark.parametrize(
  ("message", "expected_rejection"),
  [
    (
      RecordingStartedMessage(stream="stream-1", recording_id="rec-1", sample_rate_hz=16000),
      PRE_SETUP_RECORDING_START_REJECTION,
    ),
    (FlushControlMessage(stream="stream-1"), PRE_SETUP_FLUSH_REJECTION),
    (
      UtteranceCancelledMessage(stream="stream-1"),
      PRE_SETUP_UTTERANCE_CANCEL_REJECTION,
    ),
  ],
)
async def test_pre_setup_live_controls_are_rejected_without_closing_connection(
  message: RecordingStartedMessage | FlushControlMessage | UtteranceCancelledMessage,
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


def test_recording_boundary_resets_chunk_start_to_zero_for_next_recording() -> None:
  """Each accepted recording boundary must restart chunk timestamps from recording sample zero."""
  client = WebSocketStreamingClient.__new__(WebSocketStreamingClient)
  client.logger = MagicMock()
  client.buffer = AudioStreamBuffer(BufferConfig(sample_rate=10, min_chunk_duration=1.0))
  client.session = create_session("stream-1")
  client._flush_state = LiveSessionFlushState()
  client.processor = cast(StreamingProcessorLike, MagicMock())
  client._current_recording_id = None

  client._accept_recording_boundary(
    RecordingStartedMessage(stream="stream-1", recording_id="rec-1", sample_rate_hz=10)
  )
  client.buffer.add_frames(np.ones(4, dtype=np.float32))
  _, _, first_start = client.buffer.get_chunk_for_processing()
  client.buffer.advance_processed_boundary(0.4)

  client._accept_recording_boundary(
    RecordingStartedMessage(stream="stream-1", recording_id="rec-2", sample_rate_hz=10)
  )
  client.buffer.add_frames(np.ones(4, dtype=np.float32))
  _, _, second_start = client.buffer.get_chunk_for_processing()

  assert first_start == 0.0
  assert second_start == 0.0
