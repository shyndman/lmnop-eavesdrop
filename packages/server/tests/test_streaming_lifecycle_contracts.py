"""Deterministic lifecycle contract tests for ``WebSocketStreamingClient``.

These tests pin orchestration guarantees around completion handling:
- EOF ingestion drives completion and teardown.
- Pending task cancellation happens before stop orchestration.
- Stop processing/disconnect occurs before source shutdown.
- Active-task teardown remains bounded.
"""

import asyncio
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from websockets.asyncio.server import ServerConnection

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
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.client import (
  LIVE_FLUSH_ALREADY_PENDING_MESSAGE,
  LIVE_FLUSH_FILE_MODE_MESSAGE,
  LIVE_UTTERANCE_CANCEL_FILE_MODE_MESSAGE,
  WebSocketStreamingClient,
)
from eavesdrop.server.streaming.flush_state import LiveSessionFlushState
from eavesdrop.server.streaming.interfaces import TranscriptionResult
from eavesdrop.server.streaming.processor import StreamingTranscriptionProcessor
from eavesdrop.server.transcription.session import create_session
from eavesdrop.wire import (
  FlushControlMessage,
  TranscriptionSetupMessage,
  UserTranscriptionOptions,
  UtteranceCancelledMessage,
  deserialize_message,
  serialize_message,
)


class _FakeRequest:
  """Minimal request object exposing websocket headers for routing tests."""

  def __init__(self, headers: dict[str, str] | None = None) -> None:
    self.headers: dict[str, str] = headers or {}


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


def _build_client(*, processor: MagicMock, audio_source: MagicMock) -> WebSocketStreamingClient:
  """Create a minimally wired client instance for lifecycle orchestration tests."""
  client = WebSocketStreamingClient.__new__(WebSocketStreamingClient)
  client.websocket = MagicMock()
  client.stream_name = "stream-1"
  client.logger = MagicMock()
  client.session = MagicMock()
  client.buffer = MagicMock()
  client.transcription_sink = MagicMock()
  client.processor = processor
  client.audio_source = audio_source
  client._flush_state = LiveSessionFlushState()
  client._processing_task = None
  client._audio_task = None
  client._completion_task = None
  client._exit = False
  client._stopped = False
  return client


async def _run_until_cancelled(cancelled_event: asyncio.Event) -> None:
  """Block forever until cancellation, then signal deterministic cancellation observation."""
  try:
    await asyncio.Future()
  except asyncio.CancelledError:
    cancelled_event.set()
    raise


class _NoopSink:
  """Minimal sink double for processor wait-interruption tests."""

  async def send_result(self, result: TranscriptionResult) -> None:
    return

  async def send_error(self, error: str) -> None:
    return

  async def send_language_detection(self, language: str, probability: float) -> None:
    return

  async def send_server_ready(self, backend: str) -> None:
    return

  async def disconnect(self) -> None:
    return


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

  processor = MagicMock()
  processor.initialize = AsyncMock()

  async def _start_processing() -> None:
    await _run_until_cancelled(processing_cancelled)

  processor.start_processing = _start_processing
  processor.stop_processing = AsyncMock()

  audio_source = MagicMock()
  audio_source.close = MagicMock()

  client = _build_client(processor=processor, audio_source=audio_source)
  client.websocket.recv = AsyncMock(return_value=b"END_OF_AUDIO")

  completion_task = await client.start()
  await asyncio.wait_for(completion_task, timeout=0.5)

  assert processing_cancelled.is_set()
  processor.initialize.assert_awaited_once()
  processor.stop_processing.assert_awaited_once()
  audio_source.close.assert_called_once()
  assert client._exit is True


async def test_completion_cancels_pending_task_before_stop() -> None:
  """Completion wait must cancel the still-running task before invoking stop orchestration."""
  pending_cancelled = asyncio.Event()

  processor = MagicMock()
  processor.stop_processing = AsyncMock()

  audio_source = MagicMock()
  audio_source.close = MagicMock()

  client = _build_client(processor=processor, audio_source=audio_source)
  client._audio_task = asyncio.create_task(asyncio.sleep(0))
  client._processing_task = asyncio.create_task(_run_until_cancelled(pending_cancelled))

  async def _stop_guard() -> None:
    assert pending_cancelled.is_set()

  client.stop = AsyncMock(side_effect=_stop_guard)

  await asyncio.wait_for(client._wait_for_completion(), timeout=0.5)

  assert client._processing_task.cancelled()
  client.stop.assert_awaited_once()


async def test_completion_path_runs_disconnect_before_source_close() -> None:
  """Completion-triggered stop must disconnect processor output before closing audio source."""
  events: list[str] = []

  sink = MagicMock()

  async def _disconnect() -> None:
    events.append("disconnect")

  sink.disconnect = AsyncMock(side_effect=_disconnect)

  processor = MagicMock()

  async def _stop_processing() -> None:
    events.append("stop_processing")
    await sink.disconnect()

  processor.stop_processing = AsyncMock(side_effect=_stop_processing)

  audio_source = MagicMock()
  audio_source.close = MagicMock(side_effect=lambda: events.append("audio_close"))

  client = _build_client(processor=processor, audio_source=audio_source)
  client._audio_task = asyncio.create_task(asyncio.sleep(0))
  client._processing_task = asyncio.create_task(asyncio.sleep(3600))

  await asyncio.wait_for(client._wait_for_completion(), timeout=0.5)

  assert events == ["stop_processing", "disconnect", "audio_close"]
  processor.stop_processing.assert_awaited_once()
  audio_source.close.assert_called_once()


async def test_stop_during_active_processing_is_bounded() -> None:
  """Stopping during active tasks must complete quickly while cancelling both running loops."""
  processing_cancelled = asyncio.Event()
  audio_cancelled = asyncio.Event()

  processor = MagicMock()
  processor.stop_processing = AsyncMock()

  audio_source = MagicMock()
  audio_source.close = MagicMock()

  client = _build_client(processor=processor, audio_source=audio_source)
  client._processing_task = asyncio.create_task(_run_until_cancelled(processing_cancelled))
  client._audio_task = asyncio.create_task(_run_until_cancelled(audio_cancelled))

  # Yield once so both tasks begin running before stop() attempts cancellation.
  await asyncio.sleep(0)

  await asyncio.wait_for(client.stop(), timeout=0.5)

  assert processing_cancelled.is_set()
  assert audio_cancelled.is_set()
  assert client._processing_task.cancelled()
  assert client._audio_task.cancelled()
  processor.stop_processing.assert_awaited_once()
  audio_source.close.assert_called_once()


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
async def test_minimum_chunk_wait_logs_once_per_processor() -> None:
  """Repeated below-threshold polling must not spam the minimum-wait info log."""
  buffer = AudioStreamBuffer(
    BufferConfig(sample_rate=16000, min_chunk_duration=0.5, transcription_interval=0.5)
  )
  buffer.add_frames(np.zeros(1600, dtype=np.float32))
  processor = _build_processor(buffer=buffer, flush_state=LiveSessionFlushState())
  processor.logger = MagicMock()
  processor._wait_for_flush_wakeup = AsyncMock(return_value=True)

  await processor._get_next_audio_chunk()
  await processor._get_next_audio_chunk()

  minimum_wait_logs = [
    call
    for call in processor.logger.info.call_args_list
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
  client.transcription_sink = MagicMock()
  client.transcription_sink.send_error = AsyncMock()
  client._flush_state = LiveSessionFlushState()

  first_flush = serialize_message(FlushControlMessage(stream="stream-1", force_complete=False))
  second_flush = serialize_message(FlushControlMessage(stream="stream-1", force_complete=True))

  await client._handle_live_text_frame(first_flush)
  pending_flush = client._flush_state.pending()
  await client._handle_live_text_frame(second_flush)

  assert pending_flush is not None
  assert client._flush_state.pending() == pending_flush
  client.transcription_sink.send_error.assert_awaited_once_with(LIVE_FLUSH_ALREADY_PENDING_MESSAGE)


@pytest.mark.asyncio
async def test_live_utterance_cancel_discards_tail_without_merging_future_audio() -> None:
  """Accepted cancel control must drop the tail and preserve the monotonic processed cursor."""
  client = WebSocketStreamingClient.__new__(WebSocketStreamingClient)
  client.logger = MagicMock()
  client.buffer = AudioStreamBuffer(BufferConfig(sample_rate=10, min_chunk_duration=1.0))
  client.transcription_sink = MagicMock()
  client.transcription_sink.send_error = AsyncMock()
  client._flush_state = LiveSessionFlushState()

  client.buffer.add_frames(np.ones(8, dtype=np.float32))
  client.buffer.advance_processed_boundary(0.3)

  await client._handle_live_text_frame(
    serialize_message(UtteranceCancelledMessage(stream="stream-1"))
  )

  assert client.buffer.processed_up_to_time == pytest.approx(0.3)
  assert client.buffer.buffer_start_time == pytest.approx(0.3)
  assert client.buffer.available_duration == 0.0
  assert client.buffer.total_duration == 0.0

  client.buffer.add_frames(np.full(4, 0.5, dtype=np.float32))
  chunk, duration, start_time = client.buffer.get_chunk_for_processing()

  assert chunk.shape[0] == 4
  assert duration == pytest.approx(0.4)
  assert start_time == pytest.approx(0.3)
  client.transcription_sink.send_error.assert_not_awaited()


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

  result = await handler.handle_connection(cast(ServerConnection, websocket))

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
  client.transcription_sink = MagicMock()
  client.transcription_sink.send_error = AsyncMock()

  await client._handle_file_text_frame(serialize_message(message))

  client.transcription_sink.send_error.assert_awaited_once_with(expected_rejection)


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

  await server._reject_subscriber_control_frame(
    cast(ServerConnection, websocket),
    serialize_message(message),
  )

  assert websocket.close.await_count == 0
  assert len(websocket.sent_payloads) == 1
  rejection_message = deserialize_message(websocket.sent_payloads[0])
  assert rejection_message.type == "error"
  assert rejection_message.message == expected_rejection
