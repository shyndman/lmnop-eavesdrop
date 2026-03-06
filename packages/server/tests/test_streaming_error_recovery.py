"""Deterministic error-recovery contracts for ``StreamingTranscriptionProcessor``.

These tests pin failure-path orchestration behavior without loading models or using
real network sinks. They focus on loop recovery wait semantics and teardown bounds.
"""

import asyncio
from unittest.mock import AsyncMock

import numpy as np
import pytest

from eavesdrop.server.config import TranscriptionConfig
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.processor import AudioChunk, StreamingTranscriptionProcessor
from eavesdrop.server.transcription.session import create_session


@pytest.fixture
def processor_config() -> TranscriptionConfig:
  """Create a fast test config that keeps retry/teardown tests deterministic."""
  return TranscriptionConfig(
    model="distil-medium.en",
    language="en",
    buffer={"transcription_interval": 0.05, "min_chunk_duration": 0.01},
  )


@pytest.fixture
def sink_mock() -> AsyncMock:
  """Provide an async sink mock matching the processor's output contract methods."""
  sink = AsyncMock()
  sink.send_result = AsyncMock()
  sink.send_error = AsyncMock()
  sink.send_language_detection = AsyncMock()
  sink.send_server_ready = AsyncMock()
  sink.disconnect = AsyncMock()
  return sink


@pytest.fixture
def processor(
  processor_config: TranscriptionConfig,
  sink_mock: AsyncMock,
) -> StreamingTranscriptionProcessor:
  """Build a processor wired to in-memory buffer/session test doubles."""
  buffer = AudioStreamBuffer(processor_config.buffer)
  return StreamingTranscriptionProcessor(
    buffer=buffer,
    sink=sink_mock,
    config=processor_config,
    session=create_session("test-stream"),
    stream_name="test-stream",
  )


async def test_transcription_exception_triggers_recovery_sleep(
  processor: StreamingTranscriptionProcessor,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Loop exceptions must route through recovery sleep using configured interval."""
  chunk = AudioChunk(
    data=np.zeros(1600, dtype=np.float32),
    duration=0.1,
    start_time=0.0,
  )
  processor._get_next_audio_chunk = AsyncMock(return_value=chunk)
  processor._transcribe_chunk = AsyncMock(side_effect=RuntimeError("boom"))

  sleep_calls: list[float] = []

  async def fake_sleep(delay: float) -> None:
    sleep_calls.append(delay)
    processor.exit = True

  monkeypatch.setattr("eavesdrop.server.streaming.processor.asyncio.sleep", fake_sleep)

  await processor._transcription_loop()

  assert sleep_calls == [processor.buffer.config.transcription_interval]


async def test_stop_processing_remains_callable_after_disconnect_error(
  processor: StreamingTranscriptionProcessor,
  sink_mock: AsyncMock,
) -> None:
  """A disconnect failure must not prevent subsequent stop attempts."""
  sink_mock.disconnect.side_effect = [RuntimeError("disconnect failed"), None]

  with pytest.raises(RuntimeError, match="disconnect failed"):
    await asyncio.wait_for(processor.stop_processing(), timeout=0.2)

  await asyncio.wait_for(processor.stop_processing(), timeout=0.2)

  assert processor.exit is True
  assert sink_mock.disconnect.await_count == 2


async def test_stop_teardown_does_not_leave_processing_task_running(
  processor: StreamingTranscriptionProcessor,
  sink_mock: AsyncMock,
) -> None:
  """Stop teardown should not leak the processing task when disconnect raises."""

  async def idle_chunk() -> AudioChunk | None:
    await asyncio.sleep(0)
    return None

  processor._get_next_audio_chunk = idle_chunk
  sink_mock.disconnect.side_effect = RuntimeError("disconnect failed")

  processing_task = asyncio.create_task(processor.start_processing())
  await asyncio.sleep(0)

  with pytest.raises(RuntimeError, match="disconnect failed"):
    await asyncio.wait_for(processor.stop_processing(), timeout=0.2)

  await asyncio.wait_for(processing_task, timeout=0.2)

  orchestrated_tasks = [processing_task]
  assert all(task.done() for task in orchestrated_tasks)
