"""Deterministic error-recovery contracts for ``StreamingTranscriptionProcessor``.

These tests pin failure-path orchestration behavior without loading models or using
real network sinks. They focus on loop recovery wait semantics and teardown bounds.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import final, override

import numpy as np
import pytest

from eavesdrop.server.config import BufferConfig, TranscriptionConfig
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.processor import (
  AudioChunk,
  ChunkTranscriptionResult,
  StreamingTranscriptionProcessor,
)
from eavesdrop.server.transcription.session import TranscriptionSession, create_session


@final
class FakeSink:
  disconnect_outcomes: list[Exception | None]
  disconnect_calls: int

  def __init__(self) -> None:
    self.disconnect_outcomes = []
    self.disconnect_calls = 0

  async def send_result(self, result: object) -> None:
    del result

  async def send_error(self, error: str) -> None:
    del error

  async def send_language_detection(self, language: str, probability: float) -> None:
    del language, probability

  async def send_server_ready(self, backend: str) -> None:
    del backend

  async def disconnect(self) -> None:
    self.disconnect_calls += 1
    if self.disconnect_outcomes:
      outcome = self.disconnect_outcomes.pop(0)
      if outcome is not None:
        raise outcome


class ProcessorHarness(StreamingTranscriptionProcessor):
  chunk_provider: Callable[[], Awaitable[AudioChunk | None]] | None
  transcriber_behavior: Callable[[AudioChunk], Awaitable[ChunkTranscriptionResult]] | None

  def __init__(
    self,
    buffer: AudioStreamBuffer,
    sink: FakeSink,
    config: TranscriptionConfig,
    session: TranscriptionSession,
    stream_name: str,
  ) -> None:
    super().__init__(
      buffer=buffer,
      sink=sink,
      config=config,
      session=session,
      stream_name=stream_name,
    )
    self.chunk_provider = None
    self.transcriber_behavior = None

  def set_chunk_provider(self, provider: Callable[[], Awaitable[AudioChunk | None]]) -> None:
    self.chunk_provider = provider

  def set_transcriber_behavior(
    self, behavior: Callable[[AudioChunk], Awaitable[ChunkTranscriptionResult]]
  ) -> None:
    self.transcriber_behavior = behavior

  async def run_transcription_loop(self) -> None:
    await self._transcription_loop()

  @override
  async def _get_next_audio_chunk(self) -> AudioChunk | None:
    if self.chunk_provider is None:
      raise AssertionError("chunk provider not configured")
    return await self.chunk_provider()

  @override
  async def _transcribe_chunk(self, chunk: AudioChunk) -> ChunkTranscriptionResult:
    if self.transcriber_behavior is None:
      raise AssertionError("transcriber behavior not configured")
    return await self.transcriber_behavior(chunk)


@pytest.fixture
def processor_config() -> TranscriptionConfig:
  return TranscriptionConfig(
    model="distil-medium.en",
    language="en",
    buffer=BufferConfig(transcription_interval=0.05, min_chunk_duration=0.01),
  )


@pytest.fixture
def sink() -> FakeSink:
  return FakeSink()


@pytest.fixture
def processor(
  processor_config: TranscriptionConfig,
  sink: FakeSink,
) -> ProcessorHarness:
  buffer = AudioStreamBuffer(processor_config.buffer)
  return ProcessorHarness(
    buffer=buffer,
    sink=sink,
    config=processor_config,
    session=create_session("test-stream"),
    stream_name="test-stream",
  )


async def test_transcription_exception_triggers_recovery_sleep(
  processor: ProcessorHarness,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  chunk = AudioChunk(
    data=np.zeros(1600, dtype=np.float32),
    duration=0.1,
    start_time=0.0,
  )

  async def provide_chunk() -> AudioChunk | None:
    return chunk

  async def raise_transcription_error(chunk: AudioChunk) -> ChunkTranscriptionResult:
    del chunk
    raise RuntimeError("boom")

  processor.set_chunk_provider(provide_chunk)
  processor.set_transcriber_behavior(raise_transcription_error)

  sleep_calls: list[float] = []

  async def fake_sleep(delay: float) -> None:
    sleep_calls.append(delay)
    processor.exit = True

  monkeypatch.setattr("eavesdrop.server.streaming.processor.asyncio.sleep", fake_sleep)

  await processor.run_transcription_loop()

  assert sleep_calls == [processor.buffer.config.transcription_interval]


async def test_stop_processing_remains_callable_after_disconnect_error(
  processor: ProcessorHarness,
  sink: FakeSink,
) -> None:
  sink.disconnect_outcomes = [RuntimeError("disconnect failed"), None]

  with pytest.raises(RuntimeError, match="disconnect failed"):
    await asyncio.wait_for(processor.stop_processing(), timeout=0.2)

  await asyncio.wait_for(processor.stop_processing(), timeout=0.2)

  assert processor.exit is True
  assert sink.disconnect_calls == 2


async def test_stop_teardown_does_not_leave_processing_task_running(
  processor: ProcessorHarness,
  sink: FakeSink,
) -> None:
  async def idle_chunk() -> AudioChunk | None:
    await asyncio.sleep(0)
    return None

  processor.set_chunk_provider(idle_chunk)
  sink.disconnect_outcomes = [RuntimeError("disconnect failed")]

  processing_task = asyncio.create_task(processor.start_processing())
  await asyncio.sleep(0)

  with pytest.raises(RuntimeError, match="disconnect failed"):
    await asyncio.wait_for(processor.stop_processing(), timeout=0.2)

  await asyncio.wait_for(processing_task, timeout=0.2)

  assert processing_task.done()
