"""Finite-source completion contracts independent of the recognition runtime."""

import asyncio
from dataclasses import dataclass, field

import numpy as np
import pytest

from eavesdrop.server.config import BufferConfig, TranscriptionConfig
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.interfaces import TranscriptionResult
from eavesdrop.server.streaming.processor import (
  ChunkTranscriptionResult,
  StreamingTranscriptionProcessor,
  TranscriptionPassStatus,
)
from eavesdrop.server.transcription.session import create_session
from eavesdrop.wire import Segment, Word

SAMPLE_RATE = 16000


@dataclass
class RecordingSink:
  """Capture emitted results and buffered duration at emission time.

  :param buffer: Processor buffer whose discard order is under test.
  """

  buffer: AudioStreamBuffer
  results: list[TranscriptionResult] = field(default_factory=list)
  buffered_at_emission: list[float] = field(default_factory=list)
  errors: list[str] = field(default_factory=list)

  async def send_result(self, result: TranscriptionResult) -> None:
    """Capture output before the processor can discard its EOF snapshot.

    :param result: Transcript envelope sent to the client.
    """
    self.results.append(result)
    self.buffered_at_emission.append(self.buffer.total_duration)

  async def send_error(self, error: str) -> None:
    """Record a processing error for assertions.

    :param error: Client-facing error message.
    """
    self.errors.append(error)

  async def send_language_detection(self, language: str, probability: float) -> None:
    """Accept unused language notifications.

    :param language: Detected language.
    :param probability: Detection confidence.
    """
    return

  async def send_server_ready(self, backend: str) -> None:
    """Accept an unused readiness notification.

    :param backend: Recognition backend name.
    """
    return

  async def disconnect(self) -> None:
    """Complete sink cleanup without external resources."""
    return


def _processor(start: float) -> tuple[StreamingTranscriptionProcessor, RecordingSink]:
  config = TranscriptionConfig(
    buffer=BufferConfig(min_chunk_duration=0.01, transcription_interval=0.01),
    silence_completion_threshold=0.8,
  )
  buffer = AudioStreamBuffer(config.buffer)
  _ = buffer.discard_through_sample(round(start * SAMPLE_RATE))
  sink = RecordingSink(buffer)
  processor = StreamingTranscriptionProcessor(
    buffer=buffer,
    sink=sink,
    config=config,
    session=create_session("finite-source-contract"),
    stream_name="finite-source-contract",
  )
  return processor, sink


def _result(
  start: float, sample_count: int, text: str = "Final words."
) -> ChunkTranscriptionResult:
  duration = sample_count / SAMPLE_RATE
  return ChunkTranscriptionResult(
    status=TranscriptionPassStatus.TRANSCRIBED,
    chunk_start_sample=round(start * SAMPLE_RATE),
    chunk_sample_count=sample_count,
    segments=[
      Segment(
        id=0,
        start=start + 0.04,
        end=start + duration - 0.04,
        text=text,
        words=[Word(start=start + 0.04, end=start + duration - 0.04, word=text)],
        time_offset=start,
        completed=False,
      )
    ],
    info=None,
    processing_time=0.0,
    audio_duration=duration,
    speech_chunks=[{"start": 0, "end": sample_count - 320}],
  )


@pytest.mark.asyncio
@pytest.mark.parametrize("start", [0.0, 32.0])
async def test_final_source_chunk_completes_speech_then_drains_without_flush(start: float) -> None:
  """EOF closes speech without trailing silence and preserves the wire tail invariant.

  :param start: Recording-relative start, including a post-30-second chunk.
  """
  processor, sink = _processor(start)
  processor.add_audio_frames(np.ones(6400, dtype=np.float32))
  processor.mark_source_exhausted()

  await processor._process_transcription_result(_result(start, 6400))

  assert len(sink.results) == 1
  output = sink.results[0]
  assert output.flush_complete is None
  assert len(output.segments) == 2
  completed, tail = output.segments
  assert completed.text == "Final words."
  assert completed.completed is True
  assert (completed.start, completed.end) == pytest.approx((start + 0.04, start + 0.36))
  assert completed.words is not None
  assert (completed.words[0].start, completed.words[0].end) == pytest.approx(
    (start + 0.04, start + 0.36)
  )
  assert tail.text == ""
  assert tail.completed is False
  assert tail.start == pytest.approx(start + 0.4)
  assert tail.end == tail.start
  assert sum(not segment.completed for segment in output.segments) == 1
  assert processor.session.completed_segments == [completed]
  assert sink.buffered_at_emission == [0.4]
  assert processor.buffer.total_duration == 0.0
  assert processor.buffer.available_duration == 0.0
  assert processor.buffer.processed_up_to_time == pytest.approx(start + 0.4)
  assert processor.buffer.get_buffer_end_sample() == round((start + 0.4) * SAMPLE_RATE)

  await asyncio.wait_for(processor.start_processing(), timeout=1.0)

  assert len(sink.results) == 1
  assert sink.errors == []


@pytest.mark.asyncio
@pytest.mark.parametrize("start", [0.0, 32.0])
async def test_older_snapshot_cannot_finalize_or_discard_newer_eof_audio(start: float) -> None:
  """EOF alone cannot complete a snapshot that misses the final buffered samples.

  :param start: Recording-relative start of the in-flight snapshot.
  """
  processor, sink = _processor(start)
  initial_audio = np.ones(6400, dtype=np.float32)
  final_audio = np.full(3200, 0.5, dtype=np.float32)
  processor.add_audio_frames(initial_audio)
  old_result = _result(start, 6400)
  processor.add_audio_frames(final_audio)
  processor.mark_source_exhausted()

  await processor._process_transcription_result(old_result)

  assert len(sink.results) == 1
  old_output = sink.results[0]
  assert old_output.flush_complete is None
  assert len(old_output.segments) == 1
  assert old_output.segments[0].text == "Final words."
  assert old_output.segments[0].completed is False
  assert processor.session.completed_segments == []
  assert processor.buffer.processed_up_to_time == start
  assert processor.buffer.total_duration == 0.6
  assert processor.buffer.available_duration == 0.6
  remaining_audio, duration, remaining_start = processor.buffer.get_chunk_for_processing()
  np.testing.assert_array_equal(remaining_audio, np.concatenate((initial_audio, final_audio)))
  assert duration == 0.6
  assert remaining_start == start

  await processor._process_transcription_result(_result(start, 9600, "Final words continued."))

  assert len(sink.results) == 2
  final_output = sink.results[1]
  assert final_output.flush_complete is None
  assert [segment.text for segment in final_output.segments] == ["Final words continued.", ""]
  assert [segment.completed for segment in final_output.segments] == [True, False]
  assert [segment.text for segment in processor.session.completed_segments] == [
    "Final words continued."
  ]
  assert processor.buffer.total_duration == 0.0
  assert processor.buffer.available_duration == 0.0
  assert processor.buffer.processed_up_to_time == pytest.approx(start + 0.6)
  assert sink.buffered_at_emission == [0.6, 0.6]


@pytest.mark.asyncio
async def test_unexhausted_source_keeps_final_buffer_snapshot_incomplete() -> None:
  """Reaching the current buffer end does not imply EOF for an ongoing source."""
  processor, sink = _processor(0.0)
  audio = np.ones(6400, dtype=np.float32)
  processor.add_audio_frames(audio)

  await processor._process_transcription_result(_result(0.0, 6400))

  assert len(sink.results) == 1
  output = sink.results[0]
  assert output.flush_complete is None
  assert len(output.segments) == 1
  assert output.segments[0].text == "Final words."
  assert output.segments[0].completed is False
  assert processor.session.completed_segments == []
  assert processor.buffer.total_duration == 0.4
  assert processor.buffer.available_duration == 0.4
  assert processor.buffer.processed_up_to_time == 0.0
  remaining_audio, duration, start = processor.buffer.get_chunk_for_processing()
  np.testing.assert_array_equal(remaining_audio, audio)
  assert duration == 0.4
  assert start == 0.0
