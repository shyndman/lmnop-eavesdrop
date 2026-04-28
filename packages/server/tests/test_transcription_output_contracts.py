"""Contract tests for streaming transcription output envelopes.

These tests lock down the server-to-client segment envelope guarantees:
- completed history is windowed by `send_last_n_segments`
- every emission ends with exactly one incomplete segment
- start/end timestamps remain monotonic across repeated updates
"""

from dataclasses import dataclass, field

import numpy as np
import pytest

from eavesdrop.server.config import BufferConfig, TranscriptionConfig
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.flush_state import LiveSessionFlushState
from eavesdrop.server.streaming.interfaces import TranscriptionResult
from eavesdrop.server.streaming.processor import (
  ChunkTranscriptionResult,
  StreamingTranscriptionProcessor,
  TranscriptionPassStatus,
)
from eavesdrop.server.transcription.models import SpeechChunk
from eavesdrop.server.transcription.session import TranscriptionSession, create_session
from eavesdrop.wire import Segment


@dataclass
class RecordingSink:
  """Captures emitted transcription results for deterministic contract assertions."""

  results: list[TranscriptionResult] = field(default_factory=list)

  async def send_result(self, result: TranscriptionResult) -> None:
    self.results.append(result)

  async def send_error(self, error: str) -> None:
    return

  async def send_language_detection(self, language: str, probability: float) -> None:
    return

  async def send_server_ready(self, backend: str) -> None:
    return

  async def disconnect(self) -> None:
    return


def _segment(
  *,
  start: float,
  end: float,
  text: str,
  time_offset: float,
  completed: bool = False,
) -> Segment:
  """Build a lightweight segment fixture with explicit timing and completion flags."""
  return Segment(
    id=1,
    seek=0,
    start=start,
    end=end,
    text=text,
    tokens=[1, 2, 3],
    avg_logprob=-0.25,
    compression_ratio=1.0,
    words=None,
    temperature=0.0,
    time_offset=time_offset,
    completed=completed,
  )


def _create_processor(
  *, send_last_n_segments: int, session: TranscriptionSession, sink: RecordingSink
) -> StreamingTranscriptionProcessor:
  """Create a processor wired to in-memory collaborators only."""
  config = TranscriptionConfig(
    send_last_n_segments=send_last_n_segments,
    silence_completion_threshold=0.8,
  )
  buffer = AudioStreamBuffer(BufferConfig(sample_rate=16000))
  return StreamingTranscriptionProcessor(
    buffer=buffer,
    sink=sink,
    config=config,
    session=session,
    stream_name="stream-1",
  )


def _create_processor_with_buffer(
  *,
  send_last_n_segments: int,
  session: TranscriptionSession,
  sink: RecordingSink,
  buffer: AudioStreamBuffer,
  flush_state: LiveSessionFlushState,
) -> StreamingTranscriptionProcessor:
  """Create a processor using explicit buffer and flush-state collaborators."""
  config = TranscriptionConfig(
    send_last_n_segments=send_last_n_segments,
    silence_completion_threshold=0.8,
  )
  return StreamingTranscriptionProcessor(
    buffer=buffer,
    sink=sink,
    config=config,
    session=session,
    stream_name="stream-1",
    flush_state=flush_state,
  )


@pytest.mark.asyncio
async def test_output_envelope_keeps_tail_incomplete_without_completed_history() -> None:
  """When nothing is completed yet, output still contains a single incomplete tail segment."""
  session = create_session("stream-1")
  sink = RecordingSink()
  processor = _create_processor(send_last_n_segments=3, session=session, sink=sink)

  await processor._handle_transcription_output(
    result=[_segment(start=0.2, end=0.9, text="draft", time_offset=12.0)],
    duration=0.2,
    speech_chunks=None,
  )

  assert len(sink.results) == 1
  output_segments = sink.results[0].segments

  assert len(output_segments) == 1
  assert output_segments[0].text == "draft"
  assert output_segments[0].completed is False
  assert session.completed_segments == []


@pytest.mark.asyncio
async def test_send_last_n_limits_completed_history_and_appends_synthetic_tail() -> None:
  """The completed-history window is capped while preserving the tail incomplete invariant."""
  session = create_session("stream-1")
  sink = RecordingSink()
  processor = _create_processor(send_last_n_segments=2, session=session, sink=sink)

  session.add_completed_segment(
    _segment(start=0.0, end=0.3, text="one", time_offset=0.0, completed=True)
  )
  session.add_completed_segment(
    _segment(start=0.3, end=0.6, text="two", time_offset=0.0, completed=True)
  )
  session.add_completed_segment(
    _segment(start=0.6, end=0.9, text="three", time_offset=0.0, completed=True)
  )

  await processor._handle_transcription_output(
    result=[
      _segment(start=0.0, end=0.5, text="fresh", time_offset=1.2),
      _segment(start=0.5, end=0.8, text="final", time_offset=1.2),
    ],
    duration=1.0,
    speech_chunks=None,
  )

  output_segments = sink.results[-1].segments

  assert [segment.text for segment in output_segments[:-1]] == ["fresh", "final"]
  assert all(segment.completed for segment in output_segments[:-1])

  tail = output_segments[-1]
  assert tail.completed is False
  assert tail.text == ""
  assert tail.start == pytest.approx(tail.end)


@pytest.mark.asyncio
async def test_recording_relative_timestamps_remain_monotonic_across_multiple_updates() -> None:
  """Recording-relative starts and ends must continue to move forward across updates."""
  session = create_session("stream-1")
  sink = RecordingSink()
  processor = _create_processor(send_last_n_segments=3, session=session, sink=sink)

  await processor._handle_transcription_output(
    result=[
      _segment(start=30.0, end=31.0, text="alpha", time_offset=30.0),
      _segment(start=31.0, end=31.2, text="draft", time_offset=30.0),
    ],
    duration=0.2,
    speech_chunks=None,
  )

  await processor._handle_transcription_output(
    result=[
      _segment(start=31.6, end=32.2, text="bravo", time_offset=31.5),
      _segment(start=32.2, end=32.5, text="charlie", time_offset=31.5),
    ],
    duration=1.0,
    speech_chunks=None,
  )

  completed = session.completed_segments
  assert [segment.text for segment in completed] == ["alpha", "bravo", "charlie"]

  ranges = [(segment.start, segment.end) for segment in completed]
  assert ranges == pytest.approx(
    [
      (30.0, 31.0),
      (31.6, 32.2),
      (32.2, 32.5),
    ]
  )
  assert ranges[1][0] > ranges[0][1]

  latest_output = sink.results[-1].segments
  assert [segment.text for segment in latest_output[:-1]] == ["alpha", "bravo", "charlie"]
  assert latest_output[-1].completed is False


@pytest.mark.asyncio
async def test_silence_completion_converts_vad_chunk_times_to_recording_timeline() -> None:
  """Chunk-relative VAD boundaries must complete recording-relative segment timestamps."""
  session = create_session("stream-1")
  sink = RecordingSink()
  processor = _create_processor(send_last_n_segments=3, session=session, sink=sink)
  processor.buffer.advance_processed_boundary(26.934)
  speech_chunks: list[SpeechChunk] = [{"start": 720 * 16, "end": 5280 * 16}]

  await processor._handle_transcription_output(
    result=[
      _segment(
        start=28.614,
        end=32.974,
        text="account I can't remember what it's called",
        time_offset=26.934,
      )
    ],
    duration=8.612,
    speech_chunks=speech_chunks,
  )

  assert [segment.text for segment in session.completed_segments] == [
    "account I can't remember what it's called"
  ]
  assert sink.results[-1].segments[-2].completed is True
  assert sink.results[-1].segments[-1].completed is False


@pytest.mark.asyncio
async def test_incomplete_tail_blocks_silence_advancement_past_completed_history() -> None:
  """Silence skipping must not consume audio while a recording-relative tail is open."""
  session = create_session("stream-1")
  sink = RecordingSink()
  processor = _create_processor(send_last_n_segments=3, session=session, sink=sink)
  processor.buffer.advance_processed_boundary(26.934)
  session.add_completed_segment(
    _segment(
      start=22.534,
      end=26.934,
      text="I also have a retirement account",
      time_offset=19.154,
      completed=True,
    )
  )
  speech_chunks: list[SpeechChunk] = [{"start": 720 * 16, "end": 5280 * 16}]

  await processor._handle_transcription_output(
    result=[
      _segment(
        start=28.614,
        end=34.000,
        text="account I can't remember what it's called",
        time_offset=26.934,
      )
    ],
    duration=8.612,
    speech_chunks=speech_chunks,
  )

  assert session.completed_segments[-1].text == "I also have a retirement account"
  assert sink.results[-1].segments[-1].completed is False
  assert processor.buffer.processed_up_to_time == pytest.approx(26.934)


@pytest.mark.asyncio
async def test_live_results_carry_current_recording_id() -> None:
  """Emitted live transcription results must stay tagged to the active recording epoch."""
  session = create_session("stream-1")
  sink = RecordingSink()
  buffer = AudioStreamBuffer(BufferConfig(sample_rate=16000, min_chunk_duration=1.0))
  flush_state = LiveSessionFlushState()
  processor = _create_processor_with_buffer(
    send_last_n_segments=3,
    session=session,
    sink=sink,
    buffer=buffer,
    flush_state=flush_state,
  )
  generation = flush_state.start_recording("rec-1")
  processor.reset_live_recording(
    recording_id="rec-1",
    generation=generation,
    preserve_language=False,
  )

  await processor._handle_transcription_output(
    result=[_segment(start=0.0, end=0.4, text="draft", time_offset=0.0)],
    duration=0.2,
    speech_chunks=None,
  )

  assert sink.results[-1].recording_id == "rec-1"


@pytest.mark.asyncio
async def test_cancelled_generation_drops_stale_transcription_result() -> None:
  """Results from a cancelled live utterance must not reach the client sink."""
  session = create_session("stream-1")
  sink = RecordingSink()
  buffer = AudioStreamBuffer(BufferConfig(sample_rate=16000, min_chunk_duration=1.0))
  flush_state = LiveSessionFlushState()
  processor = _create_processor_with_buffer(
    send_last_n_segments=3,
    session=session,
    sink=sink,
    buffer=buffer,
    flush_state=flush_state,
  )

  stale_generation = 0
  flush_state.cancel_active_utterance()

  await processor._process_transcription_result(
    ChunkTranscriptionResult(
      status=TranscriptionPassStatus.TRANSCRIBED,
      chunk_start_sample=0,
      chunk_sample_count=int(0.5 * buffer.config.sample_rate),
      segments=[_segment(start=0.0, end=0.4, text="draft", time_offset=0.0)],
      info=None,
      processing_time=0.0,
      audio_duration=0.5,
      speech_chunks=None,
      utterance_generation=stale_generation,
    )
  )

  assert sink.results == []
  assert session.completed_segments == []


@pytest.mark.asyncio
async def test_recording_boundary_drops_stale_transcription_result_before_mutation() -> None:
  """Old-epoch transcription that finishes late must not mutate the new recording state."""
  session = create_session("stream-1")
  sink = RecordingSink()
  buffer = AudioStreamBuffer(BufferConfig(sample_rate=16000, min_chunk_duration=1.0))
  flush_state = LiveSessionFlushState()
  processor = _create_processor_with_buffer(
    send_last_n_segments=3,
    session=session,
    sink=sink,
    buffer=buffer,
    flush_state=flush_state,
  )

  old_generation = flush_state.start_recording("rec-old")
  processor.reset_live_recording(
    recording_id="rec-old",
    generation=old_generation,
    preserve_language=False,
  )
  new_generation = flush_state.start_recording("rec-new")
  processor.reset_live_recording(
    recording_id="rec-new",
    generation=new_generation,
    preserve_language=False,
  )

  await processor._process_transcription_result(
    ChunkTranscriptionResult(
      status=TranscriptionPassStatus.TRANSCRIBED,
      chunk_start_sample=0,
      chunk_sample_count=int(0.5 * buffer.config.sample_rate),
      segments=[_segment(start=0.0, end=0.4, text="stale", time_offset=0.0)],
      info=None,
      processing_time=0.0,
      audio_duration=0.5,
      speech_chunks=None,
      utterance_generation=old_generation,
      recording_id="rec-old",
    )
  )

  assert sink.results == []
  assert session.completed_segments == []
  assert buffer.processed_up_to_time == 0.0


@pytest.mark.asyncio
async def test_flush_force_complete_converts_tail_to_completed_history() -> None:
  """Force-complete flush responses finalize the tentative tail and append a fresh tail."""
  session = create_session("stream-1")
  sink = RecordingSink()
  processor = _create_processor(send_last_n_segments=3, session=session, sink=sink)

  await processor._handle_transcription_output(
    result=[
      _segment(start=0.0, end=0.4, text="alpha", time_offset=4.0),
      _segment(start=0.4, end=0.7, text="draft", time_offset=4.0),
    ],
    duration=0.2,
    speech_chunks=None,
    flush_complete=True,
    force_complete=True,
  )

  flush_result = sink.results[-1]
  output_segments = flush_result.segments

  assert flush_result.flush_complete is True
  assert [segment.text for segment in session.completed_segments] == ["alpha", "draft"]
  assert [segment.text for segment in output_segments[:-1]] == ["alpha", "draft"]
  assert all(segment.completed for segment in output_segments[:-1])
  assert sum(not segment.completed for segment in output_segments) == 1
  assert output_segments[-1].completed is False
  assert output_segments[-1].text == ""


@pytest.mark.asyncio
async def test_flush_completion_discards_audio_through_accepted_boundary() -> None:
  """A completed flush must clear all audio that belonged to the committed operation."""
  session = create_session("stream-1")
  sink = RecordingSink()
  buffer = AudioStreamBuffer(BufferConfig(sample_rate=16000, min_chunk_duration=1.0))
  flush_state = LiveSessionFlushState()
  processor = _create_processor_with_buffer(
    send_last_n_segments=3,
    session=session,
    sink=sink,
    buffer=buffer,
    flush_state=flush_state,
  )
  buffer.add_frames(np.ones(int(5.0 * buffer.config.sample_rate), dtype=np.float32))
  buffer.advance_processed_boundary(2.0)
  pending_flush = flush_state.accept(
    boundary_sample=buffer.get_buffer_end_sample(),
    force_complete=True,
  )

  assert pending_flush is not None

  await processor._process_transcription_result(
    ChunkTranscriptionResult(
      status=TranscriptionPassStatus.TRANSCRIBED,
      chunk_start_sample=int(2.0 * buffer.config.sample_rate),
      chunk_sample_count=int(3.0 * buffer.config.sample_rate),
      segments=[_segment(start=2.2, end=2.8, text="committed", time_offset=2.0)],
      info=None,
      processing_time=0.0,
      audio_duration=3.0,
      speech_chunks=None,
    )
  )

  assert sink.results[-1].flush_complete is True
  assert [segment.text for segment in session.completed_segments] == ["committed"]
  assert buffer.available_duration == 0.0
  assert buffer.total_duration == 0.0
  assert buffer.processed_up_to_time == 5.0


@pytest.mark.asyncio
async def test_flush_without_force_complete_preserves_existing_incomplete_tail() -> None:
  """Non-forcing flush responses keep the tentative tail incomplete and singular."""
  session = create_session("stream-1")
  sink = RecordingSink()
  processor = _create_processor(send_last_n_segments=3, session=session, sink=sink)
  processor.buffer.add_frames(
    np.ones(int(0.2 * processor.buffer.config.sample_rate), dtype=np.float32)
  )

  await processor._handle_transcription_output(
    result=[_segment(start=0.2, end=0.9, text="draft", time_offset=12.0)],
    duration=0.2,
    speech_chunks=None,
    flush_complete=True,
    force_complete=False,
  )
  _ = processor.buffer.discard_unprocessed_audio()

  flush_result = sink.results[-1]
  output_segments = flush_result.segments

  assert flush_result.flush_complete is True
  assert session.completed_segments == []
  assert [segment.text for segment in output_segments] == ["draft"]
  assert sum(not segment.completed for segment in output_segments) == 1
  assert output_segments[-1].completed is False
  assert processor.buffer.available_duration == 0.0
  assert processor.buffer.total_duration == 0.0
