"""Contract tests for streaming transcription output envelopes.

These tests lock down the server-to-client segment envelope guarantees:
- completed history is windowed by `send_last_n_segments`
- every emission ends with exactly one incomplete segment
- absolute timestamps remain monotonic across repeated updates
"""

from dataclasses import dataclass, field

import pytest

from eavesdrop.server.config import BufferConfig, TranscriptionConfig
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.interfaces import TranscriptionResult
from eavesdrop.server.streaming.processor import StreamingTranscriptionProcessor
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
async def test_absolute_timestamps_remain_monotonic_across_multiple_updates() -> None:
  """Absolute times must move forward even when relative timestamps reset between updates."""
  session = create_session("stream-1")
  sink = RecordingSink()
  processor = _create_processor(send_last_n_segments=3, session=session, sink=sink)

  await processor._handle_transcription_output(
    result=[
      _segment(start=0.0, end=1.0, text="alpha", time_offset=30.0),
      _segment(start=1.0, end=1.2, text="draft", time_offset=30.0),
    ],
    duration=0.2,
    speech_chunks=None,
  )

  await processor._handle_transcription_output(
    result=[
      _segment(start=0.1, end=0.7, text="bravo", time_offset=31.5),
      _segment(start=0.7, end=1.0, text="charlie", time_offset=31.5),
    ],
    duration=1.0,
    speech_chunks=None,
  )

  completed = session.completed_segments
  assert [segment.text for segment in completed] == ["alpha", "bravo", "charlie"]

  absolute_ranges = [
    (segment.absolute_start_time, segment.absolute_end_time) for segment in completed
  ]
  assert absolute_ranges == pytest.approx(
    [
      (30.0, 31.0),
      (31.6, 32.2),
      (32.2, 32.5),
    ]
  )
  assert absolute_ranges[1][0] > absolute_ranges[0][1]

  latest_output = sink.results[-1].segments
  assert [segment.text for segment in latest_output[:-1]] == ["alpha", "bravo", "charlie"]
  assert latest_output[-1].completed is False


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
async def test_flush_without_force_complete_preserves_existing_incomplete_tail() -> None:
  """Non-forcing flush responses keep the tentative tail incomplete and singular."""
  session = create_session("stream-1")
  sink = RecordingSink()
  processor = _create_processor(send_last_n_segments=3, session=session, sink=sink)

  await processor._handle_transcription_output(
    result=[_segment(start=0.2, end=0.9, text="draft", time_offset=12.0)],
    duration=0.2,
    speech_chunks=None,
    flush_complete=True,
    force_complete=False,
  )

  flush_result = sink.results[-1]
  output_segments = flush_result.segments

  assert flush_result.flush_complete is True
  assert session.completed_segments == []
  assert [segment.text for segment in output_segments] == ["draft"]
  assert sum(not segment.completed for segment in output_segments) == 1
  assert output_segments[-1].completed is False
