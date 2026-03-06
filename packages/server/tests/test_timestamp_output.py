"""Regression tests for absolute timestamp behavior in transcription output."""

import json

import pytest

from eavesdrop.server.transcription.session import create_session
from eavesdrop.wire import Segment, TranscriptionMessage, serialize_message


def _segment(*, start: float, end: float, time_offset: float) -> Segment:
  return Segment(
    id=1,
    seek=0,
    start=start,
    end=end,
    text="hello",
    tokens=[1, 2, 3],
    avg_logprob=-0.25,
    compression_ratio=1.0,
    words=None,
    temperature=0.0,
    time_offset=time_offset,
    completed=False,
  )


def test_segment_absolute_times_apply_offset_beyond_30_seconds() -> None:
  segment = _segment(start=2.5, end=4.0, time_offset=31.25)

  assert segment.absolute_start_time == pytest.approx(33.75)
  assert segment.absolute_end_time == pytest.approx(35.25)


def test_absolute_times_remain_monotonic_across_window_boundaries() -> None:
  first = _segment(start=2.0, end=4.0, time_offset=28.0)
  second = _segment(start=0.1, end=1.1, time_offset=32.0)

  assert first.absolute_start_time == pytest.approx(30.0)
  assert first.absolute_end_time == pytest.approx(32.0)
  assert second.absolute_start_time == pytest.approx(32.1)
  assert second.absolute_end_time == pytest.approx(33.1)
  assert second.absolute_start_time > first.absolute_end_time


def test_transcription_message_serializes_absolute_timestamp_fields() -> None:
  segment = _segment(start=3.0, end=5.0, time_offset=40.0)
  message = TranscriptionMessage(stream="s1", segments=[segment], language="en")

  payload = json.loads(serialize_message(message))
  serialized_segment = payload["segments"][0]

  assert serialized_segment["start"] == pytest.approx(3.0)
  assert serialized_segment["end"] == pytest.approx(5.0)
  assert serialized_segment["absolute_start_time"] == pytest.approx(43.0)
  assert serialized_segment["absolute_end_time"] == pytest.approx(45.0)


def test_session_reports_absolute_chunk_time_range() -> None:
  session = create_session("stream-1")

  session.update_audio_context(start_offset=61.0, duration=1.5)

  assert session.get_absolute_time_range() == pytest.approx((61.0, 62.5))
