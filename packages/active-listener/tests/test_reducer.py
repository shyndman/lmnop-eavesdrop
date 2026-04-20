"""Unit tests for active-listener transcription reduction helpers."""

from __future__ import annotations

from active_listener.recording.reducer import (
  OverlaySegment,
  RecordingReducerState,
  append_segment_text,
  build_transcription_update,
  reduce_new_segments,
  render_text,
)
from eavesdrop.wire import Segment


def _segment(segment_id: int, text: str) -> Segment:
  return Segment(
    id=segment_id,
    seek=0,
    start=0.0,
    end=0.1,
    text=text,
    tokens=[],
    temperature=0.0,
    avg_logprob=0.0,
    compression_ratio=1.0,
    words=None,
    completed=True,
  )


def test_reduce_new_segments_preserves_last_id_for_empty_input() -> None:
  reduction = reduce_new_segments([], last_id=7)

  assert reduction.segments == []
  assert reduction.incomplete_segment is None
  assert reduction.last_id == 7
  assert reduction.missing_last_id is False


def test_reduce_new_segments_treats_single_segment_as_tail_only() -> None:
  reduction = reduce_new_segments([_segment(10, "tail only")], last_id=None)

  assert reduction.segments == []
  assert reduction.incomplete_segment == _segment(10, "tail only")
  assert reduction.last_id is None
  assert reduction.missing_last_id is False


def test_reduce_new_segments_drops_duplicate_window_before_sentinel() -> None:
  first_window = [_segment(1, "alpha"), _segment(2, "beta"), _segment(3, "tail")]
  first_reduction = reduce_new_segments(first_window, last_id=None)

  second_window = [
    _segment(1, "alpha"),
    _segment(2, "beta"),
    _segment(3, "gamma"),
    _segment(4, "tail"),
  ]
  second_reduction = reduce_new_segments(second_window, last_id=first_reduction.last_id)

  assert [segment.id for segment in first_reduction.segments] == [1, 2]
  assert first_reduction.incomplete_segment == _segment(3, "tail")
  assert first_reduction.last_id == 2
  assert [segment.id for segment in second_reduction.segments] == [3]
  assert second_reduction.incomplete_segment == _segment(4, "tail")
  assert second_reduction.last_id == 3
  assert second_reduction.missing_last_id is False


def test_reduce_new_segments_accepts_only_segments_after_found_sentinel() -> None:
  window = [_segment(4, "old"), _segment(5, "new one"), _segment(6, "new two"), _segment(7, "tail")]

  reduction = reduce_new_segments(window, last_id=4)

  assert [segment.id for segment in reduction.segments] == [5, 6]
  assert reduction.incomplete_segment == _segment(7, "tail")
  assert reduction.last_id == 6
  assert reduction.missing_last_id is False


def test_reduce_new_segments_falls_back_when_sentinel_is_missing() -> None:
  window = [_segment(8, "reset one"), _segment(9, "reset two"), _segment(10, "tail")]

  reduction = reduce_new_segments(window, last_id=99)

  assert [segment.id for segment in reduction.segments] == [8, 9]
  assert reduction.incomplete_segment == _segment(10, "tail")
  assert reduction.last_id == 9
  assert reduction.missing_last_id is True


def test_build_transcription_update_returns_completed_delta_and_tail() -> None:
  reduction = reduce_new_segments(
    [_segment(11, "  alpha  "), _segment(12, "   "), _segment(13, " tail ")],
    last_id=None,
  )

  transcription_update = build_transcription_update(reduction)

  assert transcription_update is not None
  assert transcription_update.completed_segments == [OverlaySegment(id=11, text="alpha")]
  assert transcription_update.incomplete_segment == OverlaySegment(id=13, text="tail")


def test_append_segment_text_accumulates_stripped_non_empty_parts() -> None:
  state = RecordingReducerState(last_id=None, parts=["existing"])

  append_segment_text(
    state.parts,
    [_segment(1, "  hello  "), _segment(2, "   "), _segment(3, "world")],
  )

  assert state.parts == ["existing", "hello", "world"]
  assert render_text(state.parts) == "existing hello world"
