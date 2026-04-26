"""Unit tests for active-listener transcription reduction helpers."""

from __future__ import annotations

import pytest

from active_listener.recording.reducer import (
  CommandTextWordTimestampError,
  RecordingReducerState,
  TextRun,
  TimedWord,
  TimeSpan,
  apply_segment_reduction,
  build_transcription_update,
  classify_recording_words,
  normalize_runs,
  reduce_new_segments,
  render_text,
  serialize_text_runs,
)
from eavesdrop.wire import Segment, Word


def _segment(
  segment_id: int,
  text: str,
  *,
  start: float = 0.0,
  end: float = 0.1,
  time_offset: float = 0.0,
  words: list[Word] | None = None,
) -> Segment:
  return Segment(
    id=segment_id,
    seek=0,
    start=start,
    end=end,
    text=text,
    tokens=[],
    temperature=0.0,
    avg_logprob=0.0,
    compression_ratio=1.0,
    words=words,
    time_offset=time_offset,
    completed=True,
  )


def _word(text: str, *, start: float, end: float) -> Word:
  return Word(start=start, end=end, word=text, probability=1.0)


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


def test_build_transcription_update_returns_normalized_runs() -> None:
  reduction = reduce_new_segments(
    [_segment(11, "alpha"), _segment(12, "tail")],
    last_id=None,
  )
  state = RecordingReducerState(
    completed_words=[TimedWord(text="earlier", start_s=-0.1, end_s=-0.05, is_complete=True)],
  )
  apply_segment_reduction(state, reduction)

  transcription_update = build_transcription_update(state)

  assert transcription_update is not None
  assert transcription_update.runs == [
    TextRun(text="earlier alpha", is_command=False, is_complete=True),
    TextRun(text="tail", is_command=False, is_complete=False),
  ]


def test_apply_segment_reduction_accumulates_word_state() -> None:
  state = RecordingReducerState(
    last_id=None,
    completed_words=[
      TimedWord(text="existing", start_s=9.5, end_s=9.9, is_complete=True),
    ],
    first_word_start=9.5,
    last_word_end=9.9,
  )

  apply_segment_reduction(
    state,
    reduce_new_segments(
      [
        _segment(1, "  hello  ", start=0.0, end=0.4, time_offset=10.0),
        _segment(2, "   ", start=0.4, end=0.7, time_offset=10.0),
        _segment(3, "world", start=0.1, end=0.6, time_offset=10.5),
        _segment(4, "tail", start=0.6, end=0.8, time_offset=10.5),
      ],
      last_id=None,
    ),
  )

  assert [word.text for word in state.completed_words] == ["existing", "hello", "world"]
  assert [word.text for word in state.incomplete_words] == ["tail"]
  assert state.first_word_start == 9.5
  assert state.last_word_end == 11.1
  assert state.duration_seconds is not None
  assert abs(state.duration_seconds - 1.6) < 1e-9
  assert render_text(state.completed_words) == "existing hello world"


def test_classify_recording_words_handles_boundaries_and_open_span_tail() -> None:
  state = RecordingReducerState(
    completed_words=[
      TimedWord(text="normal", start_s=0.0, end_s=0.2, is_complete=True),
      TimedWord(text="boundary-start", start_s=1.0, end_s=1.0, is_complete=True),
      TimedWord(text="command", start_s=1.4, end_s=1.6, is_complete=True),
      TimedWord(text="boundary-end", start_s=2.0, end_s=2.0, is_complete=True),
    ],
    incomplete_words=[
      TimedWord(text="tail-command", start_s=3.2, end_s=3.4, is_complete=False),
      TimedWord(text="tail-normal", start_s=0.3, end_s=0.5, is_complete=False),
    ],
    closed_command_spans=[TimeSpan(start_s=1.0, end_s=2.0)],
    open_command_start_s=3.0,
  )

  assert [
    (word.text, word.is_command, word.is_complete) for word in classify_recording_words(state)
  ] == [
    ("normal", False, True),
    ("boundary-start", True, True),
    ("command", True, True),
    ("boundary-end", True, True),
    ("tail-command", True, False),
    ("tail-normal", False, False),
  ]


def test_apply_segment_reduction_requires_word_timestamps_for_command_text() -> None:
  state = RecordingReducerState(
    closed_command_spans=[TimeSpan(start_s=0.0, end_s=1.0)],
  )

  with pytest.raises(CommandTextWordTimestampError, match="Segment.words"):
    apply_segment_reduction(
      state,
      reduce_new_segments(
        [
          _segment(1, "spoken text", start=0.0, end=0.2),
          _segment(2, "tail", start=0.2, end=0.3),
        ],
        last_id=None,
      ),
    )


def test_normalize_runs_drops_empty_runs_and_merges_identical_flags() -> None:
  assert normalize_runs(
    [
      TextRun(text="   ", is_command=False, is_complete=True),
      TextRun(text="alpha", is_command=False, is_complete=True),
      TextRun(text=" beta ", is_command=False, is_complete=True),
      TextRun(text="gamma", is_command=True, is_complete=True),
      TextRun(text="delta", is_command=True, is_complete=True),
      TextRun(text="tail", is_command=True, is_complete=False),
    ]
  ) == [
    TextRun(text="alpha beta", is_command=False, is_complete=True),
    TextRun(text="gamma delta", is_command=True, is_complete=True),
    TextRun(text="tail", is_command=True, is_complete=False),
  ]


def test_build_transcription_update_accumulates_completed_prefix_once_and_replaces_tail() -> None:
  state = RecordingReducerState(last_id=None)

  first_reduction = reduce_new_segments(
    [
      _segment(
        1,
        "hello",
        words=[_word("hello", start=0.0, end=0.2)],
      ),
      _segment(
        2,
        "worl",
        words=[_word("worl", start=0.2, end=0.4)],
      ),
    ],
    last_id=state.last_id,
  )
  apply_segment_reduction(state, first_reduction)
  state.last_id = first_reduction.last_id
  first_update = build_transcription_update(state)

  second_reduction = reduce_new_segments(
    [
      _segment(
        1,
        "hello",
        words=[_word("hello", start=0.0, end=0.2)],
      ),
      _segment(
        2,
        "world",
        words=[_word("world", start=0.2, end=0.5)],
      ),
    ],
    last_id=state.last_id,
  )
  apply_segment_reduction(state, second_reduction)
  state.last_id = second_reduction.last_id
  second_update = build_transcription_update(state)

  third_reduction = reduce_new_segments(
    [
      _segment(
        1,
        "hello",
        words=[_word("hello", start=0.0, end=0.2)],
      ),
      _segment(
        2,
        "world",
        words=[_word("world", start=0.2, end=0.5)],
      ),
      _segment(
        3,
        "again",
        words=[_word("again", start=0.5, end=0.8)],
      ),
    ],
    last_id=state.last_id,
  )
  apply_segment_reduction(state, third_reduction)
  state.last_id = third_reduction.last_id
  third_update = build_transcription_update(state)

  assert first_update is not None
  assert first_update.runs == [
    TextRun(text="hello", is_command=False, is_complete=True),
    TextRun(text="worl", is_command=False, is_complete=False),
  ]
  assert second_update is not None
  assert second_update.runs == [
    TextRun(text="hello", is_command=False, is_complete=True),
    TextRun(text="world", is_command=False, is_complete=False),
  ]
  assert third_update is not None
  assert third_update.runs == [
    TextRun(text="hello world", is_command=False, is_complete=True),
    TextRun(text="again", is_command=False, is_complete=False),
  ]


def test_build_transcription_update_recolors_tail_without_reemitting_completed_text() -> None:
  state = RecordingReducerState(
    completed_words=[TimedWord(text="hello", start_s=0.0, end_s=0.2, is_complete=True)],
    last_id=1,
  )

  first_reduction = reduce_new_segments(
    [
      _segment(
        1,
        "hello",
        words=[_word("hello", start=0.0, end=0.2)],
      ),
      _segment(
        2,
        "draft command",
        words=[
          _word("draft", start=0.6, end=0.8),
          _word("command", start=0.8, end=1.0),
        ],
      ),
    ],
    last_id=state.last_id,
  )
  apply_segment_reduction(state, first_reduction)
  first_update = build_transcription_update(state)

  state.open_command_start_s = 0.5
  second_reduction = reduce_new_segments(
    [
      _segment(
        1,
        "hello",
        words=[_word("hello", start=0.0, end=0.2)],
      ),
      _segment(
        2,
        "draft command",
        words=[
          _word("draft", start=0.6, end=0.8),
          _word("command", start=0.8, end=1.0),
        ],
      ),
    ],
    last_id=state.last_id,
  )
  apply_segment_reduction(state, second_reduction)
  second_update = build_transcription_update(state)

  assert first_update is not None
  assert first_update.runs == [
    TextRun(text="hello", is_command=False, is_complete=True),
    TextRun(text="draft command", is_command=False, is_complete=False),
  ]
  assert second_update is not None
  assert second_update.runs == [
    TextRun(text="hello", is_command=False, is_complete=True),
    TextRun(text="draft command", is_command=True, is_complete=False),
  ]


def test_build_transcription_update_reclassifies_completed_words_after_hold_commit() -> None:
  state = RecordingReducerState(
    completed_words=[
      TimedWord(text="alpha", start_s=0.6, end_s=0.8, is_complete=True),
    ],
    incomplete_words=[
      TimedWord(text="draft", start_s=0.8, end_s=1.0, is_complete=False),
    ],
  )

  first_update = build_transcription_update(state)

  state.open_command_start_s = 0.5
  second_update = build_transcription_update(state)

  assert first_update is not None
  assert first_update.runs == [
    TextRun(text="alpha", is_command=False, is_complete=True),
    TextRun(text="draft", is_command=False, is_complete=False),
  ]
  assert second_update is not None
  assert second_update.runs == [
    TextRun(text="alpha", is_command=True, is_complete=True),
    TextRun(text="draft", is_command=True, is_complete=False),
  ]


def test_serialize_text_runs_wraps_command_text_and_preserves_punctuation() -> None:
  assert (
    serialize_text_runs(
      [
        TextRun(text="Hello,", is_command=False, is_complete=True),
        TextRun(text="scratch", is_command=True, is_complete=True),
        TextRun(text="that.", is_command=True, is_complete=True),
        TextRun(text="Bye.", is_command=False, is_complete=True),
      ]
    )
    == "Hello, <instruction>scratch that.</instruction> Bye."
  )
