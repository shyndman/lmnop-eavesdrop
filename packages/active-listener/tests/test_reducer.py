"""Unit tests for active-listener transcription reduction helpers."""

from __future__ import annotations

from active_listener.recording.reducer import (
  RecordingReducerState,
  TextRun,
  TimedWord,
  TimeSpan,
  apply_segment_reduction,
  build_completed_text_runs,
  build_incomplete_text_runs,
  classify_recording_words,
  normalize_runs,
  reduce_new_segments,
  render_text,
  serialize_runs_for_rewrite,
  serialize_runs_without_commands,
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
  completed: bool = True,
) -> Segment:
  return Segment(
    id=segment_id,
    start=start,
    end=end,
    text=text,
    words=(
      words
      if words is not None
      else [
        _word(token, start=start + time_offset, end=end + time_offset) for token in text.split()
      ]
    ),
    time_offset=time_offset,
    completed=completed,
  )


def _word(text: str, *, start: float, end: float) -> Word:
  return Word(start=start, end=end, word=text)


def test_reduce_new_segments_preserves_last_id_for_empty_input() -> None:
  reduction = reduce_new_segments([], last_id=7)

  assert reduction.segments == []
  assert reduction.incomplete_segments == []
  assert reduction.last_id == 7
  assert reduction.missing_last_id is False


def test_reduce_new_segments_keeps_single_incomplete_segment_provisional() -> None:
  segment = _segment(10, "tail only", completed=False)
  reduction = reduce_new_segments([segment], last_id=None)

  assert reduction.segments == []
  assert reduction.incomplete_segments == [segment]
  assert reduction.last_id is None
  assert reduction.missing_last_id is False


def test_reduce_new_segments_drops_duplicate_window_before_sentinel() -> None:
  first_window = [_segment(1, "alpha"), _segment(2, "beta"), _segment(3, "tail", completed=False)]
  first_reduction = reduce_new_segments(first_window, last_id=None)

  second_window = [
    _segment(1, "alpha"),
    _segment(2, "beta"),
    _segment(3, "gamma"),
    _segment(4, "tail", completed=False),
  ]
  second_reduction = reduce_new_segments(second_window, last_id=first_reduction.last_id)

  assert [segment.id for segment in first_reduction.segments] == [1, 2]
  assert first_reduction.incomplete_segments == [_segment(3, "tail", completed=False)]
  assert first_reduction.last_id == 2
  assert [segment.id for segment in second_reduction.segments] == [3]
  assert second_reduction.incomplete_segments == [_segment(4, "tail", completed=False)]
  assert second_reduction.last_id == 3
  assert second_reduction.missing_last_id is False


def test_reduce_new_segments_accepts_only_segments_after_found_sentinel() -> None:
  window = [
    _segment(4, "old"),
    _segment(5, "new one"),
    _segment(6, "new two"),
    _segment(7, "tail", completed=False),
  ]

  reduction = reduce_new_segments(window, last_id=4)

  assert [segment.id for segment in reduction.segments] == [5, 6]
  assert reduction.incomplete_segments == [_segment(7, "tail", completed=False)]
  assert reduction.last_id == 6
  assert reduction.missing_last_id is False


def test_reduce_new_segments_falls_back_when_sentinel_is_missing() -> None:
  window = [
    _segment(8, "reset one"),
    _segment(9, "reset two"),
    _segment(10, "tail", completed=False),
  ]

  reduction = reduce_new_segments(window, last_id=99)

  assert [segment.id for segment in reduction.segments] == [8, 9]
  assert reduction.incomplete_segments == [_segment(10, "tail", completed=False)]
  assert reduction.last_id == 9
  assert reduction.missing_last_id is True


def test_apply_segment_reduction_skips_duplicate_segments_after_missing_sentinel() -> None:
  state = RecordingReducerState(
    seen_segment_ids={8},
    completed_words=[TimedWord(text="reset", start_s=0.0, end_s=0.2, is_complete=True)],
    first_word_start=0.0,
    last_word_end=0.2,
  )

  reduction = reduce_new_segments(
    [
      _segment(8, "reset", words=[_word("reset", start=0.0, end=0.2)]),
      _segment(9, "two", words=[_word("two", start=0.2, end=0.4)]),
      _segment(10, "tail", words=[_word("tail", start=0.4, end=0.6)], completed=False),
    ],
    last_id=999,
  )

  apply_segment_reduction(state, reduction)

  assert reduction.missing_last_id is True
  assert [word.text for word in state.completed_words] == ["reset", "two"]
  assert [word.text for word in state.incomplete_words] == ["tail"]


def _overlay_runs(state: RecordingReducerState) -> list[TextRun]:
  """Combine shaped-source completed and incomplete runs the way the overlay does."""
  return [*build_completed_text_runs(state), *build_incomplete_text_runs(state)]


def test_overlay_runs_returns_normalized_runs() -> None:
  reduction = reduce_new_segments(
    [_segment(11, "alpha"), _segment(12, "tail", completed=False)],
    last_id=None,
  )
  state = RecordingReducerState(
    completed_words=[TimedWord(text="earlier", start_s=-0.1, end_s=-0.05, is_complete=True)],
  )
  apply_segment_reduction(state, reduction)

  assert _overlay_runs(state) == [
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
        _segment(4, "tail", start=0.6, end=0.8, time_offset=10.5, completed=False),
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


def test_multiple_provisional_revisions_commit_only_on_explicit_completion() -> None:
  """Commit revised words only when explicit completion replaces the tentative tail."""
  state = RecordingReducerState(
    closed_command_spans=[TimeSpan(start_s=0.3, end_s=0.8)],
  )
  for first, second in [("draft", "command"), ("revised", "instruction")]:
    reduction = reduce_new_segments(
      [
        _segment(1, first, start=0.0, end=0.2, completed=False),
        _segment(2, second, start=0.4, end=0.6, completed=False),
      ],
      last_id=state.last_id,
    )
    apply_segment_reduction(state, reduction)
    state.last_id = reduction.last_id
    assert state.last_id is None
    assert state.seen_segment_ids == set()
    assert state.completed_words == []
    assert _overlay_runs(state) == [
      TextRun(text=first, is_command=False, is_complete=False),
      TextRun(text=second, is_command=True, is_complete=False),
    ]

  completed = [
    _segment(1, "revised", start=0.0, end=0.2),
    _segment(2, "instruction", start=0.4, end=0.6),
  ]
  reduction = reduce_new_segments(completed, last_id=state.last_id)
  apply_segment_reduction(state, reduction)
  state.last_id = reduction.last_id
  assert state.last_id == 2
  assert state.seen_segment_ids == {1, 2}
  assert state.incomplete_words == []
  assert _overlay_runs(state) == [
    TextRun(text="revised", is_command=False, is_complete=True),
    TextRun(text="instruction", is_command=True, is_complete=True),
  ]
  apply_segment_reduction(state, reduce_new_segments(completed, last_id=state.last_id))
  assert render_text(state.completed_words) == "revised instruction"


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


def test_overlay_runs_accumulates_completed_prefix_once_and_replaces_tail() -> None:
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
        completed=False,
      ),
    ],
    last_id=state.last_id,
  )
  apply_segment_reduction(state, first_reduction)
  state.last_id = first_reduction.last_id
  first_runs = _overlay_runs(state)

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
        completed=False,
      ),
    ],
    last_id=state.last_id,
  )
  apply_segment_reduction(state, second_reduction)
  state.last_id = second_reduction.last_id
  second_runs = _overlay_runs(state)

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
        completed=False,
      ),
    ],
    last_id=state.last_id,
  )
  apply_segment_reduction(state, third_reduction)
  state.last_id = third_reduction.last_id
  third_runs = _overlay_runs(state)

  assert first_runs == [
    TextRun(text="hello", is_command=False, is_complete=True),
    TextRun(text="worl", is_command=False, is_complete=False),
  ]
  assert second_runs == [
    TextRun(text="hello", is_command=False, is_complete=True),
    TextRun(text="world", is_command=False, is_complete=False),
  ]
  assert third_runs == [
    TextRun(text="hello world", is_command=False, is_complete=True),
    TextRun(text="again", is_command=False, is_complete=False),
  ]


def test_overlay_runs_recolors_tail_without_reemitting_completed_text() -> None:
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
        completed=False,
      ),
    ],
    last_id=state.last_id,
  )
  apply_segment_reduction(state, first_reduction)
  first_runs = _overlay_runs(state)

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
        completed=False,
      ),
    ],
    last_id=state.last_id,
  )
  apply_segment_reduction(state, second_reduction)
  second_runs = _overlay_runs(state)

  assert first_runs == [
    TextRun(text="hello", is_command=False, is_complete=True),
    TextRun(text="draft command", is_command=False, is_complete=False),
  ]
  assert second_runs == [
    TextRun(text="hello", is_command=False, is_complete=True),
    TextRun(text="draft command", is_command=True, is_complete=False),
  ]


def test_overlay_runs_reclassifies_completed_words_after_hold_commit() -> None:
  state = RecordingReducerState(
    completed_words=[
      TimedWord(text="alpha", start_s=0.6, end_s=0.8, is_complete=True),
    ],
    incomplete_words=[
      TimedWord(text="draft", start_s=0.8, end_s=1.0, is_complete=False),
    ],
  )

  first_runs = _overlay_runs(state)

  state.open_command_start_s = 0.5
  second_runs = _overlay_runs(state)

  assert first_runs == [
    TextRun(text="alpha", is_command=False, is_complete=True),
    TextRun(text="draft", is_command=False, is_complete=False),
  ]
  assert second_runs == [
    TextRun(text="alpha", is_command=True, is_complete=True),
    TextRun(text="draft", is_command=True, is_complete=False),
  ]


def test_serialize_runs_for_rewrite_wraps_command_text_and_preserves_punctuation() -> None:
  assert (
    serialize_runs_for_rewrite(
      [
        TextRun(text="Hello,", is_command=False, is_complete=True),
        TextRun(text="scratch", is_command=True, is_complete=True),
        TextRun(text="that.", is_command=True, is_complete=True),
        TextRun(text="Bye.", is_command=False, is_complete=True),
      ]
    )
    == "Hello, <instruction>scratch that.</instruction> Bye."
  )


def test_serialize_runs_without_commands_drops_command_runs() -> None:
  assert (
    serialize_runs_without_commands(
      [
        TextRun(text="Hello,", is_command=False, is_complete=True),
        TextRun(text="scratch that.", is_command=True, is_complete=True),
        TextRun(text="Bye.", is_command=False, is_complete=True),
      ]
    )
    == "Hello, Bye."
  )
