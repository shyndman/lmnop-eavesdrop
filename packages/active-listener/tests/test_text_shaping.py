"""Unit tests for deterministic transcript text shaping."""

from __future__ import annotations

from active_listener.recording.reducer import TextRun
from active_listener.recording.text_shaping import shape_runs


def test_shape_runs_applies_replacement_per_run_preserving_flags() -> None:
  shaped = shape_runs(
    [TextRun(text="hillary", is_command=True, is_complete=False)],
    {},
  )

  assert [(run.text, run.is_command, run.is_complete) for run in shaped] == [
    ("hilary", True, False)
  ]


def test_shape_runs_fuses_spoken_symbols() -> None:
  shaped = shape_runs(
    [TextRun(text="release hashtag 42", is_command=False, is_complete=True)],
    {},
  )

  assert [run.text for run in shaped] == ["release#42"]


def test_shape_runs_removes_unescaped_thank_you() -> None:
  shaped = shape_runs(
    [TextRun(text="hello thank you world", is_command=False, is_complete=True)],
    {},
  )

  assert [run.text for run in shaped] == ["hello world"]


def test_shape_runs_keeps_escaped_thank_you() -> None:
  shaped = shape_runs(
    [TextRun(text="escape thank you", is_command=False, is_complete=True)],
    {},
  )

  assert [run.text for run in shaped] == ["thank you"]


def test_shape_runs_applies_passed_corrections() -> None:
  shaped = shape_runs(
    [TextRun(text="foo", is_command=False, is_complete=True)],
    {"foo": "bar"},
  )

  assert [run.text for run in shaped] == ["bar"]


def test_shape_runs_drops_run_emptied_by_thank_you_and_preserves_flags() -> None:
  shaped = shape_runs(
    [
      TextRun(text="thank you", is_command=False, is_complete=True),
      TextRun(text="scratch that", is_command=True, is_complete=False),
      TextRun(text="real text", is_command=False, is_complete=True),
    ],
    {},
  )

  assert [(run.text, run.is_command, run.is_complete) for run in shaped] == [
    ("scratch that", True, False),
    ("real text", False, True),
  ]
