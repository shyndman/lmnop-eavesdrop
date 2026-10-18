"""Unit tests for deterministic transcript text shaping."""

from __future__ import annotations

import pytest

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


@pytest.mark.parametrize(
  "text",
  [
    "thank you",
    "hello thank you world",
    "Thank you? THANK YOU, thank you!",
    "escape thank you",
    "escape THANK YOU, escape Thank you.",
  ],
)
def test_shape_runs_preserves_dictated_thank_you_phrases(text: str) -> None:
  shaped = shape_runs(
    [TextRun(text=text, is_command=False, is_complete=True)],
    {},
  )

  assert shaped == [TextRun(text=text, is_command=False, is_complete=True)]


def test_shape_runs_applies_passed_corrections() -> None:
  shaped = shape_runs(
    [TextRun(text="foo", is_command=False, is_complete=True)],
    {"foo": "bar"},
  )

  assert [run.text for run in shaped] == ["bar"]


def test_shape_runs_drops_empty_run_and_preserves_flags() -> None:
  shaped = shape_runs(
    [
      TextRun(text="  ", is_command=False, is_complete=True),
      TextRun(text="scratch that", is_command=True, is_complete=False),
      TextRun(text="real text", is_command=False, is_complete=True),
    ],
    {},
  )

  assert [(run.text, run.is_command, run.is_complete) for run in shaped] == [
    ("scratch that", True, False),
    ("real text", False, True),
  ]
