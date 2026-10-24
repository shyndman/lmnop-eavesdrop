"""Pure contracts for native sherpa token onset/duration conversion."""

from dataclasses import dataclass

import pytest

from eavesdrop.server.transcription.orukeet_timestamps import native_segments


@dataclass
class Result:
  text: str
  tokens: list[str]
  timestamps: list[float]
  durations: list[float]


def test_subwords_preserve_text_and_native_bounds() -> None:
  result = Result(
    "  Hello,\tworld!  ",
    ["▁Hel", "lo", ",", "▁world", "!"],
    [0.08, 0.16, 0.24, 0.64, 0.8],
    [0.08, 0.08, 0.0, 0.16, 0.08],
  )
  segment = native_segments(result, 1.0)[0]
  assert segment.text == result.text
  assert segment.words is not None
  assert [word.word for word in segment.words] == ["  Hello,", "\tworld!  "]
  assert [(word.start, word.end) for word in segment.words] == [(0.08, 0.24), (0.64, 0.88)]


def test_final_token_overhang_is_clipped_without_moving_onset() -> None:
  segment = native_segments(Result("Hi", ["▁Hi"], [0.16], [0.32]), 0.3)[0]
  assert (segment.start, segment.end) == (0.16, 0.3)


def test_empty_speech_is_empty() -> None:
  assert native_segments(Result("", [], [], []), 0.4) == []


@pytest.mark.parametrize(
  "result",
  [
    Result("Hi", ["▁Hi"], [], []),
    Result("Hi", ["▁Hi"], [-0.1], [0.1]),
    Result("Hi", ["▁Hi"], [float("nan")], [0.1]),
    Result("Hi", ["▁Hi"], [0.1], [-0.1]),
    Result("Hi", ["▁Hi"], [0.1], [0.8]),
    Result("Different", ["▁Hi"], [0.1], [0.1]),
  ],
)
def test_invalid_native_results_fail(result: Result) -> None:
  with pytest.raises(ValueError):
    native_segments(result, 0.5)
