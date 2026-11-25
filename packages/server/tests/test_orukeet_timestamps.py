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


def test_sentence_punctuation_splits_segments_preserving_words_and_text() -> None:
  result = Result(
    'First sentence. Second sentence! He said "Hi?" Next word.',
    [
      "▁First",
      "▁sentence",
      ".",
      "▁Second",
      "▁sentence",
      "!",
      "▁He",
      "▁said",
      '▁"Hi?"',
      "▁Next",
      "▁word",
      ".",
    ],
    [0.08, 0.16, 0.24, 0.4, 0.56, 0.64, 0.8, 0.96, 1.12, 1.36, 1.52, 1.68],
    [0.08, 0.08, 0.0, 0.08, 0.08, 0.0, 0.08, 0.08, 0.08, 0.08, 0.08, 0.0],
  )
  segments = native_segments(result, 2.0)

  assert [segment.text for segment in segments] == [
    "First sentence.",
    " Second sentence!",
    ' He said "Hi?"',
    " Next word.",
  ]
  assert "".join(segment.text for segment in segments) == result.text
  bounds = [(segment.start, segment.end) for segment in segments]
  expected = [
    (0.08, 0.24),
    (0.4, 0.64),
    (0.8, 1.20),
    (1.36, 1.68),
  ]
  assert len(bounds) == len(expected)
  for actual_pair, expected_pair in zip(bounds, expected, strict=True):
    assert actual_pair == pytest.approx(expected_pair)


def test_non_sentence_periods_do_not_split_segments() -> None:
  result = Result(
    "Use e.g. foo and v0.1.0 today.",
    ["▁Use", "▁e.g.", "▁foo", "▁and", "▁v0.1.0", "▁today", "."],
    [0.08, 0.16, 0.32, 0.48, 0.64, 0.88, 1.04],
    [0.08, 0.08, 0.08, 0.08, 0.16, 0.08, 0.0],
  )
  segments = native_segments(result, 1.5)

  assert len(segments) == 1
  assert segments[0].text == result.text
