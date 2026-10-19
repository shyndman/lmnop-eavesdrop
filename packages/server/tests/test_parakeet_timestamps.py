"""Pure conversion contracts for Parakeet native word and sentence timestamps."""

from dataclasses import dataclass

import pytest

from eavesdrop.server.transcription.parakeet_timestamps import NativeTimestamps, native_segments


@dataclass
class NativeHypothesis:
  """Provide a typed native result without importing NeMo.

  :param text: Exact recognized text.
  :param timestamp: Native word and segment timestamps in seconds.
  """

  text: str
  timestamp: NativeTimestamps


def _hypothesis(text: str = "Hello.") -> NativeHypothesis:
  return NativeHypothesis(
    text=text,
    timestamp={
      "word": [{"word": text, "start": 0.2, "end": 0.7}],
      "segment": [{"segment": text, "start": 0.1, "end": 0.8}],
    },
  )


def test_native_words_preserve_punctuation_whitespace_and_segment_bounds() -> None:
  """Every native word, segment bound, and source character survives conversion."""
  hypothesis = NativeHypothesis(
    text="  Hello,\tworld!\n Again.  ",
    timestamp={
      "word": [
        {"word": "Hello,", "start": 0.2, "end": 0.5},
        {"word": "world!", "start": 0.6, "end": 0.95},
        {"word": "Again.", "start": 1.3, "end": 1.65},
      ],
      "segment": [
        {"segment": "Hello,\tworld!", "start": 0.12, "end": 1.04},
        {"segment": "Again.", "start": 1.2, "end": 1.8},
      ],
    },
  )

  segments = native_segments(hypothesis, duration=2.0)

  assert [segment.text for segment in segments] == ["  Hello,\tworld!", "\n Again.  "]
  assert "".join(segment.text for segment in segments) == hypothesis.text
  assert [(segment.start, segment.end) for segment in segments] == [(0.12, 1.04), (1.2, 1.8)]
  words = [word for segment in segments for word in segment.words or []]
  assert [word.word for word in words] == ["  Hello,", "\tworld!", "\n Again.  "]
  assert [(word.start, word.end) for word in words] == [(0.2, 0.5), (0.6, 0.95), (1.3, 1.65)]


def test_word_only_native_output_keeps_native_word_boundaries() -> None:
  """Absent sentence metadata falls back to native words without estimated times."""
  hypothesis = _hypothesis()
  hypothesis.timestamp["segment"] = []

  segments = native_segments(hypothesis, duration=1.0)

  assert len(segments) == 1
  assert (segments[0].start, segments[0].end, segments[0].text) == (0.2, 0.7, "Hello.")


def test_zero_duration_boundary_word_belongs_to_its_native_sentence() -> None:
  """A tied TDT word timestamp cannot move a word into the previous sentence."""
  hypothesis = NativeHypothesis(
    text="Hello. I agree.",
    timestamp={
      "word": [
        {"word": "Hello.", "start": 0.0, "end": 0.4},
        {"word": "I", "start": 0.4, "end": 0.4},
        {"word": "agree.", "start": 0.4, "end": 0.8},
      ],
      "segment": [
        {"segment": "Hello.", "start": 0.0, "end": 0.4},
        {"segment": "I agree.", "start": 0.4, "end": 0.8},
      ],
    },
  )

  segments = native_segments(hypothesis, duration=1.0)

  assert [segment.text for segment in segments] == ["Hello.", " I agree."]
  assert [(segment.start, segment.end) for segment in segments] == [(0.0, 0.4), (0.4, 0.8)]
  first_words, second_words = segments[0].words, segments[1].words
  assert first_words is not None
  assert second_words is not None
  assert [word.word for word in first_words] == ["Hello."]
  assert [word.word for word in second_words] == [" I", " agree."]
  assert [(word.start, word.end) for word in second_words] == [(0.4, 0.4), (0.4, 0.8)]


def test_nonempty_text_requires_native_words() -> None:
  """Text without native word boundaries fails instead of inventing alignment."""
  hypothesis = _hypothesis()
  hypothesis.timestamp["word"] = []

  with pytest.raises(ValueError, match="word timestamps"):
    native_segments(hypothesis, duration=1.0)


def test_empty_native_hypothesis_produces_no_segments() -> None:
  """A model result containing only whitespace does not invent transcript text."""
  hypothesis = NativeHypothesis(text=" \t\n", timestamp={"word": [], "segment": []})

  assert native_segments(hypothesis, duration=1.0) == []


def test_empty_native_word_is_rejected() -> None:
  """An empty timestamp token cannot stand in for recognized text."""
  hypothesis = _hypothesis()
  hypothesis.timestamp["word"][0]["word"] = ""

  with pytest.raises(ValueError, match="text"):
    native_segments(hypothesis, duration=1.0)


@pytest.mark.parametrize("text", ["Wrong.", "Unexpected Hello.", "Hello. unexpected"])
def test_native_words_must_account_for_all_recognized_text(text: str) -> None:
  """Unmatched text before, after, or instead of native words is rejected.

  :param text: Recognized text inconsistent with the native word list.
  """
  hypothesis = _hypothesis()
  hypothesis.text = text

  with pytest.raises(ValueError, match="text"):
    native_segments(hypothesis, duration=1.0)


@pytest.mark.parametrize(
  ("start", "end"),
  [
    (float("nan"), 0.7),
    (0.2, float("inf")),
    (-0.1, 0.7),
    (0.7, 0.2),
    (0.2, 1.33),
    (1.01, 1.28),
  ],
  ids=["nan", "infinite", "negative", "reversed", "excessive-overhang", "start-past-audio"],
)
def test_invalid_native_word_bounds_are_rejected(start: float, end: float) -> None:
  """Malformed native timing fails instead of producing an invented boundary.

  :param start: Invalid native word start.
  :param end: Invalid native word end.
  """
  hypothesis = _hypothesis()
  hypothesis.timestamp["word"][0]["start"] = start
  hypothesis.timestamp["word"][0]["end"] = end

  with pytest.raises(ValueError, match="timestamp"):
    native_segments(hypothesis, duration=1.0)


def test_out_of_order_native_words_are_rejected() -> None:
  """Native words must follow their textual order on the audio timeline."""
  hypothesis = NativeHypothesis(
    text="First second",
    timestamp={
      "word": [
        {"word": "First", "start": 0.6, "end": 0.8},
        {"word": "second", "start": 0.1, "end": 0.3},
      ],
      "segment": [],
    },
  )

  with pytest.raises(ValueError, match="timestamp"):
    native_segments(hypothesis, duration=1.0)


def test_native_word_ends_cannot_move_backward() -> None:
  """Increasing word starts cannot conceal a reversed end-timestamp sequence."""
  hypothesis = NativeHypothesis(
    text="First second",
    timestamp={
      "word": [
        {"word": "First", "start": 0.1, "end": 0.8},
        {"word": "second", "start": 0.3, "end": 0.6},
      ],
      "segment": [],
    },
  )

  with pytest.raises(ValueError, match="timestamp"):
    native_segments(hypothesis, duration=1.0)


@pytest.mark.parametrize(
  ("start", "end"),
  [
    (float("nan"), 0.8),
    (0.1, float("inf")),
    (-0.1, 0.8),
    (0.8, 0.1),
    (0.1, 1.33),
    (1.01, 1.28),
  ],
  ids=["nan", "infinite", "negative", "reversed", "excessive-overhang", "start-past-audio"],
)
def test_invalid_native_segment_bounds_are_rejected(start: float, end: float) -> None:
  """Native sentence bounds are validated independently of valid word bounds.

  :param start: Invalid native segment start.
  :param end: Invalid native segment end.
  """
  hypothesis = _hypothesis()
  hypothesis.timestamp["segment"][0]["start"] = start
  hypothesis.timestamp["segment"][0]["end"] = end

  with pytest.raises(ValueError, match="timestamp"):
    native_segments(hypothesis, duration=1.0)


def test_out_of_order_native_segments_are_rejected() -> None:
  """Native segments cannot move backward even after all words are consumed."""
  hypothesis = _hypothesis()
  hypothesis.timestamp["segment"].append({"segment": "Hello.", "start": 0.0, "end": 0.9})

  with pytest.raises(ValueError, match="timestamp"):
    native_segments(hypothesis, duration=1.0)


def test_native_segment_text_must_match_its_words() -> None:
  """Inconsistent native sentence text cannot silently receive unrelated words."""
  hypothesis = _hypothesis()
  hypothesis.timestamp["segment"][0]["segment"] = "Different."

  with pytest.raises(ValueError, match="text"):
    native_segments(hypothesis, duration=1.0)


@pytest.mark.parametrize(("start", "end"), [(0.3, 0.8), (0.1, 0.6)])
def test_words_outside_native_segment_bounds_are_rejected(start: float, end: float) -> None:
  """Present sentence metadata must enclose the words assigned to that sentence.

  :param start: Native segment start, possibly after the word's start.
  :param end: Native segment end, possibly before the word's end.
  """
  hypothesis = _hypothesis()
  hypothesis.timestamp["segment"][0]["start"] = start
  hypothesis.timestamp["segment"][0]["end"] = end

  with pytest.raises(ValueError):
    native_segments(hypothesis, duration=1.0)
