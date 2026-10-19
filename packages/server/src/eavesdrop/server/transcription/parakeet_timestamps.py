"""Convert Parakeet native word and sentence timestamps into chunk-local segments.

This module does not load a model or change the input waveform's time origin.
"""

from typing import Protocol, TypedDict

import numpy as np

from eavesdrop.wire import Segment, Word

# v2 TDT durations are [0, 1, 2, 3, 4] at an 80 ms encoder stride.
MAX_TOKEN_DURATION = 0.32


class NativeWord(TypedDict):
  """Native word boundaries in seconds relative to the input waveform."""

  word: str
  start: float
  end: float


class NativeSegment(TypedDict):
  """Native sentence boundaries in seconds relative to the input waveform."""

  segment: str
  start: float
  end: float


class NativeTimestamps(TypedDict):
  word: list[NativeWord]
  segment: list[NativeSegment]


class Hypothesis(Protocol):
  text: str
  timestamp: NativeTimestamps


def native_segments(hypothesis: Hypothesis, duration: float) -> list[Segment]:
  """Translate native NeMo boundaries without alignment or invented confidence.

  :param hypothesis: NeMo result with word and segment timestamps enabled.
  :param duration: Input waveform duration, used to validate native bounds.
  :returns: Chunk-local segments with punctuation and original whitespace preserved.
  :raises ValueError: If native timing/text is missing or inconsistent.
  """
  text = hypothesis.text
  if not text.strip():
    return []
  native_words = hypothesis.timestamp["word"]
  if not native_words:
    raise ValueError("Parakeet returned text without native word timestamps")
  words: list[Word] = []
  cursor = 0
  previous_start = 0.0
  previous_end = 0.0
  for stamp in native_words:
    start, end = stamp["start"], stamp["end"]
    if not (
      np.isfinite(start)
      and np.isfinite(end)
      and previous_start <= start <= end <= duration + MAX_TOKEN_DURATION
      and start <= duration
      and previous_end <= end
    ):
      raise ValueError("Parakeet returned invalid native word timestamps")
    token = stamp["word"]
    index = text.find(token, cursor)
    if not token or index < 0 or text[cursor:index].strip():
      raise ValueError("Parakeet native words do not match the recognized text")
    stop = index + len(token)
    words.append(Word(start=start, end=min(end, duration), word=text[cursor:stop]))
    cursor = stop
    previous_start = start
    previous_end = end
  if text[cursor:].strip():
    raise ValueError("Parakeet native words omit recognized text")
  words[-1].word += text[cursor:]

  output: list[Segment] = []
  word_index = 0
  previous_segment_end = 0.0
  for stamp in hypothesis.timestamp["segment"]:
    start, end = stamp["start"], stamp["end"]
    if not (
      np.isfinite(start)
      and np.isfinite(end)
      and previous_segment_end <= start <= end <= duration + MAX_TOKEN_DURATION
      and start <= duration
    ):
      raise ValueError("Parakeet returned invalid native segment timestamps")
    selected: list[Word] = []
    expected = " ".join(stamp["segment"].split())
    joined = ""
    while word_index < len(words):
      word = words[word_index]
      if word.start < start - 1e-6 or word.end > min(end, duration) + 1e-6:
        raise ValueError("Parakeet native words fall outside their segment")
      selected.append(word)
      word_index += 1
      # TDT can emit zero-duration words at the previous sentence's exact end.
      # Match native sentence text rather than consuming every tied timestamp.
      joined = " ".join("".join(item.word for item in selected).split())
      if joined == expected:
        break
      if not expected.startswith(joined):
        raise ValueError("Parakeet native segment text disagrees with words")
    if not selected or joined != expected:
      raise ValueError("Parakeet native segment text or boundaries disagree with words")
    if selected:
      output.append(
        Segment(
          id=0,
          start=start,
          end=min(end, duration),
          text="".join(word.word for word in selected),
          words=selected,
        )
      )
    previous_segment_end = end
  # Some NeMo versions omit segment metadata for short utterances. Word timestamps
  # are sufficient to form the remaining segment without estimating any boundary.
  if word_index < len(words):
    if hypothesis.timestamp["segment"]:
      raise ValueError("Parakeet native words fall outside segment metadata")
    selected = words[word_index:]
    output.append(
      Segment(
        id=0,
        start=selected[0].start,
        end=selected[-1].end,
        text="".join(word.word for word in selected),
        words=selected,
      )
    )
  return output
