"""Group native timed recognition pieces into recording-local sentence segments."""

import re
from typing import Protocol

import numpy as np

from eavesdrop.common import get_logger
from eavesdrop.wire import Segment, Word

# TDT's maximum four-frame duration at the 80 ms encoder stride.
MAX_TOKEN_DURATION = 0.32
_SENTENCE_END_PATTERN = re.compile(r'[.!?]["\')\]”’]*$')
_SENTENCE_START_PATTERN = re.compile(r'^["\'(\[“‘]*[A-Z]')
logger = get_logger("orukeet/timestamps")


class Hypothesis(Protocol):
  """Native token pieces with waveform-local onset and duration seconds."""

  text: str
  tokens: list[str]
  timestamps: list[float]
  durations: list[float]


def _extract_timed_pieces(
  hypothesis: Hypothesis, duration: float
) -> list[tuple[int, int, float, float]]:
  text = hypothesis.text
  tokens, starts, durations = hypothesis.tokens, hypothesis.timestamps, hypothesis.durations
  if not tokens or len(tokens) != len(starts) or len(tokens) != len(durations):
    logger.error("Orukeet returned mismatched token/timestamp/duration counts")
    raise ValueError("Orukeet returned inconsistent native token timestamps")

  pieces: list[tuple[int, int, float, float]] = []
  decoded = ""
  previous_start = 0.0
  for token, start, length in zip(tokens, starts, durations, strict=True):
    end = start + length
    if not (
      np.isfinite(start)
      and np.isfinite(end)
      and previous_start <= start <= end
      and start <= duration
      and end <= duration + MAX_TOKEN_DURATION
    ):
      logger.error("Orukeet returned invalid token timing", start=start, duration=length)
      raise ValueError("Orukeet returned invalid native token timestamps")
    piece = "".join(token.replace("▁", " ").split())
    pieces.append((len(decoded), len(decoded) + len(piece), start, min(end, duration)))
    decoded += piece
    previous_start = start

  if decoded != "".join(text.split()):
    logger.error("Orukeet token text disagrees with recognition text")
    raise ValueError("Orukeet tokens do not match recognized text")

  return pieces


def _group_words(text: str, pieces: list[tuple[int, int, float, float]]) -> list[Word]:
  words: list[Word] = []
  cursor = 0
  for match in re.finditer(r"\S+", text):
    begin = len("".join(text[: match.start()].split()))
    stop = begin + len(match.group())
    selected = [p for p in pieces if p[0] < stop and p[1] > begin]
    if not selected:
      logger.error("Orukeet word has no matching timed token", word=match.group())
      raise ValueError("Orukeet word has no native token timing")
    words.append(
      Word(word=text[cursor : match.end()], start=selected[0][2], end=max(p[3] for p in selected))
    )
    cursor = match.end()

  words[-1].word += text[cursor:]
  return words


def _is_sentence_boundary(current_word: Word, next_word: Word) -> bool:
  current_stripped = current_word.word.strip()
  next_stripped = next_word.word.strip()
  return bool(
    _SENTENCE_END_PATTERN.search(current_stripped) and _SENTENCE_START_PATTERN.match(next_stripped)
  )


def _split_sentence_segments(words: list[Word]) -> list[Segment]:
  groups: list[list[Word]] = []
  current_group: list[Word] = []

  for index, word in enumerate(words):
    current_group.append(word)
    if index + 1 < len(words) and _is_sentence_boundary(word, words[index + 1]):
      groups.append(current_group)
      current_group = []

  if current_group:
    groups.append(current_group)

  return [
    Segment(
      id=0,
      text="".join(word.word for word in group),
      start=group[0].start,
      end=max(word.end for word in group),
      words=group,
    )
    for group in groups
  ]


def native_segments(hypothesis: Hypothesis, duration: float) -> list[Segment]:
  """Group timed pieces into sentence segments without changing native timing or text.

  :param hypothesis: Native token pieces with onset times and durations.
  :type hypothesis: Hypothesis
  :param duration: Original waveform duration, bounding final-token overhang.
  :type duration: float
  :returns: Waveform-local sentence segments with punctuation and whitespace preserved.
  :rtype: list[Segment]
  :raises ValueError: If token text or native timing is inconsistent.
  """
  text = hypothesis.text
  if not text.strip():
    return []

  pieces = _extract_timed_pieces(hypothesis, duration)
  words = _group_words(text, pieces)
  return _split_sentence_segments(words)
