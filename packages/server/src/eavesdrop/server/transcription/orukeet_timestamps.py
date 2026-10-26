"""Group native timed recognition pieces into recording-local words."""

import re
from typing import Protocol

import numpy as np

from eavesdrop.common import get_logger
from eavesdrop.wire import Segment, Word

# TDT's maximum four-frame duration at the 80 ms encoder stride.
MAX_TOKEN_DURATION = 0.32
logger = get_logger("orukeet/timestamps")


class Hypothesis(Protocol):
  """Native token or word pieces with waveform-local onset and duration seconds."""

  @property
  def text(self) -> str: ...

  @property
  def tokens(self) -> list[str]: ...

  @property
  def timestamps(self) -> list[float]: ...

  @property
  def durations(self) -> list[float]: ...


def native_segments(hypothesis: Hypothesis, duration: float) -> list[Segment]:
  """Group timed pieces into words without changing native timing or recognized text.

  :param hypothesis: Native token or word pieces with onset times and durations.
  :param duration: Original waveform duration, bounding final-token overhang.
  :returns: Waveform-local segments with punctuation and whitespace preserved.
  :raises ValueError: If token text or native timing is inconsistent.
  """
  text = hypothesis.text
  if not text.strip():
    return []
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
  return [
    Segment(id=0, text=text, start=words[0].start, end=max(word.end for word in words), words=words)
  ]
