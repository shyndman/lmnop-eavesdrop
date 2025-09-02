"""
Transcription data types for wire protocol communication.

Contains the core data structures used for transcription results and user configuration.
"""

from pydantic.dataclasses import dataclass


@dataclass
class Word:
  start: float
  end: float
  word: str
  probability: float


@dataclass
class Segment:
  id: int
  seek: int
  start: float
  end: float
  text: str
  tokens: list[int]
  avg_logprob: float
  compression_ratio: float
  no_speech_prob: float
  words: list[Word] | None
  temperature: float | None
  completed: bool = False


@dataclass
class UserTranscriptionOptions:
  """Transcription options that clients can specify."""

  initial_prompt: str | None = None
  hotwords: str | None = None
  beam_size: int = 5
  word_timestamps: bool = False
