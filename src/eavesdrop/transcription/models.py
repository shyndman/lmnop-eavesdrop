from typing import NotRequired, Required, TypedDict

from faster_whisper.vad import VadOptions
from pydantic.dataclasses import dataclass


class FeatureExtractorConfig(TypedDict):
  feature_size: NotRequired[int]
  sampling_rate: NotRequired[int]
  hop_length: NotRequired[int]
  chunk_length: NotRequired[int]
  n_fft: NotRequired[int]


class WordDict(TypedDict):
  start: Required[float]
  end: Required[float]
  word: Required[str]
  probability: NotRequired[float]


class SegmentDict(TypedDict):
  seek: Required[int]
  start: Required[float]
  end: Required[float]
  tokens: Required[list[int]]
  words: NotRequired[list[WordDict]]


class WordTimingDict(TypedDict):
  word: str
  tokens: list[int]
  start: float
  end: float
  probability: float


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
class TranscriptionOptions:
  beam_size: int
  best_of: int
  patience: float
  length_penalty: float
  repetition_penalty: float
  no_repeat_ngram_size: int
  log_prob_threshold: float | None
  no_speech_threshold: float | None
  compression_ratio_threshold: float | None
  condition_on_previous_text: bool
  prompt_reset_on_temperature: float
  temperatures: list[float]
  initial_prompt: str | None
  prefix: str | None
  suppress_blank: bool
  suppress_tokens: list[int] | None
  without_timestamps: bool
  max_initial_timestamp: float
  word_timestamps: bool
  prepend_punctuations: str
  append_punctuations: str
  multilingual: bool
  max_new_tokens: int | None
  hallucination_silence_threshold: float | None
  hotwords: str | None


@dataclass
class TranscriptionInfo:
  language: str
  language_probability: float
  duration: float
  duration_after_vad: float
  all_language_probs: list[tuple[str, float]] | None
  transcription_options: TranscriptionOptions
  vad_options: VadOptions
