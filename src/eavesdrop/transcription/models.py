from dataclasses import asdict, dataclass
from warnings import warn

from faster_whisper.vad import VadOptions


@dataclass
class Word:
  start: float
  end: float
  word: str
  probability: float

  def _asdict(self):
    warn(
      "Word._asdict() method is deprecated, use dataclasses.asdict(Word) instead",
      DeprecationWarning,
      2,
    )
    return asdict(self)


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

  def _asdict(self):
    warn(
      "Segment._asdict() method is deprecated, use dataclasses.asdict(Segment) instead",
      DeprecationWarning,
      2,
    )
    return asdict(self)


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
