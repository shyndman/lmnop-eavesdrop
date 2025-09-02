from dataclasses import field
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
  """Options for the transcription process."""

  # Required fields without defaults
  multilingual: bool
  """If True, enables multilingual transcription, where the model can detect and
  transcribe multiple languages within the same audio."""

  initial_prompt: str | None
  """An initial text prompt to guide the model's transcription at the beginning
  of the audio."""

  # Fields with defaults
  beam_size: int = 5
  """Number of beams to use for beam search decoding. A larger beam size increases
  the likelihood of finding a better transcription but is slower. Used when
  temperature is 0."""

  best_of: int = 5
  """Number of candidate sequences to generate and select from. This is used only
  when temperature is > 0 (sampling)."""

  patience: float = 1.0
  """Patience factor for beam search. Used to control early stopping.
  See: https://arxiv.org/abs/2204.05424"""

  length_penalty: float = 1.0
  """Penalty applied to the length of the generated sequence. A value > 1.0
  encourages longer sequences, while a value < 1.0 encourages shorter ones."""

  repetition_penalty: float = 1.0
  """Penalty applied to repeated tokens. A value > 1.0 discourages repetition."""

  no_repeat_ngram_size: int = 0
  """If set to a positive integer, this prevents the model from generating the same
  n-gram (sequence of n words) twice in a row. This is a common strategy to
  reduce repetitive loops and hallucinations. A value of 0 disables this feature."""

  log_prob_threshold: float | None = -1.0
  """If the average log probability of a segment is below this threshold, the
  transcription is considered unreliable and the system will fallback to a
  higher temperature."""

  no_speech_threshold: float | None = 0.6
  """If the probability of a segment being no-speech is higher than this
  threshold, the segment is discarded."""

  compression_ratio_threshold: float | None = 2.4
  """If the compression ratio of a segment's text is higher than this threshold,
  it's considered a hallucination or repetitive output, triggering a fallback
  to a higher temperature."""

  condition_on_previous_text: bool = True
  """If True, the model is conditioned on the previously transcribed text to
  maintain context between segments. If False, the context is reset for each
  segment."""

  prompt_reset_on_temperature: float = 0.5
  """If the generation temperature exceeds this value, the context prompt is
  reset. This helps prevent the model from getting stuck in a bad state."""

  temperatures: list[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
  """A list of temperatures to try for decoding. The system starts with the
  first temperature and falls back to subsequent ones if the output is
  unreliable (based on log_prob_threshold or compression_ratio_threshold)."""

  prefix: str | None = None
  """A text prefix to force the model to start its output with. This is useful
  for correcting errors in the first few words of a segment."""

  suppress_blank: bool = True
  """If True, suppresses the blank token during decoding. This can help with
  alignment and word-level timestamps."""

  suppress_tokens: list[int] | None = None
  """A list of token IDs to suppress during generation. -1 is a special value
  that expands to the default list of non-speech tokens. This is typically
  initialized dynamically based on the tokenizer."""

  without_timestamps: bool = False
  """If True, the model is instructed not to generate segment-level timestamp
  tokens (e.g., <|0.00|>). The resulting segment timestamps will be coarse and
  based on the audio processing window. This provides the transcribed text
  without precise timing information from the model."""

  max_initial_timestamp: float = 1.0
  """The maximum initial timestamp allowed for the first segment. This can
  prevent the model from hallucinating a long silence at the beginning of the
  audio."""

  word_timestamps: bool = False
  """If True, enables a post-processing step to compute word-level timestamps.
  This uses the model's `align()` method to find the start and end time for
  each individual word in the transcription. This is computationally more
  expensive but provides the highest level of timing detail."""

  prepend_punctuations: str = '"\'"¿([{-'
  """A string of characters that are considered prepended punctuations (e.g., '(',
  '['). These are merged with the following word during timestamp alignment."""

  append_punctuations: str = '"\'.。,，!！?？:：")]}、'
  """A string of characters that are considered appended punctuations (e.g., '.',
  ',', '?'). These are merged with the preceding word during timestamp
  alignment."""

  max_new_tokens: int | None = None
  """The maximum number of new tokens to generate in a single segment. This can
  be used to limit the length of the output."""

  hallucination_silence_threshold: float | None = None
  """Threshold in seconds for detecting and filtering hallucinations. If a
  segment is suspected to be a hallucination and is surrounded by silence
  longer than this threshold, it is discarded."""

  hotwords: str | None = "Bang"
  """A string of 'hotwords' or 'boosted tokens' to provide as context to the
  model, increasing the likelihood of these words being transcribed correctly."""


@dataclass
class TranscriptionInfo:
  language: str
  language_probability: float
  duration: float
  duration_after_vad: float
  all_language_probs: list[tuple[str, float]] | None
  transcription_options: TranscriptionOptions
  vad_options: VadOptions
