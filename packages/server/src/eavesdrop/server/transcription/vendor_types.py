from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

FloatAudio = NDArray[np.float32] | NDArray[np.float64]
WhisperFeatures = NDArray[np.float32]
VadSpeechChunk = dict[str, int]
LanguageCodeMap = dict[str, str]


class FeatureExtractorLike(Protocol):
  sampling_rate: int
  hop_length: int
  n_samples: int
  nb_max_frames: int
  time_per_frame: float
  n_fft: int
  chunk_length: int

  def __call__(self, audio: FloatAudio) -> WhisperFeatures: ...


FeatureExtractorFactory = Callable[..., FeatureExtractorLike]


class TokenizerBackendLike(Protocol):
  def token_to_id(self, token: str) -> int: ...


class TokenizerLike(Protocol):
  tokenizer: TokenizerBackendLike
  non_speech_tokens: list[int]
  transcribe: int
  translate: int
  sot: int
  sot_prev: int
  sot_lm: int
  eot: int
  timestamp_begin: int
  no_timestamps: int
  sot_sequence: list[int]
  language: int | None
  language_code: str | None

  def encode(self, text: str) -> list[int]: ...

  def decode(self, tokens: Sequence[int]) -> str: ...

  def split_to_word_tokens(self, tokens: list[int]) -> tuple[list[str], list[list[int]]]: ...


class HuggingFaceTokenizerFactory(Protocol):
  def from_buffer(self, content: bytes) -> TokenizerLike: ...

  def from_file(self, path: str) -> TokenizerLike: ...

  def from_pretrained(self, identifier: str) -> TokenizerLike: ...


class VadOptionsLike(Protocol):
  threshold: float
  neg_threshold: float | None
  min_speech_duration_ms: int
  max_speech_duration_s: float
  min_silence_duration_ms: int
  speech_pad_ms: int


class VadOptionsFactory(Protocol):
  def __call__(
    self,
    *,
    threshold: float = ...,
    neg_threshold: float | None = ...,
    min_speech_duration_ms: int = ...,
    max_speech_duration_s: float = ...,
    min_silence_duration_ms: int = ...,
    speech_pad_ms: int = ...,
  ) -> VadOptionsLike: ...


class SpeechTimestampsMapLike(Protocol):
  def get_chunk_index(self, time: float) -> int: ...

  def get_original_time(self, time: float, chunk_index: int | None = None) -> float: ...


class SpeechTimestampsMapFactory(Protocol):
  def __call__(
    self,
    chunks: list[VadSpeechChunk],
    sampling_rate: int,
  ) -> SpeechTimestampsMapLike: ...


class StorageViewLike(Protocol):
  pass


class StorageViewFactory(Protocol):
  def from_array(self, array: WhisperFeatures) -> StorageViewLike: ...


class WhisperGenerationResultLike(Protocol):
  sequences_ids: list[list[int]]
  scores: list[float]


class AlignmentResultLike(Protocol):
  text_token_probs: list[float]
  alignments: list[tuple[int, int]]


class WhisperModelLike(Protocol):
  is_multilingual: bool
  device: str
  device_index: Sequence[int]

  def generate(
    self,
    encoder_output: StorageViewLike,
    prompts: list[list[int]],
    **kwargs: object,
  ) -> list[WhisperGenerationResultLike]: ...

  def encode(self, features: StorageViewLike, *, to_cpu: bool = False) -> StorageViewLike: ...

  def detect_language(self, encoder_output: StorageViewLike) -> list[list[tuple[str, float]]]: ...

  def align(
    self,
    encoder_output: StorageViewLike,
    sot_sequence: list[int],
    text_tokens: list[list[int]],
    num_frames: list[int],
    *,
    median_filter_width: int,
  ) -> list[AlignmentResultLike]: ...


WhisperModelFactory = Callable[..., WhisperModelLike]
WhisperTokenizerFactory = Callable[..., TokenizerLike]
DownloadModel = Callable[..., str]
FormatTimestamp = Callable[[float], str]
PadOrTrim = Callable[[WhisperFeatures], WhisperFeatures]
GetEnd = Callable[[list[dict[str, object]]], float]
CollectChunks = Callable[[FloatAudio, list[VadSpeechChunk]], tuple[list[FloatAudio], object]]
GetSpeechTimestamps = Callable[[FloatAudio, VadOptionsLike], list[VadSpeechChunk]]
ContainsModel = Callable[[str], bool]
CudaDeviceCapability = tuple[int, int]


class CudaLike(Protocol):
  def is_available(self) -> bool: ...

  def device_count(self) -> int: ...

  def get_device_name(self, index: int) -> str: ...

  def get_device_capability(self, device: str) -> CudaDeviceCapability: ...


class TorchLike(Protocol):
  cuda: CudaLike


class TransformersConverterLike(Protocol):
  def convert(self, *, output_dir: str, quantization: str, force: bool) -> None: ...


TransformersConverterFactory = Callable[..., TransformersConverterLike]


class CTranslate2ModelsModuleLike(Protocol):
  Whisper: WhisperModelFactory


class CTranslate2ConvertersModuleLike(Protocol):
  TransformersConverter: TransformersConverterFactory


class CTranslate2ModuleLike(Protocol):
  models: CTranslate2ModelsModuleLike
  StorageView: StorageViewFactory
  contains_model: ContainsModel
  converters: CTranslate2ConvertersModuleLike


def _module(name: str) -> object:
  return import_module(name)


def load_feature_extractor() -> FeatureExtractorFactory:
  return cast(
    FeatureExtractorFactory,
    getattr(_module("faster_whisper.feature_extractor"), "FeatureExtractor"),
  )


def load_hf_tokenizer_factory() -> HuggingFaceTokenizerFactory:
  return cast(HuggingFaceTokenizerFactory, getattr(_module("tokenizers"), "Tokenizer"))


def load_language_codes() -> LanguageCodeMap:
  return cast(LanguageCodeMap, getattr(_module("faster_whisper.tokenizer"), "_LANGUAGE_CODES"))


def load_whisper_tokenizer() -> WhisperTokenizerFactory:
  return cast(WhisperTokenizerFactory, getattr(_module("faster_whisper.tokenizer"), "Tokenizer"))


def load_download_model() -> DownloadModel:
  return cast(DownloadModel, getattr(_module("faster_whisper.utils"), "download_model"))


def load_format_timestamp() -> FormatTimestamp:
  return cast(FormatTimestamp, getattr(_module("faster_whisper.utils"), "format_timestamp"))


def load_pad_or_trim() -> PadOrTrim:
  return cast(PadOrTrim, getattr(_module("faster_whisper.audio"), "pad_or_trim"))


def load_get_end() -> GetEnd:
  return cast(GetEnd, getattr(_module("faster_whisper.utils"), "get_end"))


def load_vad_options() -> VadOptionsFactory:
  return cast(VadOptionsFactory, getattr(_module("faster_whisper.vad"), "VadOptions"))


def load_collect_chunks() -> CollectChunks:
  return cast(CollectChunks, getattr(_module("faster_whisper.vad"), "collect_chunks"))


def load_get_speech_timestamps() -> GetSpeechTimestamps:
  return cast(GetSpeechTimestamps, getattr(_module("faster_whisper.vad"), "get_speech_timestamps"))


def load_speech_timestamps_map() -> SpeechTimestampsMapFactory:
  return cast(
    SpeechTimestampsMapFactory, getattr(_module("faster_whisper.vad"), "SpeechTimestampsMap")
  )


def load_ctranslate2_whisper() -> WhisperModelFactory:
  module = cast(CTranslate2ModuleLike, _module("ctranslate2"))
  return module.models.Whisper


def load_storage_view() -> StorageViewFactory:
  module = cast(CTranslate2ModuleLike, _module("ctranslate2"))
  return module.StorageView


def load_contains_model() -> ContainsModel:
  module = cast(CTranslate2ModuleLike, _module("ctranslate2"))
  return module.contains_model


def load_transformers_converter() -> TransformersConverterFactory:
  module = cast(CTranslate2ModuleLike, _module("ctranslate2"))
  return module.converters.TransformersConverter


def load_torch() -> TorchLike:
  return cast(TorchLike, _module("torch"))
