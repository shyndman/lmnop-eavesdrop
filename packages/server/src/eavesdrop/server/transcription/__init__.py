from typing import TYPE_CHECKING

# Re-exports for backward compatibility
from eavesdrop.server.transcription.models import (
  TranscriptionInfo,
  TranscriptionOptions,
  VadParameters,
)
from eavesdrop.server.transcription.utils import (
  get_compression_ratio,
  get_ctranslate2_storage,
  get_suppressed_tokens,
  merge_punctuations,
  restore_speech_timestamps,
)
from eavesdrop.wire import Segment, Word

if TYPE_CHECKING:
  from eavesdrop.server.transcription.pipeline import WhisperModel


def __getattr__(name: str) -> object:
  if name == "WhisperModel":
    from eavesdrop.server.transcription.pipeline import WhisperModel

    return WhisperModel

  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
  "Word",
  "Segment",
  "TranscriptionOptions",
  "TranscriptionInfo",
  "VadParameters",
  "WhisperModel",
  "restore_speech_timestamps",
  "get_ctranslate2_storage",
  "get_compression_ratio",
  "get_suppressed_tokens",
  "merge_punctuations",
]
