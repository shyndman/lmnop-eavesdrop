from dataclasses import dataclass, field
from typing import ClassVar, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class SpeechChunk(TypedDict):
  """Represents a speech chunk from Voice Activity Detection.

  ``start`` and ``end`` are sample offsets into the original, unfiltered audio
  chunk passed to VAD. They are not seconds, frames, or recording-relative
  timestamps. The chunk timeline becomes recording-relative only after
  :func:`eavesdrop.server.transcription.utils.finalize_recording_timestamps`.
  """

  start: int
  end: int


class VadParameters(BaseModel):
  """Repo-owned Silero VAD configuration kept import-light for config and tests."""

  model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

  threshold: float = Field(default=0.5, ge=0.0, le=1.0)
  min_speech_duration_ms: int = Field(default=0, ge=0)
  max_speech_duration_s: float = Field(default=15.0, gt=0.0, allow_inf_nan=False)
  min_silence_duration_ms: int = Field(default=800, ge=0, exclude=True)


@dataclass
class TranscriptionInfo:
  """Chunk metadata; speech boundaries are original chunk sample offsets."""

  language: str = "en"
  duration: float = 0.0
  duration_after_vad: float = 0.0
  speech_chunks: list[SpeechChunk] | None = field(default=None)
