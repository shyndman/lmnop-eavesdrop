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
  """Repo-owned VAD configuration kept import-light for config and tests."""

  model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

  threshold: float = Field(default=0.5, ge=0.0, le=1.0)
  neg_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
  min_speech_duration_ms: int = Field(default=0, ge=0)
  max_speech_duration_s: float = Field(default=float("inf"), gt=0.0)
  min_silence_duration_ms: int = Field(default=2000, ge=0, exclude=True)
  speech_pad_ms: int = Field(default=400, ge=0)


@dataclass
class TranscriptionInfo:
  """Chunk metadata; speech boundaries are original chunk sample offsets."""

  language: str = "en"
  duration: float = 0.0
  duration_after_vad: float = 0.0
  speech_chunks: list[SpeechChunk] = field(default_factory=list)
