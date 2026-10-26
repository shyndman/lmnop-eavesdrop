from dataclasses import dataclass, field
from typing import ClassVar, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SpeechSegment(TypedDict):
  """Represents a speech segment identified by Voice Activity Detection.

  ``start`` and ``end`` are sample offsets into the original, unfiltered audio
  chunk passed to VAD. They are not seconds, frames, or recording-relative
  timestamps. The chunk timeline becomes recording-relative only after
  :func:`eavesdrop.server.transcription.utils.finalize_recording_timestamps`.
  """

  start: int
  end: int


class VadParameters(BaseModel):
  """Repo-owned VAD configuration kept import-light for config and tests."""

  model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

  threshold: float = Field(default=0.5, ge=0.0, le=1.0)
  neg_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
  min_speech_duration_ms: int = Field(default=0, ge=0)
  max_speech_duration_s: float = Field(default=float("inf"), gt=0.0)
  min_silence_duration_ms: int = Field(default=2000, ge=0, exclude=True)
  speech_pad_ms: int = Field(default=400, ge=0)

  @field_validator("min_silence_duration_ms")
  @classmethod
  def reject_configured_silence_duration(cls, _value: int) -> int:
    """Reject configured silence duration; transcription completion owns this value."""
    raise ValueError(
      "min_silence_duration_ms cannot be specified; use silence_completion_threshold"
    )


@dataclass
class TranscriptionInfo:
  """Chunk metadata; speech boundaries are original chunk sample offsets."""

  language: str = "en"
  duration: float = 0.0
  duration_after_vad: float = 0.0
  speech_segments: list[SpeechSegment] = field(default_factory=list)
