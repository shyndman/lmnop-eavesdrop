import os
from pathlib import Path
from typing import ClassVar, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator, validate_call
from pydantic.dataclasses import dataclass
from pydantic.types import FilePath

from eavesdrop.common import get_logger
from eavesdrop.server.constants import CACHE_PATH, SINGLE_MODEL
from eavesdrop.server.transcription.models import VadParameters

logger = get_logger("cfg")


@dataclass
class BufferConfig:
  """Configuration for audio stream buffer behavior."""

  sample_rate: int = Field(default=16000, gt=0)
  """Audio sample rate in Hz."""

  max_buffer_duration: float = Field(default=45.0, gt=0.0)
  """Maximum buffer duration in seconds before cleanup."""

  cleanup_duration: float = Field(default=30.0, gt=0.0)
  """Duration of oldest audio to remove during cleanup."""

  min_chunk_duration: float = Field(default=1.0, gt=0.0)
  """Minimum chunk duration for processing in seconds."""

  transcription_interval: float = Field(default=2.0, gt=0.0)
  """Interval between transcription attempts in seconds."""

  clip_audio: bool = False
  """Whether to clip audio when transcription stalls for too long."""

  max_stall_duration: float = Field(default=25.0, gt=0.0)
  """Maximum duration without progress before clipping audio."""

  @model_validator(mode="after")
  def validate_duration_relationships(self) -> "BufferConfig":
    """Validate that cleanup_duration is less than max_buffer_duration."""
    if self.cleanup_duration >= self.max_buffer_duration:
      raise ValueError(
        f"cleanup_duration ({self.cleanup_duration}s) must be less than "
        + f"max_buffer_duration ({self.max_buffer_duration}s)"
      )
    return self


class DebugAudioConfig(BaseModel):
  """Configuration for debug audio recording behavior."""

  pre_buffer: Path | None = Field(
    default=None, description="Path prefix for pre-buffer audio files. Records audio before buffer."
  )

  post_buffer: Path | None = Field(
    default=None,
    description="Path prefix for post-buffer audio files. Records audio after dequeuing.",
  )


class TranscriptionConfig(BaseModel):
  """Configuration for transcription processing behavior."""

  send_last_n_segments: int = Field(default=10, gt=0)
  """Number of most recent segments to send to the client."""

  model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

  model: Literal["oruk/orukeet"] = "oruk/orukeet"
  """Fixed Orukeet model with CPU INT8 ONNX assets."""

  language: Literal["en"] = "en"
  """Only English transcription is supported."""

  word_timestamps: bool = True
  """Whether to include native word boundaries in output."""

  buffer: BufferConfig = Field(default_factory=BufferConfig)
  """Audio buffer configuration for streaming transcription."""

  use_vad: bool = True
  """Whether to use Voice Activity Detection."""

  vad_parameters: VadParameters = Field(default_factory=VadParameters)
  """Voice Activity Detection parameters."""

  silence_completion_threshold: float = Field(default=0.8, gt=0.0)
  """Seconds of silence after speech to mark segment as completed."""

  debug_audio: DebugAudioConfig | None = None
  """Debug audio recording configuration. When set, audio is saved at different pipeline stages."""

  @model_validator(mode="before")
  @classmethod
  def reject_derived_vad_fields(cls, values: object) -> object:
    """Reject an explicit silence duration; completion threshold owns it."""
    if isinstance(values, dict):
      settings = cast(dict[str, object], values)
      vad = settings.get("vad_parameters")
      if isinstance(vad, dict) and "min_silence_duration_ms" in vad:
        raise ValueError(
          "min_silence_duration_ms cannot be specified; use silence_completion_threshold"
        )
    return cast(object, values)

  @model_validator(mode="after")
  def align_vad_parameters(self) -> "TranscriptionConfig":
    """Align VAD silence duration with completion threshold for consistent behavior."""
    if self.buffer.sample_rate != 16000:
      raise ValueError("Orukeet requires buffer.sample_rate=16000")
    self.vad_parameters.min_silence_duration_ms = int(self.silence_completion_threshold * 1000)
    return self


class EavesdropConfig(BaseModel):
  """Top-level Eavesdrop configuration."""

  model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

  transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
  """Transcription processing configuration."""

  def _log_transcription_settings(self) -> None:
    t = self.transcription
    vad = t.vad_parameters
    logger.info("TRANSCRIPTION SETTINGS:")
    logger.info(f"  Model: {t.model}")
    logger.info(f"  Language: {t.language}")
    logger.info(f"  Word Timestamps: {t.word_timestamps}")
    logger.info(f"  Send Last N Segments: {t.send_last_n_segments}")
    logger.info(f"  Silence Completion Threshold: {t.silence_completion_threshold}s")
    logger.info(f"  Use VAD: {t.use_vad}")
    logger.info("  VAD PARAMETERS:")
    logger.info(f"    Threshold: {vad.threshold}")
    logger.info(f"    Min Speech Duration: {vad.min_speech_duration_ms}ms")
    logger.info(f"    Max Speech Duration: {vad.max_speech_duration_s}s")
    logger.info(f"    Min Silence Duration: {vad.min_silence_duration_ms}ms")

  def _log_buffer_and_debug_settings(self) -> None:
    buf = self.transcription.buffer
    logger.info("BUFFER SETTINGS:")
    logger.info(f"  Sample Rate: {buf.sample_rate}")
    logger.info(f"  Max Buffer Duration: {buf.max_buffer_duration}s")
    logger.info(f"  Cleanup Duration: {buf.cleanup_duration}s")
    logger.info(f"  Min Chunk Duration: {buf.min_chunk_duration}s")
    logger.info(f"  Transcription Interval: {buf.transcription_interval}s")
    logger.info(f"  Clip Audio: {buf.clip_audio}")
    logger.info(f"  Max Stall Duration: {buf.max_stall_duration}s")
    logger.info("DEBUG AUDIO SETTINGS:")
    if self.transcription.debug_audio:
      logger.info(f"  Pre-Buffer Path: {self.transcription.debug_audio.pre_buffer}")
      logger.info(f"  Post-Buffer Path: {self.transcription.debug_audio.post_buffer}")
    else:
      logger.info("  Debug Audio: Disabled")

  def pretty_print(self) -> None:
    """Pretty print the complete configuration at INFO level."""
    logger.info("=" * 60)
    logger.info("EAVESDROP CONFIGURATION")
    logger.info("=" * 60)
    self._log_transcription_settings()
    self._log_buffer_and_debug_settings()
    logger.info("SYSTEM CONSTANTS:")
    logger.info(f"  Cache Path: {CACHE_PATH}")
    logger.info(f"  Single Model: {SINGLE_MODEL}")
    logger.info("=" * 60)


@validate_call
def load_config_from_file(config_path: FilePath) -> EavesdropConfig:
  """Load and validate Eavesdrop configuration from YAML file."""
  logger.info("Loading Eavesdrop configuration", path=str(config_path))

  try:
    with open(config_path, "r", encoding="utf-8") as file:
      config_data = cast(object, yaml.safe_load(file))
  except yaml.YAMLError as e:
    raise ValueError(f"Invalid YAML in configuration file: {e}") from e
  except Exception as e:
    raise ValueError(f"Error reading configuration file: {e}") from e

  if config_data is None:
    raise ValueError("Configuration file is empty")

  if not isinstance(config_data, dict):
    raise ValueError("Configuration file must contain a YAML dictionary")

  config = EavesdropConfig.model_validate(config_data)
  config.pretty_print()
  return config


def get_env_float(key: str, default: float) -> float:
  """Get a float from an environment variable."""
  return float(os.getenv(key, str(default)))


def get_env_int(key: str, default: int) -> int:
  """Get an int from an environment variable."""
  return int(os.getenv(key, str(default)))


def get_env_bool(key: str, default: bool) -> bool:
  """Get a bool from an environment variable."""
  return os.getenv(key, str(default)).lower() == "true"
