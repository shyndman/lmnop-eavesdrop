import os
from pathlib import Path
from typing import Literal, cast

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


class RTSPCacheConfig(BaseModel):
  """Configuration for RTSP stream caching behavior based on listener presence."""

  waiting_for_listener_duration: float = Field(
    default=10800.0, gt=0.0, description="Cache duration in seconds when streams have no listeners"
  )

  has_listener_cache_duration: float = Field(
    default=600.0,
    gt=0.0,
    description="Cache duration in seconds when streams have active listeners",
  )


class DebugAudioConfig(BaseModel):
  """Configuration for debug audio recording behavior."""

  pre_buffer: Path | None = Field(
    default=None, description="Path prefix for pre-buffer audio files. Records audio before buffer."
  )

  post_buffer: Path | None = Field(
    default=None,
    description="Path prefix for post-buffer audio files. Records audio after dequeuing.",
  )


class RTSPConfig(BaseModel):
  """Configuration for RTSP stream behavior."""

  streams: dict[str, str] = Field(default={}, description="RTSP stream URLs")

  cache: RTSPCacheConfig = Field(
    default_factory=RTSPCacheConfig, description="RTSP cache configuration"
  )


class TranscriptionConfig(BaseModel):
  """Configuration for transcription processing behavior."""

  # Transcription behavior
  send_last_n_segments: int = Field(default=10, gt=0)
  """Number of most recent segments to send to the client."""

  model_config = ConfigDict(extra="forbid")

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
        raise ValueError("min_silence_duration_ms cannot be specified; use silence_completion_threshold")
    return values

  @model_validator(mode="after")
  def align_vad_parameters(self) -> "TranscriptionConfig":
    """Align VAD silence duration with completion threshold for consistent behavior."""
    self.vad_parameters.min_silence_duration_ms = int(self.silence_completion_threshold * 1000)
    return self



class EavesdropConfig(BaseModel):
  """Top-level Eavesdrop configuration with transcription and RTSP settings."""

  transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
  """Transcription processing configuration."""

  rtsp: RTSPConfig = Field(default_factory=RTSPConfig)
  """RTSP stream configuration."""

  def pretty_print(self) -> None:
    """
    Pretty print the complete configuration at INFO level.

    This method logs every single configuration property to provide full visibility
    into the active configuration, including defaults. This is essential for
    debugging configuration issues and verifying that settings are applied correctly.
    """
    logger.info("=" * 60)
    logger.info("EAVESDROP CONFIGURATION")
    logger.info("=" * 60)

    # Transcription settings - all properties
    logger.info("TRANSCRIPTION SETTINGS:")
    logger.info(f"  Language: {self.transcription.language}")
    logger.info(f"  Word Timestamps: {self.transcription.word_timestamps}")
    logger.info(f"  Send Last N Segments: {self.transcription.send_last_n_segments}")
    silence_threshold = self.transcription.silence_completion_threshold
    logger.info(f"  Silence Completion Threshold: {silence_threshold}s")
    logger.info(f"  Use VAD: {self.transcription.use_vad}")

    # VAD parameters - all properties
    logger.info("  VAD PARAMETERS:")
    vad = self.transcription.vad_parameters
    logger.info(f"    Threshold: {vad.threshold}")
    logger.info(f"    Neg Threshold: {vad.neg_threshold}")
    logger.info(f"    Min Speech Duration: {vad.min_speech_duration_ms}ms")
    logger.info(f"    Max Speech Duration: {vad.max_speech_duration_s}s")
    logger.info(f"    Min Silence Duration: {vad.min_silence_duration_ms}ms")
    logger.info(f"    Speech Pad: {vad.speech_pad_ms}ms")

    # Buffer settings - all properties
    logger.info("BUFFER SETTINGS:")
    logger.info(f"  Sample Rate: {self.transcription.buffer.sample_rate}")
    logger.info(f"  Max Buffer Duration: {self.transcription.buffer.max_buffer_duration}s")
    logger.info(f"  Cleanup Duration: {self.transcription.buffer.cleanup_duration}s")
    logger.info(f"  Min Chunk Duration: {self.transcription.buffer.min_chunk_duration}s")
    logger.info(f"  Transcription Interval: {self.transcription.buffer.transcription_interval}s")
    logger.info(f"  Clip Audio: {self.transcription.buffer.clip_audio}")
    logger.info(f"  Max Stall Duration: {self.transcription.buffer.max_stall_duration}s")

    # Debug audio settings
    logger.info("DEBUG AUDIO SETTINGS:")
    if self.transcription.debug_audio:
      logger.info(f"  Pre-Buffer Path: {self.transcription.debug_audio.pre_buffer}")
      logger.info(f"  Post-Buffer Path: {self.transcription.debug_audio.post_buffer}")
    else:
      logger.info("  Debug Audio: Disabled")

    # RTSP settings - all properties
    logger.info("RTSP SETTINGS:")
    if self.rtsp.streams:
      logger.info(f"  Streams ({len(self.rtsp.streams)}):")
      for name, url in self.rtsp.streams.items():
        logger.info(f"    {name}: {url}")
    else:
      logger.info("  Streams: None (WebSocket-only mode)")

    logger.info(f"  Cache Waiting Duration: {self.rtsp.cache.waiting_for_listener_duration:.1f}s")
    logger.info(f"  Cache Listener Duration: {self.rtsp.cache.has_listener_cache_duration:.1f}s")

    # System constants
    logger.info("SYSTEM CONSTANTS:")
    logger.info(f"  Cache Path: {CACHE_PATH}")
    logger.info(f"  Single Model: {SINGLE_MODEL}")

    logger.info("=" * 60)


@validate_call
def load_config_from_file(config_path: FilePath) -> EavesdropConfig:
  """Load and validate Eavesdrop configuration from YAML file."""

  logger.info("Loading Eavesdrop configuration", path=str(config_path))

  try:
    # Load YAML content
    with open(config_path, "r", encoding="utf-8") as file:
      config_data = cast(object, yaml.safe_load(file))

  except yaml.YAMLError as e:
    raise ValueError(f"Invalid YAML in configuration file: {e}") from e
  except Exception as e:
    raise ValueError(f"Error reading configuration file: {e}") from e

  # Validate configuration structure
  if config_data is None:
    raise ValueError("Configuration file is empty")

  if not isinstance(config_data, dict):
    raise ValueError("Configuration file must contain a YAML dictionary")

  # Validate and create config using Pydantic
  config = EavesdropConfig.model_validate(config_data)

  # Pretty print the config
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
