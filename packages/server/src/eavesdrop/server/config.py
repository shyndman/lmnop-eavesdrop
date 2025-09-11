import os

import torch
import yaml
from faster_whisper.vad import VadOptions
from pydantic import BaseModel, Field, model_validator, validate_call
from pydantic.dataclasses import dataclass
from pydantic.types import FilePath

from eavesdrop.server.constants import CACHE_PATH, SINGLE_MODEL, TASK
from eavesdrop.server.logs import get_logger

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
        f"max_buffer_duration ({self.max_buffer_duration}s)"
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

  no_speech_thresh: float = Field(default=0.45, ge=0.0, le=1.0)
  """Segments with no speech probability above this threshold will be discarded."""

  same_output_threshold: int = Field(default=10, gt=0)
  """Number of repeated outputs before considering it as a valid segment."""

  use_vad: bool = True
  """Whether to use Voice Activity Detection."""

  # Model configuration - these fields will be validated by model validators
  model: str | None = "distil-medium.en"
  """Whisper model size or path (mutually exclusive with custom_model)."""

  custom_model: FilePath | None = None
  """Path to custom Whisper model file (mutually exclusive with model)."""

  language: str = "en"
  """Language for transcription."""

  initial_prompt: str | None = None
  """Initial prompt for whisper inference."""

  hotwords: list[str] = Field(default_factory=list)
  """Hotwords for whisper inference to improve recognition of specific terms."""

  vad_parameters: VadOptions = Field(default_factory=VadOptions)
  """Voice Activity Detection parameters."""

  gpu_name: str | None = None
  """GPU device name to use (device_index computed from this)."""

  num_workers: int = Field(default=1, gt=0)
  """Number of workers for parallel transcription processing."""

  device_index: int = Field(default=0, ge=0, exclude=True)
  """GPU device index to use (computed from gpu_name)."""

  buffer: BufferConfig = Field(default_factory=BufferConfig)
  """Audio buffer configuration for streaming transcription."""

  @model_validator(mode="before")
  @classmethod
  def validate_model_configuration(cls, values: dict) -> dict:
    """Validate model/custom_model mutual exclusion and reject device_index input."""
    if not isinstance(values, dict):
      return values

    model = values.get("model")
    custom_model = values.get("custom_model")

    # Check mutual exclusion
    if model and custom_model:
      raise ValueError(
        "Cannot specify both 'model' and 'custom_model' in the same configuration. "
        "Use 'model' for standard Whisper models (e.g. 'distil-medium.en') or "
        "'custom_model' for a path to a local model file."
      )

    # If neither specified, use default model
    if not model and not custom_model:
      values["model"] = "distil-medium.en"

    # Reject device_index in input - it should be computed from gpu_name
    if "device_index" in values:
      raise ValueError(
        "device_index cannot be specified in config file. "
        "Use 'gpu_name' instead to specify the GPU device, and device_index will be "
        "computed automatically."
      )

    return values

  @model_validator(mode="after")
  def resolve_gpu_device_index(self) -> "TranscriptionConfig":
    """Compute device_index from gpu_name if provided."""
    if self.gpu_name:
      self.device_index = self._resolve_gpu_device_index(self.gpu_name)
    return self

  def _resolve_gpu_device_index(self, gpu_name: str) -> int:
    """Resolve GPU device index from GPU name."""
    if not torch.cuda.is_available():
      return 0

    device_count = torch.cuda.device_count()
    for i in range(device_count):
      device_name = torch.cuda.get_device_name(i)
      if gpu_name in device_name:
        return i

    available_gpus = [torch.cuda.get_device_name(i) for i in range(device_count)]
    raise ValueError(f"GPU '{gpu_name}' not found. Available GPUs: {available_gpus}")

  @property
  def model_path(self) -> str:
    """Get the final model path to use (custom_model takes precedence over model)."""
    if self.custom_model is not None:
      return str(self.custom_model)
    if self.model is not None:
      return self.model
    # This should never happen due to validators, but provide a fallback
    return "distil-medium.en"


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
    logger.info(f"  Model: {self.transcription.model}")
    logger.info(f"  Custom Model: {self.transcription.custom_model}")
    logger.info(f"  GPU Name: {self.transcription.gpu_name}")
    logger.info(f"  Language: {self.transcription.language}")
    logger.info(f"  Initial Prompt: {self.transcription.initial_prompt}")
    logger.info(f"  Hotwords: {self.transcription.hotwords}")
    logger.info(f"  Send Last N Segments: {self.transcription.send_last_n_segments}")
    logger.info(f"  No Speech Threshold: {self.transcription.no_speech_thresh}")
    logger.info(f"  Same Output Threshold: {self.transcription.same_output_threshold}")
    logger.info(f"  Num Workers: {self.transcription.num_workers}")
    logger.info(f"  Use VAD: {self.transcription.use_vad}")

    # VAD parameters - all properties
    logger.info("  VAD PARAMETERS:")
    vad = self.transcription.vad_parameters
    logger.info(f"    Onset: {vad.onset}")
    logger.info(f"    Offset: {vad.offset}")
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
    logger.info(f"  Task: {TASK}")
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
      config_data = yaml.safe_load(file)

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
