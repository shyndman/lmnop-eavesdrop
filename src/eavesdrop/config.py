import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from faster_whisper.vad import VadOptions
from pydantic import BaseModel, Field

from .constants import CACHE_PATH, SINGLE_MODEL, TASK
from .logs import get_logger


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


@dataclass
class TranscriptionConfig:
  """Configuration for transcription processing behavior."""

  # Transcription behavior
  send_last_n_segments: int
  """Number of most recent segments to send to the client."""

  no_speech_thresh: float
  """Segments with no speech probability above this threshold will be discarded."""

  same_output_threshold: int
  """Number of repeated outputs before considering it as a valid segment."""

  use_vad: bool
  """Whether to use Voice Activity Detection."""

  clip_audio: bool
  """Whether to clip audio with no valid segments."""

  # Model configuration
  model: str
  """Whisper model size or path."""

  language: str | None
  """Language for transcription."""

  initial_prompt: str | None
  """Initial prompt for whisper inference."""

  vad_parameters: VadOptions | dict | None
  """Voice Activity Detection parameters."""

  device_index: int
  """GPU device index to use."""


class EavesdropConfig:
  """
  Configuration loader and validator for Eavesdrop server.

  Handles loading, parsing, and validating YAML configuration files that define
  transcription settings and RTSP streams for the eavesdrop server.
  Provides strict validation with detailed error reporting.
  """

  def __init__(self, config_path: str):
    """
    Initialize the Eavesdrop configuration loader.

    Args:
        config_path: Path to the YAML configuration file (required)
    """
    if not config_path:
      raise ValueError("Configuration path is required")
    self.config_path = config_path
    self.streams: dict[str, str] = {}
    self.logger = get_logger("eavesdrop_config")

  def load_and_validate(self) -> tuple[dict[str, str], TranscriptionConfig, RTSPCacheConfig]:
    """
    Load and validate the Eavesdrop configuration file.

    Returns:
        Tuple of (rtsp_streams, transcription_config, rtsp_cache_config)

    Raises:
        ValueError: If the configuration is invalid or cannot be loaded
    """
    config_file = Path(self.config_path)

    # Check file existence
    if not config_file.exists():
      raise ValueError(f"Configuration file not found: {self.config_path}")

    if not config_file.is_file():
      raise ValueError(f"Configuration path is not a file: {self.config_path}")

    self.logger.info("Loading Eavesdrop configuration", path=self.config_path)

    try:
      # Load YAML content
      with open(config_file, "r", encoding="utf-8") as file:
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

    # Validate transcription section
    if "transcription" not in config_data:
      raise ValueError("Configuration file must contain a top-level 'transcription' key")

    transcription_config = self._validate_transcription_config(config_data["transcription"])

    # Validate streams section (optional)
    rtsp_streams = {}
    if "streams" in config_data:
      rtsp_streams = self._validate_streams_config(config_data["streams"])

    # Validate rtsp.cache section (optional)
    rtsp_cache_config = self._validate_rtsp_cache_config(config_data.get("rtsp", {}))

    # Pretty print the entire config at INFO level
    self._pretty_print_config(config_data, rtsp_streams, transcription_config, rtsp_cache_config)

    return rtsp_streams, transcription_config, rtsp_cache_config

  def _validate_transcription_config(
    self, transcription_data: dict[str, Any]
  ) -> TranscriptionConfig:
    """Validate transcription configuration section."""
    if not isinstance(transcription_data, dict):
      raise ValueError("'transcription' must be a dictionary")

    # Extract and validate model configuration
    model = transcription_data.get("model")
    custom_model = transcription_data.get("custom_model")

    if model and custom_model:
      raise ValueError(
        "Cannot specify both 'model' and 'custom_model' in the same configuration. "
        "Use 'model' for standard Whisper models (e.g. 'distil-medium.en') or "
        "'custom_model' for a path to a local model file."
      )

    if not model and not custom_model:
      raise ValueError(
        "Must specify either 'model' or 'custom_model' in the transcription section. "
        "Use 'model' for standard Whisper models (e.g. 'distil-medium.en') or "
        "'custom_model' for a path to a local model file."
      )

    # Use the specified model (either standard or custom)
    final_model = custom_model or model
    assert final_model is not None  # We already validated above that one must exist

    # Validate custom model path if provided
    if custom_model and not Path(custom_model).exists():
      raise ValueError(
        f"Custom model path does not exist: {custom_model}\n"
        "Please ensure the model file exists and the path is correct."
      )

    # Extract GPU configuration
    gpu_name = transcription_data.get("gpu_name")

    # Reject device_index in config - it should be computed from gpu_name
    if "device_index" in transcription_data:
      raise ValueError(
        "device_index cannot be specified in config file. "
        "Use 'gpu_name' instead to specify the GPU device, and device_index will be "
        "computed automatically."
      )

    # Compute device_index from gpu_name if provided, otherwise use default
    if gpu_name:
      device_index = self._resolve_gpu_device_index(gpu_name)
    else:
      device_index = 0  # Default to first GPU

    # Validate range constraints
    send_last_n_segments = transcription_data.get("send_last_n_segments", 10)
    no_speech_thresh = transcription_data.get("no_speech_thresh", 0.45)
    same_output_threshold = transcription_data.get("same_output_threshold", 10)

    if send_last_n_segments <= 0:
      raise ValueError(
        f"send_last_n_segments must be greater than 0, got {send_last_n_segments}. "
        "This controls how many recent transcription segments are sent to clients."
      )

    if not (0.0 <= no_speech_thresh <= 1.0):
      raise ValueError(
        f"no_speech_thresh must be between 0.0 and 1.0, got {no_speech_thresh}. "
        "This is the probability threshold above which segments are considered to contain "
        "no speech."
      )

    if same_output_threshold <= 0:
      raise ValueError(
        f"same_output_threshold must be greater than 0, got {same_output_threshold}. "
        "This controls how many repeated outputs are required before considering a segment valid."
      )

    # Create TranscriptionConfig with validated values
    return TranscriptionConfig(
      send_last_n_segments=send_last_n_segments,
      no_speech_thresh=no_speech_thresh,
      same_output_threshold=same_output_threshold,
      use_vad=transcription_data.get("use_vad", True),
      clip_audio=transcription_data.get("clip_audio", False),
      model=final_model,
      language=transcription_data.get("language"),
      initial_prompt=transcription_data.get("initial_prompt"),
      vad_parameters=transcription_data.get("vad_parameters"),
      device_index=device_index,
    )

  def _validate_streams_config(self, streams_data: dict[str, Any]) -> dict[str, str]:
    """Validate streams configuration section."""
    if not isinstance(streams_data, dict):
      raise ValueError("'streams' must be a dictionary mapping stream names to RTSP URLs")

    if not streams_data:
      self.logger.warning("No RTSP streams configured in file")
      return {}

    # Validate each stream configuration
    validated_streams = {}
    for stream_name, rtsp_url in streams_data.items():
      # Validate stream name
      if not isinstance(stream_name, str):
        raise ValueError(
          f"Stream name must be a string, got {type(stream_name).__name__}: {stream_name}"
        )

      if not stream_name.strip():
        raise ValueError("Stream name cannot be empty or whitespace")

      # Validate RTSP URL
      if not isinstance(rtsp_url, str):
        raise ValueError(
          f"RTSP URL for stream '{stream_name}' must be a string, got {type(rtsp_url).__name__}: "
          f"{rtsp_url}"
        )

      if not rtsp_url.strip():
        raise ValueError(f"RTSP URL for stream '{stream_name}' cannot be empty or whitespace")

      # Basic URL format validation
      rtsp_url = rtsp_url.strip()
      if not rtsp_url.startswith(("rtsp://", "rtmp://", "http://", "https://")):
        self.logger.warning(
          "RTSP URL does not start with expected protocol", stream=stream_name, url=rtsp_url
        )

      validated_streams[stream_name.strip()] = rtsp_url
      self.logger.debug("Validated stream", stream=stream_name, url=rtsp_url)

    self.streams = validated_streams
    self.logger.info(
      "RTSP streams validated successfully",
      stream_count=len(validated_streams),
      streams=list(validated_streams.keys()),
    )

    return validated_streams

  def _validate_rtsp_cache_config(self, rtsp_data: dict[str, Any]) -> RTSPCacheConfig:
    """Validate RTSP cache configuration section."""
    cache_data = rtsp_data.get("cache", {})

    if cache_data and not isinstance(cache_data, dict):
      raise ValueError("rtsp.cache must be a dictionary")

    try:
      # Use Pydantic model validation
      rtsp_cache_config = RTSPCacheConfig.model_validate(cache_data)

      self.logger.debug(
        "RTSP cache configuration validated",
        waiting_for_listener_duration=rtsp_cache_config.waiting_for_listener_duration,
        has_listener_cache_duration=rtsp_cache_config.has_listener_cache_duration,
      )

      return rtsp_cache_config
    except Exception as e:
      raise ValueError(f"Invalid RTSP cache configuration: {e}") from e

  def _resolve_gpu_device_index(self, gpu_name: str) -> int:
    """Resolve GPU device index from GPU name."""
    if not torch.cuda.is_available():
      self.logger.warning("CUDA not available, ignoring gpu_name setting")
      return 0

    device_count = torch.cuda.device_count()
    for i in range(device_count):
      device_name = torch.cuda.get_device_name(i)
      if gpu_name in device_name:
        self.logger.info(f"Found GPU '{gpu_name}' at device index {i}")
        return i

    available_gpus = [torch.cuda.get_device_name(i) for i in range(device_count)]
    raise ValueError(f"GPU '{gpu_name}' not found. Available GPUs: {available_gpus}")

  def _pretty_print_config(
    self,
    config_data: dict[str, Any],
    rtsp_streams: dict[str, str],
    transcription_config: TranscriptionConfig,
    rtsp_cache_config: RTSPCacheConfig,
  ) -> None:
    """Pretty print the entire configuration at INFO level."""
    self.logger.info("=" * 60)
    self.logger.info("EAVESDROP CONFIGURATION")
    self.logger.info("=" * 60)

    # Transcription settings
    self.logger.info("TRANSCRIPTION SETTINGS:")
    self.logger.info(f"  Model: {transcription_config.model}")
    self.logger.info(f"  Task: {TASK} (constant)")
    self.logger.info(f"  Language: {transcription_config.language or 'auto-detect'}")
    self.logger.info(f"  Use VAD: {transcription_config.use_vad}")
    self.logger.info(f"  Device Index: {transcription_config.device_index}")
    self.logger.info(f"  Send Last N Segments: {transcription_config.send_last_n_segments}")
    self.logger.info(f"  No Speech Threshold: {transcription_config.no_speech_thresh}")
    self.logger.info(f"  Same Output Threshold: {transcription_config.same_output_threshold}")
    self.logger.info(f"  Clip Audio: {transcription_config.clip_audio}")

    # Hardware constants
    self.logger.info("HARDWARE SETTINGS:")
    self.logger.info(f"  Cache Path: {CACHE_PATH} (constant)")
    self.logger.info(f"  Single Model: {SINGLE_MODEL} (constant)")

    # RTSP streams
    if rtsp_streams:
      self.logger.info(f"RTSP STREAMS ({len(rtsp_streams)}):")
      for name, url in rtsp_streams.items():
        self.logger.info(f"  {name}: {url}")
    else:
      self.logger.info("RTSP STREAMS: None (WebSocket-only mode)")

    # RTSP cache settings
    self.logger.info("RTSP CACHE SETTINGS:")
    self.logger.info(
      f"  Waiting for listener duration: {rtsp_cache_config.waiting_for_listener_duration:.1f}s"
    )
    self.logger.info(
      f"  Has listener cache duration: {rtsp_cache_config.has_listener_cache_duration:.1f}s"
    )

    self.logger.info("=" * 60)


def get_env_float(key: str, default: float) -> float:
  """Get a float from an environment variable."""
  return float(os.getenv(key, str(default)))


def get_env_int(key: str, default: int) -> int:
  """Get an int from an environment variable."""
  return int(os.getenv(key, str(default)))


def get_env_bool(key: str, default: bool) -> bool:
  """Get a bool from an environment variable."""
  return os.getenv(key, str(default)).lower() == "true"
