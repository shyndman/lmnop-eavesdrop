from pathlib import Path

import yaml

from .logs import get_logger


class RTSPConfig:
  """
  Configuration loader and validator for RTSP streams.

  Handles loading, parsing, and validating YAML configuration files that define
  RTSP streams for the eavesdrop server. Provides strict validation with detailed
  error reporting.
  """

  def __init__(self, config_path: str | None):
    """
    Initialize the RTSP configuration loader.

    Args:
        config_path: Path to the YAML configuration file, or None to disable RTSP
    """
    self.config_path = config_path
    self.streams: dict[str, str] = {}
    self.logger = get_logger("rtsp_config")

  def load_and_validate(self) -> dict[str, str]:
    """
    Load and validate the RTSP configuration file.

    Returns:
        Dictionary mapping stream names to RTSP URLs

    Raises:
        ValueError: If the configuration is invalid or cannot be loaded
    """
    if not self.config_path:
      self.logger.debug("No RTSP configuration path provided")
      return {}

    config_file = Path(self.config_path)

    # Check file existence
    if not config_file.exists():
      raise ValueError(f"RTSP configuration file not found: {self.config_path}")

    if not config_file.is_file():
      raise ValueError(f"RTSP configuration path is not a file: {self.config_path}")

    self.logger.info("Loading RTSP configuration", path=self.config_path)

    try:
      # Load YAML content
      with open(config_file, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    except yaml.YAMLError as e:
      raise ValueError(f"Invalid YAML in configuration file: {e}")
    except Exception as e:
      raise ValueError(f"Error reading configuration file: {e}")

    # Validate configuration structure
    if config_data is None:
      raise ValueError("Configuration file is empty")

    if not isinstance(config_data, dict):
      raise ValueError("Configuration file must contain a YAML dictionary")

    if "streams" not in config_data:
      raise ValueError("Configuration file must contain a top-level 'streams' key")

    streams_config = config_data["streams"]

    if not isinstance(streams_config, dict):
      raise ValueError("'streams' must be a dictionary mapping stream names to RTSP URLs")

    if not streams_config:
      self.logger.warning("No RTSP streams configured in file")
      return {}

    # Validate each stream configuration
    validated_streams = {}
    for stream_name, rtsp_url in streams_config.items():
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
      "RTSP configuration loaded successfully",
      stream_count=len(validated_streams),
      streams=list(validated_streams.keys()),
    )

    return validated_streams
