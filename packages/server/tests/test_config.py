"""Tests for the configuration module."""

from pathlib import Path

import pytest

from eavesdrop.server.config import (
  BufferConfig,
  EavesdropConfig,
  TranscriptionConfig,
  load_config_from_file,
)


@pytest.fixture
def fake_filesystem(fs):
  """Variable name 'fs' causes a pylint warning. Provide a longer name
  acceptable to pylint for use in tests.
  """
  yield fs


class TestBufferConfig:
  """Test BufferConfig validation and functionality."""

  def test_buffer_config_defaults(self):
    """Test that BufferConfig creates with expected defaults."""
    config = BufferConfig()

    assert config.sample_rate == 16000
    assert config.max_buffer_duration == 45.0
    assert config.cleanup_duration == 30.0
    assert config.min_chunk_duration == 1.0
    assert config.transcription_interval == 2.0
    assert config.clip_audio is False
    assert config.max_stall_duration == 25.0

  def test_buffer_config_duration_validation(self):
    """Test that cleanup_duration must be less than max_buffer_duration."""
    # Valid case
    config = BufferConfig(cleanup_duration=20.0, max_buffer_duration=30.0)
    assert config.cleanup_duration == 20.0

    # Invalid case - cleanup >= max
    with pytest.raises(
      ValueError, match="cleanup_duration.*must be less than.*max_buffer_duration"
    ):
      BufferConfig(cleanup_duration=30.0, max_buffer_duration=30.0)

    with pytest.raises(
      ValueError, match="cleanup_duration.*must be less than.*max_buffer_duration"
    ):
      BufferConfig(cleanup_duration=35.0, max_buffer_duration=30.0)

  def test_buffer_config_positive_values(self):
    """Test that numeric fields require positive values."""
    with pytest.raises(ValueError):
      BufferConfig(sample_rate=0)

    with pytest.raises(ValueError):
      BufferConfig(max_buffer_duration=-1.0)


class TestTranscriptionConfig:
  """Test TranscriptionConfig validation and functionality."""

  def test_transcription_config_defaults(self):
    """Test that TranscriptionConfig creates with expected defaults."""
    config = TranscriptionConfig()

    assert config.model == "distil-medium.en"
    assert config.custom_model is None
    assert config.language == "en"
    assert config.use_vad is True
    assert config.send_last_n_segments == 10
    assert config.device_index == 0
    assert isinstance(config.buffer, BufferConfig)

  def test_model_custom_model_mutual_exclusion(self, fake_filesystem):
    """Test that model and custom_model cannot both be specified."""
    # Valid cases
    config1 = TranscriptionConfig(model="base")
    assert config1.model == "base"
    assert config1.custom_model is None

    # Create fake files for custom_model testing
    fake_filesystem.create_file("/path/to/model")
    config2 = TranscriptionConfig(model=None, custom_model=Path("/path/to/model"))
    assert config2.model is None
    assert str(config2.custom_model) == "/path/to/model"

    # Invalid case - both specified
    fake_filesystem.create_file("/another/model")
    with pytest.raises(ValueError, match="Cannot specify both 'model' and 'custom_model'"):
      TranscriptionConfig.model_validate({"model": "base", "custom_model": "/another/model"})

  def test_device_index_rejection(self):
    """Test that device_index cannot be specified in input."""
    with pytest.raises(ValueError, match="device_index cannot be specified"):
      TranscriptionConfig.model_validate({"device_index": 1})

  def test_model_path_property(self, fake_filesystem):
    """Test that model_path property returns the correct path."""
    # Standard model
    config1 = TranscriptionConfig(model="base")
    assert config1.model_path == "base"

    # Custom model takes precedence
    fake_filesystem.create_file("/custom/path")
    config2 = TranscriptionConfig(model=None, custom_model=Path("/custom/path"))
    assert config2.model_path == "/custom/path"

    # Neither specified gets default
    config3 = TranscriptionConfig(model=None)
    assert config3.model_path == "distil-medium.en"


class TestEavesdropConfig:
  """Test EavesdropConfig integration."""

  def test_eavesdrop_config_defaults(self):
    """Test that EavesdropConfig creates with nested defaults."""
    config = EavesdropConfig()

    assert isinstance(config.transcription, TranscriptionConfig)
    assert isinstance(config.rtsp.streams, dict)
    assert len(config.rtsp.streams) == 0


class TestConfigFileLoading:
  """Test configuration file loading and validation."""

  def test_load_valid_config_file(self, fake_filesystem):
    """Test loading a valid YAML configuration file."""
    config_data = """
transcription:
  model: "base"
  language: "fr"
  use_vad: false
  buffer:
    sample_rate: 22050
    clip_audio: true

rtsp:
  streams:
    office: "rtsp://camera1:554/stream"
    lobby: "rtsp://camera2:554/stream"
"""
    fake_filesystem.create_file("/test/config.yaml", contents=config_data)
    config = load_config_from_file(Path("/test/config.yaml"))

    assert config.transcription.model == "base"
    assert config.transcription.language == "fr"
    assert config.transcription.use_vad is False
    assert config.transcription.buffer.sample_rate == 22050
    assert config.transcription.buffer.clip_audio is True
    assert len(config.rtsp.streams) == 2
    assert "office" in config.rtsp.streams
    assert "lobby" in config.rtsp.streams

  def test_hotwords_configuration(self, fake_filesystem):
    """Test that hotwords can be configured from YAML."""
    config_data = """
transcription:
  hotwords: ["hello", "world", "test"]
"""
    fake_filesystem.create_file("/test/config.yaml", contents=config_data)
    config = load_config_from_file(Path("/test/config.yaml"))

    assert config.transcription.hotwords == ["hello", "world", "test"]

  def test_hotwords_empty_default(self):
    """Test that hotwords defaults to empty list."""
    config = TranscriptionConfig()
    assert config.hotwords == []

  def test_load_empty_config_file(self, fake_filesystem):
    """Test error handling for empty config file."""
    fake_filesystem.create_file("/test/empty.yaml", contents="")

    with pytest.raises(ValueError, match="Configuration file is empty"):
      load_config_from_file(Path("/test/empty.yaml"))

  def test_load_invalid_yaml(self, fake_filesystem):
    """Test error handling for invalid YAML."""
    fake_filesystem.create_file("/test/invalid.yaml", contents="invalid: yaml: content: [")

    with pytest.raises(ValueError, match="Invalid YAML"):
      load_config_from_file(Path("/test/invalid.yaml"))

  def test_load_nonexistent_file(self, fake_filesystem):
    """Test error handling for nonexistent file."""
    with pytest.raises(ValueError, match="Path does not point to a file"):
      load_config_from_file(Path("/nonexistent/file.yaml"))

  def test_load_config_with_validation_errors(self, fake_filesystem):
    """Test error handling for config with validation errors."""
    # Create fake model file for the test
    fake_filesystem.create_file("/path/to/model")

    config_data = """
transcription:
  model: "base"
  custom_model: "/path/to/model"  # This should cause mutual exclusion error
"""
    fake_filesystem.create_file("/test/config.yaml", contents=config_data)

    with pytest.raises(ValueError, match="Cannot specify both 'model' and 'custom_model'"):
      load_config_from_file(Path("/test/config.yaml"))
