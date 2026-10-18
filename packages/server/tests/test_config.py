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

    assert config.language == "en"
    assert config.use_vad is True
    assert config.send_last_n_segments == 10
    assert config.vad_parameters.threshold == 0.5
    assert config.vad_parameters.neg_threshold is None
    assert config.vad_parameters.min_speech_duration_ms == 0
    assert config.vad_parameters.max_speech_duration_s == float("inf")
    assert config.vad_parameters.min_silence_duration_ms == 800
    assert config.vad_parameters.speech_pad_ms == 400
    assert isinstance(config.buffer, BufferConfig)

  def test_vad_parameters_validate_and_align(self):
    """Test that repo-owned VAD parameters validate and align with silence threshold."""
    config = TranscriptionConfig.model_validate(
      {
        "silence_completion_threshold": 1.2,
        "vad_parameters": {
          "threshold": 0.4,
          "neg_threshold": 0.25,
          "min_speech_duration_ms": 50,
          "max_speech_duration_s": 12.5,
          "speech_pad_ms": 250,
        },
      }
    )

    assert config.vad_parameters.threshold == 0.4
    assert config.vad_parameters.neg_threshold == 0.25
    assert config.vad_parameters.min_speech_duration_ms == 50
    assert config.vad_parameters.max_speech_duration_s == 12.5
    assert config.vad_parameters.min_silence_duration_ms == 1200
    assert config.vad_parameters.speech_pad_ms == 250

  def test_min_silence_duration_rejection(self):
    """Test that min_silence_duration_ms cannot be configured directly."""
    with pytest.raises(ValueError, match="min_silence_duration_ms cannot be specified"):
      TranscriptionConfig.model_validate(
        {
          "vad_parameters": {
            "min_silence_duration_ms": 100,
          }
        }
      )

  @pytest.mark.parametrize("option", ["initial_prompt", "hotwords", "beam_size", "same_output_threshold", "custom_model", "gpu_name", "task"])
  def test_removed_options_fail_visibly(self, option: str) -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
      TranscriptionConfig.model_validate({option: "obsolete"})

  def test_non_english_language_is_rejected(self) -> None:
    with pytest.raises(ValueError, match="Input should be 'en'"):
      TranscriptionConfig.model_validate({"language": "fr"})


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
  language: "en"
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

    assert config.transcription.language == "en"
    assert config.transcription.use_vad is False
    assert config.transcription.buffer.sample_rate == 22050
    assert config.transcription.buffer.clip_audio is True
    assert len(config.rtsp.streams) == 2
    assert "office" in config.rtsp.streams
    assert "lobby" in config.rtsp.streams

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
  custom_model: "/path/to/model"  # This should cause mutual exclusion error
"""
    fake_filesystem.create_file("/test/config.yaml", contents=config_data)

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
      load_config_from_file(Path("/test/config.yaml"))
