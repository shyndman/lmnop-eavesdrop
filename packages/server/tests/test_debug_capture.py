"""Tests for the AudioDebugCapture class."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from eavesdrop.server.streaming.debug_capture import AudioDebugCapture


@dataclass
class MockAudioChunk:
  """Mock AudioChunk for testing."""

  data: np.ndarray
  duration: float
  start_time: float


class TestAudioDebugCapture:
  """Tests for AudioDebugCapture initialization and properties."""

  @pytest.fixture
  def capture(self, tmp_path: Path) -> AudioDebugCapture:
    """Create a capture instance with a temporary directory."""
    return AudioDebugCapture(
      output_path=tmp_path / "debug",
      stream_name="test-stream",
      sample_rate=16000,
    )

  def test_initial_state(self, capture: AudioDebugCapture, tmp_path: Path) -> None:
    """Test capture initial state."""
    assert capture.output_path == tmp_path / "debug"
    assert capture.stream_name == "test-stream"
    assert capture.sample_rate == 16000
    assert capture.chunk_count == 0

  def test_creates_output_directory(self, tmp_path: Path) -> None:
    """Test that output directory is created on initialization."""
    output_path = tmp_path / "nested" / "debug" / "path"
    assert not output_path.exists()

    AudioDebugCapture(
      output_path=output_path,
      stream_name="test",
      sample_rate=16000,
    )

    assert output_path.exists()


class TestCaptureChunk:
  """Tests for AudioDebugCapture.capture() method."""

  @pytest.fixture
  def capture(self, tmp_path: Path) -> AudioDebugCapture:
    """Create a capture instance with a temporary directory."""
    return AudioDebugCapture(
      output_path=tmp_path / "debug",
      stream_name="test-stream",
      sample_rate=16000,
    )

  @pytest.fixture
  def mock_chunk(self) -> MockAudioChunk:
    """Create a mock audio chunk with 1 second of audio."""
    return MockAudioChunk(
      data=np.zeros(16000, dtype=np.float32),
      duration=1.0,
      start_time=0.0,
    )

  def test_capture_writes_wav_file(
    self, capture: AudioDebugCapture, mock_chunk: MockAudioChunk, tmp_path: Path
  ) -> None:
    """Test that capture writes a WAV file."""
    capture.capture(mock_chunk)

    wav_files = list((tmp_path / "debug").glob("*.wav"))
    assert len(wav_files) == 1
    assert "test-stream" in wav_files[0].name
    assert "_post.wav" in wav_files[0].name

  def test_capture_increments_chunk_count(
    self, capture: AudioDebugCapture, mock_chunk: MockAudioChunk
  ) -> None:
    """Test that capture increments the chunk count."""
    assert capture.chunk_count == 0

    capture.capture(mock_chunk)
    assert capture.chunk_count == 1

    capture.capture(mock_chunk)
    assert capture.chunk_count == 2

  def test_capture_creates_unique_filenames(
    self, capture: AudioDebugCapture, tmp_path: Path
  ) -> None:
    """Test that multiple captures create unique filenames."""
    chunk1 = MockAudioChunk(
      data=np.zeros(16000, dtype=np.float32),
      duration=1.0,
      start_time=0.0,
    )
    chunk2 = MockAudioChunk(
      data=np.zeros(16000, dtype=np.float32),
      duration=1.0,
      start_time=1.0,
    )

    capture.capture(chunk1)
    capture.capture(chunk2)

    wav_files = list((tmp_path / "debug").glob("*.wav"))
    assert len(wav_files) == 2
    # Filenames should be different (different start_time in chunk_id)
    assert wav_files[0].name != wav_files[1].name

  def test_capture_wav_file_is_valid(self, capture: AudioDebugCapture, tmp_path: Path) -> None:
    """Test that captured WAV file can be read back."""
    audio_data = np.random.rand(16000).astype(np.float32)
    chunk = MockAudioChunk(
      data=audio_data,
      duration=1.0,
      start_time=0.0,
    )

    capture.capture(chunk)

    wav_files = list((tmp_path / "debug").glob("*.wav"))
    assert len(wav_files) == 1

    # Read the file back and verify
    read_data, read_sr = sf.read(wav_files[0])
    assert read_sr == 16000
    assert len(read_data) == 16000
    # Check audio data matches (within WAV encoding tolerance)
    # WAV encoding introduces small precision loss, so use decimal=4
    np.testing.assert_array_almost_equal(read_data, audio_data, decimal=4)


class TestCaptureRaw:
  """Tests for AudioDebugCapture.capture_raw() method."""

  @pytest.fixture
  def capture(self, tmp_path: Path) -> AudioDebugCapture:
    """Create a capture instance with a temporary directory."""
    return AudioDebugCapture(
      output_path=tmp_path / "debug",
      stream_name="test-stream",
      sample_rate=16000,
    )

  def test_capture_raw_writes_wav_file(self, capture: AudioDebugCapture, tmp_path: Path) -> None:
    """Test that capture_raw writes a WAV file."""
    audio_data = np.zeros(16000, dtype=np.float32)

    capture.capture_raw(audio_data, start_time=0.0, duration=1.0)

    wav_files = list((tmp_path / "debug").glob("*.wav"))
    assert len(wav_files) == 1
    assert "test-stream" in wav_files[0].name

  def test_capture_raw_increments_chunk_count(self, capture: AudioDebugCapture) -> None:
    """Test that capture_raw increments the chunk count."""
    audio_data = np.zeros(16000, dtype=np.float32)

    assert capture.chunk_count == 0
    capture.capture_raw(audio_data, start_time=0.0, duration=1.0)
    assert capture.chunk_count == 1


class TestClose:
  """Tests for AudioDebugCapture.close() method."""

  @pytest.fixture
  def capture(self, tmp_path: Path) -> AudioDebugCapture:
    """Create a capture instance with a temporary directory."""
    return AudioDebugCapture(
      output_path=tmp_path / "debug",
      stream_name="test-stream",
      sample_rate=16000,
    )

  def test_close_is_safe_with_no_captures(self, capture: AudioDebugCapture) -> None:
    """Test that close works with no captures."""
    # Should not raise
    capture.close()

  def test_close_after_captures(self, capture: AudioDebugCapture) -> None:
    """Test that close works after captures."""
    chunk = MockAudioChunk(
      data=np.zeros(16000, dtype=np.float32),
      duration=1.0,
      start_time=0.0,
    )
    capture.capture(chunk)
    capture.capture(chunk)

    # Should not raise
    capture.close()
    assert capture.chunk_count == 2
