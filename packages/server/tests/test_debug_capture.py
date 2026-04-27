"""Tests for the AudioDebugCapture class."""

from pathlib import Path
from typing import Callable, cast
import wave

import numpy as np
import pytest
from numpy.typing import NDArray

from eavesdrop.server.streaming.debug_capture import AudioDebugCapture
from eavesdrop.server.streaming.processor import AudioChunk


Float32Audio = NDArray[np.float32]


def _chunk(*, start_time: float = 0.0, duration: float = 1.0) -> AudioChunk:
  return AudioChunk(
    data=np.zeros(16000, dtype=np.float32),
    duration=duration,
    start_time=start_time,
  )


def _capture_raw(
  capture: AudioDebugCapture,
  audio_data: Float32Audio,
  *,
  start_time: float,
  duration: float,
) -> None:
  capture_raw = cast(Callable[[Float32Audio, float, float], None], capture.capture_raw)
  capture_raw(audio_data, start_time, duration)


def _read_wav(path: Path) -> tuple[Float32Audio, int]:
  with wave.open(str(path), "rb") as wav_file:
    sample_rate = wav_file.getframerate()
    sample_count = wav_file.getnframes()
    frame_bytes = wav_file.readframes(sample_count)

  pcm16 = np.frombuffer(frame_bytes, dtype=np.int16)
  return pcm16.astype(np.float32) / 32768.0, sample_rate


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

    _ = AudioDebugCapture(
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
  def mock_chunk(self) -> AudioChunk:
    """Create a mock audio chunk with 1 second of audio."""
    return _chunk()

  def test_capture_writes_wav_file(
    self, capture: AudioDebugCapture, mock_chunk: AudioChunk, tmp_path: Path
  ) -> None:
    """Test that capture writes a WAV file."""
    capture.capture(mock_chunk)

    wav_files = list((tmp_path / "debug").glob("*.wav"))
    assert len(wav_files) == 1
    assert "test-stream" in wav_files[0].name
    assert "_post.wav" in wav_files[0].name

  def test_capture_increments_chunk_count(
    self, capture: AudioDebugCapture, mock_chunk: AudioChunk
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
    chunk1 = _chunk(start_time=0.0)
    chunk2 = _chunk(start_time=1.0)

    capture.capture(chunk1)
    capture.capture(chunk2)

    wav_files = list((tmp_path / "debug").glob("*.wav"))
    assert len(wav_files) == 2
    # Filenames should be different (different start_time in chunk_id)
    assert wav_files[0].name != wav_files[1].name

  def test_capture_wav_file_is_valid(self, capture: AudioDebugCapture, tmp_path: Path) -> None:
    """Test that captured WAV file can be read back."""
    audio_data = np.random.rand(16000).astype(np.float32)
    chunk = AudioChunk(
      data=audio_data,
      duration=1.0,
      start_time=0.0,
    )

    capture.capture(chunk)

    wav_files = list((tmp_path / "debug").glob("*.wav"))
    assert len(wav_files) == 1

    # Read the file back and verify
    typed_read_data, read_sr = _read_wav(wav_files[0])
    assert read_sr == 16000
    assert len(typed_read_data) == 16000
    # Check audio data matches (within WAV encoding tolerance)
    # WAV encoding introduces small precision loss, so use decimal=4
    np.testing.assert_array_almost_equal(typed_read_data, audio_data, decimal=4)


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

    _capture_raw(capture, audio_data, start_time=0.0, duration=1.0)

    wav_files = list((tmp_path / "debug").glob("*.wav"))
    assert len(wav_files) == 1
    assert "test-stream" in wav_files[0].name

  def test_capture_raw_increments_chunk_count(self, capture: AudioDebugCapture) -> None:
    """Test that capture_raw increments the chunk count."""
    audio_data = np.zeros(16000, dtype=np.float32)

    assert capture.chunk_count == 0
    _capture_raw(capture, audio_data, start_time=0.0, duration=1.0)
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
    chunk = _chunk()
    capture.capture(chunk)
    capture.capture(chunk)

    # Should not raise
    capture.close()
    assert capture.chunk_count == 2
