"""
Audio debug capture utility for post-buffer audio analysis.

Consolidates debug audio capture logic that was previously scattered
across TranscriptionServer and StreamingTranscriptionProcessor.
"""

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from eavesdrop.common import get_logger

if TYPE_CHECKING:
  from eavesdrop.server.streaming.processor import AudioChunk


class AudioDebugCapture:
  """
  Captures post-buffer audio chunks for debugging and analysis.

  Called by StreamingTranscriptionProcessor after audio chunks are dequeued
  from the buffer and before transcription. Writes each chunk to a separate
  WAV file with stream/timestamp/chunk metadata in the filename.

  :param output_path: Directory path for output WAV files.
  :type output_path: Path
  :param stream_name: Unique identifier for the stream.
  :type stream_name: str
  :param sample_rate: Audio sample rate in Hz.
  :type sample_rate: int
  """

  def __init__(
    self,
    output_path: Path,
    stream_name: str,
    sample_rate: int = 16000,
  ) -> None:
    self._output_path = output_path
    self._stream_name = stream_name
    self._sample_rate = sample_rate
    self._chunk_count = 0
    self._logger = get_logger("debug/capture", stream=stream_name)

    # Ensure output directory exists
    os.makedirs(str(output_path), exist_ok=True)

  @property
  def output_path(self) -> Path:
    """Get the output directory path."""
    return self._output_path

  @property
  def stream_name(self) -> str:
    """Get the stream name."""
    return self._stream_name

  @property
  def sample_rate(self) -> int:
    """Get the audio sample rate."""
    return self._sample_rate

  @property
  def chunk_count(self) -> int:
    """Get the number of chunks captured so far."""
    return self._chunk_count

  def capture(self, chunk: "AudioChunk") -> None:
    """
    Write an audio chunk to a WAV file.

    Creates a unique filename based on stream name, timestamp, and chunk
    timing information. The chunk's audio data is written as a single WAV file.

    :param chunk: Audio chunk with data and timing metadata.
    :type chunk: AudioChunk
    """
    timestamp = int(time.time())
    chunk_id = f"{chunk.start_time:.3f}_{chunk.duration:.3f}"
    filename = self._output_path / f"{self._stream_name}_{timestamp}_{chunk_id}_post.wav"
    import soundfile as sf

    try:
      sf.write(str(filename), chunk.data, self._sample_rate)
      self._chunk_count += 1
      self._logger.debug(
        "Post-buffer debug audio saved",
        filename=str(filename),
        duration=f"{chunk.duration:.2f}s",
      )
    except Exception:
      self._logger.exception(f"Error writing debug audio to {filename}")

  def capture_raw(
    self,
    audio_data: np.ndarray,
    start_time: float,
    duration: float,
  ) -> None:
    """
    Write raw audio data to a WAV file.

    Alternative to capture() when you have raw numpy data instead of an
    AudioChunk object.

    :param audio_data: Audio samples as float32 numpy array.
    :type audio_data: np.ndarray
    :param start_time: Start time of the audio in seconds.
    :type start_time: float
    :param duration: Duration of the audio in seconds.
    :type duration: float
    """
    timestamp = int(time.time())
    chunk_id = f"{start_time:.3f}_{duration:.3f}"
    filename = self._output_path / f"{self._stream_name}_{timestamp}_{chunk_id}_post.wav"
    import soundfile as sf

    try:
      sf.write(str(filename), audio_data, self._sample_rate)
      self._chunk_count += 1
      self._logger.debug(
        "Post-buffer debug audio saved",
        filename=str(filename),
        duration=f"{duration:.2f}s",
      )
    except Exception:
      self._logger.exception(f"Error writing debug audio to {filename}")

  def close(self) -> None:
    """
    Clean up any resources.

    Currently a no-op since we write files immediately, but included
    for interface consistency and potential future enhancements.
    """
    self._logger.debug(
      "Debug capture closed",
      total_chunks=self._chunk_count,
    )
