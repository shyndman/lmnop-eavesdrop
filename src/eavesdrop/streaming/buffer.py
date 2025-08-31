"""
Audio stream buffer management for streaming transcription.

Handles buffering of audio frames with configurable cleanup and processing parameters.
"""

import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class BufferConfig:
  """Configuration for audio stream buffer behavior."""

  sample_rate: int = 16000
  """Audio sample rate in Hz."""

  max_buffer_duration: float = 45.0
  """Maximum buffer duration in seconds before cleanup."""

  cleanup_duration: float = 30.0
  """Duration of oldest audio to remove during cleanup."""

  min_chunk_duration: float = 1.0
  """Minimum chunk duration for processing in seconds."""

  transcription_interval: float = 2.0
  """Interval between transcription attempts in seconds."""

  clip_audio: bool = False
  """Whether to clip audio with no valid segments."""

  max_stall_duration: float = 25.0
  """Maximum duration without progress before clipping audio."""


class AudioStreamBuffer:
  """
  Manages audio frame buffering for streaming transcription.

  This class handles the continuous addition of audio frames, automatic buffer
  cleanup to prevent excessive memory usage, and extraction of audio chunks
  for transcription processing.
  """

  def __init__(self, config: BufferConfig) -> None:
    self.config: BufferConfig = config

    # Audio buffer state
    self.frames_np: np.ndarray | None = None
    self.frames_offset: float = 0.0
    self.timestamp_offset: float = 0.0

    # Thread synchronization
    self.lock: threading.Lock = threading.Lock()

  def add_frames(self, frame_np: np.ndarray) -> None:
    """
    Add audio frames to the ongoing audio stream buffer.

    This method maintains the audio stream buffer, allowing continuous
    addition of audio frames as they are received. It ensures that the buffer
    does not exceed a specified size to prevent excessive memory usage.

    If the buffer size exceeds max_buffer_duration seconds of audio data,
    it discards the oldest cleanup_duration seconds of audio data to maintain
    a reasonable buffer size.

    Args:
        frame_np: The audio frame data as a NumPy array.
    """
    with self.lock:
      max_buffer_samples = self.config.max_buffer_duration * self.config.sample_rate
      if self.frames_np is not None and self.frames_np.shape[0] > max_buffer_samples:
        # Remove old audio data
        self.frames_offset += self.config.cleanup_duration
        samples_to_remove = int(self.config.cleanup_duration * self.config.sample_rate)
        self.frames_np = self.frames_np[samples_to_remove:]

        # Update timestamp offset if it hasn't progressed
        # This indicates no speech activity has been processed
        if self.timestamp_offset < self.frames_offset:
          self.timestamp_offset = self.frames_offset

      if self.frames_np is None:
        self.frames_np = frame_np.copy()
      else:
        self.frames_np = np.concatenate((self.frames_np, frame_np), axis=0)

  def get_chunk_for_processing(self) -> tuple[np.ndarray, float]:
    """
    Retrieves the next chunk of audio data for processing.

    Calculates which part of the audio data should be processed next, based on
    the difference between the current timestamp offset and the frame's offset,
    scaled by the audio sample rate.

    Returns:
        A tuple containing:
        - input_bytes: The next chunk of audio data to be processed.
        - duration: The duration of the audio chunk in seconds.
    """
    with self.lock:
      if self.frames_np is None:
        return np.array([]), 0.0

      offset_diff = self.timestamp_offset - self.frames_offset
      samples_take = max(0, int(offset_diff * self.config.sample_rate))
      input_bytes = self.frames_np[samples_take:].copy()

    duration = input_bytes.shape[0] / self.config.sample_rate
    return input_bytes, duration

  def advance_processed_boundary(self, offset: float) -> None:
    """
    Update the timestamp offset to mark audio as processed.

    Args:
        offset: Duration in seconds to advance the processed boundary.
    """
    with self.lock:
      self.timestamp_offset += offset

  def clip_if_stalled(self) -> None:
    """
    Clip audio if no valid segments have been processed for too long.

    This method updates the timestamp offset based on audio buffer status.
    If the current unprocessed chunk exceeds max_stall_duration seconds,
    it implies no valid segment has been found and clips the audio.
    """
    if not self.config.clip_audio:
      return

    with self.lock:
      if self.frames_np is None:
        return

      offset_diff = self.timestamp_offset - self.frames_offset
      unprocessed_start = int(offset_diff * self.config.sample_rate)
      unprocessed_samples = self.frames_np[unprocessed_start:].shape[0]

      if unprocessed_samples > self.config.max_stall_duration * self.config.sample_rate:
        duration = self.frames_np.shape[0] / self.config.sample_rate
        self.timestamp_offset = self.frames_offset + duration - 5.0

  def reset(self) -> None:
    """Reset the buffer to initial state."""
    with self.lock:
      self.frames_np = None
      self.frames_offset = 0.0
      self.timestamp_offset = 0.0

  @property
  def available_duration(self) -> float:
    """Duration of audio available for processing in seconds."""
    with self.lock:
      if self.frames_np is None:
        return 0.0
      offset_diff = self.timestamp_offset - self.frames_offset
      processed_samples = int(offset_diff * self.config.sample_rate)
      samples_available = max(0, self.frames_np.shape[0] - processed_samples)
      return samples_available / self.config.sample_rate

  @property
  def total_duration(self) -> float:
    """Total duration of audio in buffer in seconds."""
    with self.lock:
      if self.frames_np is None:
        return 0.0
      return self.frames_np.shape[0] / self.config.sample_rate

  @property
  def processed_duration(self) -> float:
    """Duration of audio that has been processed in seconds."""
    return self.timestamp_offset - self.frames_offset
