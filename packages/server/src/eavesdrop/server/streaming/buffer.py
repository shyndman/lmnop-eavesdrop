"""
Audio stream buffer management for streaming transcription.

Handles buffering of audio frames with configurable cleanup and processing parameters.
"""

import threading

import numpy as np

from eavesdrop.common import get_logger
from eavesdrop.server.config import BufferConfig


class AudioStreamBuffer:
  """
  Manages audio frame buffering for streaming transcription.

  This class handles the continuous addition of audio frames, automatic buffer
  cleanup to prevent excessive memory usage, and extraction of audio chunks
  for transcription processing.

  Thread Safety:
    All public methods and properties are thread-safe and protected by an
    internal lock. Multiple threads can safely call any combination of methods
    concurrently.

  TODOs:
    - Add comprehensive invariant checking with assertions instead of defensive max(0, ...)
    - Eliminate code duplication by extracting _get_processed_samples_offset() helper
    - Add extensive unit tests with concurrent access stress testing
    - Implement property-based testing to verify buffer state math
    - Consider making locks more granular for better performance if needed
    - Add debug logging for buffer state transitions during development
  """

  def __init__(self, config: BufferConfig) -> None:
    """
    Initialize the audio stream buffer.

    :param
        config: Configuration for buffer behavior including sample rate,
                cleanup policies, and processing thresholds.
    """
    self.config: BufferConfig = config
    self.logger = get_logger("snd/buf")

    # Audio buffer state
    self.frames_np: np.ndarray | None = None
    """Raw audio data as a NumPy array. None when buffer is empty."""

    self.buffer_start_time: float = 0.0
    """
    Timestamp offset for the start of the frames_np buffer in seconds.
    Tracks how much audio has been discarded from the beginning due to cleanup.
    """

    self.processed_up_to_time: float = 0.0
    """
    Timestamp offset for processed audio in seconds.
    Marks the boundary between processed and unprocessed audio within the buffer.
    Always >= frames_offset.
    """

    # Thread synchronization
    self.lock = threading.Lock()
    """Protects buffer state from concurrent access."""

  def add_frames(self, frame_np: np.ndarray) -> None:
    """
    Add new audio frames to the end of the buffer.

    Appends the provided audio data to the internal buffer and performs
    automatic cleanup if the buffer exceeds the maximum duration threshold.
    When cleanup occurs, removes the oldest audio data to prevent unbounded
    memory growth.

    :param
        frame_np: Audio data as float32 NumPy array, normalized to [-1.0, 1.0].
                  Expected to be 1D array at the configured sample rate.
    """
    with self.lock:
      max_buffer_samples = self.config.max_buffer_duration * self.config.sample_rate
      if self.frames_np is not None and self.frames_np.shape[0] > max_buffer_samples:
        # Remove old audio data
        self.buffer_start_time += self.config.cleanup_duration
        samples_to_remove = int(self.config.cleanup_duration * self.config.sample_rate)
        self.frames_np = self.frames_np[samples_to_remove:]

        # Update timestamp offset if it hasn't progressed
        # This indicates no speech activity has been processed
        if self.processed_up_to_time < self.buffer_start_time:
          self.processed_up_to_time = self.buffer_start_time

      if self.frames_np is None:
        self.frames_np = frame_np.copy()
      else:
        self.frames_np = np.concatenate((self.frames_np, frame_np), axis=0)

  def get_chunk_for_processing(self) -> tuple[np.ndarray, float, float]:
    """
    Extract all unprocessed audio data from the buffer.

    Returns all audio data that hasn't been processed yet, starting from
    the processed_up_to_time boundary to the end of the buffer. This
    represents the "next chunk" ready for transcription.

    :returns:
        A tuple containing:
        - audio_data: Unprocessed audio as float32 NumPy array, or empty array if none.
        - duration: Duration of the returned audio chunk in seconds.
        - start_time: Absolute stream time where this chunk begins.

    Note:
        The returned array is always a copy, so modifications won't affect
        the internal buffer state.
    """
    with self.lock:
      if self.frames_np is None:
        return np.array([]), 0.0, self.processed_up_to_time

      offset_diff = self.processed_up_to_time - self.buffer_start_time
      samples_take = max(0, int(offset_diff * self.config.sample_rate))
      input_bytes = self.frames_np[samples_take:].copy()
      chunk_start_time = self.processed_up_to_time

    duration = input_bytes.shape[0] / self.config.sample_rate
    return input_bytes, duration, chunk_start_time

  def advance_processed_boundary(self, offset: float) -> None:
    """
    Mark a portion of audio as processed by advancing the processed boundary.

    Moves the processed_up_to_time forward by the specified duration,
    indicating that this amount of audio has been successfully transcribed
    and no longer needs processing.

    :param
        offset: Duration in seconds to advance the processed boundary.
                Must be positive. Typically matches the duration of audio
                that was just transcribed.

    Note:
        This is typically called after successful transcription to "consume"
        the processed audio from the buffer's perspective.
    """
    with self.lock:
      self.processed_up_to_time += offset

  def clip_if_stalled(self) -> None:
    """
    Force-advance the processed boundary if transcription has stalled too long.

    When the unprocessed audio exceeds max_stall_duration, this method
    assumes that no valid speech segments exist in the stalled audio and
    skips most of it by advancing the processed boundary. This prevents
    the buffer from getting stuck on problematic audio sections.

    The method leaves the last 5 seconds of buffer unprocessed to avoid
    potentially clipping active speech.

    Behavior:
        - Only operates if clip_audio config option is enabled
        - Triggers when unprocessed audio > max_stall_duration
        - Advances processed_up_to_time to (buffer_end - 5.0 seconds)
    """
    if not self.config.clip_audio:
      return

    with self.lock:
      if self.frames_np is None:
        return

      offset_diff = self.processed_up_to_time - self.buffer_start_time
      unprocessed_start = int(offset_diff * self.config.sample_rate)
      unprocessed_samples = self.frames_np[unprocessed_start:].shape[0]

      if unprocessed_samples > self.config.max_stall_duration * self.config.sample_rate:
        duration = self.frames_np.shape[0] / self.config.sample_rate
        unprocessed_duration = unprocessed_samples / self.config.sample_rate
        clipped_duration = unprocessed_duration - 5.0

        self.logger.warning(
          "Transcription stalled, clipping audio",
          unprocessed_duration=f"{unprocessed_duration:.2f}s",
          max_stall_duration=f"{self.config.max_stall_duration:.2f}s",
          clipped_duration=f"{clipped_duration:.2f}s",
          keeping_last=5.0,
        )

        self.processed_up_to_time = self.buffer_start_time + duration - 5.0

  def reset(self) -> None:
    """
    Reset the buffer to its initial empty state.

    Clears all audio data and resets timeline tracking to zero.
    This is typically used when starting a new transcription session
    or recovering from errors.
    """
    with self.lock:
      self.frames_np = None
      self.buffer_start_time = 0.0
      self.processed_up_to_time = 0.0

  @property
  def available_duration(self) -> float:
    """
    Duration of unprocessed audio available for transcription in seconds.

    This represents the amount of audio that has been added to the buffer
    but hasn't been processed yet - essentially the "work queue" for
    transcription.

    :returns:
        Duration in seconds from processed_up_to_time to the end of the buffer.
        Returns 0.0 if buffer is empty or fully processed.

    Note:
        When this value approaches 0, it indicates transcription is "caught up"
        to live audio input.
    """
    with self.lock:
      if self.frames_np is None:
        return 0.0
      offset_diff = self.processed_up_to_time - self.buffer_start_time
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
    return self.processed_up_to_time - self.buffer_start_time
