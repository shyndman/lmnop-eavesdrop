"""
Audio stream buffer management for streaming transcription.

Handles buffering of audio frames with configurable cleanup and processing parameters.
"""

import threading
from collections import deque

import numpy as np
import structlog
from numpy.typing import NDArray

from eavesdrop.common import get_logger
from eavesdrop.server.config import BufferConfig

Float32Audio = NDArray[np.float32]


class AudioStreamBuffer:
  """Manages audio frame buffering for streaming transcription.

  Buffer times are offsets from buffer zero. In live WebSocket mode the client
  resets the buffer at each accepted recording boundary, so buffer zero is also
  recording-relative zero for the active recording.

  :param config: Buffer sizing, cleanup, and processing threshold configuration.
  :type config: BufferConfig
  """

  def __init__(self, config: BufferConfig) -> None:
    self.config: BufferConfig = config
    self.logger: structlog.stdlib.BoundLogger = get_logger("snd/buf")
    self._chunks: deque[Float32Audio] = deque()
    self._total_samples: int = 0
    self.buffer_start_time: float = 0.0
    self.processed_up_to_time: float = 0.0
    self.lock: threading.Lock = threading.Lock()

  @property
  def frames_np(self) -> Float32Audio | None:
    """Return buffered audio as a contiguous array, or ``None`` when empty."""
    with self.lock:
      return self._flatten_chunks_locked()

  def _flatten_chunks_locked(self) -> Float32Audio | None:
    if self._total_samples == 0:
      return None

    if len(self._chunks) == 1:
      return self._chunks[0]

    compacted = np.concatenate(tuple(self._chunks), axis=0)
    self._chunks.clear()
    self._chunks.append(compacted)
    return compacted

  def _trim_prefix_locked(self, samples_to_remove: int) -> None:
    remaining = min(max(0, samples_to_remove), self._total_samples)
    self._total_samples -= remaining
    while remaining > 0 and self._chunks:
      head = self._chunks[0]
      head_len = head.shape[0]
      if head_len <= remaining:
        _ = self._chunks.popleft()
        remaining -= head_len
      else:
        self._chunks[0] = head[remaining:]
        remaining = 0

  def add_frames(self, frame_np: Float32Audio) -> None:
    """Append new float32 audio frames, trimming oldest audio above the buffer limit.

    :param frame_np: Mono float32 PCM samples at the configured sample rate.
    :type frame_np: Float32Audio
    """
    with self.lock:
      max_buffer_samples = self.config.max_buffer_duration * self.config.sample_rate
      if self._total_samples > max_buffer_samples:
        self.buffer_start_time += self.config.cleanup_duration
        samples_to_remove = int(self.config.cleanup_duration * self.config.sample_rate)
        self._trim_prefix_locked(samples_to_remove)

        if self.processed_up_to_time < self.buffer_start_time:
          self.processed_up_to_time = self.buffer_start_time

      if frame_np.shape[0] > 0:
        self._chunks.append(frame_np.copy())
        self._total_samples += frame_np.shape[0]

  def get_chunk_for_processing(self) -> tuple[Float32Audio, float, float]:
    """Extract all unprocessed audio data from the buffer.

    :returns: Tuple of ``(audio_copy, duration_seconds, chunk_start_time)``.
    :rtype: tuple[Float32Audio, float, float]
    """
    with self.lock:
      if self._total_samples == 0:
        return np.array([], dtype=np.float32), 0.0, self.processed_up_to_time

      processed_offset = self.processed_up_to_time - self.buffer_start_time
      buffer_relative_start_index = max(0, int(processed_offset * self.config.sample_rate))
      chunk_start_time = self.processed_up_to_time
      if buffer_relative_start_index >= self._total_samples:
        return np.array([], dtype=np.float32), 0.0, chunk_start_time

      flattened = self._flatten_chunks_locked()
      assert flattened is not None
      input_bytes = flattened[buffer_relative_start_index:].copy()

    duration = input_bytes.shape[0] / self.config.sample_rate
    return input_bytes, duration, chunk_start_time

  def get_buffer_end_sample(self) -> int:
    """Return the current recording-relative sample index at the end of the buffer.

    :returns: Sample index representing the current buffered-audio end.
    :rtype: int
    """
    with self.lock:
      return int(self.buffer_start_time * self.config.sample_rate) + self._total_samples

  def advance_processed_boundary(self, offset: float) -> None:
    """Advance ``processed_up_to_time`` forward by ``offset`` seconds.

    :param offset: Positive duration in seconds to consume.
    :type offset: float
    """
    with self.lock:
      self.processed_up_to_time += offset

  def clip_if_stalled(self) -> None:
    """Force-advance the processed boundary if unprocessed audio exceeds ``max_stall_duration``."""
    if not self.config.clip_audio:
      return

    with self.lock:
      if self._total_samples == 0:
        return

      processed_offset = self.processed_up_to_time - self.buffer_start_time
      buffer_relative_unprocessed_start = max(0, int(processed_offset * self.config.sample_rate))
      unprocessed_samples = max(0, self._total_samples - buffer_relative_unprocessed_start)

      if unprocessed_samples > self.config.max_stall_duration * self.config.sample_rate:
        duration = self._total_samples / self.config.sample_rate
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
    """Reset the buffer to its initial empty state at recording zero."""
    with self.lock:
      self._chunks.clear()
      self._total_samples = 0
      self.buffer_start_time = 0.0
      self.processed_up_to_time = 0.0

  def discard_unprocessed_audio(self) -> float:
    """Discard buffered audio at and after the processed boundary.

    :returns: Duration in seconds discarded from the unprocessed tail.
    :rtype: float
    """
    with self.lock:
      if self._total_samples == 0:
        self.buffer_start_time = self.processed_up_to_time
        return 0.0

      processed_offset = self.processed_up_to_time - self.buffer_start_time
      buffer_relative_processed_index = min(
        max(0, int(processed_offset * self.config.sample_rate)),
        self._total_samples,
      )
      discarded_samples = self._total_samples - buffer_relative_processed_index

      self._chunks.clear()
      self._total_samples = 0
      self.buffer_start_time = self.processed_up_to_time

    return discarded_samples / self.config.sample_rate

  def discard_through_sample(self, boundary_sample: int) -> float:
    """Discard buffered audio through a recording-relative sample boundary.

    :param boundary_sample: Sample index through which buffered audio is discarded.
    :type boundary_sample: int
    :returns: Duration in seconds discarded from the unprocessed tail.
    :rtype: float
    """
    with self.lock:
      sample_rate = self.config.sample_rate
      processed_sample = int(self.processed_up_to_time * sample_rate)

      if self._total_samples == 0:
        if boundary_sample > processed_sample:
          boundary_time = boundary_sample / sample_rate
          self.buffer_start_time = boundary_time
          self.processed_up_to_time = boundary_time
        return 0.0

      buffer_start_sample = int(self.buffer_start_time * sample_rate)
      buffer_end_sample = buffer_start_sample + self._total_samples
      discard_end_sample = min(max(boundary_sample, buffer_start_sample), buffer_end_sample)
      buffer_relative_samples_to_remove = discard_end_sample - buffer_start_sample
      discarded_unprocessed_samples = max(0, discard_end_sample - processed_sample)

      if buffer_relative_samples_to_remove <= 0:
        return 0.0

      self._trim_prefix_locked(buffer_relative_samples_to_remove)

      boundary_time = max(self.buffer_start_time, discard_end_sample / sample_rate)
      self.buffer_start_time = boundary_time
      if self.processed_up_to_time < boundary_time:
        self.processed_up_to_time = boundary_time

    return discarded_unprocessed_samples / self.config.sample_rate

  @property
  def available_duration(self) -> float:
    """Duration of unprocessed audio available for transcription in seconds."""
    with self.lock:
      if self._total_samples == 0:
        return 0.0
      offset_diff = self.processed_up_to_time - self.buffer_start_time
      buffer_relative_processed_index = max(0, int(offset_diff * self.config.sample_rate))
      samples_available = max(0, self._total_samples - buffer_relative_processed_index)
      return samples_available / self.config.sample_rate

  @property
  def total_duration(self) -> float:
    """Total duration of audio in buffer in seconds."""
    with self.lock:
      return self._total_samples / self.config.sample_rate

  @property
  def processed_duration(self) -> float:
    """Duration of audio that has been processed in seconds."""
    with self.lock:
      return self.processed_up_to_time - self.buffer_start_time
