"""Bounded in-memory queue for finite-file canonical audio ingest."""

import asyncio
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from eavesdrop.server.streaming.file_decoder import CANONICAL_DTYPE, CANONICAL_SAMPLE_RATE_HZ

FILE_QUEUE_CAPACITY_SECONDS = 900.0


@dataclass(frozen=True)
class FileQueueSnapshot:
  """Queue metrics snapshot for observability logging."""

  queued_seconds: float
  fill_ratio: float
  total_enqueue_block_s: float


class FileAudioQueue:
  """Bounded queue for canonical file-session audio with blocking backpressure."""

  def __init__(self, capacity_seconds: float = FILE_QUEUE_CAPACITY_SECONDS) -> None:
    self._capacity_seconds = capacity_seconds
    self._capacity_samples = int(capacity_seconds * CANONICAL_SAMPLE_RATE_HZ)
    self._chunks: deque[np.ndarray] = deque()
    self._queued_samples = 0
    self._producer_done = False
    self._condition = asyncio.Condition()
    self._total_enqueue_block_s = 0.0

  async def enqueue(self, chunk: np.ndarray) -> float:
    """Enqueue canonical audio, blocking while at capacity.

    :param chunk: Canonical mono float32 audio chunk.
    :type chunk: np.ndarray
    :returns: Wall time blocked while waiting for queue capacity.
    :rtype: float
    :raises ValueError: If chunk is not canonical float32 mono waveform.
    """
    if chunk.dtype != CANONICAL_DTYPE:
      raise ValueError(f"Expected float32 chunk, got {chunk.dtype}")
    if chunk.ndim != 1:
      raise ValueError(f"Expected mono 1D chunk, got ndim={chunk.ndim}")

    chunk_samples = int(chunk.shape[0])
    if chunk_samples == 0:
      return 0.0

    if chunk_samples > self._capacity_samples:
      raise ValueError(
        "Chunk exceeds queue capacity; split decoded audio before enqueueing "
        f"(chunk={chunk_samples} samples, capacity={self._capacity_samples} samples)"
      )

    enqueue_wait_started = time.perf_counter()
    async with self._condition:
      while self._queued_samples + chunk_samples > self._capacity_samples:
        await self._condition.wait()

      blocked_s = time.perf_counter() - enqueue_wait_started
      self._total_enqueue_block_s += blocked_s
      self._chunks.append(chunk.copy())
      self._queued_samples += chunk_samples
      self._condition.notify_all()
      return blocked_s

  async def dequeue(self) -> np.ndarray | None:
    """Dequeue one canonical chunk, or return None once producer is done and queue is empty."""
    async with self._condition:
      while not self._chunks:
        if self._producer_done:
          return None
        await self._condition.wait()

      chunk = self._chunks.popleft()
      self._queued_samples -= int(chunk.shape[0])
      self._condition.notify_all()
      return chunk

  async def mark_producer_done(self) -> None:
    """Signal that no additional chunks will be enqueued."""
    async with self._condition:
      self._producer_done = True
      self._condition.notify_all()

  async def snapshot(self) -> FileQueueSnapshot:
    """Capture queue metrics for observability reporting."""
    async with self._condition:
      queued_seconds = self._queued_samples / CANONICAL_SAMPLE_RATE_HZ
      fill_ratio = 0.0
      if self._capacity_samples > 0:
        fill_ratio = self._queued_samples / self._capacity_samples
      return FileQueueSnapshot(
        queued_seconds=queued_seconds,
        fill_ratio=fill_ratio,
        total_enqueue_block_s=self._total_enqueue_block_s,
      )

  async def is_empty(self) -> bool:
    """Return True when no queued canonical audio remains."""
    async with self._condition:
      return self._queued_samples == 0
