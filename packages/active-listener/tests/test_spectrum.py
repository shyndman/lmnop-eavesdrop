from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from active_listener.recording.spectrum import (
  SAMPLE_RATE_HZ,
  SPECTRUM_BAR_COUNT,
  SPECTRUM_TICK_INTERVAL_SECONDS,
  WINDOW_SIZE,
  SpectrumAnalyzer,
  compute_spectrum_frame,
)


def _sine_wave(frequency_hz: float, sample_count: int) -> NDArray[np.float32]:
  sample_positions = np.arange(sample_count, dtype=np.float32) / SAMPLE_RATE_HZ
  return np.sin(2.0 * np.pi * frequency_hz * sample_positions).astype(np.float32)


def _publish_recorder(frames: list[bytes]) -> Callable[[bytes], Awaitable[None]]:
  async def publish(frame: bytes) -> None:
    frames.append(frame)

  return publish


@pytest.mark.asyncio
async def test_low_frequency_interpolation_eliminates_dead_low_bands() -> None:
  frame = compute_spectrum_frame(_sine_wave(70.0, WINDOW_SIZE))

  assert len(frame) == SPECTRUM_BAR_COUNT
  assert max(frame[:8]) > 0


@pytest.mark.asyncio
async def test_analyzer_skips_publish_until_window_is_full() -> None:
  published_frames: list[bytes] = []
  analyzer = SpectrumAnalyzer(publish=_publish_recorder(published_frames))

  _ = analyzer.start()
  analyzer.ingest(_sine_wave(220.0, WINDOW_SIZE - 1).tobytes())
  await asyncio.sleep(SPECTRUM_TICK_INTERVAL_SECONDS * 2)
  await analyzer.stop()

  assert published_frames == []


@pytest.mark.asyncio
async def test_analyzer_emits_exactly_fifty_quantized_bars() -> None:
  published_frames: list[bytes] = []
  analyzer = SpectrumAnalyzer(publish=_publish_recorder(published_frames))

  _ = analyzer.start()
  analyzer.ingest(_sine_wave(440.0, WINDOW_SIZE * 2).tobytes())
  await asyncio.sleep(SPECTRUM_TICK_INTERVAL_SECONDS * 3)
  await analyzer.stop()

  assert published_frames
  assert all(len(frame) == SPECTRUM_BAR_COUNT for frame in published_frames)
