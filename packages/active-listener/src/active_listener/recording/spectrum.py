from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

type Float32PcmChunk = bytes
type QuantizedSpectrumFrame = bytes
type AudioSamples = NDArray[np.float32]
type FrequencyBins = NDArray[np.float64]
type SpectrumMagnitudes = NDArray[np.float64]
type SpectrumDecibels = NDArray[np.float64]
type NormalizedSpectrumBars = NDArray[np.float64]
type QuantizedSpectrumBars = NDArray[np.uint8]

SAMPLE_RATE_HZ = 16000
WINDOW_SIZE = 512
SPECTRUM_TICK_INTERVAL_SECONDS = 0.016
SPECTRUM_BAR_COUNT = 50
MIN_FREQUENCY_HZ = 60.0
MAX_FREQUENCY_HZ = 8000.0
FLOOR_DB = -60.0
CEIL_DB = -18.0
EPSILON = 1e-10
ROLLING_BUFFER_SIZE = 2048

WINDOW: AudioSamples = np.hanning(WINDOW_SIZE).astype(np.float32)
FFT_FREQUENCIES_HZ: FrequencyBins = np.fft.rfftfreq(WINDOW_SIZE, d=1.0 / SAMPLE_RATE_HZ)
BAR_EDGES_HZ: FrequencyBins = np.geomspace(
  MIN_FREQUENCY_HZ,
  MAX_FREQUENCY_HZ,
  SPECTRUM_BAR_COUNT + 1,
)
BAR_CENTERS_HZ: FrequencyBins = np.sqrt(BAR_EDGES_HZ[:-1] * BAR_EDGES_HZ[1:])
# NumPy's Hann window halves a tone's coherent amplitude. Divide the FFT
# magnitudes by this factor so our bar ceiling tracks real input headroom
# instead of window-size-dependent gain.
FFT_MAGNITUDE_NORMALIZATION = float(np.sum(WINDOW) / 2.0)


@dataclass
class SpectrumAnalyzer:
  publish: Callable[[QuantizedSpectrumFrame], Awaitable[None]]
  _buffer: AudioSamples = field(
    default_factory=lambda: np.zeros(ROLLING_BUFFER_SIZE, dtype=np.float32)
  )
  _write_index: int = 0
  _sample_count: int = 0
  _active: bool = False
  _task: asyncio.Task[None] | None = None

  def start(self) -> asyncio.Task[None]:
    if self._task is not None and not self._task.done():
      return self._task

    self.reset()
    self._active = True
    self._task = asyncio.create_task(self._run())
    return self._task

  async def stop(self) -> None:
    self._active = False
    task = self._task
    self._task = None

    if task is not None and not task.done():
      _ = task.cancel()
      _ = await asyncio.gather(task, return_exceptions=True)

    self.reset()

  def reset(self) -> None:
    self._buffer.fill(0)
    self._write_index = 0
    self._sample_count = 0

  def ingest(self, chunk: Float32PcmChunk) -> None:
    if not self._active:
      return

    samples: AudioSamples = np.frombuffer(chunk, dtype=np.float32)
    sample_count = len(samples)
    if sample_count == 0:
      return

    if sample_count >= ROLLING_BUFFER_SIZE:
      self._buffer[:] = samples[-ROLLING_BUFFER_SIZE:]
      self._write_index = 0
      self._sample_count = ROLLING_BUFFER_SIZE
      return

    first_write_count = min(ROLLING_BUFFER_SIZE - self._write_index, sample_count)
    self._buffer[
      self._write_index : self._write_index + first_write_count
    ] = samples[:first_write_count]

    remaining_count = sample_count - first_write_count
    if remaining_count > 0:
      self._buffer[:remaining_count] = samples[first_write_count:]

    self._write_index = (self._write_index + sample_count) % ROLLING_BUFFER_SIZE
    self._sample_count = min(self._sample_count + sample_count, ROLLING_BUFFER_SIZE)

  async def _run(self) -> None:
    try:
      while self._active:
        await asyncio.sleep(SPECTRUM_TICK_INTERVAL_SECONDS)
        latest_window = self._latest_window()
        if latest_window is None:
          continue
        await self.publish(compute_spectrum_frame(latest_window))
    finally:
      self._task = None

  def _latest_window(self) -> AudioSamples | None:
    if self._sample_count < WINDOW_SIZE:
      return None

    start_index = (self._write_index - WINDOW_SIZE) % ROLLING_BUFFER_SIZE
    end_index = start_index + WINDOW_SIZE
    if end_index <= ROLLING_BUFFER_SIZE:
      return self._buffer[start_index:end_index].copy()

    wrap_count = end_index - ROLLING_BUFFER_SIZE
    return np.concatenate((self._buffer[start_index:], self._buffer[:wrap_count])).astype(
      np.float32,
      copy=False,
    )


def compute_spectrum_frame(latest_window: AudioSamples) -> QuantizedSpectrumFrame:
  windowed_samples: AudioSamples = latest_window * WINDOW
  spectrum: SpectrumMagnitudes = (
    np.abs(np.fft.rfft(windowed_samples)) / FFT_MAGNITUDE_NORMALIZATION
  )
  interpolated_bars: SpectrumMagnitudes = np.interp(
    BAR_CENTERS_HZ,
    FFT_FREQUENCIES_HZ,
    spectrum,
  )
  bars_db: SpectrumDecibels = 20.0 * np.log10(np.maximum(interpolated_bars, EPSILON))
  bars_normalized: NormalizedSpectrumBars = np.clip(
    (bars_db - FLOOR_DB) / (CEIL_DB - FLOOR_DB),
    0.0,
    1.0,
  )
  quantized_bars: QuantizedSpectrumBars = np.round(bars_normalized * 255.0).astype(np.uint8)
  return quantized_bars.tobytes()
