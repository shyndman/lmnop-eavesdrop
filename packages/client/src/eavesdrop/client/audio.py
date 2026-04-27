"""
Audio capture and streaming functionality for Eavesdrop client.
"""

import asyncio
from collections.abc import Callable
from importlib import import_module
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

# Audio configuration constants (WhisperLive compatible)
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.float32
BLOCKSIZE = 0

Float32Audio = NDArray[np.float32]


class InputStreamLike(Protocol):
  def start(self) -> None: ...

  def stop(self) -> None: ...

  def close(self) -> None: ...


class SoundDeviceModule(Protocol):
  def InputStream(
    self,
    *,
    device: str | int | None,
    channels: int,
    samplerate: int,
    dtype: np.dtype[np.float32] | type[np.float32],
    latency: str,
    blocksize: int,
    callback: Callable[[Float32Audio, int, object, object], None],
  ) -> InputStreamLike: ...


sd = cast(SoundDeviceModule, cast(object, import_module("sounddevice")))


class AudioCapture:
  """Handles audio capture from microphone and streaming to server."""

  def __init__(self, on_error: Callable[[str], None], audio_device: str | int | None = None):
    self.audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
    self.audio_stream: InputStreamLike | None = None
    self.recording: bool = False
    self.on_error: Callable[[str], None] = on_error
    self.audio_device: str | int | None = audio_device

  def audio_callback(
    self,
    indata: Float32Audio,
    _frames: int,
    _time_info: object,
    status: object,
  ) -> None:
    """Sounddevice audio callback."""
    if status:
      self.on_error(f"Audio status: {status}")

    if self.recording:
      # Convert to float32 and queue for transmission
      audio_data = indata.copy().astype(DTYPE)
      self.audio_queue.put_nowait(audio_data.tobytes())

  def start_recording(self):
    """Start audio capture from microphone."""
    if self.recording:
      return

    try:
      self.audio_stream = sd.InputStream(
        device=self.audio_device,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype=DTYPE,
        latency="low",
        blocksize=BLOCKSIZE,
        callback=self.audio_callback,
      )
      self.recording = True
      self.audio_stream.start()
    except Exception as e:
      self.on_error(f"Error starting audio: {e}")
      raise

  def stop_recording(self):
    """Stop audio capture."""
    if not self.recording:
      return

    self.recording = False
    if self.audio_stream:
      try:
        self.audio_stream.stop()
        self.audio_stream.close()
        self.audio_stream = None
      except Exception as e:
        self.on_error(f"Error stopping audio: {e}")

  async def get_audio_data(self, timeout: float = 0.1) -> bytes | None:
    """Get audio data from queue with timeout."""
    try:
      return await asyncio.wait_for(self.audio_queue.get(), timeout=timeout)
    except asyncio.TimeoutError:
      return None

  def is_recording(self) -> bool:
    """Check if currently recording."""
    return self.recording
