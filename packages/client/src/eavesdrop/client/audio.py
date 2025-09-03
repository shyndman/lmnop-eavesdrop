"""
Audio capture and streaming functionality for Eavesdrop client.
"""

import asyncio
from collections.abc import Callable

import numpy as np
import sounddevice as sd

# Audio configuration constants (WhisperLive compatible)
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.float32
BLOCKSIZE = 4096


class AudioCapture:
  """Handles audio capture from microphone and streaming to server."""

  def __init__(self, on_error: Callable[[str], None]):
    self.audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
    self.audio_stream: sd.InputStream | None = None
    self.recording = False
    self.on_error = on_error

  def audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
    """Sounddevice audio callback."""
    if status:
      self.on_error(f"Audio status: {status}")

    if self.recording:
      # Convert to float32 and queue for transmission
      audio_data = indata.copy().astype(DTYPE)
      try:
        self.audio_queue.put_nowait(audio_data.tobytes())
      except asyncio.QueueFull:
        pass  # Drop frame if queue is full

  def start_recording(self):
    """Start audio capture from microphone."""
    if self.recording:
      return

    try:
      self.audio_stream = sd.InputStream(
        device=None,
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
