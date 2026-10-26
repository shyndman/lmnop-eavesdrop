"""
Audio capture and streaming functionality for Eavesdrop client.
"""

import asyncio
import importlib
from collections.abc import Callable
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from structlog.stdlib import BoundLogger

from eavesdrop.common import get_logger

# Audio configuration constants for the server PCM contract
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.float32
BLOCKSIZE = 0

Float32Audio = NDArray[np.float32]


class CaptureTiming(Protocol):
  """PortAudio callback times in seconds on the stream clock.

  :ivar inputBufferAdcTime: Capture time of the first input sample.
  :ivar currentTime: Time the callback was invoked.
  :ivar outputBufferDacTime: Playback time of the first output sample.
  """

  inputBufferAdcTime: float
  currentTime: float
  outputBufferDacTime: float


class CaptureStatus(Protocol):
  """PortAudio input-stream underflow and overflow flags."""

  @property
  def input_underflow(self) -> bool: ...

  @property
  def input_overflow(self) -> bool: ...

  def __bool__(self) -> bool: ...


class InputStreamLike(Protocol):
  def start(self) -> None: ...

  def stop(self) -> None: ...

  def close(self) -> None: ...


class InputStreamFactory(Protocol):
  def __call__(
    self,
    *,
    device: str | int | None,
    channels: int,
    samplerate: int,
    dtype: np.dtype[np.float32] | type[np.float32],
    latency: str,
    blocksize: int,
    callback: Callable[[Float32Audio, int, CaptureTiming, CaptureStatus], None],
  ) -> InputStreamLike: ...


def _load_input_stream() -> InputStreamFactory:
  return cast(InputStreamFactory, importlib.import_module("sounddevice").InputStream)


class AudioCapture:
  """Handles audio capture from microphone and streaming to server."""

  def __init__(
    self, on_error: Callable[[str], None], audio_device: str | int | None = None
  ) -> None:
    """Configure capture without opening the microphone.

    :param on_error: Receives capture failures.
    :param audio_device: Device name or index; None selects the system default.
    """
    self.audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
    self.audio_stream: InputStreamLike | None = None
    self.recording: bool = False
    self.on_error: Callable[[str], None] = on_error
    self.audio_device: str | int | None = audio_device
    self._callback_loop: asyncio.AbstractEventLoop | None = None
    self._logger: BoundLogger = get_logger("client/audio", device=audio_device)

  def audio_callback(
    self,
    indata: Float32Audio,
    _frames: int,
    _time_info: CaptureTiming,
    status: CaptureStatus,
  ) -> None:
    """Copy accepted device samples into the event-loop-owned sending queue.

    :param indata: Device-owned float32 samples, valid only during this callback.
    :param _frames: Frame count supplied by PortAudio.
    :param _time_info: PortAudio callback timing on the stream clock.
    :param status: Input underflow or overflow flags.
    """
    if status:
      self.on_error(f"Audio status: {status}")

    callback_loop = self._callback_loop
    if self.recording and callback_loop is not None:
      _ = callback_loop.call_soon_threadsafe(self.audio_queue.put_nowait, indata.tobytes())

  def start_recording(self) -> None:
    """Start audio capture from microphone."""
    if self.recording:
      return

    try:
      self._callback_loop = asyncio.get_running_loop()
      input_stream = _load_input_stream()
      self.audio_stream = input_stream(
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
      self._logger.info("audio capture started")
    except Exception as e:
      self._logger.exception("audio capture start failed")
      self.on_error(f"Error starting audio: {e}")
      raise

  def stop_recording(self) -> None:
    """Stop the device and allow already accepted callbacks to reach the queue."""
    if not self.recording:
      return

    if self.audio_stream:
      try:
        self.audio_stream.stop()
        self.audio_stream.close()
        self.audio_stream = None
        self._logger.info("audio capture stopped")
      except Exception as e:
        self._logger.exception("audio capture stop failed")
        self.on_error(f"Error stopping audio: {e}")
    self.recording = False
    self._callback_loop = None

  def clear_queue(self) -> None:
    """Drop captured audio left after cancellation or disconnect."""
    while not self.audio_queue.empty():
      _ = self.audio_queue.get_nowait()

  async def get_audio_data(self, timeout: float = 0.1) -> bytes | None:
    """Wait for the next captured frame.

    :param timeout: Maximum wait in seconds.
    :returns: Captured PCM bytes, or None when no frame arrives before the timeout.
    """
    try:
      return await asyncio.wait_for(self.audio_queue.get(), timeout=timeout)
    except asyncio.TimeoutError:
      return None

  def is_recording(self) -> bool:
    """Check if currently recording."""
    return self.recording
