"""Own workstation media and spectrum effects independently of finalization."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Protocol

from structlog.stdlib import BoundLogger

from active_listener.app.ports import MediaPlaybackController
from active_listener.recording.spectrum import SpectrumAnalyzer


class SpectrumRuntime(Protocol):
  def start(self) -> asyncio.Task[None]: ...

  async def stop(self) -> None: ...


async def _publish_noop_spectrum(_bars: bytes) -> None:
  return None


def build_noop_spectrum_analyzer() -> SpectrumAnalyzer:
  return SpectrumAnalyzer(publish=_publish_noop_spectrum)


@dataclass
class WorkstationEffects:
  """Keep recording-owned media pause and spectrum task lifetimes local.

  :param media: Controller whose playback is resumed only after our own pause.
  :param spectrum: Spectrum runtime, separate from recording finalization tasks.
  :param logger: Sink for non-fatal media failures.
  """

  media: MediaPlaybackController
  spectrum: SpectrumRuntime
  logger: BoundLogger
  _paused_media: bool = field(default=False, init=False)
  _spectrum_task: asyncio.Task[None] | None = field(default=None, init=False)

  async def start(self) -> None:
    """Pause playing media, then start spectrum publication before capture."""
    try:
      self._paused_media = await self.media.pause_if_playing()
    except Exception:
      self.logger.exception("media pause failed")
      self._paused_media = False
    if self._paused_media:
      self.logger.debug("media pause requested for recording")

    spectrum_task = self.spectrum.start()
    if spectrum_task is self._spectrum_task:
      return
    self._spectrum_task = spectrum_task
    spectrum_task.add_done_callback(self._clear_spectrum_task)

  async def stop_spectrum(self) -> None:
    """Await spectrum cleanup without touching finalization tasks."""
    await self.spectrum.stop()
    self._spectrum_task = None

  async def close(self) -> None:
    """Stop spectrum and await its task even if runtime cleanup fails."""
    task = self._spectrum_task
    try:
      await self.stop_spectrum()
    finally:
      if task is not None:
        _ = task.cancel()
        _ = await asyncio.gather(task, return_exceptions=True)

  async def resume_media(self) -> None:
    """Consume the recording's pause ownership before attempting playback."""
    paused_media = self._paused_media
    self._paused_media = False
    if not paused_media:
      return
    try:
      await self.media.resume()
    except Exception:
      self.logger.exception("media resume failed")
      return
    self.logger.debug("media resume requested after recording")

  def _clear_spectrum_task(self, task: asyncio.Task[None]) -> None:
    if self._spectrum_task is task:
      self._spectrum_task = None
