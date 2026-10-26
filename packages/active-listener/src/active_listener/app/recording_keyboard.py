"""Own keyboard grabs and translate Caps Lock gestures into recording commands."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack

from structlog.stdlib import BoundLogger

from active_listener.infra.keyboard import KeyboardInput, RecordingGrabRelease
from active_listener.recording.session import RecordingTranscript

HOLD_THRESHOLD_S = 0.250


class RecordingKeyboard:
  """Coordinate hardware gestures without making transcript logic depend on hardware.

  :param keyboard: Keyboard supplying recording grabs.
  :param transcript: Current recording timeline receiving command intervals.
  :param logger: Gesture and resource failure diagnostics.
  """

  def __init__(
    self, keyboard: KeyboardInput, transcript: RecordingTranscript, logger: BoundLogger
  ) -> None:
    self.keyboard: KeyboardInput = keyboard
    self.transcript: RecordingTranscript = transcript
    self.logger: BoundLogger = logger
    self._grab_stack: AsyncExitStack | None = None
    self._release_grab: RecordingGrabRelease | None = None
    self._pending_start_s: float | None = None
    self._hold_task: asyncio.Task[None] | None = None
    self._command_open: bool = False

  async def start(self) -> None:
    """Acquire the recording grab before starting capture.

    :raises RuntimeError: If the grab is already owned.
    """
    if self._grab_stack is not None:
      self.logger.error("recording keyboard grab entered twice")
      raise RuntimeError("recording keyboard grab entered twice")
    stack = AsyncExitStack()
    try:
      release = await stack.enter_async_context(self.keyboard.recording_grab())
    except Exception:
      self.logger.exception("recording keyboard grab failed")
      await stack.aclose()
      raise
    self._grab_stack = stack
    self._release_grab = release

  async def stop(self) -> None:
    """Cancel pending gestures and release hardware before transcript teardown."""
    await self._cancel_hold()
    self._pending_start_s = None
    self._command_open = False
    release, stack = self._release_grab, self._grab_stack
    self._release_grab = None
    self._grab_stack = None
    try:
      if release is not None:
        _ = release()
    finally:
      if stack is not None:
        await stack.aclose()

  async def capslock_down(self, received_monotonic_s: float) -> None:
    """Begin distinguishing a finish tap from a command hold.

    :param received_monotonic_s: Hardware event receipt time.
    """
    if self._pending_start_s is not None or self._command_open:
      return
    start_s = self.transcript.recording_elapsed_s(received_monotonic_s)
    self._pending_start_s = start_s
    self._hold_task = asyncio.create_task(self._commit_hold(start_s))

  async def capslock_up(self, received_monotonic_s: float) -> bool:
    """Close a command hold or request finish for a short tap.

    :param received_monotonic_s: Hardware event receipt time.
    :return: True only when the gesture requests recording finish.
    """
    end_s = self.transcript.recording_elapsed_s(received_monotonic_s)
    start_s = self._pending_start_s
    if start_s is not None:
      self._pending_start_s = None
      await self._cancel_hold()
      if end_s - start_s < HOLD_THRESHOLD_S:
        return True
      self.transcript.open_command_span(start_s)
      self.transcript.close_command_span(end_s)
      return False
    if self._command_open:
      self.transcript.close_command_span(end_s)
      self._command_open = False
    return False

  async def _commit_hold(self, start_s: float) -> None:
    try:
      await asyncio.sleep(HOLD_THRESHOLD_S)
    except asyncio.CancelledError:
      return
    if self._pending_start_s != start_s:
      return
    self._pending_start_s = None
    self._command_open = True
    self.transcript.open_command_span(start_s)
    self.logger.debug("caps hold committed", start_s=start_s)

  async def _cancel_hold(self) -> None:
    task, self._hold_task = self._hold_task, None
    if task is not None and not task.done():
      _ = task.cancel()
      _ = await asyncio.gather(task, return_exceptions=True)
