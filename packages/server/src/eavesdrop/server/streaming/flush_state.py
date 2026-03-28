"""Shared live-session flush state for streaming transcription.

This module keeps the server's pending-flush state in one canonical place while
exposing separate wakeup primitives for event-loop waits and worker-thread
checkpoints.
"""

import asyncio
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class PendingFlush:
  """Canonical record for one accepted live-session flush request."""

  boundary_sample: int
  force_complete: bool


class LiveSessionFlushState:
  """Thread-safe owner for one live session's pending flush request."""

  def __init__(self) -> None:
    self._lock: threading.Lock = threading.Lock()
    self._pending_flush: PendingFlush | None = None
    self._wakeup: asyncio.Event = asyncio.Event()
    self._interrupt: threading.Event = threading.Event()

  def begin_wait(self) -> bool:
    """Prepare an async wait without racing against new flush acceptance.

    :returns: ``True`` when the caller should wait, ``False`` when a flush is
      already pending and the wait should be skipped.
    :rtype: bool
    """
    with self._lock:
      if self._pending_flush is not None:
        return False
      self._wakeup.clear()
      return True

  def accept(self, *, boundary_sample: int, force_complete: bool) -> PendingFlush | None:
    """Accept a new flush request if no other flush is already pending.

    :param boundary_sample: Inclusive sample boundary captured at acceptance time.
    :type boundary_sample: int
    :param force_complete: Whether the flush-satisfying response should force
      completion of the tentative tail segment.
    :type force_complete: bool
    :returns: The accepted flush record, or ``None`` when another flush is still pending.
    :rtype: PendingFlush | None
    """
    with self._lock:
      if self._pending_flush is not None:
        return None
      pending_flush = PendingFlush(
        boundary_sample=boundary_sample,
        force_complete=force_complete,
      )
      self._pending_flush = pending_flush
      self._wakeup.set()
      self._interrupt.set()
      return pending_flush

  def pending(self) -> PendingFlush | None:
    """Return the currently pending flush, if any."""
    with self._lock:
      return self._pending_flush

  def clear_interrupt(self) -> None:
    """Clear the worker-thread interrupt doorbell after a checkpoint consumes it."""
    self._interrupt.clear()

  def complete(self) -> PendingFlush | None:
    """Clear the pending flush only after its response has been emitted."""
    with self._lock:
      previous = self._pending_flush
      self._pending_flush = None
      self._wakeup.clear()
      self._interrupt.clear()
      return previous

  @property
  def wakeup(self) -> asyncio.Event:
    """Expose the async wakeup event for wait interruption."""
    return self._wakeup

  @property
  def interrupt(self) -> threading.Event:
    """Expose the worker-thread interrupt event for checkpoint interruption."""
    return self._interrupt
