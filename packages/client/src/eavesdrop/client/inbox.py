"""Recording-scoped transcription delivery, flush waiting, and ordered client events."""

import asyncio
from collections.abc import Awaitable, Callable

from structlog.stdlib import BoundLogger

from eavesdrop.client.events import LanguageDetectionEvent, LiveClientEvent, TranscriptionEvent
from eavesdrop.wire import LanguageDetectionMessage, TranscriptionMessage


class ClientInbox:
  """Own incoming results and keep flush failures local to their outstanding request."""

  def __init__(self, logger: BoundLogger) -> None:
    self._logger: BoundLogger = logger
    self._messages: asyncio.Queue[TranscriptionMessage] = asyncio.Queue()
    self._events: asyncio.Queue[LiveClientEvent] = asyncio.Queue()
    self._recording_id: str | None = None
    self._open: bool = False
    self._disconnected: asyncio.Event = asyncio.Event()
    self.disconnect_reason: str | None = None
    self._flush_waiting: bool = False
    self._flush_error: str | None = None

  @property
  def is_open(self) -> bool:
    """Return whether consumers should continue waiting for events."""
    return self._open

  def open(self) -> None:
    """Begin event delivery and reset transport disconnect signaling."""
    self._open = True
    self.reconnected()

  def close(self) -> None:
    """Stop waiting for new events while allowing buffered events to drain."""
    self._open = False

  def reconnected(self) -> None:
    """Reset transport signaling without clearing an outstanding flush failure."""
    self._disconnected.clear()
    self.disconnect_reason = None

  def reset(self) -> None:
    """Reset recording state after coordinated transport shutdown."""
    self.reconnected()
    self._flush_error = None
    self._recording_id = None
    self._discard_transcription_events()

  def start_recording(self, recording_id: str) -> None:
    """Accept the next recording without erasing already accepted events."""
    self._recording_id = recording_id

  def cancel_recording(self) -> None:
    """Discard cancelled results while preserving queued connection events."""
    self._recording_id = None
    self._clear_messages()
    self._discard_transcription_events()

  def publish(self, event: LiveClientEvent) -> None:
    """Append an event in callback order while the event stream is open."""
    if self._open:
      self._events.put_nowait(event)

  def accept_transcription(self, message: TranscriptionMessage) -> None:
    """Deliver only results belonging to the active recording epoch."""
    if message.recording_id != self._recording_id:
      self._logger.debug(
        "dropping stale transcription message",
        stream=message.stream,
        recording_id=message.recording_id,
        active_recording_id=self._recording_id,
      )
      return
    self._messages.put_nowait(message)
    self.publish(TranscriptionEvent(stream=message.stream, message=message))

  def accept_language(self, message: LanguageDetectionMessage) -> None:
    """Log and publish the server's detected language in arrival order."""
    self._logger.info(
      "language detected",
      stream=message.stream,
      language=message.language,
      probability=message.language_prob,
    )
    self.publish(
      LanguageDetectionEvent(
        stream=message.stream,
        language=message.language,
        probability=message.language_prob,
      )
    )

  def disconnected(self, reason: str | None) -> None:
    """Wake flush waiting and retain its first failure across reconnection."""
    self.disconnect_reason = reason
    self._disconnected.set()
    self.fail_flush(self._disconnect_message())

  def fail_flush(self, error: str) -> None:
    """Fail only an outstanding request, retaining its original failure."""
    if self._flush_waiting and self._flush_error is None:
      self._flush_error = error

  def reject_concurrent_flush(self, recording_id: str) -> None:
    """Reject overlapping flush requests before they wait on recording transitions."""
    if self._flush_waiting:
      self._logger.warning("rejecting concurrent flush", recording_id=recording_id)
      raise RuntimeError("flush() already in progress")

  async def flush(
    self,
    recording_id: str,
    send_control: Callable[[], Awaitable[None]],
    message_task: asyncio.Task[None],
  ) -> TranscriptionMessage:
    """Send one control and await its recording's terminal response.

    :param recording_id: Recording whose completion satisfies this request.
    :param send_control: Operation sending the already validated flush control.
    :param message_task: Active transport reader, used to detect terminal closure.
    :raises RuntimeError: If another flush is pending or delivery fails.
    """
    self.reject_concurrent_flush(recording_id)
    self._flush_waiting = True
    self._flush_error = None
    try:
      self._clear_messages()
      await send_control()
      while True:
        if self._flush_error is not None:
          raise RuntimeError(self._flush_error)
        if message_task.done() and self._messages.empty():
          raise RuntimeError(self._disconnect_message())
        try:
          message = await asyncio.wait_for(self._messages.get(), timeout=0.1)
        except asyncio.TimeoutError:
          if self._disconnected.is_set():
            raise RuntimeError(self._disconnect_message())
          continue
        if message.recording_id == recording_id and message.flush_complete is True:
          return message
    finally:
      self._flush_waiting = False
      self._flush_error = None

  async def next_event(self) -> LiveClientEvent:
    """Return the next ordered event, draining buffered events after closure."""
    while self._open or not self._events.empty():
      try:
        return await asyncio.wait_for(self._events.get(), timeout=0.1)
      except asyncio.TimeoutError:
        continue
    raise StopAsyncIteration

  def _clear_messages(self) -> None:
    """Discard stale flush responses before a new request or after cancellation."""
    while not self._messages.empty():
      _ = self._messages.get_nowait()

  def _discard_transcription_events(self) -> None:
    """Retain connection events in their original order when dropping transcripts."""
    preserved: list[LiveClientEvent] = []
    while not self._events.empty():
      event = self._events.get_nowait()
      if not isinstance(event, TranscriptionEvent):
        preserved.append(event)
    for event in preserved:
      self._events.put_nowait(event)

  def _disconnect_message(self) -> str:
    """Describe the transport failure responsible for an unfinished flush."""
    return self.disconnect_reason or (
      "flush() failed because the live connection closed before flush completion"
    )
