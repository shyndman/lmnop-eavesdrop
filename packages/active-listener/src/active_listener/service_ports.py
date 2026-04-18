"""Shared runtime ports for the active-listener service."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from eavesdrop.client import (
  ConnectedEvent,
  DisconnectedEvent,
  ReconnectedEvent,
  ReconnectingEvent,
  TranscriptionEvent,
)
from eavesdrop.wire import TranscriptionMessage


class ActiveListenerClient(Protocol):
  """Protocol for the live transcription client dependency."""

  async def connect(self) -> None:
    """Establish the live connection."""
    ...

  async def disconnect(self) -> None:
    """Close the live connection."""
    ...

  async def start_streaming(self) -> None:
    """Begin microphone capture and upstream streaming."""
    ...

  async def stop_streaming(self) -> None:
    """Stop microphone capture for the active recording."""
    ...

  async def cancel_utterance(self) -> None:
    """Discard the current live utterance without closing the session."""
    ...

  async def flush(self, *, force_complete: bool = True) -> TranscriptionMessage:
    """Request a committed transcription flush from the server."""
    ...

  def __aiter__(
    self,
  ) -> AsyncIterator[
    ConnectedEvent | DisconnectedEvent | ReconnectingEvent | ReconnectedEvent | TranscriptionEvent
  ]:
    """Iterate ordered client lifecycle events."""
    ...


class ActiveListenerLogger(Protocol):
  """Minimal structured logger API used by active-listener."""

  def info(self, event: str, **kwargs: object) -> None:
    """Emit an informational event."""
    ...

  def warning(self, event: str, **kwargs: object) -> None:
    """Emit a warning event."""
    ...

  def exception(self, event: str, **kwargs: object) -> None:
    """Emit an exception event with stack trace."""


class ActiveListenerRewriteClient(Protocol):
  async def rewrite_text(
    self,
    *,
    model_name: str,
    instructions: str,
    transcript: str,
  ) -> str: ...


class ActiveListenerRuntimeError(RuntimeError):
  """Raised when the service cannot satisfy runtime prerequisites."""
