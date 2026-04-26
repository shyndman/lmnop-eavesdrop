"""Runtime queue signal types for the active-listener event loop."""

from __future__ import annotations

from dataclasses import dataclass

from active_listener.infra.keyboard import KeyboardControlEvent
from eavesdrop.client import (
  ConnectedEvent,
  DisconnectedEvent,
  ReconnectedEvent,
  ReconnectingEvent,
  TranscriptionEvent,
)


@dataclass(frozen=True)
class KeyboardEventSignal:
  """Signal carrying one keyboard event into the runtime policy queue."""

  event: KeyboardControlEvent


@dataclass(frozen=True)
class ClientSignal:
  """Signal carrying a live client event into app policy."""

  event: (
    ConnectedEvent | DisconnectedEvent | ReconnectingEvent | ReconnectedEvent | TranscriptionEvent
  )


RuntimeSignal = KeyboardEventSignal | ClientSignal
