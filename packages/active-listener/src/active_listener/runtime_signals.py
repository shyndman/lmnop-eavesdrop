"""Runtime queue signal types for the active-listener event loop."""

from __future__ import annotations

from dataclasses import dataclass

from active_listener.state import KeyboardAction
from eavesdrop.client import (
  ConnectedEvent,
  DisconnectedEvent,
  ReconnectedEvent,
  ReconnectingEvent,
  TranscriptionEvent,
)


@dataclass(frozen=True)
class KeyboardSignal:
  """Signal carrying a workstation keyboard action into app policy."""

  action: KeyboardAction


@dataclass(frozen=True)
class ClientSignal:
  """Signal carrying a live client event into app policy."""

  event: (
    ConnectedEvent | DisconnectedEvent | ReconnectingEvent | ReconnectedEvent | TranscriptionEvent
  )


RuntimeSignal = KeyboardSignal | ClientSignal
