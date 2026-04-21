"""Runtime queue signal types for the active-listener event loop."""

from __future__ import annotations

from dataclasses import dataclass

from active_listener.app.state import AppAction
from eavesdrop.client import (
  ConnectedEvent,
  DisconnectedEvent,
  ReconnectedEvent,
  ReconnectingEvent,
  TranscriptionEvent,
)


@dataclass(frozen=True)
class AppActionSignal:
  """Signal carrying one app action into the runtime policy queue."""

  action: AppAction


@dataclass(frozen=True)
class ClientSignal:
  """Signal carrying a live client event into app policy."""

  event: (
    ConnectedEvent | DisconnectedEvent | ReconnectingEvent | ReconnectedEvent | TranscriptionEvent
  )


RuntimeSignal = AppActionSignal | ClientSignal
