"""Application-owned signals consumed by the listener's sequential policy queue."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from active_listener.app.ports import ActiveListenerClient
from active_listener.infra.keyboard import KeyboardControlEvent
from eavesdrop.wire import TranscriptionMessage


@dataclass(frozen=True)
class KeyboardEventSignal:
  """One keyboard event awaiting sequential application dispatch."""

  event: KeyboardControlEvent


@dataclass(frozen=True)
class ConnectedSignal:
  """The transport established a server session."""

  session_id: str


@dataclass(frozen=True)
class DisconnectedSignal:
  """The current server session disconnected."""

  session_id: str
  reason: str | None = None


@dataclass(frozen=True)
class ReconnectingSignal:
  """The transport scheduled another connection attempt."""

  session_id: str
  attempt: int
  retry_delay_s: float


@dataclass(frozen=True)
class ReconnectedSignal:
  """The transport established a replacement session."""

  session_id: str


type ConnectionSignal = (
  ConnectedSignal | DisconnectedSignal | ReconnectingSignal | ReconnectedSignal
)
type ClientSignal = ConnectionSignal | TranscriptionMessage
type RuntimeSignal = KeyboardEventSignal | ClientSignal


def bind_client_callbacks(
  client: ActiveListenerClient, queue: asyncio.Queue[RuntimeSignal]
) -> None:
  """Enqueue client push delivery before connecting the transport.

  :param client: Client exposing synchronous optional callbacks.
  :param queue: Existing application policy queue consumed by the service.
  :return: None; callback delivery never awaits application policy processing.
  """
  client.on_transcription = queue.put_nowait
  client.on_connected = lambda session_id: queue.put_nowait(ConnectedSignal(session_id))
  client.on_disconnected = lambda session_id, reason: queue.put_nowait(
    DisconnectedSignal(session_id, reason)
  )
  client.on_reconnecting = lambda session_id, attempt, delay: queue.put_nowait(
    ReconnectingSignal(session_id, attempt, delay)
  )
  client.on_reconnected = lambda session_id: queue.put_nowait(ReconnectedSignal(session_id))
