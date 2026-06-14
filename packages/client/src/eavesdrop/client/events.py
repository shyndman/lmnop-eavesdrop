"""Live client event types exposed by :mod:`eavesdrop.client`."""

from dataclasses import dataclass
from typing import Literal, TypeAlias

from eavesdrop.wire import TranscriptionMessage


@dataclass(frozen=True)
class ConnectedEvent:
  """Initial connection event for a live client session."""

  stream: str
  family: Literal["connected"] = "connected"


@dataclass(frozen=True)
class DisconnectedEvent:
  """Terminal disconnect event for the current live socket."""

  stream: str
  reason: str | None
  family: Literal["disconnected"] = "disconnected"


@dataclass(frozen=True)
class ReconnectingEvent:
  """Reconnect-attempt event for a live transcriber session."""

  stream: str
  attempt: int
  retry_delay_s: float
  family: Literal["reconnecting"] = "reconnecting"


@dataclass(frozen=True)
class ReconnectedEvent:
  """Successful reconnect event for an existing live client session."""

  stream: str
  family: Literal["reconnected"] = "reconnected"


@dataclass(frozen=True)
class TranscriptionEvent:
  """Transcription payload event emitted by the live client iterator."""

  stream: str
  message: TranscriptionMessage
  family: Literal["transcription"] = "transcription"


@dataclass(frozen=True)
class LanguageDetectionEvent:
  """Auto-detected source-language event emitted by the live client iterator."""

  stream: str
  language: str
  probability: float
  family: Literal["language_detection"] = "language_detection"


LiveClientEvent: TypeAlias = (
  ConnectedEvent
  | DisconnectedEvent
  | ReconnectingEvent
  | ReconnectedEvent
  | TranscriptionEvent
  | LanguageDetectionEvent
)


__all__ = [
  "ConnectedEvent",
  "DisconnectedEvent",
  "ReconnectingEvent",
  "ReconnectedEvent",
  "TranscriptionEvent",
  "LanguageDetectionEvent",
  "LiveClientEvent",
]
