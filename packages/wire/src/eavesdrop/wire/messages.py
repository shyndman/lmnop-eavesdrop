"""
Pydantic message protocol models for WebSocket communication.

Defines the message structure for communication between the server and live WebSocket clients.
"""

import time
from enum import StrEnum
from typing import Literal

from pydantic import Field
from pydantic.dataclasses import dataclass

from .transcription import Segment, UserTranscriptionOptions


class ClientType(StrEnum):
  """WebSocket client types."""

  TRANSCRIBER = "transcriber"
  HEALTH_CHECK = "health_check"


class WebSocketHeaders(StrEnum):
  """WebSocket header names for client communication."""

  CLIENT_TYPE = "X-Client-Type"


@dataclass
class BaseMessage:
  """Base message with common fields for all WebSocket messages."""

  timestamp: float = Field(default_factory=time.time)


@dataclass(kw_only=True)
class TranscriptionMessage(BaseMessage):
  """Message containing live transcription results."""

  type: Literal["transcription"] = "transcription"
  stream: str = Field(description="Live stream session identifier")
  segments: list[Segment] = Field(description="List of transcription segments")
  language: str | None = Field(default=None, description="Detected or specified language code")
  recording_id: str | None = Field(
    default=None,
    description="Live recording identifier",
  )
  flush_complete: bool | None = Field(
    default=None,
    description="True only for the response that satisfies an accepted flush request",
  )


@dataclass(kw_only=True)
class ErrorMessage(BaseMessage):
  """Message indicating an error condition."""

  type: Literal["error"] = "error"
  stream: str | None = Field(
    default=None,
    description="Stream name if error is stream-specific, None for connection-level errors",
  )
  message: str = Field(description="Error message description")


@dataclass(kw_only=True)
class LanguageDetectionMessage(BaseMessage):
  """Message containing language detection results."""

  type: Literal["language_detection"] = "language_detection"
  stream: str = Field(description="Stream name or client identifier")
  language: str = Field(description="Detected language code")
  language_prob: float = Field(description="Confidence score for the detection", ge=0.0, le=1.0)


@dataclass(kw_only=True)
class ServerReadyMessage(BaseMessage):
  """Message indicating server is ready for transcription."""

  type: Literal["server_ready"] = "server_ready"
  stream: str = Field(description="Stream name or client identifier")
  backend: str = Field(description="Name of the transcription backend being used")


@dataclass(kw_only=True)
class DisconnectMessage(BaseMessage):
  """Message indicating client disconnection."""

  type: Literal["disconnect"] = "disconnect"
  stream: str = Field(description="Stream name or client identifier")
  message: str | None = Field(default=None, description="Optional disconnect reason")


@dataclass(kw_only=True)
class HealthCheckRequest(BaseMessage):
  """Message requesting a health check."""

  type: Literal["health_check"] = "health_check"


@dataclass(kw_only=True)
class TranscriptionSetupMessage(BaseMessage):
  """Message containing user transcription configuration."""

  type: Literal["setup"] = "setup"
  stream: str = Field(description="Stream name or client identifier")
  options: UserTranscriptionOptions = Field(description="User-specified live transcription options")


@dataclass(kw_only=True)
class RecordingStartedMessage(BaseMessage):
  """Message declaring the start of a new live recording epoch."""

  type: Literal["control_recording_started"] = "control_recording_started"
  stream: str = Field(description="Stream name or client identifier")
  recording_id: str = Field(description="Opaque foreground recording identifier")
  sample_rate_hz: int = Field(description="Recording sample rate in Hz")


@dataclass(kw_only=True)
class FlushControlMessage(BaseMessage):
  """Message requesting an in-session live transcription flush."""

  type: Literal["control_flush"] = "control_flush"
  stream: str = Field(description="Stream name or client identifier")
  recording_id: str | None = Field(
    default=None,
    description="Live recording identifier, omitted outside active-listener live mode",
  )
  force_complete: bool = Field(
    default=True,
    description="Whether the server should force-complete the current tentative tail",
  )


@dataclass(kw_only=True)
class UtteranceCancelledMessage(BaseMessage):
  """Message declaring that the current live utterance was cancelled."""

  type: Literal["control_utterance_cancelled"] = "control_utterance_cancelled"
  stream: str = Field(description="Stream name or client identifier")
  recording_id: str | None = Field(
    default=None,
    description="Live recording identifier, omitted outside active-listener live mode",
  )
