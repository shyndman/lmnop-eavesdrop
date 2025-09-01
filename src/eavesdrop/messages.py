"""
Pydantic message protocol models for WebSocket communication.

Defines the message structure for communication between the server and WebSocket clients,
including both transcriber clients and RTSP subscriber clients.
"""

import time
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from .transcription.models import Segment


class ClientType(StrEnum):
  """WebSocket client types."""

  TRANSCRIBER = "transcriber"
  RTSP_SUBSCRIBER = "rtsp_subscriber"


class WebSocketHeaders(StrEnum):
  """WebSocket header names for client communication."""

  CLIENT_TYPE = "X-Client-Type"
  STREAM_NAMES = "X-Stream-Names"


class BaseMessage(BaseModel):
  """Base message with common fields for all WebSocket messages."""

  timestamp: float = Field(default_factory=time.time)


class TranscriptionMessage(BaseMessage):
  """Message containing transcription results sent to subscribers."""

  type: Literal["transcription"] = "transcription"
  stream: str = Field(description="Stream name for RTSP subscribers")
  segments: list[Segment] = Field(description="List of transcription segments")
  language: str | None = Field(default=None, description="Detected or specified language code")


class StreamStatusMessage(BaseMessage):
  """Message indicating status changes for RTSP streams."""

  type: Literal["stream_status"] = "stream_status"
  stream: str = Field(description="Name of the stream this status update refers to")
  status: Literal["online", "offline", "error"] = Field(description="Current status of the stream")
  message: str | None = Field(
    default=None, description="Optional descriptive message about the status change"
  )


class ErrorMessage(BaseMessage):
  """Message indicating an error condition."""

  type: Literal["error"] = "error"
  stream: str | None = Field(
    default=None,
    description="Stream name if error is stream-specific, None for connection-level errors",
  )
  message: str = Field(description="Error message description")


# Discriminated union for all outbound message types
OutboundMessage = Annotated[
  TranscriptionMessage | StreamStatusMessage | ErrorMessage, Field(discriminator="type")
]
