"""
Eavesdrop wire protocol package.

Contains shared message types and data structures used for communication
between eavesdrop clients and servers.
"""

from .messages import (
  BaseMessage,
  ClientType,
  DisconnectMessage,
  ErrorMessage,
  InboundMessage,
  LanguageDetectionMessage,
  OutboundMessage,
  ServerReadyMessage,
  StreamStatusMessage,
  TranscriptionConfigMessage,
  TranscriptionMessage,
  WebSocketHeaders,
)
from .transcription import Segment, UserTranscriptionOptions, Word

__all__ = [
  "BaseMessage",
  "ClientType",
  "DisconnectMessage",
  "ErrorMessage",
  "InboundMessage",
  "LanguageDetectionMessage",
  "OutboundMessage",
  "Segment",
  "ServerReadyMessage",
  "StreamStatusMessage",
  "TranscriptionConfigMessage",
  "TranscriptionMessage",
  "UserTranscriptionOptions",
  "WebSocketHeaders",
  "Word",
]
