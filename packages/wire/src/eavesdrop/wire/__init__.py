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
  MessageCodec,
  OutboundMessage,
  ServerReadyMessage,
  StreamStatusMessage,
  TranscriptionMessage,
  TranscriptionSetupMessage,
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
  "MessageCodec",
  "OutboundMessage",
  "Segment",
  "ServerReadyMessage",
  "StreamStatusMessage",
  "TranscriptionMessage",
  "TranscriptionSetupMessage",
  "UserTranscriptionOptions",
  "WebSocketHeaders",
  "Word",
]
