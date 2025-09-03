"""
Eavesdrop wire protocol package.

Contains shared message types and data structures used for communication
between eavesdrop clients and servers.
"""

from .codec import deserialize_message, serialize_message
from .messages import (
  BaseMessage,
  ClientType,
  DisconnectMessage,
  ErrorMessage,
  LanguageDetectionMessage,
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
  "LanguageDetectionMessage",
  "Segment",
  "ServerReadyMessage",
  "StreamStatusMessage",
  "TranscriptionMessage",
  "TranscriptionSetupMessage",
  "UserTranscriptionOptions",
  "WebSocketHeaders",
  "Word",
  "deserialize_message",
  "serialize_message",
]
