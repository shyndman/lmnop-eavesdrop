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
  FlushControlMessage,
  LanguageDetectionMessage,
  ServerReadyMessage,
  StreamStatusMessage,
  TranscriptionMessage,
  TranscriptionSetupMessage,
  UtteranceCancelledMessage,
  WebSocketHeaders,
)
from .transcription import Segment, TranscriptionSourceMode, UserTranscriptionOptions, Word

__all__ = [
  "BaseMessage",
  "ClientType",
  "DisconnectMessage",
  "ErrorMessage",
  "FlushControlMessage",
  "LanguageDetectionMessage",
  "Segment",
  "ServerReadyMessage",
  "StreamStatusMessage",
  "TranscriptionMessage",
  "TranscriptionSourceMode",
  "TranscriptionSetupMessage",
  "UtteranceCancelledMessage",
  "UserTranscriptionOptions",
  "WebSocketHeaders",
  "Word",
  "deserialize_message",
  "serialize_message",
]
