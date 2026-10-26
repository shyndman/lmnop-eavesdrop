"""
Eavesdrop wire protocol package.

Contains shared message types and data structures used for communication
between eavesdrop clients and servers.
"""

from eavesdrop.wire.codec import deserialize_message, serialize_message
from eavesdrop.wire.messages import (
  BaseMessage,
  ClientType,
  DisconnectMessage,
  ErrorMessage,
  FlushControlMessage,
  LanguageDetectionMessage,
  RecordingStartedMessage,
  ServerReadyMessage,
  TranscriptionMessage,
  TranscriptionSetupMessage,
  UtteranceCancelledMessage,
  WebSocketHeaders,
)
from eavesdrop.wire.transcription import (
  Segment,
  UserTranscriptionOptions,
  Word,
)

__all__ = [
  "BaseMessage",
  "ClientType",
  "DisconnectMessage",
  "ErrorMessage",
  "FlushControlMessage",
  "LanguageDetectionMessage",
  "RecordingStartedMessage",
  "Segment",
  "ServerReadyMessage",
  "TranscriptionMessage",
  "TranscriptionSetupMessage",
  "UtteranceCancelledMessage",
  "UserTranscriptionOptions",
  "WebSocketHeaders",
  "Word",
  "deserialize_message",
  "serialize_message",
]
