"""
Eavesdrop wire protocol package.

Contains shared message types and data structures used for communication
between eavesdrop clients and servers.
"""

from eavesdrop.wire.codec import deserialize_message, serialize_message
from eavesdrop.wire.messages import (
  BaseMessage,
  DisconnectMessage,
  ErrorMessage,
  FlushControlMessage,
  RecordingStartedMessage,
  ServerReadyMessage,
  TranscriptionMessage,
  TranscriptionSetupMessage,
  UtteranceCancelledMessage,
)
from eavesdrop.wire.transcription import (
  Segment,
  UserTranscriptionOptions,
  Word,
)

__all__ = [
  "BaseMessage",
  "DisconnectMessage",
  "ErrorMessage",
  "FlushControlMessage",
  "RecordingStartedMessage",
  "Segment",
  "ServerReadyMessage",
  "TranscriptionMessage",
  "TranscriptionSetupMessage",
  "UtteranceCancelledMessage",
  "UserTranscriptionOptions",
  "Word",
  "deserialize_message",
  "serialize_message",
]
