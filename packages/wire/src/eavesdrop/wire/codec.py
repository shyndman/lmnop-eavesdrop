"""
Message codec for wire protocol serialization and deserialization.

Provides clean public API for converting between wire protocol message objects
and JSON strings, hiding the implementation details of Pydantic serialization.
"""

from typing import Annotated

from pydantic import Field, TypeAdapter

from eavesdrop.wire.messages import (
  DisconnectMessage,
  ErrorMessage,
  FlushControlMessage,
  HealthCheckRequest,
  RecordingStartedMessage,
  ServerReadyMessage,
  TranscriptionMessage,
  TranscriptionSetupMessage,
  UtteranceCancelledMessage,
)

type Message = (
  TranscriptionMessage
  | ErrorMessage
  | RecordingStartedMessage
  | FlushControlMessage
  | UtteranceCancelledMessage
  | ServerReadyMessage
  | DisconnectMessage
  | HealthCheckRequest
  | TranscriptionSetupMessage
)

_MESSAGE_ADAPTER: TypeAdapter[Message] = TypeAdapter(
  Annotated[Message, Field(discriminator="type")]
)


def serialize_message(message: Message) -> str:
  """Serialize a wire protocol message to JSON string.

  :param message: Any wire protocol message instance.
  :type message: Message
  :returns: JSON string representation of the message.
  :rtype: str
  """
  return _MESSAGE_ADAPTER.dump_json(message, exclude_none=True).decode("utf-8")


def deserialize_message(json_str: str) -> Message:
  """Deserialize a JSON string to wire protocol message.

  :param json_str: JSON string containing the message.
  :type json_str: str
  :returns: Deserialized message instance of the appropriate type.
  :rtype: Message
  """
  return _MESSAGE_ADAPTER.validate_json(json_str)
