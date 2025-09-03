"""
Message codec for wire protocol serialization and deserialization.

Provides clean public API for converting between wire protocol message objects
and JSON strings, hiding the implementation details of Pydantic serialization.
"""

from pydantic import BaseModel, Field, TypeAdapter

from .messages import (
  DisconnectMessage,
  ErrorMessage,
  HealthCheckRequest,
  LanguageDetectionMessage,
  ServerReadyMessage,
  StreamStatusMessage,
  TranscriptionMessage,
  TranscriptionSetupMessage,
)

type Message = (
  TranscriptionMessage
  | StreamStatusMessage
  | ErrorMessage
  | LanguageDetectionMessage
  | ServerReadyMessage
  | DisconnectMessage
  | HealthCheckRequest
  | TranscriptionSetupMessage
)


class _MessageCodec(BaseModel):
  """Private message wrapper type for deserializing the discriminated union of message types."""

  message: Message = Field(discriminator="type")


def serialize_message(message: Message) -> str:
  """
  Serialize a wire protocol message to JSON string.

  Args:
      message: Any wire protocol message instance

  Returns:
      JSON string representation of the message
  """
  message_type = type(message)
  adapter = TypeAdapter(message_type)
  return adapter.dump_json(message).decode("utf-8")


def deserialize_message(json_str: str) -> Message:
  """
  Deserialize a JSON string to wire protocol message.

  Args:
      json_str: JSON string containing the message

  Returns:
      Deserialized message instance of the appropriate type
  """
  # Wrap the incoming message in the expected MessageCodec structure
  wrapped_json = f'{{"message": {json_str}}}'
  codec = _MessageCodec.model_validate_json(wrapped_json)
  return codec.message
