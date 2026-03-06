"""Smoke tests for wire protocol serialization.

These tests document the key API goal for wire models: messages must round-trip
through the codec without losing externally visible contract fields.
"""

from eavesdrop.wire import ServerReadyMessage, deserialize_message, serialize_message


def test_server_ready_message_round_trips() -> None:
  message = ServerReadyMessage(stream="stream-1", backend="faster_whisper")

  serialized = serialize_message(message)
  decoded = deserialize_message(serialized)

  assert decoded.type == "server_ready"
  assert decoded.stream == "stream-1"
  assert decoded.backend == "faster_whisper"
