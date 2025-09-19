"""Mock implementations for testing active listener components."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from eavesdrop.active_listener.client import ConnectionState
from eavesdrop.active_listener.text_manager import TypingOperation
from eavesdrop.wire.transcription import Segment, UserTranscriptionOptions


class MockYdoToolTypist:
  """Mock implementation of YdoToolTypist for testing."""

  def __init__(self, available: bool = True, should_fail: bool = False):
    self._available = available
    self._initialized = available
    self._should_fail = should_fail
    self.typed_operations: list[TypingOperation] = []
    self.typed_text_history: list[str] = []
    self.deleted_chars_history: list[int] = []

  def is_available(self) -> bool:
    return self._available

  def initialize(self) -> None:
    if self._should_fail:
      raise Exception("Mock initialization failed")
    self._initialized = True
    self._available = True

  def type_text(self, text: str) -> None:
    if not self._available:
      raise Exception("ydotool is not available")
    if self._should_fail:
      raise Exception("Mock typing failed")
    self.typed_text_history.append(text)

  def delete_characters(self, count: int) -> None:
    if not self._available:
      raise Exception("pydotool is not available")
    if self._should_fail:
      raise Exception("Mock delete failed")
    self.deleted_chars_history.append(count)

  def execute_typing_operation(self, operation: TypingOperation) -> bool:
    if not self._available or self._should_fail:
      operation.completed = False
      return False

    self.typed_operations.append(operation)
    if operation.chars_to_delete > 0:
      self.delete_characters(operation.chars_to_delete)
    if operation.text_to_type:
      self.type_text(operation.text_to_type)

    operation.completed = True
    return True

  def execute_with_retry(self, operation: TypingOperation, max_attempts: int = 3) -> bool:
    return self.execute_typing_operation(operation)

  def check_permissions(self) -> bool:
    return self._available

  def get_permission_error_message(self) -> str:
    return "Mock permission error message"

  def clear_history(self) -> None:
    """Clear all recorded history for test isolation."""
    self.typed_operations.clear()
    self.typed_text_history.clear()
    self.deleted_chars_history.clear()

  def set_available(self, available: bool) -> None:
    """Change availability for testing failure scenarios."""
    self._available = available
    self._initialized = available

  def set_should_fail(self, should_fail: bool) -> None:
    """Control whether operations should fail for testing."""
    self._should_fail = should_fail


class MockEavesdropClient:
  """Mock implementation of EavesdropClient for testing."""

  def __init__(self):
    self.host: str = "localhost"
    self.port: int = 9090
    self.audio_device: str = "default"
    self.options: UserTranscriptionOptions = UserTranscriptionOptions()
    self._connected = False
    self._streaming = False
    self.connection_state = ConnectionState()
    self.message_queue: list[Any] = []
    self.on_message_callback: Any = None

  async def connect(self) -> None:
    if self.host == "invalid" or self.port == 0:
      raise Exception("Mock connection failed")
    self._connected = True
    self.connection_state.is_connected = True

  async def disconnect(self) -> None:
    self._connected = False
    self._streaming = False
    self.connection_state.is_connected = False
    self.connection_state.is_streaming = False

  async def start_streaming(self) -> None:
    if not self._connected:
      raise Exception("Not connected")
    self._streaming = True
    self.connection_state.is_streaming = True

  async def stop_streaming(self) -> None:
    self._streaming = False
    self.connection_state.is_streaming = False

  def is_connected(self) -> bool:
    return self._connected

  def is_streaming(self) -> bool:
    return self._streaming

  def add_mock_message(self, message: Any) -> None:
    """Add a message to be yielded by the async iterator."""
    self.message_queue.append(message)

  async def __aiter__(self) -> AsyncIterator[Any]:
    """Async iterator that yields mock messages."""
    while self.message_queue:
      message = self.message_queue.pop(0)
      yield message
      await asyncio.sleep(0.001)  # Allow other coroutines to run

  def validate_audio_device(self, device: str) -> bool:
    """Mock audio device validation."""
    return device != "invalid_device"


class MockEavesdropClientWrapper:
  """Mock implementation of EavesdropClientWrapper for testing."""

  def __init__(self, host: str = "localhost", port: int = 9090, audio_device: str = "default"):
    self.host = host
    self.port = port
    self.audio_device = audio_device
    self._client = MockEavesdropClient()
    self._client.host = host
    self._client.port = port
    self._client.audio_device = audio_device
    self._initialized = False

  async def initialize(self) -> None:
    await self._client.connect()
    self._initialized = True

  async def start_streaming(self) -> None:
    if not self._initialized:
      raise Exception("Client not initialized")
    await self._client.start_streaming()

  async def shutdown(self) -> None:
    if self._client.is_streaming():
      await self._client.stop_streaming()
    if self._client.is_connected():
      await self._client.disconnect()

  def check_connection_health(self) -> bool:
    return self._client.is_connected()

  async def attempt_reconnection(self) -> bool:
    try:
      await self._client.connect()
      return True
    except Exception:
      return False

  def add_mock_message(self, message: Any) -> None:
    """Add a message to be yielded by the async iterator."""
    self._client.add_mock_message(message)

  async def __aiter__(self) -> AsyncIterator[Any]:
    """Delegate to the underlying client's async iterator."""
    async for message in self._client:
      yield message


class MockTranscriptionMessage:
  """Mock transcription message for testing."""

  def __init__(self, segments: list[Segment]):
    self.segments = segments


def create_mock_segment(
  segment_id: int, text: str, completed: bool = False, start: float = 0.0, end: float = 1.0
) -> Segment:
  """Create a mock Segment with all required fields."""
  return Segment(
    id=segment_id,
    seek=0,
    start=start,
    end=end,
    text=text,
    tokens=[],
    avg_logprob=0.0,
    compression_ratio=1.0,
    words=None,
    temperature=0.0,
    completed=completed,
  )


def create_mock_transcription_message(
  segments: list[tuple[int, str, bool]],
) -> MockTranscriptionMessage:
  """Create a mock transcription message with segments.

  Args:
      segments: List of tuples (id, text, completed)
  """
  mock_segments = [
    create_mock_segment(seg_id, text, completed) for seg_id, text, completed in segments
  ]
  return MockTranscriptionMessage(mock_segments)


class MockStructlogLogger:
  """Mock logger for testing that captures log calls."""

  def __init__(self):
    self.log_calls: list[tuple[str, dict]] = []

  def bind(self, **kwargs) -> "MockStructlogLogger":
    """Return self for method chaining."""
    return self

  def info(self, message: str, **kwargs) -> None:
    self.log_calls.append(("info", {"message": message, **kwargs}))

  def warning(self, message: str, **kwargs) -> None:
    self.log_calls.append(("warning", {"message": message, **kwargs}))

  def error(self, message: str, **kwargs) -> None:
    self.log_calls.append(("error", {"message": message, **kwargs}))

  def exception(self, message: str, **kwargs) -> None:
    self.log_calls.append(("exception", {"message": message, **kwargs}))

  def debug(self, message: str, **kwargs) -> None:
    self.log_calls.append(("debug", {"message": message, **kwargs}))

  def clear_logs(self) -> None:
    """Clear all captured log calls."""
    self.log_calls.clear()

  def get_logs_by_level(self, level: str) -> list[dict]:
    """Get all log calls for a specific level."""
    return [call[1] for call in self.log_calls if call[0] == level]
