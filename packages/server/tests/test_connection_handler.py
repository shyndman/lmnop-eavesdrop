"""Tests for the WebSocketConnectionHandler class."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from websockets.exceptions import ConnectionClosed

from eavesdrop.server.connection_handler import (
  SubscriberConnection,
  TranscriberConnection,
  WebSocketConnectionHandler,
)
from eavesdrop.wire import ClientType, WebSocketHeaders


class TestWebSocketConnectionHandler:
  """Tests for WebSocketConnectionHandler initialization and properties."""

  @pytest.fixture
  def mock_client_initializer(self) -> AsyncMock:
    """Create a mock client initializer callback."""
    return AsyncMock()

  @pytest.fixture
  def mock_subscriber_manager(self) -> MagicMock:
    """Create a mock subscriber manager."""
    manager = MagicMock()
    manager.subscribe_client = AsyncMock(return_value=(True, None))
    return manager

  @pytest.fixture
  def handler(self, mock_client_initializer: AsyncMock) -> WebSocketConnectionHandler:
    """Create a handler with no subscriber manager."""
    return WebSocketConnectionHandler(
      client_initializer=mock_client_initializer,
      subscriber_manager=None,
    )

  def test_initial_state(
    self, handler: WebSocketConnectionHandler, mock_client_initializer: AsyncMock
  ) -> None:
    """Test handler initial state."""
    assert handler.subscriber_manager is None

  def test_subscriber_manager_setter(
    self, handler: WebSocketConnectionHandler, mock_subscriber_manager: MagicMock
  ) -> None:
    """Test subscriber manager can be set after construction."""
    assert handler.subscriber_manager is None

    handler.subscriber_manager = mock_subscriber_manager

    assert handler.subscriber_manager is mock_subscriber_manager


class TestHandleTranscriberConnection:
  """Tests for transcriber connection handling."""

  @pytest.fixture
  def mock_websocket(self) -> MagicMock:
    """Create a mock WebSocket connection."""
    ws = AsyncMock()
    ws.request = MagicMock()
    ws.request.headers = {WebSocketHeaders.CLIENT_TYPE: ClientType.TRANSCRIBER}
    ws.id = "test-id-123"
    ws.recv = AsyncMock()
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    return ws

  @pytest.fixture
  def mock_client_initializer(self) -> AsyncMock:
    """Create a mock client initializer callback."""
    return AsyncMock()

  @pytest.fixture
  def handler(self, mock_client_initializer: AsyncMock) -> WebSocketConnectionHandler:
    """Create a handler."""
    return WebSocketConnectionHandler(
      client_initializer=mock_client_initializer,
      subscriber_manager=None,
    )

  async def test_handle_transcriber_connection(
    self,
    handler: WebSocketConnectionHandler,
    mock_websocket: MagicMock,
    mock_client_initializer: AsyncMock,
  ) -> None:
    """Test handling of transcriber client connections."""
    # Arrange
    mock_client = MagicMock()
    mock_client_initializer.return_value = mock_client

    # Create a mock TranscriptionSetupMessage
    mock_websocket.recv.return_value = '{"type": "transcription_setup", "stream": "test"}'

    with patch("eavesdrop.server.connection_handler.deserialize_message") as mock_deserialize:
      from eavesdrop.wire import TranscriptionSetupMessage

      mock_msg = TranscriptionSetupMessage(stream="test", options={})
      mock_deserialize.return_value = mock_msg

      # Act
      result = await handler.handle_connection(mock_websocket)

      # Assert
      assert isinstance(result, TranscriberConnection)
      assert result.client == mock_client
      mock_client_initializer.assert_called_once_with(mock_websocket, mock_msg)

  async def test_handle_connection_closed_during_handshake(
    self,
    handler: WebSocketConnectionHandler,
    mock_websocket: MagicMock,
  ) -> None:
    """Test graceful handling of connection close during handshake."""
    mock_websocket.recv.side_effect = ConnectionClosed(None, None)

    result = await handler.handle_connection(mock_websocket)

    assert result is None


class TestHandleHealthCheck:
  """Tests for health check connection handling."""

  @pytest.fixture
  def mock_websocket(self) -> MagicMock:
    """Create a mock WebSocket connection for health check."""
    ws = AsyncMock()
    ws.request = MagicMock()
    ws.request.headers = {WebSocketHeaders.CLIENT_TYPE: ClientType.HEALTH_CHECK}
    ws.id = "health-check-id"
    ws.recv = AsyncMock()
    ws.close = AsyncMock()
    return ws

  @pytest.fixture
  def mock_client_initializer(self) -> AsyncMock:
    """Create a mock client initializer callback."""
    return AsyncMock()

  @pytest.fixture
  def handler(self, mock_client_initializer: AsyncMock) -> WebSocketConnectionHandler:
    """Create a handler."""
    return WebSocketConnectionHandler(
      client_initializer=mock_client_initializer,
      subscriber_manager=None,
    )

  async def test_handle_health_check(
    self,
    handler: WebSocketConnectionHandler,
    mock_websocket: MagicMock,
  ) -> None:
    """Test health check closes connection and returns None."""
    mock_websocket.recv.return_value = '{"type": "health_check_request"}'

    with patch("eavesdrop.server.connection_handler.deserialize_message") as mock_deserialize:
      from eavesdrop.wire.messages import HealthCheckRequest

      mock_deserialize.return_value = HealthCheckRequest()

      result = await handler.handle_connection(mock_websocket)

      assert result is None
      mock_websocket.close.assert_called_once()


class TestHandleSubscriberConnection:
  """Tests for RTSP subscriber connection handling."""

  @pytest.fixture
  def mock_websocket(self) -> MagicMock:
    """Create a mock WebSocket connection for subscriber."""
    ws = AsyncMock()
    ws.request = MagicMock()
    ws.request.headers = {
      WebSocketHeaders.CLIENT_TYPE: ClientType.RTSP_SUBSCRIBER,
      WebSocketHeaders.STREAM_NAMES: "office,lobby",
    }
    ws.id = "subscriber-id"
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    return ws

  @pytest.fixture
  def mock_client_initializer(self) -> AsyncMock:
    """Create a mock client initializer callback."""
    return AsyncMock()

  @pytest.fixture
  def mock_subscriber_manager(self) -> MagicMock:
    """Create a mock subscriber manager."""
    manager = MagicMock()
    manager.subscribe_client = AsyncMock(return_value=(True, None))
    return manager

  async def test_handle_subscriber_without_manager(
    self,
    mock_websocket: MagicMock,
    mock_client_initializer: AsyncMock,
  ) -> None:
    """Test subscriber connection fails when no manager configured."""
    handler = WebSocketConnectionHandler(
      client_initializer=mock_client_initializer,
      subscriber_manager=None,
    )

    result = await handler.handle_connection(mock_websocket)

    assert result is None
    mock_websocket.send.assert_called()  # Error message sent
    mock_websocket.close.assert_called()

  async def test_handle_subscriber_with_manager(
    self,
    mock_websocket: MagicMock,
    mock_client_initializer: AsyncMock,
    mock_subscriber_manager: MagicMock,
  ) -> None:
    """Test subscriber connection succeeds with manager."""
    handler = WebSocketConnectionHandler(
      client_initializer=mock_client_initializer,
      subscriber_manager=mock_subscriber_manager,
    )

    result = await handler.handle_connection(mock_websocket)

    assert isinstance(result, SubscriberConnection)
    mock_subscriber_manager.subscribe_client.assert_called_once()

  async def test_handle_subscriber_missing_stream_names(
    self,
    mock_client_initializer: AsyncMock,
    mock_subscriber_manager: MagicMock,
  ) -> None:
    """Test subscriber connection fails without stream names header."""
    ws = AsyncMock()
    ws.request = MagicMock()
    ws.request.headers = {
      WebSocketHeaders.CLIENT_TYPE: ClientType.RTSP_SUBSCRIBER,
      # Missing STREAM_NAMES header
    }
    ws.send = AsyncMock()
    ws.close = AsyncMock()

    handler = WebSocketConnectionHandler(
      client_initializer=mock_client_initializer,
      subscriber_manager=mock_subscriber_manager,
    )

    result = await handler.handle_connection(ws)

    assert result is None
    ws.send.assert_called()  # Error message sent
    ws.close.assert_called()

  async def test_handle_subscriber_subscription_failure(
    self,
    mock_websocket: MagicMock,
    mock_client_initializer: AsyncMock,
    mock_subscriber_manager: MagicMock,
  ) -> None:
    """Test subscriber connection fails when subscription fails."""
    mock_subscriber_manager.subscribe_client.return_value = (
      False,
      "Stream not found",
    )

    handler = WebSocketConnectionHandler(
      client_initializer=mock_client_initializer,
      subscriber_manager=mock_subscriber_manager,
    )

    result = await handler.handle_connection(mock_websocket)

    assert result is None
    mock_websocket.send.assert_called()  # Error message sent
    mock_websocket.close.assert_called()


class TestSendErrorAndClose:
  """Tests for error sending and connection closing."""

  @pytest.fixture
  def mock_websocket(self) -> MagicMock:
    """Create a mock WebSocket connection."""
    ws = AsyncMock()
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    return ws

  @pytest.fixture
  def mock_client_initializer(self) -> AsyncMock:
    """Create a mock client initializer callback."""
    return AsyncMock()

  @pytest.fixture
  def handler(self, mock_client_initializer: AsyncMock) -> WebSocketConnectionHandler:
    """Create a handler."""
    return WebSocketConnectionHandler(
      client_initializer=mock_client_initializer,
      subscriber_manager=None,
    )

  async def test_send_error_and_close_sends_message(
    self,
    handler: WebSocketConnectionHandler,
    mock_websocket: MagicMock,
  ) -> None:
    """Test that error message is sent and connection is closed."""
    await handler._send_error_and_close(mock_websocket, "Test error message")

    mock_websocket.send.assert_called_once()
    mock_websocket.close.assert_called_once()

  async def test_send_and_close_handles_exception(
    self,
    handler: WebSocketConnectionHandler,
    mock_websocket: MagicMock,
  ) -> None:
    """Test that exceptions during send are handled gracefully."""
    mock_websocket.send.side_effect = Exception("Connection lost")

    # Should not raise
    await handler._send_error_and_close(mock_websocket, "Test error")
