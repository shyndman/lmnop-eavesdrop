"""Tests for the WebSocketConnectionHandler class."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast, final

import pytest
from websockets.asyncio.server import ServerConnection
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed

from eavesdrop.server.connection_handler import (
  ClientInitializer,
  SubscriberConnection,
  TranscriberConnection,
  WebSocketConnectionHandler,
)
from eavesdrop.wire import (
  ClientType,
  TranscriptionSetupMessage,
  UserTranscriptionOptions,
  WebSocketHeaders,
)
from eavesdrop.wire.messages import HealthCheckRequest

if TYPE_CHECKING:
  from eavesdrop.server.rtsp.subscriber import RTSPSubscriberManager
  from eavesdrop.server.streaming import WebSocketStreamingClient


@final
@dataclass
class FakeRequest:
  headers: Headers


@final
@dataclass
class FakeWebSocket:
  headers: dict[str, str] = field(default_factory=dict)
  id: str = "test-id"
  recv_messages: list[str] = field(default_factory=list)
  recv_exception: Exception | None = None
  send_exception: Exception | None = None
  sent_messages: list[str] = field(default_factory=list)
  close_calls: int = 0
  request: FakeRequest = field(init=False)

  def __post_init__(self) -> None:
    self.request = FakeRequest(Headers(self.headers))

  async def recv(self, decode: bool = True) -> str:
    del decode
    if self.recv_exception is not None:
      raise self.recv_exception
    return self.recv_messages.pop(0)

  async def send(self, message: str) -> None:
    if self.send_exception is not None:
      raise self.send_exception
    self.sent_messages.append(message)

  async def close(self) -> None:
    self.close_calls += 1


class RecordingClientInitializer:
  client: "WebSocketStreamingClient"

  def __init__(self, client: "WebSocketStreamingClient") -> None:
    self.client = client
    self.calls: list[tuple[ServerConnection, TranscriptionSetupMessage]] = []

  async def __call__(
    self, websocket: ServerConnection, message: TranscriptionSetupMessage
  ) -> "WebSocketStreamingClient":
    self.calls.append((websocket, message))
    return self.client


class FakeSubscriberManager:
  result: tuple[bool, str | None]

  def __init__(self, result: tuple[bool, str | None] = (True, None)) -> None:
    self.result = result
    self.calls: list[tuple[ServerConnection, list[str]]] = []

  async def subscribe_client(
    self, websocket: ServerConnection, stream_names: list[str]
  ) -> tuple[bool, str | None]:
    self.calls.append((websocket, stream_names))
    return self.result


class PublicErrorHandler(WebSocketConnectionHandler):
  async def send_error_and_close(self, websocket: ServerConnection, error_message: str) -> None:
    await self._send_error_and_close(websocket, error_message)


def as_server_connection(websocket: FakeWebSocket) -> ServerConnection:
  return cast(ServerConnection, cast(object, websocket))


class TestWebSocketConnectionHandler:
  @pytest.fixture
  def client_initializer(self) -> RecordingClientInitializer:
    return RecordingClientInitializer(client=cast("WebSocketStreamingClient", object()))

  @pytest.fixture
  def subscriber_manager(self) -> FakeSubscriberManager:
    return FakeSubscriberManager()

  @pytest.fixture
  def handler(self, client_initializer: RecordingClientInitializer) -> WebSocketConnectionHandler:
    return WebSocketConnectionHandler(
      client_initializer=cast(ClientInitializer, client_initializer),
      subscriber_manager=None,
    )

  def test_initial_state(self, handler: WebSocketConnectionHandler) -> None:
    assert handler.subscriber_manager is None

  def test_subscriber_manager_setter(
    self, handler: WebSocketConnectionHandler, subscriber_manager: FakeSubscriberManager
  ) -> None:
    assert handler.subscriber_manager is None

    handler.subscriber_manager = cast("RTSPSubscriberManager", cast(object, subscriber_manager))

    assert handler.subscriber_manager is subscriber_manager


class TestHandleTranscriberConnection:
  @pytest.fixture
  def websocket(self) -> FakeWebSocket:
    return FakeWebSocket(
      headers={WebSocketHeaders.CLIENT_TYPE: ClientType.TRANSCRIBER},
      id="test-id-123",
      recv_messages=['{"type": "transcription_setup", "stream": "test"}'],
    )

  @pytest.fixture
  def client_initializer(self) -> RecordingClientInitializer:
    return RecordingClientInitializer(client=cast("WebSocketStreamingClient", object()))

  @pytest.fixture
  def handler(self, client_initializer: RecordingClientInitializer) -> WebSocketConnectionHandler:
    return WebSocketConnectionHandler(
      client_initializer=cast(ClientInitializer, client_initializer),
      subscriber_manager=None,
    )

  async def test_handle_transcriber_connection(
    self,
    handler: WebSocketConnectionHandler,
    websocket: FakeWebSocket,
    client_initializer: RecordingClientInitializer,
    monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    mock_message = TranscriptionSetupMessage(stream="test", options=UserTranscriptionOptions())

    def deserialize_transcriber_message(raw_message: str) -> TranscriptionSetupMessage:
      del raw_message
      return mock_message

    monkeypatch.setattr(
      "eavesdrop.server.connection_handler.deserialize_message",
      deserialize_transcriber_message,
    )

    result = await handler.handle_connection(as_server_connection(websocket))

    assert isinstance(result, TranscriberConnection)
    assert result.client is client_initializer.client
    assert client_initializer.calls == [(as_server_connection(websocket), mock_message)]

  async def test_handle_connection_closed_during_handshake(
    self,
    handler: WebSocketConnectionHandler,
    websocket: FakeWebSocket,
  ) -> None:
    websocket.recv_exception = ConnectionClosed(None, None)

    result = await handler.handle_connection(as_server_connection(websocket))

    assert result is None


class TestHandleHealthCheck:
  @pytest.fixture
  def websocket(self) -> FakeWebSocket:
    return FakeWebSocket(
      headers={WebSocketHeaders.CLIENT_TYPE: ClientType.HEALTH_CHECK},
      id="health-check-id",
      recv_messages=['{"type": "health_check_request"}'],
    )

  @pytest.fixture
  def handler(self) -> WebSocketConnectionHandler:
    return WebSocketConnectionHandler(
      client_initializer=cast(
        ClientInitializer,
        RecordingClientInitializer(client=cast("WebSocketStreamingClient", object())),
      ),
      subscriber_manager=None,
    )

  async def test_handle_health_check(
    self,
    handler: WebSocketConnectionHandler,
    websocket: FakeWebSocket,
    monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    def deserialize_health_check(raw_message: str) -> HealthCheckRequest:
      del raw_message
      return HealthCheckRequest()

    monkeypatch.setattr(
      "eavesdrop.server.connection_handler.deserialize_message",
      deserialize_health_check,
    )

    result = await handler.handle_connection(as_server_connection(websocket))

    assert result is None
    assert websocket.close_calls == 1


class TestHandleSubscriberConnection:
  @pytest.fixture
  def websocket(self) -> FakeWebSocket:
    return FakeWebSocket(
      headers={
        WebSocketHeaders.CLIENT_TYPE: ClientType.RTSP_SUBSCRIBER,
        WebSocketHeaders.STREAM_NAMES: "office,lobby",
      },
      id="subscriber-id",
    )

  @pytest.fixture
  def client_initializer(self) -> RecordingClientInitializer:
    return RecordingClientInitializer(client=cast("WebSocketStreamingClient", object()))

  @pytest.fixture
  def subscriber_manager(self) -> FakeSubscriberManager:
    return FakeSubscriberManager()

  async def test_handle_subscriber_without_manager(
    self,
    websocket: FakeWebSocket,
    client_initializer: RecordingClientInitializer,
  ) -> None:
    handler = WebSocketConnectionHandler(
      client_initializer=cast(ClientInitializer, client_initializer),
      subscriber_manager=None,
    )

    result = await handler.handle_connection(as_server_connection(websocket))

    assert result is None
    assert len(websocket.sent_messages) == 1
    assert websocket.close_calls == 1

  async def test_handle_subscriber_with_manager(
    self,
    websocket: FakeWebSocket,
    client_initializer: RecordingClientInitializer,
    subscriber_manager: FakeSubscriberManager,
  ) -> None:
    handler = WebSocketConnectionHandler(
      client_initializer=cast(ClientInitializer, client_initializer),
      subscriber_manager=cast("RTSPSubscriberManager", cast(object, subscriber_manager)),
    )

    result = await handler.handle_connection(as_server_connection(websocket))

    assert isinstance(result, SubscriberConnection)
    assert subscriber_manager.calls == [(as_server_connection(websocket), ["office", "lobby"])]

  async def test_handle_subscriber_missing_stream_names(
    self,
    client_initializer: RecordingClientInitializer,
    subscriber_manager: FakeSubscriberManager,
  ) -> None:
    websocket = FakeWebSocket(headers={WebSocketHeaders.CLIENT_TYPE: ClientType.RTSP_SUBSCRIBER})
    handler = WebSocketConnectionHandler(
      client_initializer=cast(ClientInitializer, client_initializer),
      subscriber_manager=cast("RTSPSubscriberManager", cast(object, subscriber_manager)),
    )

    result = await handler.handle_connection(as_server_connection(websocket))

    assert result is None
    assert len(websocket.sent_messages) == 1
    assert websocket.close_calls == 1

  async def test_handle_subscriber_subscription_failure(
    self,
    websocket: FakeWebSocket,
    client_initializer: RecordingClientInitializer,
  ) -> None:
    subscriber_manager = FakeSubscriberManager(result=(False, "Stream not found"))
    handler = WebSocketConnectionHandler(
      client_initializer=cast(ClientInitializer, client_initializer),
      subscriber_manager=cast("RTSPSubscriberManager", cast(object, subscriber_manager)),
    )

    result = await handler.handle_connection(as_server_connection(websocket))

    assert result is None
    assert len(websocket.sent_messages) == 1
    assert websocket.close_calls == 1


class TestSendErrorAndClose:
  @pytest.fixture
  def handler(self) -> PublicErrorHandler:
    return PublicErrorHandler(
      client_initializer=cast(
        ClientInitializer,
        RecordingClientInitializer(client=cast("WebSocketStreamingClient", object())),
      ),
      subscriber_manager=None,
    )

  async def test_send_error_and_close_sends_message(
    self,
    handler: PublicErrorHandler,
  ) -> None:
    websocket = FakeWebSocket()

    await handler.send_error_and_close(as_server_connection(websocket), "Test error message")

    assert len(websocket.sent_messages) == 1
    assert websocket.close_calls == 1

  async def test_send_and_close_handles_exception(
    self,
    handler: PublicErrorHandler,
  ) -> None:
    websocket = FakeWebSocket(send_exception=Exception("Connection lost"))

    await handler.send_error_and_close(as_server_connection(websocket), "Test error")
