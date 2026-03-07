"""
WebSocket connection routing and client type dispatch.

Handles the initial handshake phase for all WebSocket connections,
delegating to appropriate handlers based on client type headers.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from eavesdrop.common import get_logger
from eavesdrop.wire import (
  ClientType,
  ErrorMessage,
  TranscriptionSetupMessage,
  TranscriptionSourceMode,
  WebSocketHeaders,
  deserialize_message,
  serialize_message,
)
from eavesdrop.wire.codec import Message
from eavesdrop.wire.messages import HealthCheckRequest

if TYPE_CHECKING:
  from eavesdrop.server.rtsp.subscriber import RTSPSubscriberManager
  from eavesdrop.server.streaming import WebSocketStreamingClient


@dataclass
class TranscriberConnection:
  """Result of a successful transcriber connection."""

  client: "WebSocketStreamingClient"
  source_mode: TranscriptionSourceMode


@dataclass
class SubscriberConnection:
  """Result of a successful subscriber connection."""

  pass


ConnectionResult = TranscriberConnection | SubscriberConnection | None

ClientInitializer = Callable[
  [ServerConnection, TranscriptionSetupMessage],
  Awaitable["WebSocketStreamingClient"],
]


class WebSocketConnectionHandler:
  """
  Routes WebSocket connections based on client type header.

  Handles the initial handshake phase for all WebSocket connections,
  delegating to appropriate handlers based on the X-Client-Type header.

  :param client_initializer: Callback to initialize transcriber clients.
  :type client_initializer: ClientInitializer
  :param subscriber_manager: Manager for RTSP subscriber connections (optional).
  :type subscriber_manager: RTSPSubscriberManager | None
  """

  def __init__(
    self,
    client_initializer: ClientInitializer,
    subscriber_manager: "RTSPSubscriberManager | None" = None,
  ) -> None:
    self._client_initializer = client_initializer
    self._subscriber_manager = subscriber_manager
    self._logger = get_logger("svr/conn")

  @property
  def subscriber_manager(self) -> "RTSPSubscriberManager | None":
    """Get the subscriber manager for RTSP connections."""
    return self._subscriber_manager

  @subscriber_manager.setter
  def subscriber_manager(self, value: "RTSPSubscriberManager | None") -> None:
    """Set the subscriber manager for RTSP connections."""
    self._subscriber_manager = value

  async def handle_connection(self, websocket: ServerConnection) -> ConnectionResult:
    """
    Route a new WebSocket connection to the appropriate handler.

    :param websocket: The incoming WebSocket connection.
    :type websocket: ServerConnection
    :returns: Connection result indicating type, or None on failure.
    :rtype: ConnectionResult
    """
    try:
      self._logger.info("handle_connection: New client connected")

      # Check WebSocket headers to determine client type
      headers = websocket.request.headers if websocket.request else {}
      client_type = headers.get(WebSocketHeaders.CLIENT_TYPE, ClientType.TRANSCRIBER)

      if client_type == ClientType.RTSP_SUBSCRIBER:
        result = await self._handle_subscriber_connection(websocket, dict(headers))
        return SubscriberConnection() if result else None

      raw_msg: str = await websocket.recv(decode=True)
      message = deserialize_message(raw_msg)

      match (client_type, message):
        case (ClientType.TRANSCRIBER, TranscriptionSetupMessage()):
          client = await self._handle_transcriber_connection(websocket, message)
          if not client:
            return None
          return TranscriberConnection(client=client, source_mode=message.options.source_mode)

        case (ClientType.HEALTH_CHECK, HealthCheckRequest()):
          await self._handle_health_check(websocket, message)
          return None

    except ConnectionClosed:
      self._logger.info("handle_connection: Connection closed by client")
      return None

    except (KeyboardInterrupt, SystemExit):
      raise

    return None

  async def _handle_health_check(
    self, websocket: ServerConnection, message: HealthCheckRequest
  ) -> None:
    """
    Handle health check requests by logging and closing the connection.

    :param websocket: The WebSocket connection.
    :type websocket: ServerConnection
    :param message: The health check request message.
    :type message: HealthCheckRequest
    """
    self._logger.info("Health check successful", websocket_id=websocket.id)
    await websocket.close()

  async def _handle_transcriber_connection(
    self, websocket: ServerConnection, message: TranscriptionSetupMessage
  ) -> "WebSocketStreamingClient | None":
    """
    Handle traditional transcriber client connections.

    :param websocket: The WebSocket connection.
    :type websocket: ServerConnection
    :param message: The transcription setup message.
    :type message: TranscriptionSetupMessage
    :returns: Initialized streaming client, or None on failure.
    :rtype: WebSocketStreamingClient | None
    """
    return await self._client_initializer(websocket, message)

  async def _handle_subscriber_connection(
    self, websocket: ServerConnection, headers: dict[str, str]
  ) -> bool:
    """
    Handle RTSP subscriber client connections.

    :param websocket: The WebSocket connection.
    :type websocket: ServerConnection
    :param headers: The WebSocket request headers.
    :type headers: dict[str, str]
    :returns: True if subscription succeeded, False otherwise.
    :rtype: bool
    """
    if not self._subscriber_manager:
      error_msg = "RTSP subscription not available: no RTSP streams configured"
      await self._send_error_and_close(websocket, error_msg)
      return False

    # Parse stream names from header
    stream_names_header = headers.get(WebSocketHeaders.STREAM_NAMES, "")
    if not stream_names_header.strip():
      error_msg = f"{WebSocketHeaders.STREAM_NAMES} header is required for RTSP subscribers"
      await self._send_error_and_close(websocket, error_msg)
      return False

    stream_names = [name.strip() for name in stream_names_header.split(",") if name.strip()]
    if not stream_names:
      error_msg = f"No valid stream names provided in {WebSocketHeaders.STREAM_NAMES} header"
      await self._send_error_and_close(websocket, error_msg)
      return False

    # Subscribe the client
    success, error_message = await self._subscriber_manager.subscribe_client(
      websocket, stream_names
    )

    if not success:
      await self._send_error_and_close(websocket, error_message or "Subscription failed")
      return False

    return True

  async def _send_error_and_close(self, websocket: ServerConnection, error_message: str) -> None:
    """
    Send error message and close WebSocket connection.

    :param websocket: The WebSocket connection.
    :type websocket: ServerConnection
    :param error_message: The error message to send.
    :type error_message: str
    """
    await self._send_and_close(websocket, ErrorMessage(message=error_message))
    self._logger.warning("Sent error and closed connection", error=error_message)

  async def _send_and_close(self, websocket: ServerConnection, message: Message) -> None:
    """
    Send a message and close the WebSocket connection.

    :param websocket: The WebSocket connection.
    :type websocket: ServerConnection
    :param message: The message to send.
    :type message: Message
    """
    try:
      await websocket.send(serialize_message(message))
      await websocket.close()
    except Exception:
      self._logger.exception("Error sending message and closing connection")
