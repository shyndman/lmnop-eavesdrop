"""
WebSocket connection routing and client type dispatch.

Handles the initial handshake phase for all WebSocket connections,
delegating to appropriate handlers based on client type headers.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from structlog.stdlib import BoundLogger
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from eavesdrop.common import get_logger
from eavesdrop.wire import (
  ClientType,
  ErrorMessage,
  FlushControlMessage,
  RecordingStartedMessage,
  TranscriptionSetupMessage,
  UtteranceCancelledMessage,
  WebSocketHeaders,
  deserialize_message,
  serialize_message,
)
from eavesdrop.wire.codec import Message
from eavesdrop.wire.messages import HealthCheckRequest

if TYPE_CHECKING:
  from eavesdrop.server.streaming import WebSocketStreamingClient


@dataclass
class TranscriberConnection:
  """Result of a successful transcriber connection."""

  client: "WebSocketStreamingClient"


ConnectionResult = TranscriberConnection | None

ClientInitializer = Callable[
  [ServerConnection, TranscriptionSetupMessage],
  Awaitable["WebSocketStreamingClient"],
]

PRE_SETUP_FLUSH_REJECTION = (
  "Flush rejected: control_flush is unavailable before live transcriber setup completes"
)
PRE_SETUP_RECORDING_START_REJECTION = (
  "Recording start rejected: control_recording_started is unavailable before "
  "live transcriber setup completes"
)
PRE_SETUP_UTTERANCE_CANCEL_REJECTION = (
  "Utterance cancel rejected: control_utterance_cancelled is unavailable before "
  "live transcriber setup completes"
)


class WebSocketConnectionHandler:
  """
  Routes WebSocket connections based on client type header.

  Handles the initial handshake phase for all WebSocket connections,
  delegating to appropriate handlers based on the X-Client-Type header.

  :param client_initializer: Callback to initialize transcriber clients.
  :type client_initializer: ClientInitializer
  """

  def __init__(
    self,
    client_initializer: ClientInitializer,
  ) -> None:
    self._client_initializer: ClientInitializer = client_initializer
    self._logger: BoundLogger = get_logger("svr/conn")

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
      headers = self._request_headers(websocket)
      client_type = headers.get(WebSocketHeaders.CLIENT_TYPE, ClientType.TRANSCRIBER)

      while True:
        raw_msg: str = await websocket.recv(decode=True)
        message = deserialize_message(raw_msg)

        if client_type == ClientType.TRANSCRIBER:
          if isinstance(message, TranscriptionSetupMessage):
            client = await self._handle_transcriber_connection(websocket, message)
            if not client:
              return None
            return TranscriberConnection(client=client)

          if isinstance(message, FlushControlMessage):
            await self._send_error(websocket, PRE_SETUP_FLUSH_REJECTION)
            continue

          if isinstance(message, RecordingStartedMessage):
            await self._send_error(websocket, PRE_SETUP_RECORDING_START_REJECTION)
            continue

          if isinstance(message, UtteranceCancelledMessage):
            await self._send_error(websocket, PRE_SETUP_UTTERANCE_CANCEL_REJECTION)
            continue

          return None

        if client_type == ClientType.HEALTH_CHECK and isinstance(message, HealthCheckRequest):
          await self._handle_health_check(websocket, message)
          return None

        return None

    except ValueError as error:
      await self._send_error_and_close(websocket, f"Setup rejected: {error}")
      return None

    except ConnectionClosed:
      self._logger.info("handle_connection: Connection closed by client")
      return None

    except (KeyboardInterrupt, SystemExit):
      raise

  async def _handle_health_check(
    self, websocket: ServerConnection, _message: HealthCheckRequest
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

  def _request_headers(self, websocket: ServerConnection) -> dict[str, str]:
    """Extract request headers as a plain string map."""
    if websocket.request is None:
      return {}

    return {key: value for key, value in websocket.request.headers.raw_items()}

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
    try:
      return await self._client_initializer(websocket, message)
    except ValueError as err:
      await self._send_error_and_close(websocket, str(err))
      return None

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

  async def _send_error(self, websocket: ServerConnection, error_message: str) -> None:
    """Send an error message without closing the websocket."""
    try:
      await websocket.send(serialize_message(ErrorMessage(message=error_message)))
    except Exception:
      self._logger.exception("Error sending message")

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
