"""
RTSP subscriber management for WebSocket clients.

Handles WebSocket clients that subscribe to transcription results from named RTSP streams
rather than sending audio for transcription.
"""

from websockets.asyncio.server import ServerConnection

from .logs import get_logger
from .messages import ErrorMessage, OutboundMessage, StreamStatusMessage, TranscriptionMessage


class RTSPSubscriberManager:
  """
  Manages WebSocket subscribers that receive transcription results from RTSP streams.

  Enforces the single listener policy where each RTSP stream can have at most one
  WebSocket subscriber at a time.
  """

  def __init__(self, available_streams: set[str]) -> None:
    """
    Initialize the RTSP subscriber manager.

    Args:
        available_streams: Set of available RTSP stream names
    """
    self.available_streams = available_streams
    self.stream_subscribers: dict[str, ServerConnection] = {}
    self.subscriber_streams: dict[ServerConnection, set[str]] = {}
    self.logger = get_logger("rtsp_subscriber_manager")

  def validate_stream_names(self, requested_streams: list[str]) -> tuple[list[str], list[str]]:
    """
    Validate requested stream names against available streams.

    Args:
        requested_streams: List of stream names requested by client

    Returns:
        Tuple of (valid_streams, invalid_streams)
    """
    valid_streams = []
    invalid_streams = []

    for stream_name in requested_streams:
      if stream_name in self.available_streams:
        valid_streams.append(stream_name)
      else:
        invalid_streams.append(stream_name)

    return valid_streams, invalid_streams

  async def subscribe_client(
    self, websocket: ServerConnection, stream_names: list[str]
  ) -> tuple[bool, str | None]:
    """
    Subscribe a WebSocket client to the specified RTSP streams.

    Implements the single listener policy by disconnecting any existing subscribers
    for the requested streams.

    Args:
        websocket: WebSocket connection for the subscriber
        stream_names: List of stream names to subscribe to

    Returns:
        Tuple of (success, error_message)
    """
    # Validate stream names
    valid_streams, invalid_streams = self.validate_stream_names(stream_names)

    if invalid_streams:
      available_list = ", ".join(sorted(self.available_streams))
      error_msg = (
        f"Unknown streams: {', '.join(invalid_streams)}. Available streams: {available_list}"
      )
      return False, error_msg

    if not valid_streams:
      return False, "No valid streams specified"

    self.logger.info(
      "Subscribing client to RTSP streams", client=id(websocket), streams=valid_streams
    )

    # Disconnect any existing subscribers for these streams (single listener policy)
    disconnected_clients = set()
    for stream_name in valid_streams:
      if stream_name in self.stream_subscribers:
        existing_websocket = self.stream_subscribers[stream_name]
        if existing_websocket != websocket:
          disconnected_clients.add(existing_websocket)

    # Disconnect previous subscribers
    for old_websocket in disconnected_clients:
      await self._disconnect_subscriber(
        old_websocket, "Disconnected: another client subscribed to your stream(s)"
      )

    # Subscribe the new client
    self.subscriber_streams[websocket] = set(valid_streams)
    for stream_name in valid_streams:
      self.stream_subscribers[stream_name] = websocket

    self.logger.info(
      "Client subscribed successfully",
      client=id(websocket),
      streams=valid_streams,
      disconnected_previous=len(disconnected_clients),
    )

    return True, None

  async def unsubscribe_client(self, websocket: ServerConnection) -> None:
    """
    Unsubscribe a WebSocket client from all its streams.

    Args:
        websocket: WebSocket connection to unsubscribe
    """
    if websocket not in self.subscriber_streams:
      return

    stream_names = self.subscriber_streams[websocket]

    self.logger.info(
      "Unsubscribing client from RTSP streams", client=id(websocket), streams=list(stream_names)
    )

    # Remove from stream mappings
    for stream_name in stream_names:
      if self.stream_subscribers.get(stream_name) == websocket:
        del self.stream_subscribers[stream_name]

    # Remove from subscriber mappings
    del self.subscriber_streams[websocket]

  async def send_to_subscribers(self, stream_name: str, message: OutboundMessage) -> None:
    """
    Send a message to all subscribers of a specific stream.

    Args:
        stream_name: Name of the stream
        message: Message to send
    """
    if stream_name not in self.stream_subscribers:
      return

    websocket = self.stream_subscribers[stream_name]

    try:
      message_json = message.model_dump_json()
      await websocket.send(message_json)

      self.logger.debug(
        "Sent message to subscriber",
        stream=stream_name,
        message_type=message.type,
        client=id(websocket),
      )

    except Exception as e:
      self.logger.warning(
        "Failed to send message to subscriber",
        stream=stream_name,
        client=id(websocket),
        error=str(e),
      )

      # Remove the failed subscriber
      await self.unsubscribe_client(websocket)

  async def send_stream_status(
    self, stream_name: str, status: str, message: str | None = None
  ) -> None:
    """
    Send stream status update to subscribers.

    Args:
        stream_name: Name of the stream
        status: Stream status ('online', 'offline', 'error')
        message: Optional status message
    """
    status_message = StreamStatusMessage(
      stream=stream_name,
      status=status,  # type: ignore
      message=message,
    )

    await self.send_to_subscribers(stream_name, status_message)

  async def send_transcription(
    self, stream_name: str, segments: list, language: str | None = None
  ) -> None:
    """
    Send transcription result to subscribers.

    Args:
        stream_name: Name of the stream
        segments: List of transcription segments
        language: Detected language
    """
    transcription_message = TranscriptionMessage(
      stream=stream_name, segments=segments, language=language
    )

    await self.send_to_subscribers(stream_name, transcription_message)

  async def _disconnect_subscriber(self, websocket: ServerConnection, reason: str) -> None:
    """
    Disconnect a subscriber with an error message.

    Args:
        websocket: WebSocket connection to disconnect
        reason: Reason for disconnection
    """
    try:
      error_message = ErrorMessage(message=reason)
      message_json = error_message.model_dump_json()
      await websocket.send(message_json)
      await websocket.close()

      self.logger.info("Disconnected subscriber", client=id(websocket), reason=reason)

    except Exception as e:
      self.logger.warning("Error disconnecting subscriber", client=id(websocket), error=str(e))
    finally:
      # Always clean up the subscription
      await self.unsubscribe_client(websocket)

  def get_subscriber_count(self) -> int:
    """Get total number of active subscribers."""
    return len(self.subscriber_streams)

  def get_stream_subscriber_count(self, stream_name: str) -> int:
    """Get number of subscribers for a specific stream (0 or 1)."""
    return 1 if stream_name in self.stream_subscribers else 0

  def get_subscribed_streams(self) -> set[str]:
    """Get set of streams that have active subscribers."""
    return set(self.stream_subscribers.keys())

  def get_subscriber_info(self) -> dict:
    """Get detailed subscriber information for debugging."""
    return {
      "total_subscribers": len(self.subscriber_streams),
      "stream_mappings": {
        stream: id(websocket) for stream, websocket in self.stream_subscribers.items()
      },
      "subscriber_mappings": {
        id(websocket): list(streams) for websocket, streams in self.subscriber_streams.items()
      },
    }
