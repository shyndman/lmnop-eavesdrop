"""
RTSP subscriber management for WebSocket clients.

Handles WebSocket clients that subscribe to transcription results from named RTSP streams
rather than sending audio for transcription.
"""

from typing import TYPE_CHECKING

from websockets.asyncio.server import ServerConnection

from eavesdrop.server.logs import get_logger
from eavesdrop.wire import (
  ErrorMessage,
  StreamStatusMessage,
  TranscriptionMessage,
  serialize_message,
)
from eavesdrop.wire.codec import Message

if TYPE_CHECKING:
  from eavesdrop.server.rtsp.cache import RTSPTranscriptionCache


class RTSPSubscriberManager:
  """
  Manages WebSocket subscribers that receive transcription results from RTSP streams.

  Enforces the single listener policy where each RTSP stream can have at most one
  WebSocket subscriber at a time.
  """

  def __init__(
    self, available_streams: set[str], transcription_cache: "RTSPTranscriptionCache"
  ) -> None:
    """
    Initialize the RTSP subscriber manager.

    :param available_streams: Set of available RTSP stream names
    :param transcription_cache: Cache for storing and retrieving transcription history
    """
    self.available_streams = available_streams
    self.transcription_cache = transcription_cache
    self.stream_subscribers: dict[str, ServerConnection] = {}
    self.subscriber_streams: dict[ServerConnection, set[str]] = {}
    self.logger = get_logger("rtsp/mgr")

  def validate_stream_names(self, requested_streams: list[str]) -> tuple[list[str], list[str]]:
    """
    Validate requested stream names against available streams.

    :param requested_streams: List of stream names requested by client
    :returns: Tuple of (valid_streams, invalid_streams)
    """
    valid_streams = [s for s in requested_streams if s in self.available_streams]
    invalid_streams = [s for s in requested_streams if s not in self.available_streams]
    return valid_streams, invalid_streams

  async def subscribe_client(
    self, websocket: ServerConnection, stream_names: list[str]
  ) -> tuple[bool, str | None]:
    """
    Subscribe a WebSocket client to the specified RTSP streams.

    Implements the single listener policy by disconnecting any existing subscribers
    for the requested streams.

    :param websocket: WebSocket connection for the subscriber
    :param stream_names: List of stream names to subscribe to
    :returns: Tuple of (success, error_message)
    """
    # Validate stream names
    valid_streams, invalid_streams = self.validate_stream_names(stream_names)
    if invalid_streams:
      available_list = ", ".join(sorted(self.available_streams))
      return (
        False,
        f"Unknown streams: {', '.join(invalid_streams)}. Available streams: {available_list}",
      )
    if not valid_streams:
      return False, "No valid streams specified"

    self.logger.info(
      "Subscribing client to RTSP streams", client=id(websocket), streams=valid_streams
    )

    # Disconnect any existing subscribers for these streams (single listener policy)
    disconnected_clients = set()
    for stream_name in valid_streams:
      if existing_websocket := self.stream_subscribers.get(stream_name, None):
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

    # Update cache and send history
    await self._update_cache_listener_presence(valid_streams, True)
    history_count = await self._send_buffered_history(websocket, valid_streams)

    self.logger.info(
      "Client subscribed successfully",
      client=id(websocket),
      streams=valid_streams,
      disconnected_previous=len(disconnected_clients),
      history_messages_sent=history_count,
    )

    return True, None

  async def unsubscribe_client(self, websocket: ServerConnection) -> None:
    """
    Unsubscribe a WebSocket client from all its streams.

    :param websocket: WebSocket connection to unsubscribe
    """
    if websocket not in self.subscriber_streams:
      return

    stream_names = self.subscriber_streams[websocket]

    self.logger.info(
      "Unsubscribing client from RTSP streams", client=id(websocket), streams=list(stream_names)
    )

    # Update cache listener presence for unsubscribed streams
    await self._update_cache_listener_presence(list(stream_names), False)

    # Remove from stream mappings
    for stream_name in stream_names:
      if self.stream_subscribers.get(stream_name) == websocket:
        del self.stream_subscribers[stream_name]

    # Remove from subscriber mappings
    del self.subscriber_streams[websocket]

  async def send_to_subscriber(self, stream_name: str, message: Message) -> None:
    """
    Send a message to all subscribers of a specific stream.

    :param stream_name: Name of the stream
    :param message: Message to send
    """
    if stream_name not in self.stream_subscribers:
      return

    websocket = self.stream_subscribers[stream_name]
    success = await self._send_message_to_websocket(websocket, message)

    if not success:
      await self.unsubscribe_client(websocket)

  async def send_stream_status(
    self, stream_name: str, status: str, message: str | None = None
  ) -> None:
    """
    Send stream status update to subscribers.

    :param stream_name: Name of the stream
    :param status: Stream status ('online', 'offline', 'error')
    :param message: Optional status message
    """
    await self.send_to_subscriber(
      stream_name,
      StreamStatusMessage(
        stream=stream_name,
        status=status,  # type: ignore
        message=message,
      ),
    )

  async def send_transcription(
    self, stream_name: str, segments: list, language: str | None = None
  ) -> None:
    """
    Send transcription result to subscribers.

    :param stream_name: Name of the stream
    :param segments: List of transcription segments
    :param language: Detected language
    """
    transcription_message = TranscriptionMessage(
      stream=stream_name, segments=segments, language=language
    )

    await self.send_to_subscriber(stream_name, transcription_message)

  async def _disconnect_subscriber(self, websocket: ServerConnection, reason: str) -> None:
    """
    Disconnect a subscriber with an error message.

    :param websocket: WebSocket connection to disconnect
    :param reason: Reason for disconnection
    """
    try:
      await self._send_message_to_websocket(websocket, ErrorMessage(message=reason))
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

  async def _update_cache_listener_presence(
    self, stream_names: list[str], has_listeners: bool
  ) -> None:
    """Update listener presence in cache for the specified streams."""
    for stream_name in stream_names:
      try:
        await self.transcription_cache.update_listener_presence(stream_name, has_listeners)
      except Exception as e:
        self.logger.warning(
          "Failed to update cache listener presence",
          stream=stream_name,
          has_listeners=has_listeners,
          error=str(e),
        )

  async def _send_buffered_history(
    self, websocket: ServerConnection, stream_names: list[str]
  ) -> int:
    """Send buffered transcription history to a subscriber. Returns count of messages sent."""
    total_sent = 0

    for stream_name in stream_names:
      try:
        recent_messages = await self.transcription_cache.get_recent_messages(stream_name)

        for message in recent_messages:
          if await self._send_message_to_websocket(websocket, message):
            total_sent += 1
          else:
            return total_sent

      except Exception as e:
        self.logger.warning(
          "Failed to retrieve history for stream",
          stream=stream_name,
          client=id(websocket),
          error=str(e),
        )

    return total_sent

  async def _send_message_to_websocket(self, websocket: ServerConnection, message: Message) -> bool:
    """
    Send a message to a specific websocket connection.

    :param websocket: WebSocket connection to send to
    :param message: Message to send
    :returns: True if message was sent successfully, False otherwise
    """
    try:
      await websocket.send(serialize_message(message))

      self.logger.debug(
        "Sent message to websocket",
        message_type=message.type,
        client=id(websocket),
      )
      return True

    except Exception as e:
      self.logger.warning(
        "Failed to send message to websocket",
        client=id(websocket),
        error=str(e),
      )
      return False
