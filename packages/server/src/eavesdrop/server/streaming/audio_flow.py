"""
WebSocket implementation of streaming transcription output sinks.
"""

from typing import override

import structlog
from websockets.asyncio.server import ServerConnection
from websockets.protocol import State

from eavesdrop.common import get_logger
from eavesdrop.server.streaming.interfaces import (
  TranscriptionResult,
  TranscriptionSink,
)
from eavesdrop.wire import (
  DisconnectMessage,
  ErrorMessage,
  ServerReadyMessage,
  TranscriptionMessage,
  serialize_message,
)
from eavesdrop.wire.codec import Message


class WebSocketTranscriptionSink(TranscriptionSink):
  """WebSocket implementation of :class:`TranscriptionSink`."""

  def __init__(self, websocket: ServerConnection, stream_name: str) -> None:
    """Initialize WebSocket transcription sink.

    :param websocket: WebSocket connection to send to.
    :type websocket: ServerConnection
    :param stream_name: Unique identifier for the client.
    :type stream_name: str
    """
    self.websocket: ServerConnection = websocket
    self.stream_name: str = stream_name
    self.logger: structlog.stdlib.BoundLogger = get_logger("ws/sink")
    self._closed: bool = False

  @override
  async def send_result(self, result: TranscriptionResult) -> None:
    await self.send_message(
      TranscriptionMessage(
        stream=self.stream_name,
        segments=result.segments,
        language=result.language,
        recording_id=result.recording_id,
        flush_complete=result.flush_complete,
      )
    )

  @override
  async def send_error(self, error: str) -> None:
    await self.send_message(
      ErrorMessage(
        stream=self.stream_name,
        message=error,
      )
    )

  @override
  async def send_server_ready(self, backend: str) -> None:
    await self.send_message(
      ServerReadyMessage(
        stream=self.stream_name,
        backend=backend,
      )
    )

  @override
  async def disconnect(self) -> None:
    """Send disconnect notification and clean up resources."""
    try:
      if self._closed or not self.websocket or self.websocket.state is not State.OPEN:
        return

      try:
        await self.websocket.send(serialize_message(DisconnectMessage(stream=self.stream_name)))
      except Exception:
        pass
    finally:
      self._closed = True
      self.logger.info("WebSocket transcription sink disconnected")

  async def send_message(self, message: Message) -> None:
    """Send a message to the WebSocket client."""
    if self._closed or not self.websocket:
      return

    try:
      await self.websocket.send(serialize_message(message))
    except Exception:
      self.logger.exception("Error sending message to client")
