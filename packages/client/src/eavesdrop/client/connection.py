"""
WebSocket connection and protocol handling for Eavesdrop client.
"""

import asyncio
import time
from collections.abc import Callable

import websockets
from structlog.stdlib import BoundLogger
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed

from eavesdrop.common import get_logger
from eavesdrop.wire import (
  DisconnectMessage,
  ErrorMessage,
  FlushControlMessage,
  RecordingStartedMessage,
  ServerReadyMessage,
  TranscriptionMessage,
  TranscriptionSetupMessage,
  UserTranscriptionOptions,
  UtteranceCancelledMessage,
  deserialize_message,
  serialize_message,
)

DEFAULT_SOCKET_CLOSE_REASON = "websocket connection closed"


class WebSocketConnection:
  """Manages WebSocket connection and protocol communication with server."""

  def __init__(
    self,
    host: str,
    port: int,
    stream_name: str,
    on_error: Callable[[str], None],
    on_transcription: Callable[[TranscriptionMessage], None] | None = None,
    on_disconnect: Callable[[str | None], None] | None = None,
  ) -> None:
    """Initialize one live connection.

    :param host: Server hostname.
    :param port: Server port.
    :param stream_name: Session identifier carried by wire messages.
    :param on_error: Receives server and transport failures.
    :param on_transcription: Receives wire transcript messages without text flattening.
    :param on_disconnect: Receives terminal socket closure once.
    """
    self.host: str = host
    self.port: int = port
    self.stream_name: str = stream_name
    self.on_transcription: Callable[[TranscriptionMessage], None] | None = on_transcription
    self.on_error: Callable[[str], None] = on_error
    self.on_disconnect: Callable[[str | None], None] | None = on_disconnect

    self._logger: BoundLogger = get_logger(
      "client/conn",
      stream=stream_name,
      host=host,
      port=port,
    )

    self.ws: ClientConnection | None = None
    self.connected: bool = False
    self._disconnect_notified: bool = False
    self._send_lock: asyncio.Lock = asyncio.Lock()

    # Latency tracking
    self.first_audio_sent_time: float | None = None
    self.session_end_sent_time: float | None = None
    self.first_response_received: bool = False
    self.session_end_received: bool = False
    self.audio_sending_started: bool = False

  async def connect(self, transcription_options: UserTranscriptionOptions | None = None) -> None:
    """Establish WebSocket connection to server."""
    socket_url = f"ws://{self.host}:{self.port}"

    self._logger.debug("connecting websocket", socket_url=socket_url)
    self.ws = await websockets.connect(socket_url)
    self.connected = True
    self._disconnect_notified = False
    self._logger.debug("websocket connected", socket_url=socket_url)

    options = transcription_options or UserTranscriptionOptions()
    config_message = TranscriptionSetupMessage(
      stream=self.stream_name,
      options=options,
    )
    await self.ws.send(serialize_message(config_message))

  async def disconnect(self) -> None:
    """Close WebSocket connection."""
    if self.ws:
      ws = self.ws
      self.ws = None
      self.connected = False
      self._disconnect_notified = True
      await ws.close()
      self._logger.debug("websocket disconnected")

  async def handle_messages(self) -> None:
    """Handle incoming WebSocket messages from server."""
    if not self.ws:
      return

    try:
      while self.connected:
        await self._process_message(await self.ws.recv(decode=True))
    except ConnectionClosed:
      self.connected = False
      self._notify_disconnect(self.ws.close_reason or DEFAULT_SOCKET_CLOSE_REASON)
    except Exception as e:
      self.connected = False
      self._logger.exception("websocket message handling failed")
      self.on_error(f"Message handling error: {e}")
      self._notify_disconnect(str(e))
    else:
      self.connected = False
      self._notify_disconnect(DEFAULT_SOCKET_CLOSE_REASON)

  async def _process_message(self, message_json: str) -> None:
    """Route wire messages without introducing a parallel event model."""
    message = deserialize_message(message_json)
    match message:
      case ServerReadyMessage(stream=stream, backend=backend) if stream == self.stream_name:
        self._logger.info("transcription server ready", backend=backend)
      case TranscriptionMessage(stream=stream) as transcription if stream == self.stream_name:
        if self.on_transcription is not None:
          self.on_transcription(transcription)
      case ErrorMessage(stream=stream, message=error) if stream in (None, self.stream_name):
        self.on_error(error)
      case DisconnectMessage(stream=stream, message=reason) if stream == self.stream_name:
        self.session_end_received = True
        self.connected = False
        self._notify_disconnect(reason)
      case _:
        self._logger.warning("ignoring unrelated server message", message_type=message.type)

  async def send_audio_data(self, audio_data: bytes) -> None:
    """Send audio data to server."""
    if not self.ws or not self.connected:
      return

    try:
      if not self.audio_sending_started:
        self.audio_sending_started = True
        self.first_audio_sent_time = time.time()

      await self._send_bytes(audio_data)
    except Exception as e:
      self._logger.exception("sending audio failed")
      self.on_error(f"Error sending audio data: {e}")

  async def send_recording_started(self, recording_id: str, sample_rate_hz: int) -> None:
    """Send the live recording boundary control before any audio bytes."""
    if not self.ws or not self.connected:
      return

    try:
      await self._send_text(
        serialize_message(
          RecordingStartedMessage(
            stream=self.stream_name,
            recording_id=recording_id,
            sample_rate_hz=sample_rate_hz,
          )
        )
      )
    except Exception as e:
      self._logger.exception("sending recording boundary failed", recording_id=recording_id)
      self.on_error(f"Error sending recording start control: {e}")
      raise

  async def send_flush_control(self, recording_id: str, *, force_complete: bool = True) -> None:
    """Send a live flush control message over the websocket.

    :param force_complete: Whether the server should force-complete the current tail segment.
    :type force_complete: bool
    """
    if not self.ws or not self.connected:
      return

    try:
      flush_message = FlushControlMessage(
        stream=self.stream_name,
        recording_id=recording_id,
        force_complete=force_complete,
      )
      await self._send_text(serialize_message(flush_message))
    except Exception as e:
      self._logger.exception("sending flush failed", recording_id=recording_id)
      self.on_error(f"Error sending flush control: {e}")
      raise

  async def send_utterance_cancelled(self, recording_id: str) -> None:
    """Send a live utterance-cancel control message over the websocket."""
    if not self.ws or not self.connected:
      return

    try:
      cancel_message = UtteranceCancelledMessage(
        stream=self.stream_name,
        recording_id=recording_id,
      )
      await self._send_text(serialize_message(cancel_message))
    except Exception as e:
      self._logger.exception("sending cancellation failed", recording_id=recording_id)
      self.on_error(f"Error sending utterance cancel control: {e}")
      raise

  async def _send_text(self, payload: str) -> None:
    if not self.ws:
      return

    async with self._send_lock:
      await self.ws.send(payload)

  async def _send_bytes(self, payload: bytes) -> None:
    if not self.ws:
      return

    async with self._send_lock:
      await self.ws.send(payload)

  def reset_session_tracking(self) -> None:
    """Reset session tracking variables."""
    self.first_response_received = False
    self.session_end_received = False
    self.audio_sending_started = False
    self.first_audio_sent_time = None
    self.session_end_sent_time = None

  def is_connected(self) -> bool:
    """Check if WebSocket is connected."""
    return self.connected and self.ws is not None

  def _notify_disconnect(self, reason: str | None) -> None:
    """Notify the client exactly once about a terminal websocket disconnect.

    :param reason: Disconnect reason from the server or transport layer.
    :type reason: str | None
    """
    if self._disconnect_notified:
      return

    self._disconnect_notified = True
    if self.on_disconnect:
      self.on_disconnect(reason)
