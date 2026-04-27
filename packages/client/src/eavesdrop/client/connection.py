"""
WebSocket connection and protocol handling for Eavesdrop client.
"""

import time
from collections.abc import Callable

import websockets
from structlog.stdlib import BoundLogger
from websockets.asyncio.client import ClientConnection

from eavesdrop.common import get_logger
from eavesdrop.wire import (
  ClientType,
  DisconnectMessage,
  ErrorMessage,
  FlushControlMessage,
  ServerReadyMessage,
  TranscriptionMessage,
  TranscriptionSetupMessage,
  UserTranscriptionOptions,
  UtteranceCancelledMessage,
  WebSocketHeaders,
  deserialize_message,
  serialize_message,
)

FILE_UPLOAD_CHUNK_BYTES = 64 * 1024
DEFAULT_SOCKET_CLOSE_REASON = "websocket connection closed"


class WebSocketConnection:
  """Manages WebSocket connection and protocol communication with server."""

  def __init__(
    self,
    host: str,
    port: int,
    stream_name: str,
    on_ready: Callable[[str], None],
    on_transcription: Callable[[str], None],
    on_error: Callable[[str], None],
    client_type: ClientType = ClientType.TRANSCRIBER,
    stream_names: list[str] | None = None,
    on_transcription_message: Callable[[TranscriptionMessage], None] | None = None,
    on_disconnect: Callable[[str | None], None] | None = None,
  ):
    self.host: str = host
    self.port: int = port
    self.stream_name: str = stream_name
    self.on_ready: Callable[[str], None] = on_ready
    self.on_transcription: Callable[[str], None] = on_transcription
    self.on_error: Callable[[str], None] = on_error
    self.client_type: ClientType = client_type
    self.stream_names: list[str] = stream_names or []
    self.on_transcription_message: Callable[[TranscriptionMessage], None] | None = (
      on_transcription_message
    )
    self.on_disconnect: Callable[[str | None], None] | None = on_disconnect

    self._logger: BoundLogger = get_logger(
      "client/conn",
      stream=stream_name,
      client_type=client_type.value,
      host=host,
      port=port,
    )

    self.ws: ClientConnection | None = None
    self.connected: bool = False
    self._disconnect_notified: bool = False

    # Latency tracking
    self.first_audio_sent_time: float | None = None
    self.session_end_sent_time: float | None = None
    self.first_response_received: bool = False
    self.session_end_received: bool = False
    self.audio_sending_started: bool = False

  async def connect(self, transcription_options: UserTranscriptionOptions | None = None):
    """Establish WebSocket connection to server."""
    socket_url = f"ws://{self.host}:{self.port}"
    headers = {WebSocketHeaders.CLIENT_TYPE.value: self.client_type.value}

    # Add stream names for subscriber mode
    if self.client_type == ClientType.RTSP_SUBSCRIBER:
      headers[WebSocketHeaders.STREAM_NAMES.value] = ",".join(self.stream_names)

    self._logger.debug("connecting websocket", socket_url=socket_url)
    self.ws = await websockets.connect(socket_url, additional_headers=headers)
    self.connected = True
    self._disconnect_notified = False
    self._logger.debug("websocket connected", socket_url=socket_url)

    # Send client configuration for transcriber mode
    if self.client_type == ClientType.TRANSCRIBER:
      options = transcription_options or UserTranscriptionOptions()

      config_message = TranscriptionSetupMessage(
        stream=self.stream_name,
        options=options,
      )

      await self.ws.send(serialize_message(config_message))

  async def disconnect(self):
    """Close WebSocket connection."""
    if self.ws:
      ws = self.ws
      self.ws = None
      self.connected = False
      self._disconnect_notified = True
      await ws.close()
      self._logger.debug("websocket disconnected")

  async def handle_messages(self):
    """Handle incoming WebSocket messages from server."""
    if not self.ws:
      return

    try:
      async for m in self.ws:
        # Convert bytes to string if necessary
        if isinstance(m, str):
          message = m
        else:
          message = m.decode("utf-8")

        await self._process_message(message)
    except Exception as e:
      self.connected = False
      self.on_error(f"Message handling error: {e}")
      self._notify_disconnect(str(e))
    else:
      self.connected = False
      self._notify_disconnect(DEFAULT_SOCKET_CLOSE_REASON)

  # TODO This is UG-LY. Needs a refactor.
  async def _process_message(self, message_json: str):
    """Process individual WebSocket message."""
    try:
      message = deserialize_message(message_json)

      match message:
        case ServerReadyMessage() as ready_msg:
          if ready_msg.stream == self.stream_name:
            self.on_ready(ready_msg.backend)

        case TranscriptionMessage() as transcription:
          # Handle transcription messages differently for different client types
          if self.client_type == ClientType.TRANSCRIBER:
            # For transcriber mode, check stream name match
            if transcription.stream == self.stream_name and transcription.segments:
              # Call both callbacks for backward compatibility
              if self.on_transcription_message:
                self.on_transcription_message(transcription)

              # Legacy text callback
              text_parts = [seg.text.strip() for seg in transcription.segments if seg.text.strip()]
              if text_parts:
                text = " ".join(text_parts)
                self.on_transcription(text)

          elif self.client_type == ClientType.RTSP_SUBSCRIBER:
            # For subscriber mode, accept any stream we're subscribed to
            if transcription.stream in self.stream_names:
              if self.on_transcription_message:
                self.on_transcription_message(transcription)

              # Legacy text callback (include stream name)
              text_parts = [seg.text.strip() for seg in transcription.segments if seg.text.strip()]
              if text_parts:
                text = " ".join(text_parts)
                self.on_transcription(f"[{transcription.stream}] {text}")

        case ErrorMessage() as error:
          if error.stream == self.stream_name or error.stream is None:
            self.on_error(error.message)

        case DisconnectMessage() as disconnect:
          if disconnect.stream == self.stream_name:
            self.session_end_received = True
            self.connected = False
            self._notify_disconnect(disconnect.message)

        case _:
          # Handle unexpected message types
          self.on_error(f"Received unexpected message: {type(message)}")

    except Exception:
      raise

  async def send_audio_data(self, audio_data: bytes):
    """Send audio data to server."""
    if not self.ws or not self.connected:
      return

    try:
      if not self.audio_sending_started:
        self.audio_sending_started = True
        self.first_audio_sent_time = time.time()

      await self.ws.send(audio_data)
    except Exception as e:
      self.on_error(f"Error sending audio data: {e}")

  async def send_end_of_audio(self):
    """Send end of audio signal to server."""
    if not self.ws or not self.connected:
      return

    try:
      await self.ws.send(b"END_OF_AUDIO")
    except Exception as e:
      self.on_error(f"Error sending END_OF_AUDIO: {e}")

  async def send_flush_control(self, *, force_complete: bool = True) -> None:
    """Send a live flush control message over the websocket.

    :param force_complete: Whether the server should force-complete the current tail segment.
    :type force_complete: bool
    """
    if not self.ws or not self.connected:
      return

    try:
      flush_message = FlushControlMessage(
        stream=self.stream_name,
        force_complete=force_complete,
      )
      await self.ws.send(serialize_message(flush_message))
    except Exception as e:
      self.on_error(f"Error sending flush control: {e}")
      raise

  async def send_utterance_cancelled(self) -> None:
    """Send a live utterance-cancel control message over the websocket."""
    if not self.ws or not self.connected:
      return

    try:
      cancel_message = UtteranceCancelledMessage(stream=self.stream_name)
      await self.ws.send(serialize_message(cancel_message))
    except Exception as e:
      self.on_error(f"Error sending utterance cancel control: {e}")
      raise

  async def send_file_bytes(
    self,
    file_bytes: bytes,
    chunk_size: int = FILE_UPLOAD_CHUNK_BYTES,
  ) -> None:
    """Upload file payload bytes over websocket in deterministic chunk sizes.

    :param file_bytes: Raw bytes from the local transcription input file.
    :type file_bytes: bytes
    :param chunk_size: Chunk size used for websocket sends.
    :type chunk_size: int
    """
    if not self.ws or not self.connected:
      return

    if chunk_size <= 0:
      raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    try:
      for offset in range(0, len(file_bytes), chunk_size):
        chunk = file_bytes[offset : offset + chunk_size]
        await self.ws.send(chunk)
    except Exception as e:
      self.on_error(f"Error sending file bytes: {e}")
      raise

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
