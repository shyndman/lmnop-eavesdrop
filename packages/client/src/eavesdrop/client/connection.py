"""
WebSocket connection and protocol handling for Eavesdrop client.
"""

import time
from collections.abc import Callable

import websockets
from websockets.asyncio.client import ClientConnection

from eavesdrop.wire import (
  ClientType,
  ErrorMessage,
  ServerReadyMessage,
  TranscriptionMessage,
  TranscriptionSetupMessage,
  UserTranscriptionOptions,
  WebSocketHeaders,
  deserialize_message,
  serialize_message,
)


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
  ):
    self.host = host
    self.port = port
    self.stream_name = stream_name
    self.on_ready = on_ready
    self.on_transcription = on_transcription
    self.on_error = on_error

    self.ws: ClientConnection | None = None
    self.connected = False

    # Latency tracking
    self.first_audio_sent_time: float | None = None
    self.session_end_sent_time: float | None = None
    self.first_response_received = False
    self.session_end_received = False
    self.audio_sending_started = False

  async def connect(self):
    """Establish WebSocket connection to server."""
    socket_url = f"ws://{self.host}:{self.port}"
    headers = {WebSocketHeaders.CLIENT_TYPE.value: ClientType.TRANSCRIBER.value}

    self.ws = await websockets.connect(socket_url, additional_headers=headers)
    self.connected = True

    # Send client configuration
    config_message = TranscriptionSetupMessage(
      stream=self.stream_name,
      options=UserTranscriptionOptions(
        initial_prompt=None,
        hotwords=None,
        beam_size=5,
        word_timestamps=False,
      ),
    )

    await self.ws.send(serialize_message(config_message))

  async def disconnect(self):
    """Close WebSocket connection."""
    if self.ws:
      await self.ws.close()
      self.ws = None
      self.connected = False

  async def handle_messages(self):
    """Handle incoming WebSocket messages from server."""
    if not self.ws:
      return

    try:
      async for message in self.ws:
        # Convert bytes to string if necessary
        message_str = message if isinstance(message, str) else message.decode("utf-8")
        await self._process_message(message_str)
    except Exception as e:
      self.on_error(f"Message handling error: {e}")

  async def _process_message(self, message_json: str):
    """Process individual WebSocket message."""
    try:
      message = deserialize_message(message_json)

      match message:
        case ServerReadyMessage() as ready_msg:
          if ready_msg.stream == self.stream_name:
            self.on_ready(ready_msg.backend)

        case TranscriptionMessage() as transcription:
          if transcription.stream == self.stream_name and transcription.segments:
            # Get the text from segments
            text_parts = [seg.text.strip() for seg in transcription.segments if seg.text.strip()]
            if text_parts:
              text = " ".join(text_parts)
              self.on_transcription(text)

        case ErrorMessage() as error:
          if error.stream == self.stream_name or error.stream is None:
            self.on_error(error.message)

        case _:
          # Handle unexpected message types
          self.on_error(f"Received unexpected message: {type(message)}")

    except Exception as e:
      self.on_error(f"Error processing message: {e}")

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

  def reset_session_tracking(self):
    """Reset session tracking variables."""
    self.first_response_received = False
    self.session_end_received = False
    self.audio_sending_started = False
    self.first_audio_sent_time = None
    self.session_end_sent_time = None

  def is_connected(self) -> bool:
    """Check if WebSocket is connected."""
    return self.connected and self.ws is not None
