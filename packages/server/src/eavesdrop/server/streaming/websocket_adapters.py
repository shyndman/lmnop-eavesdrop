"""
WebSocket implementations of streaming transcription interfaces.

Provides WebSocket-specific implementations of AudioSource and TranscriptionSink
that integrate with the existing WebSocket server infrastructure.
"""

from collections.abc import Awaitable, Callable

import numpy as np
from websockets.asyncio.server import ServerConnection

from eavesdrop.server.logs import get_logger
from eavesdrop.server.messages import (
  DisconnectMessage,
  ErrorMessage,
  LanguageDetectionMessage,
  ServerReadyMessage,
  TranscriptionMessage,
)
from eavesdrop.server.streaming.interfaces import (
  AudioSource,
  TranscriptionResult,
  TranscriptionSink,
)


class WebSocketAudioSource(AudioSource):
  """
  WebSocket implementation of AudioSource protocol.

  A pass-through adapter that reads pre-processed audio data from WebSocket
  connections. The WebSocket server (TranscriptionServer) handles the raw
  audio conversion, so this source primarily validates and forwards numpy
  arrays to the streaming transcription processor.

  Audio Flow:
    WebSocket client → TranscriptionServer → WebSocketAudioSource → StreamingProcessor

  Data Format:
    - Input: numpy.ndarray (float32, normalized [-1.0, 1.0]) from server callback
    - Output: Same numpy array passed through, or None for end-of-stream
    - Sample Rate: 16kHz (managed by server, not validated here)

  Error Handling:
    - Returns None on any errors, signaling end-of-stream to processor
    - Logs exceptions but doesn't raise them (graceful degradation)
    - Closed state prevents further audio reading

  Threading:
    - Safe for single-threaded async use within StreamingTranscriptionProcessor
    - Not thread-safe for concurrent access from multiple tasks
  """

  def __init__(
    self,
    websocket: ServerConnection,
    get_audio_func: Callable[[ServerConnection], Awaitable[np.ndarray | bool]],
  ) -> None:
    """
    Initialize WebSocket audio source.

    Args:
        websocket: WebSocket connection to read from.
        get_audio_func: Function to get audio from websocket (from TranscriptionServer).
    """
    self.websocket: ServerConnection = websocket
    self.get_audio_func: Callable[[ServerConnection], Awaitable[np.ndarray | bool]] = get_audio_func
    self.logger = get_logger("ws/audiosrc")
    self._closed: bool = False

  async def read_audio(self) -> np.ndarray | None:
    """
    Read audio data from the WebSocket connection.

    Returns:
        Audio data as numpy array, or None for end-of-stream.
    """
    if self._closed:
      return None

    try:
      audio_data = await self.get_audio_func(self.websocket)

      # Handle end-of-audio signal
      if audio_data is False:
        self.logger.debug("Received end-of-audio signal")
        return None

      # audio_data should be np.ndarray at this point
      if isinstance(audio_data, np.ndarray):
        return audio_data
      else:
        self.logger.error(f"Unexpected audio data type: {type(audio_data)}")
        return None

    except Exception:
      self.logger.exception("Error reading audio from WebSocket")
      return None

  def close(self) -> None:
    """Close the audio source and clean up resources."""
    self._closed = True
    self.logger.debug("WebSocket audio source closed")


class WebSocketTranscriptionSink(TranscriptionSink):
  """
  WebSocket implementation of TranscriptionSink protocol.

  Sends transcription results and control messages to WebSocket clients using
  the established JSON message format. Handles the bidirectional communication
  between the streaming transcription processor and WebSocket clients.

  Message Flow:
    StreamingProcessor → WebSocketTranscriptionSink → WebSocket client

  Message Types:
    - Transcription results: segments with text, timestamps, completion status
    - Error messages: transcription failures, model loading issues
    - Language detection: detected language and confidence scores
    - Server status: ready notifications, disconnect signals

  JSON Message Format:
    - All messages include 'uid' field for client identification
    - Result messages: {"uid": str, "segments": [...]}
    - Error messages: {"uid": str, "status": "ERROR", "message": str}
    - Language detection: {"uid": str, "language": str, "language_prob": float}
    - Server ready: {"uid": str, "message": "SERVER_READY", "backend": str}

  Error Handling:
    - Graceful failure on WebSocket send errors (logs but doesn't raise)
    - Closed state prevents further message sending
    - WebSocket connection failures are handled transparently

  Threading:
    - Safe for single-threaded async use within StreamingTranscriptionProcessor
    - WebSocket send operations are properly awaited and serialized
  """

  def __init__(self, websocket: ServerConnection, client_uid: str) -> None:
    """
    Initialize WebSocket transcription sink.

    Args:
        websocket: WebSocket connection to send to.
        client_uid: Unique identifier for the client.
    """
    self.websocket: ServerConnection = websocket
    self.client_uid: str = client_uid
    self.logger = get_logger("ws/sink")
    self._closed: bool = False

  async def send_result(self, result: TranscriptionResult) -> None:
    """
    Send transcription result to the WebSocket client.

    Args:
        result: The transcription result to send.
    """
    if self._closed or not self.websocket:
      return

    try:
      # Use the Pydantic TranscriptionMessage for proper serialization
      transcription_message = TranscriptionMessage(
        stream=self.client_uid, segments=result.segments, language=result.language
      )

      await self.websocket.send(transcription_message.model_dump_json())

    except Exception:
      self.logger.exception("Error sending transcription result to client")

  async def send_error(self, error: str) -> None:
    """
    Send error message to the WebSocket client.

    Args:
        error: Error message to send.
    """
    if self._closed or not self.websocket:
      return

    try:
      error_message = ErrorMessage(stream=self.client_uid, message=error)
      await self.websocket.send(error_message.model_dump_json())

    except Exception:
      self.logger.exception("Error sending error message to client")

  async def send_language_detection(self, language: str, probability: float) -> None:
    """
    Send language detection result to the WebSocket client.

    Args:
        language: Detected language code.
        probability: Confidence score for the detection.
    """
    if self._closed or not self.websocket:
      return

    try:
      language_message = LanguageDetectionMessage(
        stream=self.client_uid, language=language, language_prob=probability
      )
      await self.websocket.send(language_message.model_dump_json())

    except Exception:
      self.logger.exception("Error sending language detection to client")

  async def send_server_ready(self, backend: str) -> None:
    """
    Send server ready notification to the WebSocket client.

    Args:
        backend: Name of the transcription backend being used.
    """
    if self._closed or not self.websocket:
      return

    try:
      server_ready_message = ServerReadyMessage(stream=self.client_uid, backend=backend)
      await self.websocket.send(server_ready_message.model_dump_json())

    except Exception:
      self.logger.exception("Error sending server ready message to client")

  async def disconnect(self) -> None:
    """Send disconnect notification and clean up resources."""
    if self._closed or not self.websocket:
      return

    try:
      disconnect_message = DisconnectMessage(stream=self.client_uid)
      await self.websocket.send(disconnect_message.model_dump_json())

    except Exception:
      self.logger.exception("Error sending disconnect message to client")
    finally:
      self._closed = True
      self.logger.debug("WebSocket transcription sink disconnected")
