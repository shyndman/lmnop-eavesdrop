"""
WebSocket implementations of streaming transcription interfaces.

Provides WebSocket-specific implementations of AudioSource and TranscriptionSink
that integrate with the existing WebSocket server infrastructure.
"""

import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict

import numpy as np
from websockets.asyncio.server import ServerConnection

from eavesdrop.server.logs import get_logger
from eavesdrop.server.streaming.interfaces import (
  AudioSource,
  TranscriptionResult,
  TranscriptionSink,
)
from eavesdrop.wire import (
  DisconnectMessage,
  ErrorMessage,
  LanguageDetectionMessage,
  OutboundMessage,
  ServerReadyMessage,
  TranscriptionMessage,
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
    - All messages include 'stream' field for client identification
    - Result messages: {"stream": str, "segments": [...]}
    - Error messages: {"stream": str, "status": "ERROR", "message": str}
    - Language detection: {"stream": str, "language": str, "language_prob": float}
    - Server ready: {"stream": str, "message": "SERVER_READY", "backend": str}

  Error Handling:
    - Graceful failure on WebSocket send errors (logs but doesn't raise)
    - Closed state prevents further message sending
    - WebSocket connection failures are handled transparently

  Threading:
    - Safe for single-threaded async use within StreamingTranscriptionProcessor
    - WebSocket send operations are properly awaited and serialized
  """

  def __init__(self, websocket: ServerConnection, stream_name: str) -> None:
    """
    Initialize WebSocket transcription sink.

    Args:
        websocket: WebSocket connection to send to.
        stream_name: Unique identifier for the client.
    """
    self.websocket: ServerConnection = websocket
    self.stream_name: str = stream_name
    self.logger = get_logger("ws/sink")
    self._closed: bool = False

  async def send_result(self, result: TranscriptionResult) -> None:
    await self.send_message(
      TranscriptionMessage(
        stream=self.stream_name,
        segments=result.segments,
        language=result.language,
      )
    )

  async def send_error(self, error: str) -> None:
    await self.send_message(
      ErrorMessage(
        stream=self.stream_name,
        message=error,
      )
    )

  async def send_language_detection(self, language: str, probability: float) -> None:
    await self.send_message(
      LanguageDetectionMessage(
        stream=self.stream_name,
        language=language,
        language_prob=probability,
      )
    )

  async def send_server_ready(self, backend: str) -> None:
    await self.send_message(
      ServerReadyMessage(
        stream=self.stream_name,
        backend=backend,
      )
    )

  async def disconnect(self) -> None:
    """Send disconnect notification and clean up resources."""
    try:
      if self._closed or not self.websocket:
        await self.send_message(DisconnectMessage(stream=self.stream_name))
    finally:
      self._closed = True
      self.logger.info("WebSocket transcription sink disconnected")

  async def send_message(self, message: OutboundMessage) -> None:
    """Send a message to the WebSocket client."""
    if self._closed or not self.websocket:
      return

    try:
      await self.websocket.send(json.dumps(asdict(message)))

    except Exception:
      self.logger.exception("Error sending message to client")
