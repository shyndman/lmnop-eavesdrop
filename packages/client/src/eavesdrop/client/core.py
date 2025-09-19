"""
EavesdropClient: Unified programmatic interface for eavesdrop transcription services.

Provides factory methods for creating mode-specific clients with async iterator
and context manager protocols for streaming transcription results.
"""

import asyncio
import secrets
from collections.abc import AsyncIterator
from typing import Any

from eavesdrop.client.audio import AudioCapture
from eavesdrop.client.connection import WebSocketConnection
from eavesdrop.common import get_logger
from eavesdrop.wire import (
  ClientType,
  TranscriptionMessage,
  UserTranscriptionOptions,
)

logger = get_logger("cli")


class EavesdropClient:
  """
  Unified client for eavesdrop transcription services.

  Supports both transcriber mode (sending audio for transcription) and
  subscriber mode (receiving transcriptions from RTSP streams).
  """

  def __init__(
    self,
    client_type: ClientType,
    host: str = "localhost",
    port: int = 8080,
    stream_names: list[str] | None = None,
    audio_device: str | None = None,
    transcription_options: UserTranscriptionOptions | None = None,
  ):
    """
    Initialize EavesdropClient.

    Args:
        host: Server hostname
        port: Server port
        client_type: Type of client (TRANSCRIBER or RTSP_SUBSCRIBER)
        stream_names: For subscriber mode, list of stream names to subscribe to
        audio_device: For transcriber mode, audio device to use
        transcription_options: Transcription configuration options
    """
    self._host = host
    self._port = port
    self._client_type = client_type
    self._stream_names = stream_names or []
    self._audio_device = audio_device
    self._transcription_options: UserTranscriptionOptions = (
      transcription_options or UserTranscriptionOptions()
    )

    # Internal state
    self._connection: WebSocketConnection | None = None
    self._audio_capture: AudioCapture | None = None
    self._connected = False
    self._streaming = False
    self._message_queue: asyncio.Queue[TranscriptionMessage] = asyncio.Queue()
    self._background_tasks: set[asyncio.Task[Any]] = set()

    # Validate configuration
    self._validate_configuration()

  def _validate_configuration(self) -> None:
    """Validate client configuration parameters."""
    if self._client_type == ClientType.RTSP_SUBSCRIBER:
      if not self._stream_names:
        raise ValueError("Subscriber mode requires at least one stream name")
    elif self._client_type == ClientType.TRANSCRIBER:
      if not self._audio_device:
        raise ValueError("Transcriber mode requires audio device specification")

    if self._port <= 0 or self._port > 65535:
      raise ValueError(f"Invalid port number: {self._port}")

  @classmethod
  def transcriber(
    cls,
    host: str = "localhost",
    port: int = 8080,
    audio_device: str = "default",
    word_timestamps: bool = False,
    initial_prompt: str | None = None,
    hotwords: list[str] | None = None,
    send_last_n_segments: int = 3,
  ) -> "EavesdropClient":
    """
    Create a transcriber client for sending audio for transcription.

    Args:
        host: Server hostname
        port: Server port
        audio_device: Audio device to capture from
        beam_size: Beam search size for transcription
        word_timestamps: Enable word-level timestamps
        initial_prompt: Initial prompt for transcription context
        hotwords: Comma-separated hotwords for improved recognition

    Returns:
        Configured EavesdropClient in transcriber mode
    """
    transcription_options = UserTranscriptionOptions(
      word_timestamps=word_timestamps,
      initial_prompt=initial_prompt,
      hotwords=hotwords,
      send_last_n_segments=send_last_n_segments,
    )

    return cls(
      client_type=ClientType.TRANSCRIBER,
      host=host,
      port=port,
      audio_device=audio_device,
      transcription_options=transcription_options,
    )

  @classmethod
  def subscriber(
    cls,
    stream_names: list[str],
    host: str = "localhost",
    port: int = 8080,
  ) -> "EavesdropClient":
    """
    Create a subscriber client for receiving RTSP stream transcriptions.

    Args:
        host: Server hostname
        port: Server port
        stream_names: List of RTSP stream names to subscribe to

    Returns:
        Configured EavesdropClient in subscriber mode
    """
    if not stream_names:
      raise ValueError("stream_names cannot be empty")

    return cls(
      client_type=ClientType.RTSP_SUBSCRIBER,
      host=host,
      port=port,
      stream_names=stream_names,
    )

  async def connect(self) -> None:
    """Establish WebSocket connection to server."""
    if self._connected:
      return

    # Create connection based on client type
    if self._client_type == ClientType.TRANSCRIBER:
      await self._connect_transcriber()
    else:
      await self._connect_subscriber()

    self._connected = True

  async def _connect_transcriber(self) -> None:
    """Connect in transcriber mode."""
    # Initialize audio capture for transcriber mode
    self._audio_capture = AudioCapture(on_error=self._on_error, audio_device=self._audio_device)

    # Create WebSocket connection
    self._connection = WebSocketConnection(
      host=self._host,
      port=self._port,
      stream_name=f"t{secrets.token_hex(2)}",
      on_ready=self._on_ready,
      on_transcription=self._on_transcription_text,
      on_error=self._on_error,
      client_type=ClientType.TRANSCRIBER,
      on_transcription_message=self._on_transcription_message,
    )

    await self._connection.connect(self._transcription_options)

    # Start message handling task
    task = asyncio.create_task(self._connection.handle_messages())
    self._background_tasks.add(task)
    task.add_done_callback(self._background_tasks.discard)

  async def _connect_subscriber(self) -> None:
    """Connect in subscriber mode."""
    # Create WebSocket connection for subscriber mode
    self._connection = WebSocketConnection(
      host=self._host,
      port=self._port,
      stream_name=f"s{secrets.token_hex(2)}",
      on_ready=self._on_ready,
      on_transcription=self._on_transcription_text,
      on_error=self._on_error,
      client_type=ClientType.RTSP_SUBSCRIBER,
      stream_names=self._stream_names,
      on_transcription_message=self._on_transcription_message,
    )

    await self._connection.connect()

    # Start message handling task
    task = asyncio.create_task(self._connection.handle_messages())
    self._background_tasks.add(task)
    task.add_done_callback(self._background_tasks.discard)

  async def disconnect(self) -> None:
    """Close WebSocket connection and cleanup resources."""
    if not self._connected:
      return

    self._connected = False
    self._streaming = False

    # Stop audio capture if active
    if self._audio_capture:
      self._audio_capture.stop_recording()
      self._audio_capture = None

    # Close WebSocket connection
    if self._connection:
      await self._connection.disconnect()
      self._connection = None

    # Cancel background tasks
    for task in self._background_tasks:
      task.cancel()

    if self._background_tasks:
      await asyncio.gather(*self._background_tasks, return_exceptions=True)
    self._background_tasks.clear()

  async def start_streaming(self) -> None:
    """Start audio streaming (transcriber mode only)."""
    if self._client_type != ClientType.TRANSCRIBER:
      raise RuntimeError("start_streaming() only supported in transcriber mode")

    if not self._connected:
      raise RuntimeError("Must connect() before start_streaming()")

    if self._streaming:
      return

    if not self._audio_capture:
      raise RuntimeError("Audio capture not initialized")

    # Start audio capture
    self._audio_capture.start_recording()
    self._streaming = True

    # Start audio streaming task
    task = asyncio.create_task(self._audio_streaming_loop())
    self._background_tasks.add(task)
    task.add_done_callback(self._background_tasks.discard)

  async def stop_streaming(self) -> None:
    """Stop audio streaming while maintaining connection (transcriber mode only)."""
    if self._client_type != ClientType.TRANSCRIBER:
      raise RuntimeError("stop_streaming() only supported in transcriber mode")

    if not self._streaming:
      return

    self._streaming = False

    if self._audio_capture:
      self._audio_capture.stop_recording()

  async def _audio_streaming_loop(self) -> None:
    """Background task to stream audio data to server."""
    if not self._connection or not self._audio_capture:
      return

    try:
      while self._streaming and self._connected:
        audio_data = await self._audio_capture.get_audio_data(timeout=0.1)
        if audio_data:
          await self._connection.send_audio_data(audio_data)
    except Exception as e:
      self._on_error(f"Audio streaming error: {e}")

  def is_connected(self) -> bool:
    """Check if client is connected to server."""
    return self._connected

  def is_streaming(self) -> bool:
    """Check if client is actively streaming (transcriber mode only)."""
    if self._client_type != ClientType.TRANSCRIBER:
      return False
    return self._streaming

  # Async iterator protocol
  def __aiter__(self) -> AsyncIterator[TranscriptionMessage]:
    """Return async iterator for transcription messages."""
    return self

  async def __anext__(self) -> TranscriptionMessage:
    """Get next transcription message."""
    if not self._connected:
      raise StopAsyncIteration

    try:
      # Wait for next message from queue with timeout to allow interruption
      message = await asyncio.wait_for(self._message_queue.get(), timeout=0.5)
      return message
    except asyncio.TimeoutError:
      # Check if still connected, continue loop if so
      if self._connected:
        return await self.__anext__()
      else:
        raise StopAsyncIteration
    except Exception:
      raise StopAsyncIteration

  # Async context manager protocol
  async def __aenter__(self) -> "EavesdropClient":
    """Enter async context manager."""
    await self.connect()
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    """Exit async context manager."""
    await self.disconnect()

  # Callback handlers
  def _on_ready(self, backend: str) -> None:
    """Handle server ready callback."""
    # Server is ready for transcription
    pass

  def _on_transcription_message(self, message: TranscriptionMessage) -> None:
    """Handle full TranscriptionMessage from connection."""
    try:
      self._message_queue.put_nowait(message)
    except asyncio.QueueFull:
      # Drop message if queue is full
      pass

  def _on_transcription_text(self, text: str) -> None:
    """Handle legacy transcription text callback from existing connection."""
    # This is primarily for backward compatibility
    # The main message flow now uses _on_transcription_message
    pass

  def _on_error(self, error: str) -> None:
    """Handle error callback."""
    # For now, just print error
    # In full implementation, could raise exceptions or call user error handlers
    logger.error(f"EavesdropClient error: {error}")
