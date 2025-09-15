"""
WebSocket streaming client facade.

Provides a high-level interface that combines all streaming transcription components
to replace the existing ServeClientFasterWhisper functionality.
"""

import asyncio
import queue
from collections.abc import Awaitable, Callable

import numpy as np
from websockets.asyncio.server import ServerConnection

from eavesdrop.common import get_logger
from eavesdrop.server.config import TranscriptionConfig
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.processor import StreamingTranscriptionProcessor
from eavesdrop.server.streaming.websocket_adapters import (
  WebSocketAudioSource,
  WebSocketTranscriptionSink,
)
from eavesdrop.server.transcription.session import create_session


class WebSocketStreamingClient:
  """
  High-level facade for WebSocket streaming transcription.

  This class combines AudioStreamBuffer, StreamingTranscriptionProcessor,
  and WebSocket adapters to provide a complete transcription solution
  that can replace ServeClientFasterWhisper.
  """

  def __init__(
    self,
    websocket: ServerConnection,
    stream_name: str,
    get_audio_func: Callable[[ServerConnection], Awaitable[np.ndarray | bool]],
    transcription_config: TranscriptionConfig,
    translation_queue: queue.Queue[dict] | None = None,
  ) -> None:
    """
    Initialize WebSocket streaming client.

    :param websocket: WebSocket connection for the client.
    :param stream_name: Unique identifier for the client.
    :param get_audio_func: Function to get audio from websocket.
    :param transcription_config: Configuration for transcription processing (required).
    :param translation_queue: Optional queue for translation pipeline.
    """
    self.websocket = websocket
    self.stream_name = stream_name
    self.translation_queue = translation_queue
    self.logger = get_logger("ws/client")

    # Initialize session and components
    self.session = create_session(stream_name)
    self.buffer = AudioStreamBuffer(transcription_config.buffer)
    self.audio_source = WebSocketAudioSource(websocket, get_audio_func)
    self.transcription_sink = WebSocketTranscriptionSink(websocket, stream_name)
    self.processor = StreamingTranscriptionProcessor(
      buffer=self.buffer,
      sink=self.transcription_sink,
      config=transcription_config,
      stream_name=stream_name,
      translation_queue=translation_queue,
      logger_name=f"ws/proc.{stream_name[0:4]}",
      session=self.session,
    )

    # State tracking
    self._processing_task: asyncio.Task | None = None
    self._audio_task: asyncio.Task | None = None
    self._completion_task: asyncio.Task | None = None
    self._exit = False

  async def start(self) -> asyncio.Task:
    """Start the streaming transcription process and return a task to await completion."""
    self.logger.info("Starting WebSocket streaming client", stream=self.stream_name)

    # Initialize the processor (loads model and sends server ready)
    await self.processor.initialize()

    # Start processing tasks
    self._processing_task = asyncio.create_task(self.processor.start_processing())
    self._audio_task = asyncio.create_task(self._audio_ingestion_loop())

    # Create a task that waits for either task to complete (indicating the client is done)
    completion_task = asyncio.create_task(self._wait_for_completion())

    self.logger.info("WebSocket streaming client started", stream=self.stream_name)
    return completion_task

  async def _wait_for_completion(self) -> None:
    """Wait for either the audio or processing task to complete."""
    try:
      # Ensure both tasks are available
      if not self._audio_task or not self._processing_task:
        raise RuntimeError("Tasks not properly initialized")

      # Wait for the first task to complete (either audio ends or processing stops)
      done, pending = await asyncio.wait(
        [self._audio_task, self._processing_task], return_when=asyncio.FIRST_COMPLETED
      )

      # Cancel any remaining tasks
      for task in pending:
        task.cancel()
        try:
          await task
        except asyncio.CancelledError:
          pass

    except Exception:
      self.logger.exception("Error in completion wait")
    finally:
      await self.stop()

  async def stop(self) -> None:
    """Stop the streaming transcription process and clean up."""
    self.logger.info("Stopping WebSocket streaming client", stream=self.stream_name)

    self._exit = True

    # Stop processor
    await self.processor.stop_processing()

    # Cancel tasks
    if self._processing_task and not self._processing_task.done():
      self._processing_task.cancel()
      try:
        await self._processing_task
      except asyncio.CancelledError:
        pass

    if self._audio_task and not self._audio_task.done():
      self._audio_task.cancel()
      try:
        await self._audio_task
      except asyncio.CancelledError:
        pass

    # Clean up resources
    self.audio_source.close()

    self.logger.info("WebSocket streaming client stopped", stream=self.stream_name)

  def add_frames(self, frames: np.ndarray) -> None:
    """
    Add audio frames to the buffer for processing.

    :param frames: Audio frames as numpy array.
    """
    self.processor.add_audio_frames(frames)

  async def _audio_ingestion_loop(self) -> None:
    """
    Continuously read audio from the source and add to buffer.

    This loop replaces the audio frame processing that was done
    in TranscriptionServer.process_audio_frames().
    """
    while not self._exit:
      try:
        audio_data = await self.audio_source.read_audio()

        if audio_data is None:
          # End of stream
          self.logger.debug("End of audio stream received")
          break

        self.add_frames(audio_data)

      except Exception:
        self.logger.exception("Error in audio ingestion loop")
        await asyncio.sleep(0.1)

  @property
  def language(self) -> str | None:
    """Get the detected language."""
    return self.processor.language

  def cleanup(self) -> None:
    """
    Perform cleanup tasks.
    """
    self.logger.info("Cleaning up WebSocket streaming client", stream=self.stream_name)
    # The actual cleanup is handled in stop() method
