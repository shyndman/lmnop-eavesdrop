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

from eavesdrop.server.config import TranscriptionConfig
from eavesdrop.server.logs import get_logger
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.processor import StreamingTranscriptionProcessor
from eavesdrop.server.streaming.websocket_adapters import (
  WebSocketAudioSource,
  WebSocketTranscriptionSink,
)


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
    client_uid: str,
    get_audio_func: Callable[[ServerConnection], Awaitable[np.ndarray | bool]],
    transcription_config: TranscriptionConfig,
    translation_queue: queue.Queue[dict] | None = None,
  ) -> None:
    """
    Initialize WebSocket streaming client.

    Args:
        websocket: WebSocket connection for the client.
        client_uid: Unique identifier for the client.
        get_audio_func: Function to get audio from websocket.
        transcription_config: Configuration for transcription processing (required).
        translation_queue: Optional queue for translation pipeline.
    """
    self.websocket = websocket
    self.client_uid = client_uid
    self.translation_queue = translation_queue
    self.logger = get_logger("ws/client")

    # Initialize components
    self.buffer = AudioStreamBuffer(transcription_config.buffer)
    self.audio_source = WebSocketAudioSource(websocket, get_audio_func)
    self.transcription_sink = WebSocketTranscriptionSink(websocket, client_uid)
    self.processor = StreamingTranscriptionProcessor(
      buffer=self.buffer,
      sink=self.transcription_sink,
      config=transcription_config,
      client_uid=client_uid,
      translation_queue=translation_queue,
      logger_name=f"ws/proc.{client_uid[0:4]}",
    )

    # State tracking
    self._processing_task: asyncio.Task | None = None
    self._audio_task: asyncio.Task | None = None
    self._completion_task: asyncio.Task | None = None
    self._exit = False

  async def start(self) -> asyncio.Task:
    """Start the streaming transcription process and return a task to await completion."""
    self.logger.info("Starting WebSocket streaming client", client_uid=self.client_uid)

    # Initialize the processor (loads model and sends server ready)
    await self.processor.initialize()

    # Start processing tasks
    self._processing_task = asyncio.create_task(self.processor.start_processing())
    self._audio_task = asyncio.create_task(self._audio_ingestion_loop())

    # Create a task that waits for either task to complete (indicating the client is done)
    completion_task = asyncio.create_task(self._wait_for_completion())

    self.logger.info("WebSocket streaming client started", client_uid=self.client_uid)
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
    self.logger.info("Stopping WebSocket streaming client", client_uid=self.client_uid)

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

    self.logger.info("WebSocket streaming client stopped", client_uid=self.client_uid)

  def add_frames(self, frames: np.ndarray) -> None:
    """
    Add audio frames to the buffer for processing.

    Args:
        frames: Audio frames as numpy array.
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
    self.logger.info("Cleaning up WebSocket streaming client", client_uid=self.client_uid)
    # The actual cleanup is handled in stop() method
