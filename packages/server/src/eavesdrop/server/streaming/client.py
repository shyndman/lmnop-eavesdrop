"""WebSocket streaming client facade.

Provides a high-level interface that combines all streaming transcription components
for live-stream and finite-file session lifecycles.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum

import numpy as np
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from eavesdrop.common import get_logger
from eavesdrop.server.config import TranscriptionConfig
from eavesdrop.server.streaming.audio_flow import (
  WebSocketAudioSource,
  WebSocketTranscriptionSink,
)
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.file_decoder import (
  CANONICAL_SAMPLE_RATE_HZ,
  FileDecodeError,
  decode_file_bytes_to_canonical_audio,
)
from eavesdrop.server.streaming.file_queue import FileAudioQueue
from eavesdrop.server.streaming.flush_state import LiveSessionFlushState
from eavesdrop.server.streaming.processor import StreamingTranscriptionProcessor
from eavesdrop.server.transcription.session import create_session
from eavesdrop.wire import (
  FlushControlMessage,
  TranscriptionSourceMode,
  deserialize_message,
)

FILE_QUEUE_ENQUEUE_CHUNK_SECONDS = 2.0
FILE_OBSERVABILITY_INTERVAL_SECONDS = 5.0
FILE_QUEUE_WARNING_FILL_RATIO = 0.85
LIVE_FLUSH_ALREADY_PENDING_MESSAGE = "Flush rejected: another flush is already pending"
LIVE_FLUSH_FILE_MODE_MESSAGE = "Flush rejected: control_flush is unsupported during file upload"
LIVE_FLUSH_UNEXPECTED_MESSAGE = "Flush rejected: unsupported live control message"


class FileLifecycleState(StrEnum):
  """Finite-source lifecycle states for deterministic drain/finalization."""

  INGESTING = "ingesting"
  DRAINING = "draining"
  FINALIZING = "finalizing"
  TERMINAL = "terminal"


class WebSocketStreamingClient:
  """High-level facade for WebSocket streaming transcription."""

  def __init__(
    self,
    websocket: ServerConnection,
    stream_name: str,
    get_audio_func: Callable[[ServerConnection], Awaitable[np.ndarray | bool]],
    transcription_config: TranscriptionConfig,
    source_mode: TranscriptionSourceMode = TranscriptionSourceMode.LIVE,
  ) -> None:
    """Initialize WebSocket streaming client.

    :param websocket: WebSocket connection for the client.
    :type websocket: ServerConnection
    :param stream_name: Unique identifier for the client.
    :type stream_name: str
    :param get_audio_func: Function to get live audio frames from websocket.
    :type get_audio_func: Callable[[ServerConnection], Awaitable[np.ndarray | bool]]
    :param transcription_config: Configuration for transcription processing.
    :type transcription_config: TranscriptionConfig
    :param source_mode: Session source mode used for routing live vs file lifecycle.
    :type source_mode: TranscriptionSourceMode
    """
    self.websocket = websocket
    self.stream_name = stream_name
    self.source_mode = source_mode
    self.logger = get_logger("ws/client", stream=stream_name)

    # Initialize session and components
    self.session = create_session(stream_name)
    self.buffer = AudioStreamBuffer(transcription_config.buffer)
    self._flush_state = LiveSessionFlushState()
    self.audio_source = WebSocketAudioSource(websocket, get_audio_func)
    self.transcription_sink = WebSocketTranscriptionSink(websocket, stream_name)
    self.processor = StreamingTranscriptionProcessor(
      buffer=self.buffer,
      sink=self.transcription_sink,
      config=transcription_config,
      stream_name=stream_name,
      session=self.session,
      flush_state=self._flush_state,
    )

    # Shared state tracking
    self._processing_task: asyncio.Task[None] | None = None
    self._audio_task: asyncio.Task[None] | None = None
    self._completion_task: asyncio.Task[None] | None = None
    self._exit = False
    self._stopped = False

    # File-mode state tracking
    self._file_state: FileLifecycleState = FileLifecycleState.TERMINAL
    self._file_queue: FileAudioQueue | None = None
    self._file_ingest_task: asyncio.Task[None] | None = None
    self._file_feed_task: asyncio.Task[None] | None = None
    self._file_observability_task: asyncio.Task[None] | None = None
    self._file_ingested_seconds = 0.0

  async def start(self) -> asyncio.Task[None]:
    """Start the streaming transcription process and return completion task."""
    source_mode = getattr(self, "source_mode", TranscriptionSourceMode.LIVE)
    self.logger.info(
      "Starting WebSocket streaming client",
      stream=self.stream_name,
      source_mode=source_mode,
    )

    await self.processor.initialize()

    if source_mode == TranscriptionSourceMode.FILE:
      completion_task = asyncio.create_task(self._run_file_mode_lifecycle())
    else:
      self._processing_task = asyncio.create_task(self.processor.start_processing())
      self._audio_task = asyncio.create_task(self._audio_ingestion_loop())
      completion_task = asyncio.create_task(self._wait_for_completion())

    self._completion_task = completion_task
    self.logger.info("WebSocket streaming client started", stream=self.stream_name)
    return completion_task

  async def _run_file_mode_lifecycle(self) -> None:
    """Run finite-file ingest, drain, and deterministic terminal finalization."""
    # <intent>
    # Accept a finite audio file upload and produce the complete transcription result for that file.
    # This function coordinates ingest, drain, and finalization so the caller gets one
    # deterministic terminal result.
    # </intent>
    self._file_state = FileLifecycleState.INGESTING
    self._file_queue = FileAudioQueue()

    self._processing_task = asyncio.create_task(self.processor.start_processing())
    self._file_ingest_task = asyncio.create_task(self._ingest_file_upload())
    self._file_feed_task = asyncio.create_task(self._feed_file_queue_to_processor())
    self._file_observability_task = asyncio.create_task(self._file_observability_loop())

    try:
      await self._file_ingest_task

      self._file_state = FileLifecycleState.DRAINING
      await self._file_feed_task

      self.processor.mark_source_exhausted()
      if not self._processing_task:
        raise RuntimeError("File lifecycle missing processing task")
      await self._processing_task

      self._file_state = FileLifecycleState.FINALIZING
      await self.transcription_sink.disconnect()

    except Exception as exc:
      self.logger.exception("File mode lifecycle failed", state=self._file_state)
      await self.transcription_sink.send_error(f"File-mode transcription failed: {exc}")
      raise
    finally:
      await self._cancel_file_mode_aux_tasks()
      self._file_state = FileLifecycleState.TERMINAL
      await self.stop()

  async def _ingest_file_upload(self) -> None:
    """Receive uploaded bytes, decode to canonical audio, and enqueue bounded chunks."""
    if not self._file_queue:
      raise RuntimeError("File queue not initialized for file-mode ingestion")

    uploaded_bytes = bytearray()
    while True:
      frame_data = await self.websocket.recv()

      if isinstance(frame_data, str):
        await self._handle_file_text_frame(frame_data)
        continue

      if isinstance(frame_data, bytes):
        frame_bytes = frame_data
      else:
        frame_bytes = bytes(frame_data)

      if frame_bytes == b"END_OF_AUDIO":
        break

      uploaded_bytes.extend(frame_bytes)

    try:
      decoded_audio = await decode_file_bytes_to_canonical_audio(bytes(uploaded_bytes))
    except FileDecodeError:
      await self._file_queue.mark_producer_done()
      raise

    chunk_size_samples = int(FILE_QUEUE_ENQUEUE_CHUNK_SECONDS * CANONICAL_SAMPLE_RATE_HZ)
    for chunk_start in range(0, decoded_audio.shape[0], chunk_size_samples):
      chunk_end = chunk_start + chunk_size_samples
      chunk = decoded_audio[chunk_start:chunk_end]
      await self._file_queue.enqueue(chunk)
      self._file_ingested_seconds += chunk.shape[0] / CANONICAL_SAMPLE_RATE_HZ

    await self._file_queue.mark_producer_done()

  async def _feed_file_queue_to_processor(self) -> None:
    """Consume queued canonical chunks and feed them into the existing processor buffer."""
    if not self._file_queue:
      raise RuntimeError("File queue not initialized for file-mode feeder")

    while not self._exit:
      queued_chunk = await self._file_queue.dequeue()
      if queued_chunk is None:
        break
      self.processor.add_audio_frames(queued_chunk)

  async def _file_observability_loop(self) -> None:
    """Emit periodic file-session queue and throughput health metrics."""
    if not self._file_queue:
      raise RuntimeError("File queue not initialized for file observability")

    previous_wall_time = time.monotonic()
    previous_ingested_seconds = self._file_ingested_seconds
    previous_processed_seconds = self.buffer.processed_up_to_time
    previous_block_seconds = 0.0

    while self._file_state in (FileLifecycleState.INGESTING, FileLifecycleState.DRAINING):
      await asyncio.sleep(FILE_OBSERVABILITY_INTERVAL_SECONDS)

      snapshot = await self._file_queue.snapshot()
      current_wall_time = time.monotonic()
      elapsed_wall = max(current_wall_time - previous_wall_time, 0.001)

      ingest_rate = (self._file_ingested_seconds - previous_ingested_seconds) / elapsed_wall
      processed_seconds = self.buffer.processed_up_to_time
      process_rate = (processed_seconds - previous_processed_seconds) / elapsed_wall
      enqueue_block_delta = snapshot.total_enqueue_block_s - previous_block_seconds

      log_method = self.logger.warning
      if snapshot.fill_ratio < FILE_QUEUE_WARNING_FILL_RATIO:
        log_method = self.logger.info

      log_method(
        "File session observability",
        state=self._file_state,
        queue_fill_ratio=f"{snapshot.fill_ratio:.3f}",
        queued_seconds=f"{snapshot.queued_seconds:.2f}",
        ingest_rate_s_per_s=f"{ingest_rate:.2f}",
        process_rate_s_per_s=f"{process_rate:.2f}",
        enqueue_block_s=f"{enqueue_block_delta:.3f}",
      )

      previous_wall_time = current_wall_time
      previous_ingested_seconds = self._file_ingested_seconds
      previous_processed_seconds = processed_seconds
      previous_block_seconds = snapshot.total_enqueue_block_s

  async def _wait_for_completion(self) -> None:
    """Wait for either audio or processing task completion in live mode."""
    try:
      if not self._audio_task or not self._processing_task:
        raise RuntimeError("Tasks not properly initialized")

      done, pending = await asyncio.wait(
        [self._audio_task, self._processing_task],
        return_when=asyncio.FIRST_COMPLETED,
      )

      for task in pending:
        task.cancel()
        try:
          await task
        except asyncio.CancelledError:
          pass

      for task in done:
        if task.exception():
          raise task.exception()

    except Exception:
      self.logger.exception("Error in completion wait")
    finally:
      await self.stop()

  async def stop(self) -> None:
    """Stop processing tasks and clean up resources."""
    if getattr(self, "_stopped", False):
      return

    self._stopped = True
    self._exit = True

    self.logger.info("Stopping WebSocket streaming client", stream=self.stream_name)

    await self._cancel_file_mode_aux_tasks()

    await self.processor.stop_processing()

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

    self.audio_source.close()
    self._file_state = FileLifecycleState.TERMINAL

    self.logger.info("WebSocket streaming client stopped", stream=self.stream_name)

  async def _cancel_file_mode_aux_tasks(self) -> None:
    """Cancel file-mode auxiliary tasks that are not part of live mode."""
    auxiliary_tasks = [
      getattr(self, "_file_ingest_task", None),
      getattr(self, "_file_feed_task", None),
      getattr(self, "_file_observability_task", None),
    ]

    for auxiliary_task in auxiliary_tasks:
      if auxiliary_task and not auxiliary_task.done():
        auxiliary_task.cancel()
        try:
          await auxiliary_task
        except asyncio.CancelledError:
          pass

  def add_frames(self, frames: np.ndarray) -> None:
    """Add audio frames to the processor buffer."""
    self.processor.add_audio_frames(frames)

  async def _handle_file_text_frame(self, message_json: str) -> None:
    """Reject control frames that are illegal during file uploads."""
    try:
      message = deserialize_message(message_json)
    except Exception:
      await self.transcription_sink.send_error("Unexpected text frame during file upload")
      return

    if isinstance(message, FlushControlMessage):
      await self.transcription_sink.send_error(LIVE_FLUSH_FILE_MODE_MESSAGE)
      return

    await self.transcription_sink.send_error(
      f"Unexpected text frame during file upload: {message.type}"
    )

  async def _handle_live_text_frame(self, message_json: str) -> None:
    """Handle post-setup live control frames without adding another recv owner."""
    try:
      message = deserialize_message(message_json)
    except Exception:
      await self.transcription_sink.send_error("Invalid live control frame")
      return

    if not isinstance(message, FlushControlMessage):
      await self.transcription_sink.send_error(f"{LIVE_FLUSH_UNEXPECTED_MESSAGE}: {message.type}")
      return

    pending_flush = self._flush_state.accept(
      boundary_sample=self.buffer.get_buffer_end_sample(),
      force_complete=message.force_complete,
    )
    if pending_flush is None:
      await self.transcription_sink.send_error(LIVE_FLUSH_ALREADY_PENDING_MESSAGE)
      return

    self.logger.info(
      "Accepted live flush request",
      boundary_sample=pending_flush.boundary_sample,
      force_complete=pending_flush.force_complete,
    )

  async def _audio_ingestion_loop(self) -> None:
    """Continuously read live audio and add it to the processing buffer."""
    while not self._exit:
      try:
        frame_data = await self.websocket.recv()

        if isinstance(frame_data, str):
          await self._handle_live_text_frame(frame_data)
          continue

        frame_bytes = frame_data if isinstance(frame_data, bytes) else bytes(frame_data)
        if frame_bytes == b"END_OF_AUDIO":
          self.logger.debug("End of audio stream received")
          break

        self.add_frames(np.frombuffer(frame_bytes, dtype=np.float32))

      except ConnectionClosed:
        self.logger.info("Live websocket closed during audio ingestion")
        break
      except Exception:
        self.logger.exception("Error in audio ingestion loop")
        await asyncio.sleep(0.1)

  @property
  def language(self) -> str | None:
    """Get the detected language."""
    return self.processor.language

  def cleanup(self) -> None:
    """Perform cleanup tasks."""
    self.logger.info("Cleaning up WebSocket streaming client", stream=self.stream_name)
