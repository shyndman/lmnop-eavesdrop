"""WebSocket streaming client facade for live transcription sessions."""

import asyncio
from typing import Protocol, cast

import numpy as np
import structlog
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from eavesdrop.common import get_logger
from eavesdrop.server.config import TranscriptionConfig
from eavesdrop.server.streaming.audio_flow import WebSocketTranscriptionSink
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.flush_state import LiveSessionFlushState
from eavesdrop.server.streaming.interfaces import Float32Audio
from eavesdrop.server.streaming.processor import StreamingTranscriptionProcessor
from eavesdrop.server.transcription.session import TranscriptionSession, create_session
from eavesdrop.wire import (
  FlushControlMessage,
  RecordingStartedMessage,
  UtteranceCancelledMessage,
  deserialize_message,
)

LIVE_FLUSH_ALREADY_PENDING_MESSAGE = "Flush rejected: another flush is already pending"
LIVE_RECORDING_START_REQUIRED_MESSAGE = (
  "Live control rejected: control_recording_started is required before live controls"
)
LIVE_RECORDING_ID_MISMATCH_MESSAGE = (
  "Live control rejected: recording_id does not match the active recording epoch"
)
LIVE_AUDIO_BEFORE_RECORDING_START_MESSAGE = (
  "Live audio rejected: control_recording_started is required before audio frames"
)
LIVE_FLUSH_UNEXPECTED_MESSAGE = "Flush rejected: unsupported live control message"


class StreamingProcessorLike(Protocol):
  language: str | None

  async def initialize(self) -> None: ...
  async def start_processing(self) -> None: ...
  async def stop_processing(self) -> None: ...
  def add_audio_frames(self, frames: Float32Audio) -> None: ...
  def reset_live_recording(
    self,
    *,
    recording_id: str | None,
    generation: int,
    preserve_language: bool = True,
  ) -> None: ...


class WebSocketStreamingClient:
  """High-level facade for live WebSocket streaming transcription."""

  def __init__(
    self,
    websocket: ServerConnection,
    stream_name: str,
    transcription_config: TranscriptionConfig,
  ) -> None:
    """Initialize WebSocket streaming client.

    :param websocket: WebSocket connection for the client.
    :type websocket: ServerConnection
    :param stream_name: Unique identifier for the client.
    :type stream_name: str
    :param transcription_config: Configuration for transcription processing.
    :type transcription_config: TranscriptionConfig
    """
    self.websocket: ServerConnection = websocket
    self.stream_name: str = stream_name
    self.logger: structlog.stdlib.BoundLogger = get_logger("ws/client", stream=stream_name)

    self.session: TranscriptionSession = create_session(stream_name)
    self.buffer: AudioStreamBuffer = AudioStreamBuffer(transcription_config.buffer)
    self._flush_state: LiveSessionFlushState = LiveSessionFlushState()
    self.transcription_sink: WebSocketTranscriptionSink = WebSocketTranscriptionSink(
      websocket, stream_name
    )
    self.processor: StreamingProcessorLike = cast(
      StreamingProcessorLike,
      cast(
        object,
        StreamingTranscriptionProcessor(
          buffer=self.buffer,
          sink=self.transcription_sink,
          config=transcription_config,
          stream_name=stream_name,
          session=self.session,
          flush_state=self._flush_state,
        ),
      ),
    )

    self._processing_task: asyncio.Task[None] | None = None
    self._audio_task: asyncio.Task[None] | None = None
    self._completion_task: asyncio.Task[None] | None = None
    self._exit: bool = False
    self._stopped: bool = False
    self._current_recording_id: str | None = None

  @property
  def completion_task(self) -> asyncio.Task[None] | None:
    """Completion task created by :meth:`start`, or ``None`` before start."""
    return self._completion_task

  async def start(self) -> asyncio.Task[None]:
    """Start the streaming transcription process and return completion task."""
    self.logger.info("Starting WebSocket streaming client", stream=self.stream_name)
    await self.processor.initialize()

    self._processing_task = asyncio.create_task(self.processor.start_processing())
    self._audio_task = asyncio.create_task(self._audio_ingestion_loop())
    completion_task = asyncio.create_task(self._wait_for_completion())

    self._completion_task = completion_task
    self.logger.info("WebSocket streaming client started", stream=self.stream_name)
    return completion_task

  async def _wait_for_completion(self) -> None:
    """Wait for either audio or processing task completion in live mode."""
    try:
      if not self._audio_task or not self._processing_task:
        raise RuntimeError("Tasks not properly initialized")

      tasks: list[asyncio.Task[None]] = [self._audio_task, self._processing_task]
      done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

      for task in pending:
        _ = task.cancel()
        try:
          await task
        except asyncio.CancelledError:
          pass

      for task in done:
        if task.cancelled():
          raise asyncio.CancelledError

        task_exception = task.exception()
        if task_exception is not None:
          raise task_exception
    except Exception:
      self.logger.exception("Error in completion wait")
    finally:
      await self.stop()

  async def stop(self) -> None:
    """Stop processing tasks and clean up resources."""
    if self._stopped:
      return

    self._stopped = True
    self._exit = True

    self.logger.info("Stopping WebSocket streaming client", stream=self.stream_name)
    await self.processor.stop_processing()

    if self._processing_task and not self._processing_task.done():
      _ = self._processing_task.cancel()
      try:
        await self._processing_task
      except asyncio.CancelledError:
        pass

    if self._audio_task and not self._audio_task.done():
      _ = self._audio_task.cancel()
      try:
        await self._audio_task
      except asyncio.CancelledError:
        pass

    self.logger.info("WebSocket streaming client stopped", stream=self.stream_name)

  def add_frames(self, frames: Float32Audio) -> None:
    """Add audio frames to the processor buffer."""
    self.processor.add_audio_frames(frames)

  async def _handle_live_text_frame(self, message_json: str) -> None:
    """Handle post-setup live control frames without adding another recv owner."""
    try:
      message = deserialize_message(message_json)
    except Exception:
      await self.transcription_sink.send_error("Invalid live control frame")
      return

    if isinstance(message, RecordingStartedMessage):
      self._accept_recording_boundary(message)
      return

    if not isinstance(message, (FlushControlMessage, UtteranceCancelledMessage)):
      await self.transcription_sink.send_error(f"{LIVE_FLUSH_UNEXPECTED_MESSAGE}: {message.type}")
      return

    if self._current_recording_id is None:
      await self.transcription_sink.send_error(LIVE_RECORDING_START_REQUIRED_MESSAGE)
      return

    if message.recording_id != self._current_recording_id:
      await self.transcription_sink.send_error(LIVE_RECORDING_ID_MISMATCH_MESSAGE)
      return

    if isinstance(message, UtteranceCancelledMessage):
      active_utterance_generation = self._flush_state.cancel_active_utterance()
      self.session.reset_utterance()
      discarded_duration = self.buffer.discard_unprocessed_audio()
      self.processor.reset_live_recording(
        recording_id=self._current_recording_id,
        generation=active_utterance_generation,
        preserve_language=False,
      )
      self.logger.info(
        "Discarded live utterance tail",
        active_utterance_generation=active_utterance_generation,
        discarded_duration_s=f"{discarded_duration:.3f}",
        processed_up_to_time_s=f"{self.buffer.processed_up_to_time:.3f}",
      )
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
      recording_relative_boundary_sample=pending_flush.boundary_sample,
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

        if frame_data == b"END_OF_AUDIO":
          self.logger.debug("End of audio stream received")
          break

        if self._current_recording_id is None:
          await self.transcription_sink.send_error(LIVE_AUDIO_BEFORE_RECORDING_START_MESSAGE)
          continue

        self.add_frames(np.frombuffer(frame_data, dtype=np.float32))
      except ConnectionClosed:
        self.logger.info("Live websocket closed during audio ingestion")
        break
      except Exception:
        self.logger.exception("Error in audio ingestion loop")
        await asyncio.sleep(0.1)

  @property
  def language(self) -> str | None:
    """Get the configured language."""
    return self.processor.language

  def cleanup(self) -> None:
    """Perform cleanup tasks."""
    self.logger.info("Cleaning up WebSocket streaming client", stream=self.stream_name)

  def _accept_recording_boundary(self, message: RecordingStartedMessage) -> None:
    """Reset all live recording-local state for a newly accepted epoch."""
    if message.sample_rate_hz != self.buffer.config.sample_rate:
      raise ValueError(
        "control_recording_started sample_rate_hz does not match server buffer sample rate"
      )

    generation = self._flush_state.start_recording(message.recording_id)
    self._current_recording_id = message.recording_id
    self.buffer.reset()
    self.session.reset_utterance()
    self.session.update_audio_context(0.0, 0.0)
    self.processor.reset_live_recording(
      recording_id=message.recording_id,
      generation=generation,
      preserve_language=False,
    )
    self.logger.info(
      "Accepted live recording boundary",
      recording_id=message.recording_id,
      sample_rate_hz=message.sample_rate_hz,
      generation=generation,
    )
