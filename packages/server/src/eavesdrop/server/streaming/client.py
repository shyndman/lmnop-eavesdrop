"""Connection-owned live WebSocket transcription lifecycle."""

import asyncio

import numpy as np
from structlog.stdlib import BoundLogger
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


class WebSocketStreamingClient:
  """Own one receiver, ordered ingestion, and recognition for a connection."""

  def __init__(
    self,
    websocket: ServerConnection,
    stream_name: str,
    transcription_config: TranscriptionConfig,
  ) -> None:
    self.websocket: ServerConnection = websocket
    self.stream_name: str = stream_name
    self.logger: BoundLogger = get_logger("ws/client", stream=stream_name)
    self.session: TranscriptionSession = create_session(stream_name)
    self.buffer: AudioStreamBuffer = AudioStreamBuffer(transcription_config.buffer)
    self._flush_state: LiveSessionFlushState = LiveSessionFlushState()
    self.transcription_sink: WebSocketTranscriptionSink = WebSocketTranscriptionSink(
      websocket, stream_name
    )
    self.processor: StreamingTranscriptionProcessor = StreamingTranscriptionProcessor(
      buffer=self.buffer,
      sink=self.transcription_sink,
      config=transcription_config,
      stream_name=stream_name,
      session=self.session,
      flush_state=self._flush_state,
    )
    self._frames: asyncio.Queue[str | bytes | None] = asyncio.Queue()
    self._tasks: list[asyncio.Task[None]] = []
    self._completion_task: asyncio.Task[None] | None = None
    self._stopped: bool = False
    self._current_recording_id: str | None = None

  @property
  def completion_task(self) -> asyncio.Task[None] | None:
    return self._completion_task

  async def start(self) -> asyncio.Task[None]:
    await self.processor.initialize()
    self._tasks = [
      asyncio.create_task(self.processor.start_processing()),
      asyncio.create_task(self._receive_frames()),
      asyncio.create_task(self._audio_ingestion_loop()),
    ]
    self._completion_task = asyncio.create_task(self._wait_for_completion())
    return self._completion_task

  async def _receive_frames(self) -> None:
    """Keep receiving while a prior recording drains; only ingestion mutates state."""
    try:
      async for frame in self.websocket:
        self._frames.put_nowait(frame)
    except ConnectionClosed:
      self.logger.info("Live websocket closed during reception")
    finally:
      self._frames.put_nowait(None)

  async def _wait_for_completion(self) -> None:
    try:
      done, _ = await asyncio.wait(self._tasks, return_when=asyncio.FIRST_COMPLETED)
      for task in done:
        if not task.cancelled() and task.exception() is not None:
          task.result()
    except Exception:
      self.logger.exception("Live connection task failed")
    finally:
      await self.stop()

  async def stop(self) -> None:
    if self._stopped:
      return
    self._stopped = True
    self.processor.exit = True
    for task in self._tasks:
      if task is not asyncio.current_task() and not task.done():
        _ = task.cancel()
    _ = await asyncio.gather(
      *(task for task in self._tasks if task is not asyncio.current_task()),
      return_exceptions=True,
    )
    await self.processor.stop_processing()
    if self._completion_task is not None and self._completion_task is not asyncio.current_task():
      await self._completion_task

  def add_frames(self, frames: Float32Audio) -> None:
    self.processor.add_audio_frames(frames)

  async def _audio_ingestion_loop(self) -> None:
    while True:
      frame = await self._frames.get()
      if frame is None:
        return
      if isinstance(frame, str):
        await self._handle_live_text_frame(frame)
      elif self._current_recording_id is None:
        await self.transcription_sink.send_error(LIVE_AUDIO_BEFORE_RECORDING_START_MESSAGE)
      else:
        self.add_frames(np.frombuffer(frame, dtype=np.float32))

  async def _handle_live_text_frame(self, message_json: str) -> None:
    try:
      message = deserialize_message(message_json)
    except Exception:
      await self.transcription_sink.send_error("Invalid live control frame")
      return
    if isinstance(message, RecordingStartedMessage):
      if message.sample_rate_hz != self.buffer.config.sample_rate:
        await self.transcription_sink.send_error(
          "control_recording_started sample_rate_hz does not match server buffer sample rate"
        )
        return
      if self._current_recording_id is not None:
        # A boundary never invalidates accepted A work. The receiver remains
        # independent and queues B's frames while A's final response is sent.
        await self._flush_state.wait_until_complete()
        if self.buffer.available_duration > 0 or self.processor.has_pending_segments:
          _ = self._flush_state.accept(
            boundary_sample=self.buffer.get_buffer_end_sample(), force_complete=True
          )
          await self._flush_state.wait_until_complete()
        await self.processor.wait_until_idle()
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
      generation = self._flush_state.cancel_active_utterance()
      _ = self.buffer.discard_unprocessed_audio()
      # The worker retains its old session until it exits; do not mutate it
      # underneath native recognition or reuse its trace/transcript state.
      self.session = create_session(self.stream_name)
      self.processor.session = self.session
      self.processor.reset_live_recording(
        recording_id=self._current_recording_id,
        generation=generation,
        preserve_language=False,
      )
      return
    pending = self._flush_state.accept(
      boundary_sample=self.buffer.get_buffer_end_sample(), force_complete=message.force_complete
    )
    if pending is None:
      await self.transcription_sink.send_error(LIVE_FLUSH_ALREADY_PENDING_MESSAGE)

  def _accept_recording_boundary(self, message: RecordingStartedMessage) -> None:
    generation = self._flush_state.start_recording(message.recording_id)
    self._current_recording_id = message.recording_id
    self.buffer.reset()
    self.session.reset_utterance()
    self.session.update_audio_context(0.0, 0.0)
    self.processor.reset_live_recording(
      recording_id=message.recording_id, generation=generation, preserve_language=False
    )

  @property
  def language(self) -> str | None:
    return self.processor.language

  def cleanup(self) -> None:
    self.logger.info("Cleaned up WebSocket streaming client", stream=self.stream_name)
