"""Coordinate scheduled recognition, identity validation, and result delivery."""

from __future__ import annotations

import asyncio
import time

from structlog.stdlib import BoundLogger

from eavesdrop.common import get_logger
from eavesdrop.server.config import TranscriptionConfig
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.flush_state import LiveSessionFlushState, PendingFlush
from eavesdrop.server.streaming.interfaces import Float32Audio, TranscriptionSink
from eavesdrop.server.streaming.processing import models
from eavesdrop.server.streaming.processing.output import TranscriptOutput
from eavesdrop.server.streaming.processing.recognition import RecognitionWorker
from eavesdrop.server.streaming.processing.scheduling import ProcessingSchedule
from eavesdrop.server.transcription.session import TranscriptionSession


class StreamingTranscriptionProcessor:
  """Sequence live recognition and delivery without exposing worker-owned state."""

  def __init__(
    self,
    buffer: AudioStreamBuffer,
    sink: TranscriptionSink,
    config: TranscriptionConfig,
    session: TranscriptionSession,
    stream_name: str,
    flush_state: LiveSessionFlushState | None = None,
  ) -> None:
    self.buffer: AudioStreamBuffer = buffer
    self.sink: TranscriptionSink = sink
    self.config: TranscriptionConfig = config
    self.stream_name: str = stream_name
    self.flush_state: LiveSessionFlushState | None = flush_state
    self.logger: BoundLogger = get_logger("proc", stream=self.stream_name)

    self.exit: bool = False
    self.recording_id: str | None = None
    self._idle: asyncio.Event = asyncio.Event()
    self._idle.set()
    self.output: TranscriptOutput = TranscriptOutput(buffer, sink, config, session, self.logger)
    self.recognition: RecognitionWorker = RecognitionWorker(
      buffer, config, self.logger, flush_state
    )
    self.schedule: ProcessingSchedule = ProcessingSchedule(buffer, config, self.logger, flush_state)

  @property
  def session(self) -> TranscriptionSession:
    """Return the current output session, independently of captured worker sessions."""
    return self.output.session

  @session.setter
  def session(self, session: TranscriptionSession) -> None:
    """Route recording replacement to output without changing in-flight workers."""
    self.output.session = session

  async def initialize(self) -> None:
    """Initialize the Orukeet model off the event loop and announce readiness."""
    try:
      await self.recognition.initialize()
    except Exception:
      self.logger.exception("Failed to load Orukeet")
      await self.sink.send_error("Failed to load Orukeet")
      raise
    await self.sink.send_server_ready("orukeet_tdt")

  async def start_processing(self) -> None:
    """Start the transcription processing loop."""
    self.logger.info("Starting transcription processing", stream=self.stream_name)
    await self._transcription_loop()

  async def stop_processing(self) -> None:
    """Stop the transcription processing loop."""
    self.logger.info("Stopping transcription processing", stream=self.stream_name)
    self.exit = True
    await self.sink.disconnect()

  def add_audio_frames(self, frames: Float32Audio) -> None:
    """Add audio frames to the buffer for processing."""
    self.buffer.add_frames(frames)

  def reset_live_recording(
    self,
    *,
    recording_id: str | None,
    generation: int,
    preserve_language: bool = True,
  ) -> None:
    """Reset processor-local recording state after a new epoch or cancellation.

    A recording boundary resets the upstream buffer/session to recording-relative
    zero. Utterance cancellation preserves that recording-relative base while
    discarding only the unprocessed tail.
    """
    self.recording_id = recording_id
    self.output.reset(recording_id, preserve_language=preserve_language)

    self.logger.info(
      "processor live recording reset",
      recording_id=recording_id,
      generation=generation,
      preserve_language=preserve_language,
    )

  @property
  def has_pending_segments(self) -> bool:
    """Report whether provisional transcription awaits completion.

    :returns: Whether the active recording contains pending segments.
    :rtype: bool
    """
    return self.output.has_pending_segments

  async def wait_until_idle(self) -> None:
    """Wait until the active recognition pass and result handling finish.

    :returns: None.
    :rtype: None
    """
    _ = await self._idle.wait()

  async def _transcription_loop(self) -> None:
    """Main processing loop: get audio → transcribe → send results → wait."""
    while not self.exit:
      loop_start = time.monotonic()
      try:
        chunk_fetch_start = time.monotonic()
        audio_chunk = await self._get_next_audio_chunk()
        chunk_fetch_duration = time.monotonic() - chunk_fetch_start
        if audio_chunk is None:
          continue

        result_processing_start = time.monotonic()
        self._idle.clear()
        try:
          transcription_result = await self._transcribe_chunk(audio_chunk)
          await self._process_transcription_result(transcription_result)
        finally:
          self._idle.set()
        result_processing_duration = time.monotonic() - result_processing_start

        if transcription_result.status is models.TranscriptionPassStatus.INTERRUPTED_BEFORE_COMMIT:
          continue

        interval_wait_start = time.monotonic()
        await self.schedule.wait_for_interval(transcription_result.processing_time)
        interval_wait_duration = time.monotonic() - interval_wait_start

        loop_duration = time.monotonic() - loop_start
        self.logger.info(
          "Transcription loop timing",
          chunk_duration_s=f"{audio_chunk.duration:.3f}",
          chunk_fetch_s=f"{chunk_fetch_duration:.3f}",
          inference_and_result_s=f"{result_processing_duration:.3f}",
          inference_s=f"{transcription_result.processing_time:.3f}",
          interval_wait_s=f"{interval_wait_duration:.3f}",
          total_loop_s=f"{loop_duration:.3f}",
        )
      except Exception:
        await self._handle_transcription_error()

    self.logger.info("Exiting speech to text thread")

  async def _get_next_audio_chunk(self) -> models.AudioChunk | None:
    """Select audio or deliver an empty flush before acknowledging control."""
    chunk = await self.schedule.next_chunk()
    pending_flush = self.schedule.pending_flush()
    if chunk is not None and chunk.duration == 0 and pending_flush is not None:
      await self.output.flush_empty(force_complete=pending_flush.force_complete)
      if self.flush_state is not None:
        _ = self.flush_state.complete()
      return None
    return chunk

  async def _transcribe_chunk(self, chunk: models.AudioChunk) -> models.ChunkTranscriptionResult:
    """Capture the active identity and session before scheduling native work."""
    return await self.recognition.transcribe(
      chunk, self.schedule.current_generation(), self.recording_id, self.session
    )

  async def _process_transcription_result(self, result: models.ChunkTranscriptionResult) -> None:
    """Process transcription results, update language, and handle segments."""
    if result.status is models.TranscriptionPassStatus.INTERRUPTED_BEFORE_COMMIT:
      self.logger.info("Skipping interrupted transcription pass before commit")
      return

    active_generation = self.schedule.current_generation()
    if result.utterance_generation != active_generation:
      self.logger.info(
        "Dropping stale transcription result after utterance cancel",
        result_generation=result.utterance_generation,
        active_generation=active_generation,
      )
      return

    if result.recording_id != self.recording_id:
      self.logger.info(
        "Dropping stale transcription result after recording boundary",
        result_recording_id=result.recording_id,
        active_recording_id=self.recording_id,
      )
      return

    pending_flush = self.schedule.pending_flush()
    flush_boundary_reached = pending_flush is not None and self._covers_flush_boundary(
      result, pending_flush
    )
    force_complete = bool(flush_boundary_reached and pending_flush and pending_flush.force_complete)

    await self.output.handle(
      result.segments or [],
      result.audio_duration,
      result.speech_chunks,
      emit_result=pending_flush is None or flush_boundary_reached,
      flush_complete=flush_boundary_reached,
      force_complete=force_complete,
    )

    if flush_boundary_reached:
      if force_complete:
        self._discard_flushed_audio(pending_flush)
      if self.flush_state is not None:
        _ = self.flush_state.complete()

  def _discard_flushed_audio(self, pending_flush: PendingFlush | None) -> None:
    """Drop buffered audio through the accepted flush boundary after emitting the response.

    The boundary is an absolute buffer sample index. In live WebSocket mode the
    buffer was reset at recording start, so the sample index is recording-relative.
    """
    if pending_flush is None:
      raise RuntimeError("Cannot discard flushed audio without a pending flush boundary")

    discarded_duration = self.buffer.discard_through_sample(pending_flush.boundary_sample)
    self.logger.info(
      "Discarded flushed audio",
      discarded_duration_s=f"{discarded_duration:.3f}",
      recording_relative_boundary_sample=pending_flush.boundary_sample,
    )

  async def _handle_transcription_error(self) -> None:
    """Handle transcription errors with logging and recovery."""
    self.logger.exception("Failed to transcribe audio chunk")
    await asyncio.sleep(self.buffer.config.transcription_interval)

  def _covers_flush_boundary(
    self, result: models.ChunkTranscriptionResult, pending_flush: PendingFlush
  ) -> bool:
    """Return whether the transcribed chunk covers the accepted flush boundary."""
    chunk_end_sample = result.chunk_start_sample + result.chunk_sample_count
    boundary_reached = chunk_end_sample >= pending_flush.boundary_sample
    self.logger.debug(
      "Evaluated flush boundary coverage",
      recording_relative_chunk_start_sample=result.chunk_start_sample,
      recording_relative_chunk_end_sample=chunk_end_sample,
      recording_relative_boundary_sample=pending_flush.boundary_sample,
      boundary_reached=boundary_reached,
    )
    return boundary_reached
