"""
Streaming transcription processor with integrated Orukeet speech recognition.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Lock

import numpy as np
from numpy.typing import NDArray
from structlog.stdlib import BoundLogger

from eavesdrop.common import get_logger
from eavesdrop.server.config import TranscriptionConfig
from eavesdrop.server.constants import SINGLE_MODEL
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.debug_capture import AudioDebugCapture
from eavesdrop.server.streaming.flush_state import LiveSessionFlushState, PendingFlush
from eavesdrop.server.streaming.interfaces import TranscriptionResult, TranscriptionSink
from eavesdrop.server.streaming.tracing import (
  log_client_segments,
  log_generated_segments,
  log_vad_analysis,
  tracing_logger,
)
from eavesdrop.server.transcription.models import SpeechChunk, TranscriptionInfo, VadParameters
from eavesdrop.server.transcription.orukeet import OrukeetTranscriber
from eavesdrop.server.transcription.session import TranscriptionSession
from eavesdrop.wire import Segment
from eavesdrop.wire.transcription import compute_segment_chain_id

Float32Audio = NDArray[np.float32]


@dataclass
class AudioChunk:
  """Audio data with associated metadata for processing."""

  data: Float32Audio
  duration: float
  start_time: float


@dataclass
class ChunkTranscriptionResult:
  """Result of transcribing an audio chunk."""

  status: "TranscriptionPassStatus"
  chunk_start_sample: int
  chunk_sample_count: int
  segments: list[Segment] | None
  info: TranscriptionInfo | None
  processing_time: float
  audio_duration: float
  speech_chunks: list[SpeechChunk] | None = None
  utterance_generation: int = 0
  recording_id: str | None = None


class TranscriptionPassStatus(str, Enum):
  """Outcome for one transcription pass over a buffered audio chunk."""

  TRANSCRIBED = "transcribed"
  INTERRUPTED_BEFORE_COMMIT = "interrupted_before_commit"


class StreamingTranscriptionProcessor:
  """Orchestrates live streaming audio buffering and Orukeet transcription."""

  _models: dict[str, OrukeetTranscriber] = {}
  _model_lock: Lock = Lock()

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
    self.session: TranscriptionSession = session
    self.flush_state: LiveSessionFlushState | None = flush_state
    self.logger: BoundLogger = get_logger("proc", stream=self.stream_name)

    self.exit: bool = False
    self._minimum_chunk_wait_logged: bool = False
    self.language: str | None = config.language
    self.recording_id: str | None = None
    self.transcriber: OrukeetTranscriber | None = None
    self.vad_parameters: VadParameters = config.vad_parameters

    self._debug_capture: AudioDebugCapture | None = None
    if config.debug_audio and config.debug_audio.post_buffer:
      self._debug_capture = AudioDebugCapture(
        output_path=Path(str(config.debug_audio.post_buffer)),
        stream_name=stream_name,
        sample_rate=buffer.config.sample_rate,
      )

  async def initialize(self) -> None:
    """Initialize the Orukeet model off the event loop and announce readiness."""
    try:
      self.transcriber = await asyncio.to_thread(self._load_model)
    except Exception:
      self.logger.exception("Failed to load Orukeet")
      await self.sink.send_error("Failed to load Orukeet")
      raise
    await self.sink.send_server_ready("orukeet_tdt")

  def _load_model(self) -> OrukeetTranscriber:
    if not SINGLE_MODEL:
      return OrukeetTranscriber.load(self.config)
    key = self.config.model
    with self._model_lock:
      if key not in self._models:
        self._models[key] = OrukeetTranscriber.load(self.config)
      return self._models[key]

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
    """Reset processor-local recording state after a new epoch or cancellation."""
    self.recording_id = recording_id
    if not preserve_language:
      self.language = self.config.language

    self.logger.info(
      "processor live recording reset",
      recording_id=recording_id,
      generation=generation,
      preserve_language=preserve_language,
    )

  async def _transcription_loop(self) -> None:
    """Main processing loop: get audio -> transcribe -> send results -> wait."""
    while not self.exit:
      loop_start = time.monotonic()
      try:
        chunk_fetch_start = time.monotonic()
        audio_chunk = await self._get_next_audio_chunk()
        chunk_fetch_duration = time.monotonic() - chunk_fetch_start
        if audio_chunk is None:
          continue

        result_processing_start = time.monotonic()
        transcription_result = await self._transcribe_chunk(audio_chunk)
        await self._process_transcription_result(transcription_result)
        result_processing_duration = time.monotonic() - result_processing_start

        if transcription_result.status is TranscriptionPassStatus.INTERRUPTED_BEFORE_COMMIT:
          continue

        interval_wait_start = time.monotonic()
        await self._wait_for_next_interval(transcription_result.processing_time)
        interval_wait_duration = time.monotonic() - interval_wait_start

        self.logger.info(
          "Transcription loop timing",
          chunk_duration_s=f"{audio_chunk.duration:.3f}",
          chunk_fetch_s=f"{chunk_fetch_duration:.3f}",
          inference_and_result_s=f"{result_processing_duration:.3f}",
          inference_s=f"{transcription_result.processing_time:.3f}",
          interval_wait_s=f"{interval_wait_duration:.3f}",
          total_loop_s=f"{time.monotonic() - loop_start:.3f}",
        )
      except Exception as exc:
        await self._handle_transcription_error(exc)
        break

    self.logger.info("Exiting speech to text thread")

  async def _get_next_audio_chunk(self) -> AudioChunk | None:
    """Get the next audio chunk, handling buffer management and minimum duration."""
    self.buffer.clip_if_stalled()
    input_bytes, duration, start_time = self.buffer.get_chunk_for_processing()

    if duration < self.buffer.config.min_chunk_duration:
      return await self._wait_for_min_chunk(input_bytes, duration, start_time)

    tracing_logger.info(
      "Processing audio chunk",
      recording_relative_chunk_start=f"{start_time:.2f}s",
      recording_relative_chunk_end=f"{start_time + duration:.2f}s",
      chunk_duration=f"{duration:.2f}s",
      buffer_available=f"{self.buffer.available_duration:.2f}s",
      buffer_total=f"{self.buffer.total_duration:.2f}s",
      buffer_processed=f"{self.buffer.processed_duration:.2f}s",
    )
    return AudioChunk(data=input_bytes, duration=duration, start_time=start_time)

  async def _wait_for_min_chunk(
    self, input_bytes: Float32Audio, duration: float, start_time: float
  ) -> AudioChunk | None:
    remaining_wait = self.buffer.config.min_chunk_duration - duration
    if not self._minimum_chunk_wait_logged:
      self.logger.info(
        "Transcription loop minimum chunk wait",
        buffered_duration_s=f"{duration:.3f}",
        required_duration_s=f"{self.buffer.config.min_chunk_duration:.3f}",
        wait_s=f"{remaining_wait:.3f}",
      )
      self._minimum_chunk_wait_logged = True

    observed_generation = self._current_utterance_generation()
    wait_completed = await self._wait_for_live_control_wakeup(
      remaining_wait,
      observed_generation=observed_generation,
    )
    pending_flush = self._pending_flush()
    if not wait_completed and pending_flush is not None and duration > 0:
      self.logger.info(
        "Pending flush interrupts minimum chunk wait",
        buffered_duration_s=f"{duration:.3f}s",
        boundary_sample=pending_flush.boundary_sample,
      )
      return AudioChunk(data=input_bytes, duration=duration, start_time=start_time)
    if not wait_completed and self._current_utterance_generation() != observed_generation:
      self.logger.info(
        "Cancelled utterance interrupts minimum chunk wait",
        buffered_duration_s=f"{duration:.3f}s",
        utterance_generation=observed_generation,
      )
    return None

  async def _transcribe_chunk(self, chunk: AudioChunk) -> ChunkTranscriptionResult:
    """Transcribe an audio chunk and return results with timing information."""
    if self._debug_capture:
      self._debug_capture.capture(chunk)

    transcription_start = time.monotonic()
    pass_result = await asyncio.to_thread(self._run_transcription_pass, chunk)
    processing_time = time.monotonic() - transcription_start

    if pass_result.status is TranscriptionPassStatus.INTERRUPTED_BEFORE_COMMIT:
      return pass_result

    self.logger.debug(
      "Transcription performance",
      audio_duration=f"{chunk.duration:.2f}s",
      transcription_time=f"{processing_time:.2f}s",
      speed_vs_realtime=f"{chunk.duration / processing_time:.1f}x faster",
    )
    pass_result.processing_time = processing_time
    return pass_result

  async def _process_transcription_result(self, result: ChunkTranscriptionResult) -> None:
    """Process transcription results and handle segments."""
    if result.status is TranscriptionPassStatus.INTERRUPTED_BEFORE_COMMIT:
      self.logger.info("Skipping interrupted transcription pass before commit")
      return

    active_generation = self._current_utterance_generation()
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

    pending_flush = self._pending_flush()
    flush_boundary_reached = pending_flush is not None and self._covers_flush_boundary(
      result, pending_flush
    )

    await self._handle_transcription_output(
      result.segments or [],
      result.audio_duration,
      result.speech_chunks,
      emit_result=pending_flush is None or flush_boundary_reached,
      flush_complete=flush_boundary_reached,
      force_complete=bool(
        flush_boundary_reached and pending_flush and pending_flush.force_complete
      ),
    )

    if flush_boundary_reached:
      self._discard_flushed_audio(pending_flush)
      if self.flush_state is not None:
        _ = self.flush_state.complete()

  def _discard_flushed_audio(self, pending_flush: PendingFlush | None) -> None:
    """Drop buffered audio through the accepted flush boundary after emitting the response."""
    if pending_flush is None:
      raise RuntimeError("Cannot discard flushed audio without a pending flush boundary")

    discarded_duration = self.buffer.discard_through_sample(pending_flush.boundary_sample)
    self.logger.info(
      "Discarded flushed audio",
      discarded_duration_s=f"{discarded_duration:.3f}",
      recording_relative_boundary_sample=pending_flush.boundary_sample,
    )

  async def _wait_for_next_interval(self, processing_time: float) -> None:
    """Wait to maintain consistent transcription intervals."""
    remaining_wait = self.buffer.config.transcription_interval - processing_time
    if remaining_wait <= 0:
      self.logger.warning("Transcription took longer than interval, proceeding immediately")
      return

    self.logger.debug(f"Waiting {remaining_wait:.2f}s to maintain transcription interval")
    observed_generation = self._current_utterance_generation()
    wait_completed = await self._wait_for_live_control_wakeup(
      remaining_wait,
      observed_generation=observed_generation,
    )
    if not wait_completed:
      if self._pending_flush() is not None:
        self.logger.info("Pending flush interrupts interval wait")
      elif self._current_utterance_generation() != observed_generation:
        self.logger.info(
          "Cancelled utterance interrupts interval wait",
          utterance_generation=observed_generation,
        )

  async def _handle_transcription_error(self, exc: Exception) -> None:
    """Log recognition failure, notify the client, and stop processing."""
    self.logger.exception("Failed to transcribe audio chunk")
    await self.sink.send_error(f"Transcription failed: {exc}")

  def _pending_flush(self) -> PendingFlush | None:
    if self.flush_state is None:
      return None
    return self.flush_state.pending()

  def _chunk_start_sample(self, chunk: AudioChunk) -> int:
    return int(round(chunk.start_time * self.buffer.config.sample_rate))

  def _consume_precommit_interrupt(self) -> bool:
    if self.flush_state is None or not self.flush_state.interrupt.is_set():
      return False
    _ = self.flush_state.clear_interrupt()
    self.logger.info("Live-session control interrupts worker pass before commit")
    return True

  def _current_utterance_generation(self) -> int:
    if self.flush_state is None:
      return 0
    return self.flush_state.current_generation()

  def _run_transcription_pass(self, chunk: AudioChunk) -> ChunkTranscriptionResult:
    """Run one worker-thread transcription pass or report interruption before commit."""
    utterance_generation = self._current_utterance_generation()
    if self._consume_precommit_interrupt():
      return ChunkTranscriptionResult(
        status=TranscriptionPassStatus.INTERRUPTED_BEFORE_COMMIT,
        chunk_start_sample=self._chunk_start_sample(chunk),
        chunk_sample_count=int(chunk.data.shape[0]),
        segments=None,
        info=None,
        processing_time=0.0,
        audio_duration=chunk.duration,
        speech_chunks=None,
        utterance_generation=utterance_generation,
        recording_id=self.recording_id,
      )

    result, info, speech_chunks = self._transcribe_audio(chunk)
    return ChunkTranscriptionResult(
      status=TranscriptionPassStatus.TRANSCRIBED,
      chunk_start_sample=self._chunk_start_sample(chunk),
      chunk_sample_count=int(chunk.data.shape[0]),
      segments=result,
      info=info,
      processing_time=0.0,
      audio_duration=chunk.duration,
      speech_chunks=speech_chunks,
      utterance_generation=utterance_generation,
      recording_id=self.recording_id,
    )

  def _covers_flush_boundary(
    self, result: ChunkTranscriptionResult, pending_flush: PendingFlush
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

  async def _wait_for_live_control_wakeup(
    self,
    delay: float,
    *,
    observed_generation: int,
  ) -> bool:
    """Sleep unless live-session control needs the loop to resume immediately."""
    if delay <= 0:
      return True
    if self.flush_state is None:
      await asyncio.sleep(delay)
      return True
    if not self.flush_state.begin_wait(observed_generation=observed_generation):
      return False

    sleep_task = asyncio.create_task(asyncio.sleep(delay))
    wake_task = asyncio.create_task(self.flush_state.wakeup.wait())
    try:
      done, pending = await asyncio.wait(
        {sleep_task, wake_task},
        return_when=asyncio.FIRST_COMPLETED,
      )
      for task in pending:
        _ = task.cancel()
      for task in pending:
        try:
          await task
        except asyncio.CancelledError:
          pass
      return sleep_task in done
    finally:
      if not sleep_task.done():
        _ = sleep_task.cancel()
      if not wake_task.done():
        _ = wake_task.cancel()

  def _transcribe_audio(
    self, chunk: AudioChunk
  ) -> tuple[list[Segment] | None, TranscriptionInfo, list[SpeechChunk] | None]:
    """Transcribe audio sample using Orukeet."""
    if not self.transcriber:
      raise RuntimeError("Transcriber not initialized")

    self.session.update_recording_context(self.recording_id)
    result, info = self.transcriber.transcribe(
      chunk.data, config=self.config, absolute_stream_start=chunk.start_time, session=self.session
    )
    result_list = list(result) if result else None
    speech_chunks = info.speech_chunks
    log_vad_analysis(chunk.data, chunk.duration, speech_chunks)
    return result_list, info, speech_chunks

  async def _handle_transcription_output(
    self,
    result: list[Segment],
    duration: float,
    speech_chunks: list[SpeechChunk] | None = None,
    *,
    emit_result: bool = True,
    flush_complete: bool = False,
    force_complete: bool = False,
  ) -> None:
    """Handle transcription output and send to client."""
    silence_threshold = self.config.silence_completion_threshold

    if not result and not flush_complete:
      self.logger.debug("No segments to send to client")
      self.buffer.advance_processed_boundary(duration)
      return

    if not result:
      recent_completed = list(
        self.session.most_recent_completed_segments(self.config.send_last_n_segments)
      )
      synthetic_segment = self._create_synthetic_incomplete_segment(
        speech_chunks, silence_threshold, duration
      )
      segments_for_client = [*recent_completed, synthetic_segment]
      await self._emit_segments(
        segments_for_client,
        emit_result=emit_result,
        flush_complete=True,
      )
      return

    log_generated_segments(result, duration)
    current_incomplete_segment = self._apply_segment_completion(
      result, speech_chunks, silence_threshold, duration, force_complete=force_complete
    )
    recent_completed = list(
      self.session.most_recent_completed_segments(self.config.send_last_n_segments)
    )
    tail_segment = current_incomplete_segment or self._create_synthetic_incomplete_segment(
      speech_chunks, silence_threshold, duration
    )
    segments_for_client = [*recent_completed, tail_segment]

    self._advance_buffer_by_completed_segments(
      speech_chunks, silence_threshold, duration, current_incomplete_segment
    )
    await self._emit_segments(
      segments_for_client,
      emit_result=emit_result,
      flush_complete=flush_complete,
    )

  def _apply_segment_completion(
    self,
    result: list[Segment],
    speech_chunks: list[SpeechChunk] | None,
    silence_threshold: float,
    duration: float,
    *,
    force_complete: bool,
  ) -> Segment | None:
    current_incomplete_segment: Segment | None = None
    for index, segment in enumerate(result):
      is_last = index == len(result) - 1
      should_complete = (
        self._should_complete_segment_by_silence(
          segment, speech_chunks, silence_threshold, duration
        )
        if is_last
        else True
      )
      if should_complete and not segment.completed:
        preceding_segment = self.session.get_last_completed_segment()
        segment.mark_completed(preceding_segment)
        self.session.add_completed_segment(segment)
        reason = "positional" if not is_last else "silence_analysis"
        tracing_logger.info(
          "Segment marked complete",
          segment_id=segment.id,
          reason=reason,
          text=segment.text[:50] + ("..." if len(segment.text) > 50 else ""),
          time_range=f"{segment.start:.2f}-{segment.end:.2f}s",
        )
      else:
        current_incomplete_segment = segment
        if is_last:
          tracing_logger.info(
            "Last segment remains incomplete",
            segment_id=segment.id,
            reason="insufficient_silence",
            text=segment.text[:50] + ("..." if len(segment.text) > 50 else ""),
            time_range=f"{segment.start:.2f}-{segment.end:.2f}s",
          )

    if force_complete and current_incomplete_segment is not None:
      preceding_segment = self.session.get_last_completed_segment()
      current_incomplete_segment.mark_completed(preceding_segment)
      self.session.add_completed_segment(current_incomplete_segment)
      current_incomplete_segment = None
      tracing_logger.info("Forced completion of tentative tail at explicit boundary")

    return current_incomplete_segment

  async def _emit_segments(
    self,
    segments_for_client: list[Segment],
    *,
    emit_result: bool,
    flush_complete: bool,
  ) -> None:
    if not emit_result:
      self.logger.debug("Suppressing ordinary emission while flush remains pending")
      return

    log_client_segments(segments_for_client)
    transcription_result = TranscriptionResult(
      segments=segments_for_client,
      language=self.language,
      recording_id=self.recording_id,
      flush_complete=flush_complete or None,
    )
    self.logger.debug("Sending segments to client", segments=segments_for_client)
    await self.sink.send_result(transcription_result)

  def _should_complete_segment_by_silence(
    self,
    segment: Segment,
    speech_chunks: list[SpeechChunk] | None,
    silence_threshold: float,
    audio_duration: float,
  ) -> bool:
    """Determine if segment should be completed based on silence after speech."""
    if not self.config.use_vad:
      return False

    if not speech_chunks:
      if segment.text.strip() and audio_duration >= silence_threshold:
        tracing_logger.info(
          "Completing segment due to silence chunk",
          segment_id=segment.id,
          silence_duration=f"{audio_duration:.2f}s",
          threshold=f"{silence_threshold:.2f}s",
        )
        return True
      return False

    chunk_start_recording_time = self.buffer.processed_up_to_time
    last_speech_chunk_time = self._last_speech_chunk_time(speech_chunks)
    last_speech_recording_time = chunk_start_recording_time + last_speech_chunk_time
    chunk_end_recording_time = chunk_start_recording_time + audio_duration
    silence_duration = chunk_end_recording_time - last_speech_recording_time

    if (
      segment.end <= last_speech_recording_time + silence_threshold
      and silence_duration >= silence_threshold
    ):
      tracing_logger.info(
        "Completing segment due to silence after speech",
        segment_id=segment.id,
        silence_duration=f"{silence_duration:.2f}s",
        last_speech_recording_time=f"{last_speech_recording_time:.2f}s",
        threshold=f"{silence_threshold:.2f}s",
      )
      return True

    return False

  def _last_speech_chunk_time(self, speech_chunks: list[SpeechChunk]) -> float:
    """Return the last VAD speech end as chunk-relative seconds."""
    sample_rate = self.buffer.config.sample_rate
    return max(chunk["end"] for chunk in speech_chunks) / sample_rate

  def _advance_buffer_by_completed_segments(
    self,
    speech_chunks: list[SpeechChunk] | None,
    silence_threshold: float,
    audio_duration: float,
    current_incomplete_segment: Segment | None,
  ) -> None:
    """Advance buffer pointer based on speech boundaries and completed segments."""
    last_completed = self.session.get_last_completed_segment()
    if not last_completed:
      return

    completed_end_time = last_completed.end
    current_processed_time = self.buffer.processed_up_to_time
    if completed_end_time > current_processed_time:
      advance_amount = completed_end_time - current_processed_time
      self.buffer.advance_processed_boundary(advance_amount)
      tracing_logger.info(
        "Buffer advanced by completed segment",
        advance_amount=f"{advance_amount:.2f}s",
        old_processed_time=f"{current_processed_time:.2f}s",
        new_processed_time=f"{completed_end_time:.2f}s",
        last_completed_id=last_completed.id,
      )

    if speech_chunks and current_incomplete_segment is None:
      last_speech_time = self._last_speech_chunk_time(speech_chunks)
      silence_after_speech = audio_duration - last_speech_time
      if silence_after_speech > silence_threshold * 2:
        additional_advance = silence_after_speech - silence_threshold
        self.buffer.advance_processed_boundary(additional_advance)
        tracing_logger.info(
          "Buffer advanced through silence",
          silence_duration=f"{silence_after_speech:.2f}s",
          additional_advance=f"{additional_advance:.2f}s",
          silence_threshold=f"{silence_threshold:.2f}s",
        )

  def _create_synthetic_incomplete_segment(
    self,
    speech_chunks: list[SpeechChunk] | None,
    silence_threshold: float,
    duration: float,
  ) -> Segment:
    """Create a synthetic incomplete segment to maintain client state machine invariant."""
    if speech_chunks:
      last_speech_time = self._last_speech_chunk_time(speech_chunks)
      synthetic_start = last_speech_time + silence_threshold
    else:
      synthetic_start = duration

    actual_start = min(synthetic_start, duration)
    recording_start = self.buffer.processed_up_to_time + actual_start
    return Segment(
      id=compute_segment_chain_id(0, ""),
      start=recording_start,
      end=recording_start,
      text="",
      words=None,
      completed=False,
    )
