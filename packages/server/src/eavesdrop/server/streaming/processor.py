"""
Streaming transcription processor with integrated speech recognition transcriber.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from structlog.stdlib import BoundLogger

from eavesdrop.common import Pretty, Range, Seconds, get_logger
from eavesdrop.server.config import TranscriptionConfig
from eavesdrop.server.constants import SINGLE_MODEL
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.flush_state import LiveSessionFlushState, PendingFlush
from eavesdrop.server.streaming.interfaces import TranscriptionResult, TranscriptionSink
from eavesdrop.server.transcription.models import (
  SpeechChunk,
  TranscriptionInfo,
  VadParameters,
)
from eavesdrop.server.transcription.orukeet import OrukeetTranscriber
from eavesdrop.server.transcription.session import TranscriptionSession
from eavesdrop.wire import Segment

tracing_logger = get_logger("tracing")

Float32Audio = NDArray[np.float32]


class ClientSegmentDict(TypedDict):
  id: str
  start: str
  end: str
  text: str
  completed: bool


class FormattedSegmentDict(TypedDict):
  start: str
  end: str
  text: str
  completed: bool


class TracingClientSegmentDict(TypedDict):
  id: int
  start: str
  end: str
  text: str
  completed: bool
  synthetic: bool


@dataclass
class AudioChunk:
  """Audio data with associated metadata for processing.

  ``start_time`` is a buffer offset. In live WebSocket mode recording boundaries
  reset the buffer, so the value is recording-relative for the active recording.
  """

  data: Float32Audio
  duration: float
  start_time: float


@dataclass
class ChunkTranscriptionResult:
  """Result of transcribing an audio chunk."""

  status: "TranscriptionPassStatus"
  # Absolute sample indexes are in the buffer timeline. In live WebSocket mode,
  # recording boundaries reset the buffer, so these are recording-relative samples.
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
  """
  Orchestrates streaming transcription processing with integrated speech recognition.

  This class combines audio buffering, transcription processing, and result
  delivery by combining an AudioStreamBuffer with integrated speech recognition
  model management.
  """

  # Class variable for single model mode
  _models: dict[tuple[str, str, bool], OrukeetTranscriber] = {}
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

    # Transcription state
    self.exit: bool = False
    self._minimum_chunk_wait_logged: bool = False
    self.language: str | None = config.language
    self.text: list[str] = []
    self.recording_id: str | None = None
    self._pending_segments: list[Segment] = []
    self._undelivered_completed_count: int = 0
    self._idle: asyncio.Event = asyncio.Event()
    self._idle.set()

    # Model attributes
    self.transcriber: OrukeetTranscriber | None = None
    self.vad_parameters: VadParameters = config.vad_parameters

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
    key = (self.config.model, self.config.device, self.config.use_vad)
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
    """Reset processor-local recording state after a new epoch or cancellation.

    A recording boundary resets the upstream buffer/session to recording-relative
    zero. Utterance cancellation preserves that recording-relative base while
    discarding only the unprocessed tail.
    """
    self.recording_id = recording_id
    self.text = []
    self._pending_segments = []
    self._undelivered_completed_count = 0
    if not preserve_language:
      self.language = self.config.language

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
    return bool(self._pending_segments)

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

        if transcription_result.status is TranscriptionPassStatus.INTERRUPTED_BEFORE_COMMIT:
          continue

        interval_wait_start = time.monotonic()
        await self._wait_for_next_interval(transcription_result.processing_time)
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

  async def _get_next_audio_chunk(self) -> AudioChunk | None:
    """Get the next audio chunk, handling buffer management and minimum duration."""
    pending_flush = self._pending_flush()
    if self.config.use_vad and pending_flush is None:
      self.buffer.clip_if_stalled()
    input_bytes, duration, start_time = self.buffer.get_chunk_for_processing()
    if pending_flush is not None:
      boundary_offset = pending_flush.boundary_sample - int(
        round(start_time * self.buffer.config.sample_rate)
      )
      input_bytes = input_bytes[: max(0, boundary_offset)]
      duration = input_bytes.shape[0] / self.buffer.config.sample_rate
    if pending_flush is not None and duration == 0:
      await self._handle_transcription_output(
        self._pending_segments.copy(),
        0.0,
        flush_complete=True,
        force_complete=pending_flush.force_complete,
      )
      if self.flush_state is not None:
        _ = self.flush_state.complete()
      return None

    if pending_flush is not None and duration > 0:
      return AudioChunk(data=input_bytes, duration=duration, start_time=start_time)

    # Debug logging to track buffer catchup patterns
    available_duration = self.buffer.available_duration
    total_duration = self.buffer.total_duration
    processed_duration = self.buffer.processed_duration

    self.logger.debug(
      "Buffer status before transcription",
      available_for_processing=f"{available_duration:.2f}s",
      total_buffered=f"{total_duration:.2f}s",
      already_processed=f"{processed_duration:.2f}s",
      is_caught_up=available_duration < 0.5,
    )

    if duration < self.buffer.config.min_chunk_duration:
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

    # TRACING: Audio chunk range being processed
    tracing_logger.info("\n\n")
    tracing_logger.info(
      "Processing audio chunk",
      recording_relative_chunk_start=f"{start_time:.2f}s",
      recording_relative_chunk_end=f"{start_time + duration:.2f}s",
      chunk_duration=f"{duration:.2f}s",
      buffer_available=f"{available_duration:.2f}s",
      buffer_total=f"{total_duration:.2f}s",
      buffer_processed=f"{processed_duration:.2f}s",
    )

    return AudioChunk(data=input_bytes, duration=duration, start_time=start_time)

  async def _transcribe_chunk(self, chunk: AudioChunk) -> ChunkTranscriptionResult:
    """Transcribe an audio chunk and return results with timing information."""
    transcription_start = time.monotonic()
    worker = asyncio.create_task(
      asyncio.to_thread(
        self._run_transcription_pass,
        chunk,
        self._current_utterance_generation(),
        self.recording_id,
        self.session,
      )
    )
    try:
      pass_result = await asyncio.shield(worker)
    except asyncio.CancelledError:
      # Cancelling to_thread does not stop native inference. Await it before
      # releasing connection/session ownership, but never commit its output.
      await worker
      raise
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
    """Process transcription results, update language, and handle segments."""
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
    force_complete = bool(flush_boundary_reached and pending_flush and pending_flush.force_complete)

    await self._handle_transcription_output(
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

  async def _wait_for_next_interval(self, processing_time: float) -> None:
    """Wait to maintain consistent transcription intervals."""
    remaining_wait = self.buffer.config.transcription_interval - processing_time
    if remaining_wait > 0:
      self.logger.debug(f"Waiting {remaining_wait:.2f}s to maintain transcription interval")
      observed_generation = self._current_utterance_generation()
      wait_completed = await self._wait_for_live_control_wakeup(
        remaining_wait,
        observed_generation=observed_generation,
      )
      if not wait_completed:
        pending_flush = self._pending_flush()
        if pending_flush is not None:
          self.logger.info("Pending flush interrupts interval wait")
        elif self._current_utterance_generation() != observed_generation:
          self.logger.info(
            "Cancelled utterance interrupts interval wait",
            utterance_generation=observed_generation,
          )
    else:
      self.logger.warning("Transcription took longer than interval, proceeding immediately")

  async def _handle_transcription_error(self) -> None:
    """Handle transcription errors with logging and recovery."""
    self.logger.exception("Failed to transcribe audio chunk")
    await asyncio.sleep(self.buffer.config.transcription_interval)

  def _pending_flush(self) -> PendingFlush | None:
    """Return the currently pending flush request, if any."""
    if self.flush_state is None:
      return None
    return self.flush_state.pending()

  def _chunk_start_sample(self, chunk: AudioChunk) -> int:
    """Convert a chunk's buffer start offset to an absolute buffer sample index.

    Live WebSocket recording boundaries reset the buffer, so this sample index
    is recording-relative in live mode.
    """
    return int(round(chunk.start_time * self.buffer.config.sample_rate))

  def _consume_precommit_interrupt(self) -> bool:
    """Consume one live-session control interrupt before expensive work begins."""
    if self.flush_state is None or not self.flush_state.interrupt.is_set():
      return False
    _ = self.flush_state.clear_interrupt()
    self.logger.info("Live-session control interrupts worker pass before commit")
    return True

  def _current_utterance_generation(self) -> int:
    """Return the active utterance generation for the live session."""
    if self.flush_state is None:
      return 0
    return self.flush_state.current_generation()

  def _run_transcription_pass(
    self,
    chunk: AudioChunk,
    utterance_generation: int,
    recording_id: str | None,
    session: TranscriptionSession,
  ) -> ChunkTranscriptionResult:
    """Run one worker-thread pass with identity captured before scheduling."""
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
        recording_id=recording_id,
      )

    result, info, speech_chunks = self._transcribe_audio(chunk, session, recording_id)
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
      recording_id=recording_id,
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
      _ = await asyncio.gather(sleep_task, wake_task, return_exceptions=True)

  def _transcribe_audio(
    self, chunk: AudioChunk, session: TranscriptionSession, recording_id: str | None
  ) -> tuple[list[Segment] | None, TranscriptionInfo, list[SpeechChunk] | None]:
    """Transcribe audio sample using the speech recognition model."""
    if not self.transcriber:
      raise RuntimeError("Transcriber not initialized")

    session.update_recording_context(recording_id)

    result, info = self.transcriber.transcribe(
      chunk.data, config=self.config, absolute_stream_start=chunk.start_time, session=session
    )
    self.logger.debug(
      "Returned from transcriber.transcribe",
      info_present=True,
      speech_chunk_count=len(info.speech_chunks) if info.speech_chunks else 0,
    )
    result_list = list(result) if result else None
    self.logger.debug(
      "Materialized transcriber result",
      segment_count=len(result_list) if result_list else 0,
    )

    # Store VAD speech chunks for silence analysis
    speech_chunks = info.speech_chunks

    # TRACING: Audio characteristics analysis
    sample_rate = 16000  # Required sample rate
    rms = np.sqrt(np.mean(chunk.data**2)) if len(chunk.data) > 0 else 0.0
    peak = np.max(np.abs(chunk.data)) if len(chunk.data) > 0 else 0.0
    zero_crossings = np.sum(np.diff(np.signbit(chunk.data))) if len(chunk.data) > 1 else 0
    zcr = zero_crossings / len(chunk.data) * sample_rate if len(chunk.data) > 0 else 0

    # TRACING: VAD speech detection results
    if speech_chunks:
      speech_ranges = [
        f"{chunk['start'] / sample_rate:.2f}-{chunk['end'] / sample_rate:.2f}s"
        for chunk in speech_chunks
      ]
      total_speech_duration = sum(
        (chunk["end"] - chunk["start"]) / sample_rate for chunk in speech_chunks
      )

      # Build timeline showing speech vs silence (S=speech, ~=silence)
      timeline = ["~"] * int(chunk.duration * 10)  # 100ms resolution
      for speech_chunk in speech_chunks:
        start_idx = int((speech_chunk["start"] / sample_rate) * 10)
        end_idx = int((speech_chunk["end"] / sample_rate) * 10)
        for i in range(start_idx, min(end_idx, len(timeline))):
          timeline[i] = "S"
      timeline_str = "".join(timeline)

      tracing_logger.info(
        "VAD analysis",
        speech_ranges=speech_ranges,
        speech_duration=f"{total_speech_duration:.2f}s",
        silence_duration=f"{chunk.duration - total_speech_duration:.2f}s",
        audio_duration=f"{chunk.duration:.2f}s",
        speech_ratio=f"{total_speech_duration / chunk.duration:.1%}",
        timeline=timeline_str,
        audio_rms=f"{rms:.6f}",
        audio_peak=f"{peak:.6f}",
        audio_zcr=f"{zcr:.1f}",
      )
    else:
      tracing_logger.info(
        "VAD analysis",
        speech_ranges=[],
        speech_duration="0.00s",
        silence_duration=f"{chunk.duration:.2f}s",
        audio_duration=f"{chunk.duration:.2f}s",
        speech_ratio="0.0%",
        timeline="~" * int(chunk.duration * 10),
        audio_rms=f"{rms:.6f}",
        audio_peak=f"{peak:.6f}",
        audio_zcr=f"{zcr:.1f}",
      )

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
    result_count = len(result) if result else 0
    self.logger.debug(
      "Handling transcription output",
      duration=f"{duration:.2f}s",
      segments=result_count,
    )

    if not result and not flush_complete:
      self.logger.debug("No segments to send to client")
      if self.config.use_vad:
        self.buffer.advance_processed_boundary(duration)
      return

    if not result:
      result = self._pending_segments.copy()

    if result:
      # TRACING: Raw speech recognition segments generated
      raw_segments = [
        {
          "id": seg.id,
          "timespan": Range(Seconds(seg.start), Seconds(seg.end)),
          "text": seg.text[:50] + ("..." if len(seg.text) > 50 else ""),
          "completed": seg.completed,
        }
        for seg in result
      ]
      tracing_logger.info(
        "speech recognition generated segments",
        segment_count=len(result),
        segments=Pretty(raw_segments),
        audio_duration=Seconds(duration),
      )

    self.logger.debug("Processing transcription segments")
    # Update the internal transcript.
    self._update_segments(result, duration)

    current_incomplete_segment: Segment | None = None
    silence_threshold = self.config.silence_completion_threshold
    incomplete_segments: list[Segment] = []

    for i, segment in enumerate(result):
      should_complete = force_complete or (
        self.config.use_vad
        and (
          i < len(result) - 1
          or self._should_complete_segment_by_silence(
            segment, speech_chunks, silence_threshold, duration
          )
        )
      )
      if should_complete:
        if not segment.completed:
          segment.mark_completed(self.session.get_last_completed_segment())
          self.session.add_completed_segment(segment)
          self._undelivered_completed_count += 1
      else:
        segment.completed = False
        incomplete_segments.append(segment)

    self._pending_segments = incomplete_segments
    if incomplete_segments:
      current_incomplete_segment = incomplete_segments[0]

    if not result and self.config.use_vad:
      self.buffer.advance_processed_boundary(duration)

    # Keep the history window without truncating completions not yet delivered.
    segments_for_client: list[Segment] = []

    recent_completed_segments: list[Segment] = list(
      self.session.most_recent_completed_segments(
        max(self.config.send_last_n_segments, self._undelivered_completed_count)
      )
    )
    segments_for_client.extend(recent_completed_segments)

    if incomplete_segments:
      segments_for_client.extend(incomplete_segments)
    else:
      # Create synthetic incomplete segment to maintain client state machine invariant
      synthetic_segment = self._create_synthetic_incomplete_segment(
        speech_chunks, silence_threshold, duration
      )
      segments_for_client.append(synthetic_segment)

    if self.config.use_vad or force_complete:
      self._advance_buffer_by_completed_segments(
        speech_chunks, silence_threshold, duration, current_incomplete_segment
      )

    if not emit_result:
      self.logger.debug("Suppressing ordinary emission while flush remains pending")
      return

    # TRACING: Final client output
    client_segments: list[TracingClientSegmentDict] = [
      {
        "id": seg.id,
        "start": f"{seg.start:.2f}s",
        "end": f"{seg.end:.2f}s",
        "text": seg.text[:30] + ("..." if len(seg.text) > 30 else ""),
        "completed": seg.completed,
        "synthetic": seg.text == "" and seg.start == seg.end,
      }
      for seg in segments_for_client
    ]
    completed_count = sum(1 for seg in segments_for_client if seg.completed)
    incomplete_count = len(segments_for_client) - completed_count

    tracing_logger.info(
      "Sending segments to client",
      total_segments=len(segments_for_client),
      completed_segments=completed_count,
      incomplete_segments=incomplete_count,
      segments=client_segments,
    )

    # Send rich Segment objects to client
    transcription_result = TranscriptionResult(
      segments=segments_for_client,
      language=self.language,
      recording_id=self.recording_id,
      flush_complete=flush_complete or None,
    )
    self.logger.debug("Sending segments to client", segments=segments_for_client)
    await self.sink.send_result(transcription_result)
    self._undelivered_completed_count = 0

  def _update_segments(self, segments: list[Segment], _duration: float) -> None:
    """Process segments and update transcript."""
    # Update the internal transcript.
    for segment in segments:
      if segment.completed and segment.text.strip():
        self.text.append(segment.text)

  def _prepare_segments(
    self, last_segment: ClientSegmentDict | None = None
  ) -> list[ClientSegmentDict]:
    """Prepare segments for client using session completed segments."""
    segments: list[ClientSegmentDict] = []

    if self.session and self.session.completed_segments:
      # Convert Segment objects to dict format for client
      completed_dicts: list[ClientSegmentDict] = []
      for seg in self.session.completed_segments:
        # Compatibility helpers expose the same canonical recording-relative
        # segment times as ``seg.start``/``seg.end`` for completed segments.
        completed_dicts.append(
          {
            "id": str(seg.id),
            "start": "{:.3f}".format(seg.absolute_start_time),
            "end": "{:.3f}".format(seg.absolute_end_time),
            "text": seg.text,
            "completed": seg.completed,
          }
        )

      # Apply send_last_n_segments limit
      if len(completed_dicts) >= self.config.send_last_n_segments:
        segments = completed_dicts[-self.config.send_last_n_segments :].copy()
      else:
        segments = completed_dicts.copy()

    if last_segment is not None:
      segments = segments + [last_segment]

    return segments

  def _should_complete_segment_by_silence(
    self,
    segment: Segment,
    speech_chunks: list[SpeechChunk] | None,
    silence_threshold: float,
    audio_duration: float,
  ) -> bool:
    """Determine if segment should be completed based on silence after speech."""
    if not speech_chunks:
      # No speech in current chunk - check if we have a segment from previous speech
      # If the segment exists and we have sufficient silence duration, complete it
      if segment.text.strip() and audio_duration >= silence_threshold:
        tracing_logger.info(
          "Completing segment due to silence chunk",
          segment_id=segment.id,
          silence_duration=f"{audio_duration:.2f}s",
          threshold=f"{silence_threshold:.2f}s",
        )
        return True
      return False

    # VAD speaks chunk-relative samples. Segment timestamps are recording-relative;
    # convert before comparing because recording time is the canonical timeline.
    chunk_start_recording_time = self.buffer.processed_up_to_time
    last_speech_chunk_time = self._last_speech_chunk_time(speech_chunks)
    last_speech_recording_time = chunk_start_recording_time + last_speech_chunk_time

    # Time since last speech in this audio chunk.
    chunk_end_recording_time = chunk_start_recording_time + audio_duration
    silence_duration = chunk_end_recording_time - last_speech_recording_time

    # If we have enough silence after the segment ends, mark it complete
    if (
      segment.absolute_end_time <= last_speech_recording_time + silence_threshold
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
    # Get the last completed segment to determine how far we can advance
    last_completed = self.session.get_last_completed_segment()

    if not last_completed:
      return

    # Native word ends can precede the VAD speech end. Once every segment in
    # this chunk is complete, consume its entire speech span so a later flush
    # cannot recognize the already-completed word's remaining audio again.
    current_processed_time = self.buffer.processed_up_to_time
    completed_end_time = last_completed.absolute_end_time
    if speech_chunks and current_incomplete_segment is None:
      completed_end_time = max(
        completed_end_time,
        current_processed_time + self._last_speech_chunk_time(speech_chunks),
      )

    # Only advance if we're moving forward
    if completed_end_time > current_processed_time:
      advance_amount = completed_end_time - current_processed_time
      self.buffer.advance_processed_boundary(advance_amount)

      # TRACING: Buffer pointer advancement
      tracing_logger.info(
        "Buffer advanced by completed segment",
        advance_amount=f"{advance_amount:.2f}s",
        old_processed_time=f"{current_processed_time:.2f}s",
        new_processed_time=f"{completed_end_time:.2f}s",
        last_completed_id=last_completed.id,
      )

    # Additionally, if we have pure silence after completion, advance through it
    if speech_chunks and current_incomplete_segment is None:
      last_speech_time = self._last_speech_chunk_time(speech_chunks)
      chunk_end_time = audio_duration
      silence_after_speech = chunk_end_time - last_speech_time

      # This advance amount is chunk-relative by nature. Only apply it after the
      # recording-relative tail is closed, or the client can lose an open segment.
      if silence_after_speech > silence_threshold * 2:
        additional_advance = silence_after_speech - silence_threshold
        self.buffer.advance_processed_boundary(additional_advance)

        # TRACING: Additional silence advancement
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
    from eavesdrop.wire.transcription import compute_segment_chain_id

    # Determine where the synthetic segment should start
    if speech_chunks:
      # Start after the last detected speech + threshold
      last_speech_time = self._last_speech_chunk_time(speech_chunks)
      synthetic_start = last_speech_time + silence_threshold
    else:
      # No speech detected, start at current buffer position
      synthetic_start = duration

    # Always create to maintain client invariant - place at end if necessary
    actual_start = min(synthetic_start, duration)
    recording_start = self.buffer.processed_up_to_time + actual_start

    return Segment(
      id=compute_segment_chain_id(0, ""),  # Baseline ID for incomplete
      start=recording_start,
      end=recording_start,  # Zero-duration synthetic segment
      text="",  # Empty text is fine for client state machine
      words=[],
      completed=False,
      time_offset=0.0,
    )

  def _format_segment(
    self, start: float, end: float, text: str, completed: bool = False
  ) -> FormattedSegmentDict:
    """Format a transcription segment."""
    return {
      "start": "{:.3f}".format(start),
      "end": "{:.3f}".format(end),
      "text": text,
      "completed": completed,
    }
