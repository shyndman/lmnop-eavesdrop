"""Transcript completion, retained history, sink delivery, and buffer advancement."""

from __future__ import annotations

from typing import TypedDict

from structlog.stdlib import BoundLogger

from eavesdrop.common import Pretty, Range, Seconds, get_logger
from eavesdrop.server.config import TranscriptionConfig
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.interfaces import TranscriptionResult, TranscriptionSink
from eavesdrop.server.transcription.models import SpeechChunk
from eavesdrop.server.transcription.session import TranscriptionSession
from eavesdrop.wire import Segment

tracing_logger = get_logger("tracing")


class TracingClientSegmentDict(TypedDict):
  id: int
  start: str
  end: str
  text: str
  completed: bool
  synthetic: bool


class TranscriptOutput:
  """Own completion accounting until results have actually reached the sink."""

  def __init__(
    self,
    buffer: AudioStreamBuffer,
    sink: TranscriptionSink,
    config: TranscriptionConfig,
    session: TranscriptionSession,
    logger: BoundLogger,
  ) -> None:
    self.buffer: AudioStreamBuffer = buffer
    self.sink: TranscriptionSink = sink
    self.config: TranscriptionConfig = config
    self.session: TranscriptionSession = session
    self.logger: BoundLogger = logger
    self.language: str | None = config.language
    self.recording_id: str | None = None
    self.text: list[str] = []
    self._pending_segments: list[Segment] = []
    self._undelivered_completed_count: int = 0

  def reset(self, recording_id: str | None, *, preserve_language: bool) -> None:
    """Clear recording-local completion state without replacing the session."""
    self.recording_id = recording_id
    self.text = []
    self._pending_segments = []
    self._undelivered_completed_count = 0
    if not preserve_language:
      self.language = self.config.language

  @property
  def has_pending_segments(self) -> bool:
    """Return whether provisional segments await completion."""
    return bool(self._pending_segments)

  async def flush_empty(self, *, force_complete: bool) -> None:
    """Deliver pending completions when a flush has no remaining audio."""
    await self.handle(
      self._pending_segments.copy(),
      0.0,
      flush_complete=True,
      force_complete=force_complete,
    )

  async def handle(
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
