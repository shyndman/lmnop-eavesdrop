"""Transcript completion, retained history, sink delivery, and buffer advancement."""

from __future__ import annotations

from typing import TypedDict

from structlog.stdlib import BoundLogger

from eavesdrop.common import get_logger
from eavesdrop.server.config import TranscriptionConfig
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.interfaces import TranscriptionResult, TranscriptionSink
from eavesdrop.server.transcription.models import SpeechSegment
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
    self._pending_segment: Segment | None = None
    self._undelivered_completed_count: int = 0

  def reset(self, recording_id: str | None, *, preserve_language: bool) -> None:
    """Clear recording-local completion state without replacing the session."""
    self.recording_id = recording_id
    self._pending_segment = None
    self._undelivered_completed_count = 0
    if not preserve_language:
      self.language = self.config.language

  @property
  def has_pending_segments(self) -> bool:
    """Return whether the single revisable timed segment awaits completion."""
    return self._pending_segment is not None

  async def flush_empty(self, *, force_complete: bool) -> None:
    """Deliver pending completions when a flush has no remaining audio."""
    await self.handle([], 0.0, flush_complete=True, force_complete=force_complete)

  async def handle(
    self,
    result: list[Segment],
    duration: float,
    speech_segments: list[SpeechSegment] | None = None,
    *,
    emit_result: bool = True,
    flush_complete: bool = False,
    force_complete: bool = False,
  ) -> None:
    """Revise one pending timed segment and deliver accepted completion work."""
    if not result:
      result = [self._pending_segment] if self._pending_segment is not None else []
    if not result and not flush_complete:
      if self.config.use_vad:
        self.buffer.advance_processed_boundary(duration)
      return
    if result and not self.config.use_vad:
      result = [self._combine_portions(result)]
    self._complete_segments(result, duration, speech_segments, force_complete)
    segments_for_client = list(
      self.session.most_recent_completed_segments(
        max(self.config.send_last_n_segments, self._undelivered_completed_count)
      )
    )
    segments_for_client.append(
      self._pending_segment
      if self._pending_segment is not None
      else self._create_synthetic_incomplete_segment(
        speech_segments, self.config.silence_completion_threshold, duration
      )
    )
    if self.config.use_vad or force_complete:
      self._advance_buffer_by_completed_segments(
        speech_segments,
        self.config.silence_completion_threshold,
        duration,
        self._pending_segment,
      )
    if emit_result:
      await self._send_result(segments_for_client, flush_complete)

  def _combine_portions(self, portions: list[Segment]) -> Segment:
    """Combine recognized portions without changing any recording word timestamp."""
    if len(portions) == 1:
      return portions[0]
    return Segment(
      id=portions[0].id,
      start=min(segment.start for segment in portions),
      end=max(segment.end for segment in portions),
      text=" ".join(segment.text for segment in portions),
      words=[word for segment in portions for word in segment.words],
      completed=False,
    )

  def _complete_segments(
    self,
    segments: list[Segment],
    duration: float,
    speech_segments: list[SpeechSegment] | None,
    force_complete: bool,
  ) -> None:
    """Retain the revisable tail while recording each newly completed segment once."""
    self._pending_segment = None
    for i, segment in enumerate(segments):
      should_complete = force_complete or (
        self.config.use_vad
        and (
          i < len(segments) - 1
          or self._should_complete_segment_by_silence(
            segment, speech_segments, self.config.silence_completion_threshold, duration
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
        self._pending_segment = segment

  async def _send_result(self, segments_for_client: list[Segment], flush_complete: bool) -> None:
    """Deliver history and the single revisable tail before releasing delivery accounting."""

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

  def _should_complete_segment_by_silence(
    self,
    segment: Segment,
    speech_segments: list[SpeechSegment] | None,
    silence_threshold: float,
    audio_duration: float,
  ) -> bool:
    """Determine if segment should be completed based on silence after speech."""
    if not speech_segments:
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
    last_speech_chunk_time = self._last_speech_end_time(speech_segments)
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

  def _last_speech_end_time(self, speech_segments: list[SpeechSegment]) -> float:
    """Return the last VAD speech end as chunk-relative seconds."""
    sample_rate = self.buffer.config.sample_rate
    return max(chunk["end"] for chunk in speech_segments) / sample_rate

  def _advance_buffer_by_completed_segments(
    self,
    speech_segments: list[SpeechSegment] | None,
    silence_threshold: float,
    audio_duration: float,
    current_incomplete_segment: Segment | None,
  ) -> None:
    """Advance buffer pointer based on speech boundaries and completed segments."""
    last_completed = self.session.get_last_completed_segment()

    if not last_completed:
      return

    # Native word ends can precede the VAD speech end. Once every segment in
    # this chunk is complete, consume its entire speech span so a later flush
    # cannot recognize the already-completed word's remaining audio again.
    current_processed_time = self.buffer.processed_up_to_time
    completed_end_time = last_completed.absolute_end_time
    if speech_segments and current_incomplete_segment is None:
      completed_end_time = max(
        completed_end_time,
        current_processed_time + self._last_speech_end_time(speech_segments),
      )

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

    if speech_segments and current_incomplete_segment is None:
      last_speech_time = self._last_speech_end_time(speech_segments)
      silence_after_speech = audio_duration - last_speech_time

      # This advance amount is chunk-relative by nature. Only apply it after the
      # recording-relative tail is closed, or the client can lose an open segment.
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
    speech_segments: list[SpeechSegment] | None,
    silence_threshold: float,
    duration: float,
  ) -> Segment:
    """Create a synthetic incomplete segment to maintain client state machine invariant."""
    from eavesdrop.wire.transcription import compute_segment_chain_id

    # Determine where the synthetic segment should start
    if speech_segments:
      # Start after the last detected speech + threshold
      last_speech_time = self._last_speech_end_time(speech_segments)
      synthetic_start = last_speech_time + silence_threshold
    else:
      # Without detected speech boundaries, place the empty tail at the chunk end.
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
