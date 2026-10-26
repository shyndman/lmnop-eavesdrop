"""Stage-specific timing, structured logging, and Langfuse observations."""

import time
from collections.abc import Iterable
from contextlib import AbstractContextManager
from types import TracebackType
from typing import TYPE_CHECKING, Self, override

from eavesdrop.server.langfuse_tracing import ServerObservation
from eavesdrop.server.transcription.models import SpeechChunk
from eavesdrop.server.transcription.tracing.context import TracingContext

if TYPE_CHECKING:
  from eavesdrop.wire.transcription import Segment


class BaseTracer:
  """Base class for stage-specific tracers that handle timing and logging."""

  def __init__(self, session: TracingContext, stage_name: str):
    """Initialize tracer with session and stage name.

    :param session: The shared transcription tracing context to update.
    :type session: TracingContext
    :param stage_name: Name of the processing stage.
    :type stage_name: str
    """
    self.session: TracingContext = session
    self.stage_name: str = stage_name
    self.start_time: float = 0.0
    self.langfuse_context: AbstractContextManager[ServerObservation | None] | None = None
    self.langfuse_observation: ServerObservation | None = None

  def __enter__(self) -> Self:
    """Start timing the stage."""
    self.start_time = time.perf_counter()
    self.langfuse_context = self.session.start_langfuse_observation(self.stage_name)
    self.langfuse_observation = self.langfuse_context.__enter__()
    return self

  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc_val: BaseException | None,
    exc_tb: TracebackType | None,
  ) -> None:
    """Stop timing and record the duration."""
    elapsed = time.perf_counter() - self.start_time
    # Record timing even on exceptions for debugging purposes
    self.session.stage_timings[self.stage_name] = elapsed if exc_type is None else -1.0
    if self.langfuse_observation is not None:
      metadata: dict[str, object] = {
        "duration_ms": elapsed * 1000,
        "failed": exc_type is not None,
      }
      if self.stage_name == "total_pipeline":
        metadata["stage_timings_ms"] = {
          stage_name: timing * 1000
          for stage_name, timing in self.session.stage_timings.items()
          if timing >= 0
        }
        metadata["time_range"] = self.session.format_time_range()
      if exc_val is not None:
        metadata["error"] = str(exc_val)
      _ = self.langfuse_observation.update(
        metadata=metadata,
        level="ERROR" if exc_type is not None else None,
        status_message=str(exc_val) if exc_val is not None else None,
      )
    if self.langfuse_context is not None:
      _ = self.langfuse_context.__exit__(exc_type, exc_val, exc_tb)


class VadStageTracer(BaseTracer):
  """Tracer for VAD audio preprocessing stage."""

  def __call__(
    self,
    speech_chunks: list[SpeechChunk] | None,
    sample_rate: int,
    total_samples: int,
  ) -> None:
    """Record VAD results and generate visualization.

    :param speech_chunks: Speech chunks detected by VAD, or None if no VAD.
    :type speech_chunks: list[SpeechChunk] | None
    :param sample_rate: Audio sample rate for time calculations.
    :type sample_rate: int
    :param total_samples: Total number of audio samples processed.
    :type total_samples: int
    """
    audio_duration = total_samples / sample_rate
    time_start, time_end = self.session.get_recording_time_range()

    if speech_chunks:
      speech_duration = sum(
        (chunk["end"] - chunk["start"]) / sample_rate for chunk in speech_chunks
      )
      silence_duration = audio_duration - speech_duration
      vad_viz = self.session.format_vad_visualization(speech_chunks, sample_rate, total_samples)

      self.session.logger.info(
        "Voice detected!",
        t_start=time_start,
        t_end=time_end,
        chunk_count=len(speech_chunks),
        total=audio_duration,
        speech=speech_duration,
        silence=silence_duration,
        vad=vad_viz,
      )
      if self.langfuse_observation is not None:
        _ = self.langfuse_observation.update(
          output={
            "speech_chunks": speech_chunks,
            "speech_chunk_count": len(speech_chunks),
            "speech_duration_seconds": speech_duration,
            "silence_duration_seconds": silence_duration,
            "audio_duration_seconds": audio_duration,
            "vad_visualization": vad_viz,
          }
        )
      return

    if speech_chunks == []:
      self.session.logger.info(
        "No voice activity",
        t_start=time_start,
        t_end=time_end,
        total=audio_duration,
      )
      if self.langfuse_observation is not None:
        _ = self.langfuse_observation.update(
          output={
            "speech_chunks": [],
            "speech_chunk_count": 0,
            "speech_duration_seconds": 0.0,
            "silence_duration_seconds": audio_duration,
            "audio_duration_seconds": audio_duration,
          }
        )
      return

    self.session.logger.info(
      "VAD not applied",
      t_start=time_start,
      t_end=time_end,
      total=audio_duration,
    )
    if self.langfuse_observation is not None:
      _ = self.langfuse_observation.update(
        output={
          "speech_chunks": None,
          "audio_duration_seconds": audio_duration,
          "vad_applied": False,
        }
      )


class FeatureStageTracer(BaseTracer):
  """Tracer for feature extraction stage."""

  def __call__(self) -> None:
    """Record feature extraction completion (no specific data to log)."""
    if self.langfuse_observation is not None:
      _ = self.langfuse_observation.update(output={"features_extracted": True})


class InferenceStageTracer(BaseTracer):
  """Tracer for model inference stage."""

  def __call__(self) -> None:
    """Record completion of model inference."""
    if self.langfuse_observation is not None:
      _ = self.langfuse_observation.update(output={"inference_completed": True})


class PipelineTracer(BaseTracer):
  """Tracer for the entire transcription pipeline."""

  @override
  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc_val: BaseException | None,
    exc_tb: TracebackType | None,
  ) -> None:
    """Stop timing and log the complete pipeline summary."""
    # Call parent to record timing.
    super().__exit__(exc_type, exc_val, exc_tb)

    # Format stage timings with "stage_" prefix and millisecond formatting
    stage_fields: dict[str, str] = {}
    for stage_name, timing in self.session.stage_timings.items():
      if timing >= 0:
        stage_fields[f"stage_{stage_name}"] = f"{timing * 1000:.3f}ms"
      else:
        stage_fields[f"stage_{stage_name}"] = "FAILED"

    # Add generation metadata if available
    extra_fields: dict[str, int | str] = {}

    time_range = self.session.format_time_range()

    self.session.logger.info(
      "Pipeline completed",
      time_range=time_range,
      **stage_fields,
      **extra_fields,
    )

  def __call__(self) -> None:
    """No-op for pipeline tracer - summary is logged in __exit__."""
    return


class SegmentStageTracer(BaseTracer):
  """Tracer for segment processing stage."""

  def __call__(self, segments: Iterable["Segment"]) -> None:
    """Record final segment results and log summary.

    :param segments: Final transcribed segments.
    :type segments: Iterable[Segment]
    """
    # Convert to list for logging analysis
    segments_list = list(segments)
    segment_count = len(segments_list)
    total_text_chars = sum(len(seg.text) for seg in segments_list)

    # Log segment generation results
    self.session.logger.info(
      "Transcription segments generated",
      segment_count=segment_count,
      total_text_chars=total_text_chars,
      avg_chars_per_segment=total_text_chars / segment_count if segment_count > 0 else 0,
    )
    if self.langfuse_observation is not None:
      _ = self.langfuse_observation.update(
        output={
          "segment_count": segment_count,
          "total_text_chars": total_text_chars,
          "segments": [
            {
              "id": segment.id,
              "start": segment.start,
              "end": segment.end,
              "text": segment.text,
              "completed": segment.completed,
            }
            for segment in segments_list
          ],
        }
      )

    # Log comprehensive pipeline summary with structured timing data
    self.session.logger.info(
      "Pipeline completed",
      **self.session.stage_timings,
    )
