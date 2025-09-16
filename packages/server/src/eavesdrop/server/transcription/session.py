"""Transcription session context for timing and logging coordination.

This module provides session context that bridges client connection timing
with audio buffer timing to enable comprehensive transcription logging.
"""

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from structlog.stdlib import BoundLogger

from eavesdrop.common import get_logger
from eavesdrop.server.transcription.models import SpeechChunk

if TYPE_CHECKING:
  from eavesdrop.wire.transcription import Segment

# Type alias for stage timing measurements
type StageTimings = dict[str, float]


class TracerProtocol(Protocol):
  """Protocol for stage-specific tracers with context manager and callable interfaces.

  Tracers automatically time their associated stage using context manager protocol
  and can be called with stage-specific data to perform logging or other actions.

  :example:

    with session.trace_vad_stage() as tracer:
      # Processing happens here - timing is automatic
      result = process_audio(audio)
      # Call tracer with results for logging
      tracer(result.speech_chunks, sample_rate, total_samples)
  """

  def __enter__(self) -> "TracerProtocol":
    """Enter context manager and start timing the stage.

    :returns: The tracer instance for method chaining.
    :rtype: TracerProtocol
    """
    ...

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    """Exit context manager and record elapsed time.

    :param exc_type: Exception type if an exception occurred.
    :param exc_val: Exception value if an exception occurred.
    :param exc_tb: Exception traceback if an exception occurred.
    """
    ...

  def __call__(self, *args, **kwargs) -> None:
    """Process stage-specific results and perform logging or other actions.

    The signature and behavior varies by tracer type - see specific tracer
    implementations for details.

    :param args: Stage-specific positional arguments.
    :param kwargs: Stage-specific keyword arguments.
    """
    ...


class TranscriptionSessionProtocol(Protocol):
  """Protocol defining the interface for transcription sessions.

  Transcription sessions coordinate timing and logging across the entire pipeline,
  bridging client connection timing with audio buffer timing. They provide stage
  tracers for measurement and structured logging of pipeline execution.

  Sessions support both real implementations (TranscriptionSession) and no-op
  implementations (NoopSession) to eliminate conditional logic in pipeline code.
  """

  def get_absolute_time_range(self) -> tuple[float, float]:
    """Get the absolute time range of the current audio chunk.

    "Absolute" means relative to the client connection start time, not relative
    to the current buffer or chunk. This gives the timeline position within
    the overall transcription session.

    :returns: Tuple of (start_time, end_time) in seconds since connection.
    :rtype: tuple[float, float]
    """
    ...

  def format_time_range(self) -> str:
    """Format the current time range as a human-readable string.

    :returns: Formatted string like "+3.45s to +7.20s".
    :rtype: str
    """
    ...

  def format_vad_visualization(
    self, speech_chunks: list[SpeechChunk] | None, sample_rate: int, total_samples: int
  ) -> str:
    """Format VAD results as ASCII art visualization.

    Creates a visualization showing speech/silence patterns like:
    +3.45s|-1.20s-|~0.80s~|-0.50s-|+7.20s

    :param speech_chunks: Speech chunks from VAD detection, or None if no VAD applied.
    :type speech_chunks: list[SpeechChunk] | None
    :param sample_rate: Audio sample rate for time calculations.
    :type sample_rate: int
    :param total_samples: Total number of audio samples.
    :type total_samples: int
    :returns: ASCII art visualization of speech/silence patterns.
    :rtype: str
    """
    ...

  def update_audio_context(self, start_offset: float, duration: float) -> None:
    """Update the audio timing context for the current chunk.

    :param start_offset: Start time offset from connection start in seconds.
    :type start_offset: float
    :param duration: Duration of the audio chunk in seconds.
    :type duration: float
    """
    ...

  def record_generation_result(self, attempts: int, final_temp: float) -> None:
    """Record the results of temperature fallback generation.

    :param attempts: Number of temperature attempts made during generation.
    :type attempts: int
    :param final_temp: Final temperature that succeeded.
    :type final_temp: float
    """
    ...

  def format_pipeline_summary(self) -> str:
    """Format a comprehensive pipeline timing summary.

    :returns: Single-line summary of all pipeline timings and generation results.
    :rtype: str
    """
    ...

  # Tracer factory methods
  def trace_pipeline(self) -> TracerProtocol:
    """Create a tracer for the entire transcription pipeline.

    :returns: Pipeline tracer that measures total execution time.
    :rtype: TracerProtocol
    """
    ...

  def trace_vad_stage(self) -> TracerProtocol:
    """Create a tracer for VAD audio preprocessing stage.

    :returns: VAD tracer that logs speech activity and visualization.
    :rtype: TracerProtocol
    """
    ...

  def trace_feature_stage(self) -> TracerProtocol:
    """Create a tracer for feature extraction stage.

    :returns: Feature extraction tracer.
    :rtype: TracerProtocol
    """
    ...

  def trace_inference_stage(self) -> TracerProtocol:
    """Create a tracer for model inference stage.

    :returns: Inference tracer that records generation attempts and temperature.
    :rtype: TracerProtocol
    """
    ...

  def trace_segment_stage(self) -> TracerProtocol:
    """Create a tracer for segment processing stage.

    :returns: Segment tracer that logs final results and pipeline summary.
    :rtype: TracerProtocol
    """
    ...


@dataclass
class TranscriptionSession:
  """Context object that tracks timing and metadata for a transcription session.

  Bridges the gap between client connection time and audio buffer time to provide
  comprehensive logging across both WebSocket and RTSP clients.

  :param connection_start_time: When the client connection was established (perf_counter).
  :type connection_start_time: float
  :param stream_name: Unique identifier for this transcription stream.
  :type stream_name: str
  """

  connection_start_time: float
  stream_name: str

  # Session logger for all tracing output
  logger: BoundLogger = field(init=False)

  # Timing tracking
  stage_timings: StageTimings = field(default_factory=dict)
  generation_attempts: int = 0
  final_temperature: float = 0.0

  # Segment chain tracking
  completed_segments: list["Segment"] = field(default_factory=list)

  def __post_init__(self):
    """Initialize the session logger."""
    self.logger = get_logger("trace", stream=self.stream_name)

  # Audio context
  audio_start_offset: float = 0.0
  audio_duration: float = 0.0

  # Segment chain management
  def get_last_completed_segment(self) -> "Segment | None":
    """Get the most recently completed segment for chain ID computation.

    :returns: The last completed segment, or None if no segments completed yet
    :rtype: Segment | None
    """
    return self.completed_segments[-1] if self.completed_segments else None

  def most_recent_completed_segments(self, n: int) -> Iterable["Segment"]:
    """Get the last N completed segments for sending to the client.

    :param n: Number of recent completed segments to retrieve.
    :type n: int
    :returns: Iterable of the last N completed segments.
    :rtype: Iterable[Segment]
    """
    return self.completed_segments[-n:]

  def add_completed_segment(self, segment: "Segment") -> None:
    """Add a completed segment to the chain.

    This should be called after a segment has been marked as completed
    and assigned its chain-based ID.

    :param segment: The completed segment to add to the chain
    :type segment: Segment
    """
    self.completed_segments.append(segment)

  def get_absolute_time_range(self) -> tuple[float, float]:
    """Get the absolute time range of the current audio chunk.

    "Absolute" means relative to the client connection start time, not relative
    to the current buffer or chunk. This gives the timeline position within
    the overall transcription session.

    :returns: Tuple of (start_time, end_time) in seconds since connection.
    :rtype: tuple[float, float]
    """
    return (self.audio_start_offset, self.audio_start_offset + self.audio_duration)

  def format_time_range(self) -> str:
    """Format the time range for logging.

    :returns: Formatted string like "+3.45s to +7.20s".
    :rtype: str
    """
    start, end = self.get_absolute_time_range()
    return f"+{start:.2f}s to +{end:.2f}s"

  def format_vad_visualization(
    self, speech_chunks: list[SpeechChunk] | None, sample_rate: int, total_samples: int
  ) -> str:
    """Format VAD results as ASCII art visualization.

    Creates a visualization like:
    +3.45s|-1.20s-|~0.80s~|-0.50s-|+7.20s

    :param speech_chunks: Speech chunks from VAD detection, or None if no VAD.
    :type speech_chunks: list[SpeechChunk] | None
    :param sample_rate: Audio sample rate for time calculations.
    :type sample_rate: int
    :param total_samples: Total number of audio samples.
    :type total_samples: int
    :returns: ASCII art visualization of speech/silence patterns.
    :rtype: str
    """
    if not speech_chunks:
      # No VAD applied - show entire duration as unanalyzed
      duration = total_samples / sample_rate
      start, _ = self.get_absolute_time_range()
      return f"+{start:.2f}s~{duration * 1000:.3f}ms~+{start + duration:.2f}s"

    start_time, _ = self.get_absolute_time_range()
    visualization_parts = [f"+{start_time:.2f}s"]

    last_end_sample = 0

    for chunk in speech_chunks:
      # Add silence before this speech chunk
      if chunk["start"] > last_end_sample:
        silence_duration = (chunk["start"] - last_end_sample) / sample_rate
        visualization_parts.append(f"|-{silence_duration * 1000:.3f}ms-|")

      # Add the speech chunk
      speech_duration = (chunk["end"] - chunk["start"]) / sample_rate
      visualization_parts.append(f"~{speech_duration * 1000:.3f}ms~")

      last_end_sample = chunk["end"]

    # Add final silence if any
    if last_end_sample < total_samples:
      silence_duration = (total_samples - last_end_sample) / sample_rate
      visualization_parts.append(f"|-{silence_duration:.2f}s-|")

    # Add end time
    end_time = start_time + (total_samples / sample_rate)
    visualization_parts.append(f"+{end_time:.2f}s")

    return "".join(visualization_parts)

  def update_audio_context(self, start_offset: float, duration: float) -> None:
    """Update the audio timing context for the current chunk.

    :param start_offset: Start time offset from connection start in seconds.
    :type start_offset: float
    :param duration: Duration of the audio chunk in seconds.
    :type duration: float
    """
    self.audio_start_offset = start_offset
    self.audio_duration = duration

  def record_generation_result(self, attempts: int, final_temp: float) -> None:
    """Record the results of temperature fallback generation.

    :param attempts: Number of temperature attempts made.
    :type attempts: int
    :param final_temp: Final temperature that succeeded.
    :type final_temp: float
    """
    self.generation_attempts = attempts
    self.final_temperature = final_temp

  def format_pipeline_summary(self) -> str:
    """Format a comprehensive pipeline timing summary.

    :returns: Single-line summary of all pipeline timings and results.
    :rtype: str
    """
    timing_parts = [f"{stage}={time * 1000:.3f}ms" for stage, time in self.stage_timings.items()]
    timing_str = " ".join(timing_parts)

    result_parts = []
    if self.generation_attempts > 0:
      result_parts.append(f"attempts={self.generation_attempts}")
    if self.final_temperature > 0:
      result_parts.append(f"temp={self.final_temperature:.1f}")

    result_str = " ".join(result_parts)
    parts = ["Pipeline:", timing_str]
    if result_str:
      parts.append(result_str)

    return " ".join(parts)

  def format_pipeline_summary_with_total(self) -> str:
    """Format pipeline summary including total time.

    :returns: Summary with individual stages and total pipeline time.
    :rtype: str
    """
    timing_parts = [
      f"{stage}={time * 1000:.3f}ms"
      for stage, time in self.stage_timings.items()
      if stage != "total_pipeline"
    ]
    timing_str = " ".join(timing_parts)

    result_parts = []
    if self.generation_attempts > 0:
      result_parts.append(f"attempts={self.generation_attempts}")
    if self.final_temperature > 0:
      result_parts.append(f"temp={self.final_temperature:.1f}")

    result_str = " ".join(result_parts)
    parts = ["Pipeline:", timing_str]
    if result_str:
      parts.append(result_str)

    # Add total time if available
    if "total_pipeline" in self.stage_timings:
      total_time = self.stage_timings["total_pipeline"]
      parts.append(f"total={total_time * 1000:.3f}ms")

    return " ".join(parts)

  # Tracer factory methods
  def trace_pipeline(self) -> "PipelineTracer":
    """Create a tracer for the entire transcription pipeline."""
    return PipelineTracer(self, "total_pipeline")

  def trace_vad_stage(self) -> "VadStageTracer":
    """Create a tracer for VAD audio preprocessing stage."""
    return VadStageTracer(self, "audio_preprocessing")

  def trace_feature_stage(self) -> "FeatureStageTracer":
    """Create a tracer for feature extraction stage."""
    return FeatureStageTracer(self, "feature_extraction")

  def trace_inference_stage(self) -> "InferenceStageTracer":
    """Create a tracer for model inference stage."""
    return InferenceStageTracer(self, "model_inference")

  def trace_segment_stage(self) -> "SegmentStageTracer":
    """Create a tracer for segment processing stage."""
    return SegmentStageTracer(self, "segment_processing")


class BaseTracer:
  """Base class for stage-specific tracers that handle timing and logging."""

  def __init__(self, session: TranscriptionSession, stage_name: str):
    """Initialize tracer with session and stage name.

    :param session: The transcription session to update.
    :type session: TranscriptionSession
    :param stage_name: Name of the processing stage.
    :type stage_name: str
    """
    self.session = session
    self.stage_name = stage_name
    self.start_time: float = 0.0

  def __enter__(self):
    """Start timing the stage."""
    self.start_time = time.perf_counter()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Stop timing and record the duration."""
    elapsed = time.perf_counter() - self.start_time
    # Record timing even on exceptions for debugging purposes
    self.session.stage_timings[self.stage_name] = elapsed if exc_type is None else -1.0


class VadStageTracer(BaseTracer):
  """Tracer for VAD audio preprocessing stage."""

  def __call__(self, speech_chunks: list[SpeechChunk] | None, sample_rate: int, total_samples: int):
    """Record VAD results and generate visualization.

    :param speech_chunks: Speech chunks detected by VAD, or None if no VAD.
    :type speech_chunks: list[SpeechChunk] | None
    :param sample_rate: Audio sample rate for time calculations.
    :type sample_rate: int
    :param total_samples: Total number of audio samples processed.
    :type total_samples: int
    """
    audio_duration = total_samples / sample_rate
    time_start, time_end = self.session.get_absolute_time_range()

    if speech_chunks is not None:  # VAD was applied
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
    else:  # No VAD applied
      self.session.logger.info(
        "No voice activity",
        t_start=time_start,
        t_end=time_end,
        total=audio_duration,
      )


class FeatureStageTracer(BaseTracer):
  """Tracer for feature extraction stage."""

  def __call__(self):
    """Record feature extraction completion (no specific data to log)."""
    pass


class InferenceStageTracer(BaseTracer):
  """Tracer for model inference stage."""

  def __call__(self, attempts: int, final_temperature: float):
    """Record generation results for pipeline summary.

    :param attempts: Number of temperature attempts made during generation.
    :type attempts: int
    :param final_temperature: Final temperature that succeeded.
    :type final_temperature: float
    """
    self.session.record_generation_result(attempts, final_temperature)


class PipelineTracer(BaseTracer):
  """Tracer for the entire transcription pipeline."""

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Stop timing and log the complete pipeline summary."""
    # Call parent to record timing
    super().__exit__(exc_type, exc_val, exc_tb)

    # Format stage timings with "stage_" prefix and millisecond formatting
    stage_fields = {}
    for stage_name, timing in self.session.stage_timings.items():
      if timing >= 0:
        stage_fields[f"stage_{stage_name}"] = f"{timing * 1000:.3f}ms"
      else:
        stage_fields[f"stage_{stage_name}"] = "FAILED"

    # Add generation metadata if available
    extra_fields = {}
    if self.session.generation_attempts > 0:
      extra_fields["generation_attempts"] = self.session.generation_attempts
    if self.session.final_temperature > 0:
      extra_fields["final_temperature"] = f"{self.session.final_temperature:.1f}"

    time_range = self.session.format_time_range()

    self.session.logger.info(
      "Pipeline completed",
      time_range=time_range,
      **stage_fields,
      **extra_fields,
    )

  def __call__(self):
    """No-op for pipeline tracer - summary is logged in __exit__."""
    pass


class SegmentStageTracer(BaseTracer):
  """Tracer for segment processing stage."""

  def __call__(self, segments: Iterable["Segment"]):
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

    # Log comprehensive pipeline summary with structured timing data
    self.session.logger.info(
      "Pipeline completed",
      **self.session.stage_timings,
      generation_attempts=self.session.generation_attempts,
      final_temperature=self.session.final_temperature,
    )


class NoopTracer:
  """No-op tracer that does nothing - for use with NoopSession."""

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    pass

  def __call__(self, *args, **kwargs):
    pass


class NoopSession:
  """No-op session that provides the same API as TranscriptionSession but does nothing."""

  def get_absolute_time_range(self) -> tuple[float, float]:
    return (0.0, 0.0)

  def format_time_range(self) -> str:
    return ""

  def format_vad_visualization(
    self, speech_chunks: list[SpeechChunk] | None, sample_rate: int, total_samples: int
  ) -> str:
    return ""

  def update_audio_context(self, start_offset: float, duration: float) -> None:
    pass

  def record_generation_result(self, attempts: int, final_temp: float) -> None:
    pass

  def format_pipeline_summary(self) -> str:
    return ""

  # Tracer factory methods - all return NoopTracer
  def trace_pipeline(self) -> NoopTracer:
    return NoopTracer()

  def trace_vad_stage(self) -> NoopTracer:
    return NoopTracer()

  def trace_feature_stage(self) -> NoopTracer:
    return NoopTracer()

  def trace_inference_stage(self) -> NoopTracer:
    return NoopTracer()

  def trace_segment_stage(self) -> NoopTracer:
    return NoopTracer()


# Default no-op session instance
_noop_session = NoopSession()


def create_session(stream_name: str) -> TranscriptionSession:
  """Create a new transcription session with current time as connection start.

  :param stream_name: Unique identifier for the transcription stream.
  :type stream_name: str
  :returns: New transcription session instance.
  :rtype: TranscriptionSession
  """
  return TranscriptionSession(connection_start_time=time.perf_counter(), stream_name=stream_name)
