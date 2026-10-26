"""Transcription session coordinating segment history and shared tracing state."""

import time
from collections.abc import Iterable
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

from structlog.stdlib import BoundLogger

from eavesdrop.server.langfuse_tracing import ServerObservation
from eavesdrop.server.transcription.models import SpeechChunk
from eavesdrop.server.transcription.tracing import stages
from eavesdrop.server.transcription.tracing.context import StageTimings, TracingContext

if TYPE_CHECKING:
  from eavesdrop.wire.transcription import Segment


class TranscriptionSession:
  """Keep segment history alongside one shared, mutable tracing context.

  :param connection_start_time: Connection establishment time from perf_counter.
  :param stream_name: Unique identifier for this transcription stream.
  """

  def __init__(
    self,
    connection_start_time: float,
    stream_name: str,
    stage_timings: StageTimings | None = None,
    completed_segments: list["Segment"] | None = None,
    audio_start_offset: float = 0.0,
    audio_duration: float = 0.0,
    recording_id: str | None = None,
  ) -> None:
    """Initialize history and the state shared by all stage tracers."""
    self._tracing: TracingContext = TracingContext(
      connection_start_time=connection_start_time,
      stream_name=stream_name,
      stage_timings={} if stage_timings is None else stage_timings,
      audio_start_offset=audio_start_offset,
      audio_duration=audio_duration,
      recording_id=recording_id,
    )
    self.completed_segments: list["Segment"] = (
      [] if completed_segments is None else completed_segments
    )

  @property
  def connection_start_time(self) -> float:
    """Return the shared tracing connection start time."""
    return self._tracing.connection_start_time

  @connection_start_time.setter
  def connection_start_time(self, value: float) -> None:
    self._tracing.connection_start_time = value

  @property
  def stream_name(self) -> str:
    """Return the shared tracing stream name."""
    return self._tracing.stream_name

  @stream_name.setter
  def stream_name(self, value: str) -> None:
    self._tracing.stream_name = value

  @property
  def logger(self) -> BoundLogger:
    """Return the shared tracing logger."""
    return self._tracing.logger

  @logger.setter
  def logger(self, value: BoundLogger) -> None:
    self._tracing.logger = value

  @property
  def stage_timings(self) -> StageTimings:
    """Return the shared tracing stage timings."""
    return self._tracing.stage_timings

  @stage_timings.setter
  def stage_timings(self, value: StageTimings) -> None:
    self._tracing.stage_timings = value

  @property
  def audio_start_offset(self) -> float:
    """Return the shared tracing audio start offset."""
    return self._tracing.audio_start_offset

  @audio_start_offset.setter
  def audio_start_offset(self, value: float) -> None:
    self._tracing.audio_start_offset = value

  @property
  def audio_duration(self) -> float:
    """Return the shared tracing audio duration."""
    return self._tracing.audio_duration

  @audio_duration.setter
  def audio_duration(self, value: float) -> None:
    self._tracing.audio_duration = value

  @property
  def recording_id(self) -> str | None:
    """Return the shared tracing recording id."""
    return self._tracing.recording_id

  @recording_id.setter
  def recording_id(self, value: str | None) -> None:
    self._tracing.recording_id = value

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

  def reset_utterance(self) -> None:
    """Clear utterance-local completed history after the live utterance is discarded."""
    self.completed_segments.clear()

  def get_recording_time_range(self) -> tuple[float, float]:
    """Get recording time range using the shared tracing context."""
    return self._tracing.get_recording_time_range()

  def format_time_range(self) -> str:
    """Format time range using the shared tracing context."""
    return self._tracing.format_time_range()

  def format_vad_visualization(
    self, speech_chunks: list[SpeechChunk] | None, sample_rate: int, total_samples: int
  ) -> str:
    """Format vad visualization using the shared tracing context."""
    return self._tracing.format_vad_visualization(speech_chunks, sample_rate, total_samples)

  def update_audio_context(self, start_offset: float, duration: float) -> None:
    """Update audio context using the shared tracing context."""
    return self._tracing.update_audio_context(start_offset, duration)

  def update_recording_context(self, recording_id: str | None) -> None:
    """Update recording context using the shared tracing context."""
    return self._tracing.update_recording_context(recording_id)

  def start_langfuse_observation(
    self, stage_name: str
  ) -> AbstractContextManager[ServerObservation | None]:
    """Start langfuse observation using the shared tracing context."""
    return self._tracing.start_langfuse_observation(stage_name)

  def format_pipeline_summary(self) -> str:
    """Format pipeline summary using the shared tracing context."""
    return self._tracing.format_pipeline_summary()

  def format_pipeline_summary_with_total(self) -> str:
    """Format pipeline summary with total using the shared tracing context."""
    return self._tracing.format_pipeline_summary_with_total()

  # Tracer factory methods
  def trace_pipeline(self) -> "stages.PipelineTracer":
    """Create a tracer for the entire transcription pipeline."""
    return stages.PipelineTracer(self._tracing, "total_pipeline")

  def trace_vad_stage(self) -> "stages.VadStageTracer":
    """Create a tracer for VAD audio preprocessing stage."""
    return stages.VadStageTracer(self._tracing, "audio_preprocessing")

  def trace_feature_stage(self) -> "stages.FeatureStageTracer":
    """Create a tracer for feature extraction stage."""
    return stages.FeatureStageTracer(self._tracing, "feature_extraction")

  def trace_inference_stage(self) -> "stages.InferenceStageTracer":
    """Create a tracer for model inference stage."""
    return stages.InferenceStageTracer(self._tracing, "model_inference")

  def trace_segment_stage(self) -> "stages.SegmentStageTracer":
    """Create a tracer for segment processing stage."""
    return stages.SegmentStageTracer(self._tracing, "segment_processing")


def create_session(stream_name: str) -> TranscriptionSession:
  """Create a new transcription session with current time as connection start.

  :param stream_name: Unique identifier for the transcription stream.
  :type stream_name: str
  :returns: New transcription session instance.
  :rtype: TranscriptionSession
  """
  return TranscriptionSession(connection_start_time=time.perf_counter(), stream_name=stream_name)
