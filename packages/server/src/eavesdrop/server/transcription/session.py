"""Transcription session context for timing and logging coordination.

This module provides session context that bridges recording-relative chunk timing
with audio buffer timing to enable comprehensive transcription logging.
"""

import time
from collections.abc import Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from structlog.stdlib import BoundLogger

from eavesdrop.common import get_logger
from eavesdrop.server.langfuse_tracing import (
  ServerObservation,
  start_recording_observation,
  start_stage_observation,
)
from eavesdrop.server.transcription.models import SpeechChunk
from eavesdrop.server.transcription.tracers import (
  InferenceStageTracer,
  PipelineTracer,
  SegmentStageTracer,
  TracerProtocol,
  VadStageTracer,
)

if TYPE_CHECKING:
  from eavesdrop.wire.transcription import Segment

type StageTimings = dict[str, float]


class TranscriptionSessionProtocol(Protocol):
  """Protocol defining the interface for transcription sessions."""

  def get_recording_time_range(self) -> tuple[float, float]: ...

  def format_time_range(self) -> str: ...

  def format_vad_visualization(
    self, speech_chunks: list[SpeechChunk] | None, sample_rate: int, total_samples: int
  ) -> str: ...

  def update_audio_context(self, start_offset: float, duration: float) -> None: ...

  def update_recording_context(self, recording_id: str | None) -> None: ...

  def format_pipeline_summary(self) -> str: ...

  def trace_pipeline(self) -> TracerProtocol: ...

  def trace_vad_stage(self) -> TracerProtocol: ...

  def trace_inference_stage(self) -> TracerProtocol: ...

  def trace_segment_stage(self) -> TracerProtocol: ...


@dataclass
class TranscriptionSession:
  """Context object that tracks timing and metadata for a live transcription session.

  :param connection_start_time: When the client connection was established (perf_counter).
  :type connection_start_time: float
  :param stream_name: Unique identifier for this transcription stream.
  :type stream_name: str
  """

  connection_start_time: float
  stream_name: str
  logger: BoundLogger = field(init=False)
  stage_timings: StageTimings = field(default_factory=dict)
  completed_segments: list["Segment"] = field(default_factory=list)
  audio_start_offset: float = 0.0
  audio_duration: float = 0.0
  recording_id: str | None = None

  def __post_init__(self) -> None:
    """Initialize the session logger."""
    self.logger = get_logger("trace", stream=self.stream_name)

  def get_last_completed_segment(self) -> "Segment | None":
    """Get the most recently completed segment for chain ID computation."""
    return self.completed_segments[-1] if self.completed_segments else None

  def most_recent_completed_segments(self, n: int) -> Iterable["Segment"]:
    """Get the last N completed segments for sending to the client."""
    return self.completed_segments[-n:]

  def add_completed_segment(self, segment: "Segment") -> None:
    """Add a completed segment to the chain."""
    self.completed_segments.append(segment)

  def reset_utterance(self) -> None:
    """Clear utterance-local completed history after the live utterance is discarded."""
    self.completed_segments.clear()

  def get_recording_time_range(self) -> tuple[float, float]:
    """Get the recording-relative time range of the current audio chunk."""
    return (self.audio_start_offset, self.audio_start_offset + self.audio_duration)

  def format_time_range(self) -> str:
    """Format the time range for logging."""
    start, end = self.get_recording_time_range()
    return f"+{start:.2f}s to +{end:.2f}s"

  def format_vad_visualization(
    self, speech_chunks: list[SpeechChunk] | None, sample_rate: int, total_samples: int
  ) -> str:
    """Format VAD results as ASCII art visualization."""
    if not speech_chunks:
      duration = total_samples / sample_rate
      start, _ = self.get_recording_time_range()
      return f"+{start:.2f}s~{duration * 1000:.3f}ms~+{start + duration:.2f}s"

    start_time, _ = self.get_recording_time_range()
    visualization_parts: list[str] = [f"+{start_time:.2f}s"]
    last_end_sample = 0

    for chunk in speech_chunks:
      if chunk["start"] > last_end_sample:
        silence_duration = (chunk["start"] - last_end_sample) / sample_rate
        visualization_parts.append(f"|-{silence_duration * 1000:.3f}ms-|")

      speech_duration = (chunk["end"] - chunk["start"]) / sample_rate
      visualization_parts.append(f"~{speech_duration * 1000:.3f}ms~")
      last_end_sample = chunk["end"]

    if last_end_sample < total_samples:
      silence_duration = (total_samples - last_end_sample) / sample_rate
      visualization_parts.append(f"|-{silence_duration:.2f}s-|")

    end_time = start_time + (total_samples / sample_rate)
    visualization_parts.append(f"+{end_time:.2f}s")
    return "".join(visualization_parts)

  def update_audio_context(self, start_offset: float, duration: float) -> None:
    """Update the audio timing context for the current chunk."""
    self.audio_start_offset = start_offset
    self.audio_duration = duration

  def update_recording_context(self, recording_id: str | None) -> None:
    """Update the current live recording identifier used for trace correlation."""
    self.recording_id = recording_id

  def start_langfuse_observation(
    self,
    stage_name: str,
  ) -> AbstractContextManager[ServerObservation | None]:
    """Start a Langfuse observation for the current transcription stage."""
    metadata: dict[str, object] = {
      "recording_start_seconds": self.audio_start_offset,
      "recording_end_seconds": self.audio_start_offset + self.audio_duration,
      "audio_duration_seconds": self.audio_duration,
    }
    if stage_name == "total_pipeline":
      return start_recording_observation(
        stream=self.stream_name,
        recording_id=self.recording_id,
        name="server-transcription-chunk",
        input={
          "stream": self.stream_name,
          "recording_id": self.recording_id,
          "recording_start_seconds": self.audio_start_offset,
          "recording_end_seconds": self.audio_start_offset + self.audio_duration,
          "audio_duration_seconds": self.audio_duration,
        },
        metadata=metadata,
      )

    return start_stage_observation(
      stream=self.stream_name,
      recording_id=self.recording_id,
      name=f"server-{stage_name.replace('_', '-')}",
      input={
        "stage": stage_name,
        "recording_start_seconds": self.audio_start_offset,
        "recording_end_seconds": self.audio_start_offset + self.audio_duration,
      },
      metadata=metadata,
    )

  def format_pipeline_summary(self) -> str:
    """Format a comprehensive pipeline timing summary."""
    timing_parts = [f"{stage}={time * 1000:.3f}ms" for stage, time in self.stage_timings.items()]
    return f"Pipeline: {' '.join(timing_parts)}"

  def trace_pipeline(self) -> PipelineTracer:
    """Create a tracer for the entire transcription pipeline."""
    return PipelineTracer(self, "total_pipeline")

  def trace_vad_stage(self) -> VadStageTracer:
    """Create a tracer for VAD audio preprocessing stage."""
    return VadStageTracer(self, "audio_preprocessing")

  def trace_inference_stage(self) -> InferenceStageTracer:
    """Create a tracer for model inference stage."""
    return InferenceStageTracer(self, "model_inference")

  def trace_segment_stage(self) -> SegmentStageTracer:
    """Create a tracer for segment processing stage."""
    return SegmentStageTracer(self, "segment_processing")


def create_session(stream_name: str) -> TranscriptionSession:
  """Create a new transcription session with current time as connection start.

  :param stream_name: Unique identifier for the transcription stream.
  :type stream_name: str
  :returns: New transcription session instance.
  :rtype: TranscriptionSession
  """
  return TranscriptionSession(connection_start_time=time.perf_counter(), stream_name=stream_name)
