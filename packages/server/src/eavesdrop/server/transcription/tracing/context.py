"""Transcription session context for timing and logging coordination.

This module provides session context that bridges recording-relative chunk timing
with audio buffer timing to enable comprehensive transcription logging.
"""

from contextlib import AbstractContextManager
from dataclasses import dataclass, field

from langfuse import LangfuseSpan
from pydantic import JsonValue
from structlog.stdlib import BoundLogger

from eavesdrop.common import get_logger
from eavesdrop.server.langfuse_tracing import (
  start_recording_observation,
  start_stage_observation,
)
from eavesdrop.server.transcription.models import SpeechSegment

# Type alias for stage timing measurements
type StageTimings = dict[str, float]


@dataclass
class TracingContext:
  """Context object that tracks timing and metadata for a transcription session.

  Bridges the gap between recording-relative chunk time and audio buffer time to provide
  comprehensive logging for live WebSocket clients.

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

  def __post_init__(self) -> None:
    """Initialize the session logger."""
    self.logger = get_logger("trace", stream=self.stream_name)

  # Audio context
  audio_start_offset: float = 0.0
  audio_duration: float = 0.0
  recording_id: str | None = None

  def get_recording_time_range(self) -> tuple[float, float]:
    """Get the recording-relative time range of the current audio chunk.

    Recording-relative time is the canonical timeline used for wire-facing
    transcription segment and word timestamps. It is not relative to the current
    buffer or decoder chunk.

    :returns: Tuple of (start_time, end_time) in recording-relative seconds.
    :rtype: tuple[float, float]
    """
    return (self.audio_start_offset, self.audio_start_offset + self.audio_duration)

  def format_time_range(self) -> str:
    """Format the time range for logging.

    :returns: Formatted recording-relative range like "+3.45s to +7.20s".
    :rtype: str
    """
    start, end = self.get_recording_time_range()
    return f"+{start:.2f}s to +{end:.2f}s"

  def format_vad_visualization(
    self, speech_segments: list[SpeechSegment] | None, sample_rate: int, total_samples: int
  ) -> str:
    """Format VAD results as ASCII art visualization.

    Creates a visualization like:
    +3.45s|-1.20s-|~0.80s~|-0.50s-|+7.20s

    :param speech_segments: VAD-detected speech segments, or None if VAD was disabled.
    :type speech_segments: list[SpeechSegment] | None
    :param sample_rate: Audio sample rate for time calculations.
    :type sample_rate: int
    :param total_samples: Total number of audio samples.
    :type total_samples: int
    :returns: ASCII art visualization of speech/silence patterns.
    :rtype: str
    """
    if speech_segments is None:
      duration = total_samples / sample_rate
      start, _ = self.get_recording_time_range()
      return f"+{start:.2f}s[unanalyzed {duration:.2f}s]+{start + duration:.2f}s"

    start_time, _ = self.get_recording_time_range()
    visualization_parts: list[str] = [f"+{start_time:.2f}s"]

    last_end_sample = 0

    for chunk in speech_segments:
      # Add silence before this speech segment.
      if chunk["start"] > last_end_sample:
        silence_duration = (chunk["start"] - last_end_sample) / sample_rate
        visualization_parts.append(f"|-{silence_duration * 1000:.3f}ms-|")

      # Add the detected speech segment.
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

  def start_langfuse_observation(
    self,
    stage_name: str,
  ) -> AbstractContextManager[LangfuseSpan | None]:
    """Start a Langfuse observation for the current transcription stage.

    :param stage_name: Canonical transcription stage name.
    :type stage_name: str
    :returns: Context manager for a Langfuse observation, or a no-op when disabled.
    :rtype: AbstractContextManager[LangfuseSpan | None]
    """
    metadata: dict[str, JsonValue] = {
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
