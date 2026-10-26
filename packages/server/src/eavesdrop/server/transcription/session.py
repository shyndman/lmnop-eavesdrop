"""Recording transcript history alongside its directly owned tracing context."""

import time
from collections.abc import Iterable

from eavesdrop.server.transcription.tracing.context import TracingContext
from eavesdrop.wire import Segment


class TranscriptionSession:
  """Own completed transcript history and recording-local tracing state.

  :param connection_start_time: Connection establishment time from perf_counter.
  :param stream_name: Connection session identifier.
  """

  def __init__(self, connection_start_time: float, stream_name: str) -> None:
    self.tracing: TracingContext = TracingContext(connection_start_time, stream_name)
    self.completed_segments: list[Segment] = []

  def get_last_completed_segment(self) -> Segment | None:
    """Return the preceding completed timed segment for chain identity."""
    return self.completed_segments[-1] if self.completed_segments else None

  def most_recent_completed_segments(self, n: int) -> Iterable[Segment]:
    """Return the most recent ``n`` completed timed segments."""
    return self.completed_segments[-n:]

  def add_completed_segment(self, segment: Segment) -> None:
    """Retain a completed timed segment with its assigned chain identity."""
    self.completed_segments.append(segment)

  def reset_utterance(self) -> None:
    """Clear history after cancellation or an accepted recording boundary."""
    self.completed_segments.clear()


def create_session(stream_name: str) -> TranscriptionSession:
  """Create recording state using the current connection start time."""
  return TranscriptionSession(time.perf_counter(), stream_name)
