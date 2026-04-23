"""Pure helpers for reducing per-recording transcription windows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from eavesdrop.wire import Segment


@dataclass
class RecordingReducerState:
  """Per-recording reducer state consumed by app policy.

  :ivar last_id: Last committed segment identifier.
  :vartype last_id: int | None
  :ivar parts: Accumulated committed text parts.
  :vartype parts: list[str]
  :ivar first_segment_start: Earliest absolute start time among committed text segments.
  :vartype first_segment_start: float | None
  :ivar last_segment_end: Latest absolute end time among committed text segments.
  :vartype last_segment_end: float | None
  """

  last_id: int | None = None
  parts: list[str] = field(default_factory=list)
  first_segment_start: float | None = None
  last_segment_end: float | None = None

  @property
  def duration_seconds(self) -> float | None:
    if self.first_segment_start is None or self.last_segment_end is None:
      return None

    return self.last_segment_end - self.first_segment_start


@dataclass(frozen=True)
class OverlaySegment:
  """Minimal transcript segment payload published to the local overlay.

  :ivar id: Stable segment identifier.
  :vartype id: int
  :ivar text: Stripped transcript text ready for display.
  :vartype text: str
  """

  id: int
  text: str


@dataclass(frozen=True)
class TranscriptionUpdate:
  """Live transcript delta surfaced to UI integrations.

  :ivar completed_segments: Newly completed transcript segments since the last window.
  :vartype completed_segments: list[OverlaySegment]
  :ivar incomplete_segment: Current in-progress tail segment.
  :vartype incomplete_segment: OverlaySegment
  """

  completed_segments: list[OverlaySegment]
  incomplete_segment: OverlaySegment


@dataclass(frozen=True)
class SegmentReduction:
  """Committed segments extracted from the latest window.

  :ivar segments: Newly committed non-tail segments.
  :vartype segments: list[Segment]
  :ivar incomplete_segment: Current in-progress tail segment.
  :vartype incomplete_segment: Segment | None
  :ivar last_id: Identifier to store for the next reduction.
  :vartype last_id: int | None
  :ivar missing_last_id: True when the previous sentinel was absent.
  :vartype missing_last_id: bool
  """

  segments: list[Segment]
  incomplete_segment: Segment | None
  last_id: int | None
  missing_last_id: bool


def reduce_new_segments(
  segments: Sequence[Segment],
  last_id: int | None,
) -> SegmentReduction:
  """Return newly committed segments from a transcription window.

  The transcriber keeps the final segment as the in-progress tail, so only the
  completed prefix is eligible for emission.

  :param segments: Window returned by the transcriber.
  :type segments: Sequence[Segment]
  :param last_id: Previously committed segment identifier.
  :type last_id: int | None
  :returns: Newly committed non-tail segments and sentinel status.
  :rtype: SegmentReduction
  """

  incomplete_segment = segments[-1] if segments else None
  committed_prefix = list(segments[:-1])
  if incomplete_segment is None:
    return SegmentReduction(
      segments=[],
      incomplete_segment=None,
      last_id=last_id,
      missing_last_id=False,
    )

  if not committed_prefix:
    return SegmentReduction(
      segments=[],
      incomplete_segment=incomplete_segment,
      last_id=last_id,
      missing_last_id=False,
    )

  missing_last_id = False
  new_segments = committed_prefix
  if last_id is not None:
    sentinel_index = next(
      (index for index, segment in enumerate(committed_prefix) if segment.id == last_id),
      None,
    )
    if sentinel_index is None:
      missing_last_id = True
    else:
      new_segments = committed_prefix[sentinel_index + 1 :]

  return SegmentReduction(
    segments=new_segments,
    incomplete_segment=incomplete_segment,
    last_id=committed_prefix[-1].id,
    missing_last_id=missing_last_id,
  )


def build_transcription_update(reduction: SegmentReduction) -> TranscriptionUpdate | None:
  """Build a display-ready live transcript delta from a reduced window.

  :param reduction: Reduced transcription window with committed delta and tail state.
  :type reduction: SegmentReduction
  :returns: Overlay-ready transcript update, or None when no tail segment exists.
  :rtype: TranscriptionUpdate | None
  """

  incomplete_segment = reduction.incomplete_segment
  if incomplete_segment is None:
    return None

  return TranscriptionUpdate(
    completed_segments=[
      OverlaySegment(id=segment.id, text=segment.text.strip())
      for segment in reduction.segments
      if segment.text.strip()
    ],
    incomplete_segment=OverlaySegment(
      id=incomplete_segment.id,
      text=incomplete_segment.text.strip(),
    ),
  )


def append_segment_text(state: RecordingReducerState, segments: Sequence[Segment]) -> None:
  """Append stripped committed text parts and timing in place.

  :param state: Accumulated reducer state for the current recording.
  :type state: RecordingReducerState
  :param segments: Newly committed segments to ingest.
  :type segments: Sequence[Segment]
  """

  for segment in segments:
    text = segment.text.strip()
    if text:
      state.parts.append(text)
      segment_start = segment.absolute_start_time
      segment_end = segment.absolute_end_time
      if state.first_segment_start is None or segment_start < state.first_segment_start:
        state.first_segment_start = segment_start
      if state.last_segment_end is None or segment_end > state.last_segment_end:
        state.last_segment_end = segment_end


def render_text(parts: Sequence[str]) -> str:
  """Render accumulated committed parts for emission.

  :param parts: Previously appended committed text parts.
  :type parts: Sequence[str]
  :returns: Space-joined committed text.
  :rtype: str
  """

  return " ".join(parts)
