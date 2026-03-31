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
  """

  last_id: int | None = None
  parts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SegmentReduction:
  """Committed segments extracted from the latest window.

  :ivar segments: Newly committed non-tail segments.
  :vartype segments: list[Segment]
  :ivar last_id: Identifier to store for the next reduction.
  :vartype last_id: int | None
  :ivar missing_last_id: True when the previous sentinel was absent.
  :vartype missing_last_id: bool
  """

  segments: list[Segment]
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

  committed_prefix = list(segments[:-1])
  if not committed_prefix:
    return SegmentReduction(segments=[], last_id=last_id, missing_last_id=False)

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
    last_id=committed_prefix[-1].id,
    missing_last_id=missing_last_id,
  )


def append_segment_text(parts: list[str], segments: Sequence[Segment]) -> None:
  """Append stripped committed text parts in place.

  :param parts: Accumulated committed text parts.
  :type parts: list[str]
  :param segments: Newly committed segments to ingest.
  :type segments: Sequence[Segment]
  """

  for segment in segments:
    text = segment.text.strip()
    if text:
      parts.append(text)


def render_text(parts: Sequence[str]) -> str:
  """Render accumulated committed parts for emission.

  :param parts: Previously appended committed text parts.
  :type parts: Sequence[str]
  :returns: Space-joined committed text.
  :rtype: str
  """

  return " ".join(parts)
