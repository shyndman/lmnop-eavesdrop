"""Text state management and diffing algorithms.

Handles incremental transcription updates and calculates minimal typing operations
to maintain consistency between transcribed text and desktop applications.
"""

from __future__ import annotations

import dataclasses
from collections import OrderedDict
from enum import Enum

from pydantic import NonNegativeInt
from pydantic.dataclasses import dataclass
from structlog import get_logger

from eavesdrop.wire.transcription import Segment


class UpdateType(Enum):
  """Classification of text update operations."""

  REPLACE_SUFFIX = "replace_suffix"  # Update end of in-progress segment (common case)
  REPLACE_ALL = "replace_all"  # Complete replacement of in-progress segment
  NEW_SEGMENT = "new_segment"  # Start typing new segment after completion
  NO_CHANGE = "no_change"  # No typing action needed


@dataclass
class TextUpdate:
  """Represents typing actions needed to update desktop text."""

  chars_to_delete: NonNegativeInt
  """Number of characters to backspace from current position"""

  text_to_type: str
  """New text content to type after deletions"""

  operation_type: UpdateType
  """Classification of update for logging/debugging purposes"""


@dataclass
class TypingOperation:
  """Encapsulates a single desktop typing action with error recovery."""

  operation_id: str  # Unique identifier for operation tracking
  chars_to_delete: NonNegativeInt  # Backspace operations to perform
  text_to_type: str  # Text content to type
  timestamp: float  # When operation was initiated
  completed: bool  # Whether operation finished successfully


logger = get_logger("txt")


@dataclasses.dataclass
class TextState:
  """Manages the complete state of text that has been typed to the desktop."""

  current_segment: Segment | None = None
  """The currently in-progress (not yet finalized) transcription segment"""

  completed_segment_ids: OrderedDict[int, Segment] = dataclasses.field(
    default_factory=lambda: OrderedDict([])
  )
  """Set of finalized transcription segments"""

  def process_segment(self, segment: Segment) -> TextUpdate | None:
    """Process a new transcription segment and return required text update."""

    # Case 1. The segment completed, and we already have it marked completed
    if segment.completed and segment.id in self.completed_segment_ids:
      logger.warn("Still completed")
      return None

    # Case 2. The segment is newly completed, transitioning from in-progress
    if segment.completed and segment.id not in self.completed_segment_ids:
      logger.warn("New completed", previous_in_progress=self.current_segment)
      assert self.current_segment is not None
      self.completed_segment_ids[segment.id] = segment
      update = calculate_text_update(from_segment=self.current_segment, to_segment=segment)
      if segment.id == self.current_segment.id:
        self.current_segment = None
      return update

    # Case 3. We are receiving the latest in-progress update for the current segment
    if not segment.completed and self.current_segment:
      logger.warn("Updated in-progress")
      assert self.current_segment.id == segment.id
      update = calculate_text_update(from_segment=self.current_segment, to_segment=segment)
      self.current_segment = segment  # Update to the new segment state
      return update

    # Case 4. We are receiving the first in-progress segment
    if not segment.completed and not self.current_segment:
      logger.warn("New in-progress")
      self.current_segment = segment
      return calculate_text_update(from_segment=None, to_segment=segment)

    raise ValueError(
      "Received segment state that does not match any known case.",
    )

  def get_complete_text(self) -> str:
    """Returns full text as currently typed (completed + in-progress)."""
    completed_text = " ".join([segment.text for segment in self.completed_segment_ids.values()])
    return f"{completed_text} {self.current_segment.text if self.current_segment else ''}".strip()


def calculate_text_update(from_segment: Segment | None, to_segment: Segment) -> TextUpdate:
  """Determines typing actions needed to update from current to new segment state."""

  if from_segment is None:
    # Handle new in-progress segment
    return TextUpdate(
      chars_to_delete=0,
      text_to_type=to_segment.text,
      operation_type=UpdateType.NEW_SEGMENT,
    )

  return calculate_text_diff(from_segment.text, to_segment.text)


def find_common_prefix(str1: str, str2: str) -> str:
  """Find the common prefix between two strings."""
  if not str1 or not str2:
    return ""

  min_length = min(len(str1), len(str2))

  for i in range(min_length):
    if str1[i] != str2[i]:
      return str1[:i]

  return str1[:min_length]


def calculate_text_diff(old_text: str, new_text: str) -> TextUpdate:
  """Calculate minimal typing operations needed to transform old_text to new_text."""
  # Handle empty cases
  if not old_text and not new_text:
    return TextUpdate(0, "", UpdateType.NO_CHANGE)

  if not old_text:
    return TextUpdate(0, new_text, UpdateType.NEW_SEGMENT)

  if not new_text:
    return TextUpdate(len(old_text), "", UpdateType.REPLACE_ALL)

  # Same text
  if old_text == new_text:
    return TextUpdate(0, "", UpdateType.NO_CHANGE)

  # Find common prefix
  common_prefix = find_common_prefix(old_text, new_text)

  # If no common prefix, replace everything
  if not common_prefix:
    return TextUpdate(
      chars_to_delete=len(old_text), text_to_type=new_text, operation_type=UpdateType.REPLACE_ALL
    )

  # Calculate suffix replacement
  chars_to_delete = len(old_text) - len(common_prefix)
  text_to_type = new_text[len(common_prefix) :]

  return TextUpdate(
    chars_to_delete=chars_to_delete,
    text_to_type=text_to_type,
    operation_type=UpdateType.REPLACE_SUFFIX,
  )
