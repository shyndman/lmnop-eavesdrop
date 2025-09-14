"""Text state management and diffing algorithms.

Handles incremental transcription updates and calculates minimal typing operations
to maintain consistency between transcribed text and desktop applications.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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

  chars_to_delete: int  # Number of characters to backspace from current position
  text_to_type: str  # New text content to type after deletions
  operation_type: UpdateType  # Classification of update for logging/debugging

  def __post_init__(self):
    """Validate TextUpdate fields after initialization."""
    if self.chars_to_delete < 0:
      raise ValueError("chars_to_delete must be >= 0")
    if not isinstance(self.text_to_type, str):
      raise ValueError("text_to_type must be a valid UTF-8 string")


@dataclass
class TypingOperation:
  """Encapsulates a single desktop typing action with error recovery."""

  operation_id: str  # Unique identifier for operation tracking
  chars_to_delete: int  # Backspace operations to perform
  text_to_type: str  # Text content to type
  timestamp: float  # When operation was initiated
  completed: bool  # Whether operation finished successfully

  def __post_init__(self):
    """Validate TypingOperation fields after initialization."""
    if self.chars_to_delete < 0:
      raise ValueError("chars_to_delete must be >= 0")


@dataclass
class ConnectionState:
  """Tracks the health and status of the eavesdrop server connection."""

  is_connected: bool = False  # Current WebSocket connection status
  is_streaming: bool = False  # Whether audio streaming is active
  last_message_time: float = 0.0  # Timestamp of last received message for health monitoring
  reconnection_attempts: int = 0  # Count of connection retry attempts
  error_message: str | None = None  # Last error encountered, if any


class TextState:
  """Manages the complete state of text that has been typed to the desktop."""

  def __init__(self):
    self.completed_segments: list[str] = []  # List of finalized transcription segments
    self.current_in_progress_text: str = ""  # Text of current in-progress segment
    self.current_segment_id: int | None = None  # ID of current in-progress segment
    self.total_typed_length: int = 0  # Total character count of all typed text

  def get_complete_text(self) -> str:
    """Returns full text as currently typed (completed + in-progress)."""
    completed_text = "".join(self.completed_segments)
    return completed_text + self.current_in_progress_text

  def calculate_update(self, new_segment: Segment) -> TextUpdate:
    """Determines typing actions needed to update from current to new segment state."""
    # Handle new segment case
    if self.current_segment_id is None or new_segment.id != self.current_segment_id:
      return self._handle_new_segment(new_segment)

    # Handle in-progress segment update
    return self._handle_in_progress_update(new_segment)

  def apply_segment_completion(self, completed_segment: str) -> None:
    """Moves in-progress text to completed segments list."""
    if self.current_in_progress_text:
      self.completed_segments.append(completed_segment)
      self.total_typed_length += len(completed_segment)
      self.current_in_progress_text = ""
      self.current_segment_id = None

  def reset_in_progress(self, new_segment: Segment) -> None:
    """Starts tracking new in-progress segment."""
    self.current_segment_id = new_segment.id
    self.current_in_progress_text = new_segment.text

  def _handle_new_segment(self, segment: Segment) -> TextUpdate:
    """Handle case where we're starting a new segment."""
    self.reset_in_progress(segment)
    return TextUpdate(
      chars_to_delete=0, text_to_type=segment.text, operation_type=UpdateType.NEW_SEGMENT
    )

  def _handle_in_progress_update(self, segment: Segment) -> TextUpdate:
    """Handle case where we're updating the current in-progress segment."""
    old_text = self.current_in_progress_text
    new_text = segment.text

    # Check if no change needed
    if old_text == new_text:
      return TextUpdate(chars_to_delete=0, text_to_type="", operation_type=UpdateType.NO_CHANGE)

    # Calculate diff and update internal state
    self.current_in_progress_text = new_text
    return calculate_text_diff(old_text, new_text)


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
