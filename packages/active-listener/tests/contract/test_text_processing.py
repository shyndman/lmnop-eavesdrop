"""Contract tests for text processing functions.

Tests the text state management and diffing algorithms that handle
incremental transcription updates and calculate minimal typing operations.

CRITICAL: These tests must fail until implementation is complete.
"""

import pytest

from eavesdrop.active_listener.text_manager import (
  TextState,
  TextUpdate,
  UpdateType,
  calculate_text_diff,
  find_common_prefix,
)
from eavesdrop.wire.transcription import Segment


class TestTextProcessingContracts:
  """Test text processing function contracts."""

  def test_find_common_prefix_behavior(self):
    """Test find_common_prefix function with various input scenarios."""
    # Partial match scenario
    result = find_common_prefix("hello world", "hello there")
    assert result == "hello "

    # Empty string scenarios
    result = find_common_prefix("", "hello")
    assert result == ""

    result = find_common_prefix("hello", "")
    assert result == ""

    # Identical strings
    result = find_common_prefix("same", "same")
    assert result == "same"

    # No common prefix
    result = find_common_prefix("abc", "def")
    assert result == ""

    # One string is prefix of another
    result = find_common_prefix("hello", "hello world")
    assert result == "hello"

    result = find_common_prefix("hello world", "hello")
    assert result == "hello"

  def test_text_state_get_complete_text(self):
    """Test that get_complete_text returns correct combined text."""
    state = TextState()

    # Add completed segments using the actual data structure
    segment1 = Segment(
      id=1,
      seek=0,
      start=0.0,
      end=1.0,
      text="Hello",
      tokens=[1],
      avg_logprob=-0.5,
      compression_ratio=1.0,
      words=None,
      temperature=None,
      completed=True,
    )
    segment2 = Segment(
      id=2,
      seek=10,
      start=1.0,
      end=2.0,
      text="world",
      tokens=[2],
      avg_logprob=-0.5,
      compression_ratio=1.0,
      words=None,
      temperature=None,
      completed=True,
    )
    state.completed_segment_ids[1] = segment1
    state.completed_segment_ids[2] = segment2

    # Set current in-progress segment
    state.current_segment = Segment(
      id=3,
      seek=20,
      start=2.0,
      end=3.0,
      text="this is",
      tokens=[3],
      avg_logprob=-0.5,
      compression_ratio=1.0,
      words=None,
      temperature=None,
      completed=False,
    )

    result = state.get_complete_text()
    assert result == "Hello world this is"

  def test_text_state_process_segment_new_segment(self):
    """Test process_segment for new segment scenario."""
    state = TextState()

    # Create segment with all required fields
    segment = Segment(
      id=1,
      seek=0,
      start=0.0,
      end=1.0,
      text="Hello",
      tokens=[1, 2, 3],
      avg_logprob=-0.5,
      compression_ratio=1.0,
      words=None,
      temperature=None,
      completed=False,
    )

    update = state.process_segment(segment)

    assert isinstance(update, TextUpdate)
    assert update.operation_type == UpdateType.NEW_SEGMENT
    assert update.text_to_type == "Hello"
    assert update.chars_to_delete == 0

  def test_text_state_process_segment_in_progress_change(self):
    """Test process_segment for in-progress segment changes."""
    state = TextState()

    # Set up initial in-progress segment
    initial_segment = Segment(
      id=1,
      seek=0,
      start=0.0,
      end=1.0,
      text="Hell",
      tokens=[1, 2, 3],
      avg_logprob=-0.5,
      compression_ratio=1.0,
      words=None,
      temperature=None,
      completed=False,
    )
    state.current_segment = initial_segment

    # Create updated segment with all required fields
    segment = Segment(
      id=1,
      seek=0,
      start=0.0,
      end=2.0,
      text="Hello world",
      tokens=[1, 2, 3, 4],
      avg_logprob=-0.5,
      compression_ratio=1.0,
      words=None,
      temperature=None,
      completed=False,
    )
    update = state.process_segment(segment)

    assert isinstance(update, TextUpdate)
    assert update.operation_type in [UpdateType.REPLACE_SUFFIX, UpdateType.REPLACE_ALL]
    assert update.chars_to_delete >= 0
    assert len(update.text_to_type) > 0

  def test_text_state_segment_completion(self):
    """Test segment completion moves text correctly."""
    state = TextState()

    # Set up initial in-progress segment
    segment = Segment(
      id=1,
      seek=0,
      start=0.0,
      end=1.0,
      text="Hello world",
      tokens=[1, 2, 3],
      avg_logprob=-0.5,
      compression_ratio=1.0,
      words=None,
      temperature=None,
      completed=False,
    )
    state.current_segment = segment

    # Now complete the segment
    completed_segment = Segment(
      id=1,
      seek=0,
      start=0.0,
      end=1.0,
      text="Hello world",
      tokens=[1, 2, 3],
      avg_logprob=-0.5,
      compression_ratio=1.0,
      words=None,
      temperature=None,
      completed=True,
    )

    state.process_segment(completed_segment)

    assert 1 in state.completed_segment_ids
    assert state.completed_segment_ids[1].text == "Hello world"
    assert state.current_segment is None

  def test_text_state_new_in_progress_segment(self):
    """Test new in-progress segment tracking."""
    state = TextState()

    segment = Segment(
      id=2,
      seek=100,
      start=2.0,
      end=3.0,
      text="New text",
      tokens=[10, 20, 30],
      avg_logprob=-0.3,
      compression_ratio=1.2,
      words=None,
      temperature=None,
      completed=False,
    )

    update = state.process_segment(segment)

    assert state.current_segment is not None
    assert state.current_segment.id == 2
    assert state.current_segment.text == "New text"
    assert update is not None
    assert update.operation_type == UpdateType.NEW_SEGMENT

  def test_text_update_validation_rules(self):
    """Test TextUpdate validation rules from specification."""
    # This will fail until validation is implemented

    # chars_to_delete must be >= 0
    with pytest.raises(ValueError):
      TextUpdate(chars_to_delete=-1, text_to_type="test", operation_type=UpdateType.NO_CHANGE)

    # text_to_type must be valid string
    update = TextUpdate(
      chars_to_delete=0, text_to_type="valid", operation_type=UpdateType.NO_CHANGE
    )
    assert update.text_to_type == "valid"

  def test_complex_text_diff_scenarios(self):
    """Test text diffing with complex real-world scenarios."""
    # This will fail until implementation

    # Prefix matching scenario
    old_text = "The quick brown"
    new_text = "The quick brown fox jumps"

    result = calculate_text_diff(old_text, new_text)
    assert result.chars_to_delete == 0
    assert result.text_to_type == " fox jumps"
    assert result.operation_type == UpdateType.REPLACE_SUFFIX

    # Complete replacement scenario
    old_text = "Hello world"
    new_text = "Goodbye universe"

    result = calculate_text_diff(old_text, new_text)
    assert result.chars_to_delete == len(old_text)
    assert result.text_to_type == "Goodbye universe"
    assert result.operation_type == UpdateType.REPLACE_ALL

  def test_unicode_handling(self):
    """Test that text processing handles Unicode characters correctly."""
    unicode_text = "Hello 世界 🌍"
    state = TextState()

    # Set current segment with Unicode text
    state.current_segment = Segment(
      id=1,
      seek=0,
      start=0.0,
      end=1.0,
      text=unicode_text,
      tokens=[1],
      avg_logprob=-0.5,
      compression_ratio=1.0,
      words=None,
      temperature=None,
      completed=False,
    )

    result = state.get_complete_text()
    assert unicode_text in result

    # Test diffing with Unicode
    old_text = "Hello 世"
    new_text = "Hello 世界"

    result = calculate_text_diff(old_text, new_text)
    assert result.text_to_type == "界"
    assert result.chars_to_delete == 0

  def test_empty_and_edge_cases(self):
    """Test edge cases with empty strings and boundary conditions."""
    # This will fail until edge case handling is implemented

    # Empty to non-empty
    result = calculate_text_diff("", "Hello")
    assert result.chars_to_delete == 0
    assert result.text_to_type == "Hello"
    assert result.operation_type == UpdateType.NEW_SEGMENT

    # Non-empty to empty
    result = calculate_text_diff("Hello", "")
    assert result.chars_to_delete == 5
    assert result.text_to_type == ""
    assert result.operation_type == UpdateType.REPLACE_ALL

    # Same text
    result = calculate_text_diff("Same", "Same")
    assert result.chars_to_delete == 0
    assert result.text_to_type == ""
    assert result.operation_type == UpdateType.NO_CHANGE
