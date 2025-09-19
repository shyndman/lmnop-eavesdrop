"""Unit tests for text diffing algorithms in text_manager module."""

import pytest

from eavesdrop.active_listener.text_manager import (
  TextState,
  TextUpdate,
  UpdateType,
  calculate_text_diff,
  find_common_prefix,
)
from eavesdrop.wire.transcription import Segment


class TestFindCommonPrefix:
  """Test find_common_prefix function."""

  def test_empty_strings(self):
    assert find_common_prefix("", "") == ""
    assert find_common_prefix("hello", "") == ""
    assert find_common_prefix("", "world") == ""

  def test_identical_strings(self):
    assert find_common_prefix("hello", "hello") == "hello"
    assert find_common_prefix("test", "test") == "test"

  def test_no_common_prefix(self):
    assert find_common_prefix("hello", "world") == ""
    assert find_common_prefix("abc", "def") == ""

  def test_partial_common_prefix(self):
    assert find_common_prefix("hello", "help") == "hel"
    assert find_common_prefix("testing", "test") == "test"
    assert find_common_prefix("prefix", "prelude") == "pre"

  def test_one_string_is_prefix_of_other(self):
    assert find_common_prefix("test", "testing") == "test"
    assert find_common_prefix("testing", "test") == "test"

  def test_single_character_common(self):
    assert find_common_prefix("apple", "ant") == "a"
    assert find_common_prefix("x", "xyz") == "x"


class TestCalculateTextDiff:
  """Test calculate_text_diff function."""

  def test_empty_to_empty(self):
    result = calculate_text_diff("", "")
    assert result.chars_to_delete == 0
    assert result.text_to_type == ""
    assert result.operation_type == UpdateType.NO_CHANGE

  def test_empty_to_text(self):
    result = calculate_text_diff("", "hello")
    assert result.chars_to_delete == 0
    assert result.text_to_type == "hello"
    assert result.operation_type == UpdateType.NEW_SEGMENT

  def test_text_to_empty(self):
    result = calculate_text_diff("hello", "")
    assert result.chars_to_delete == 5
    assert result.text_to_type == ""
    assert result.operation_type == UpdateType.REPLACE_ALL

  def test_identical_text(self):
    result = calculate_text_diff("hello", "hello")
    assert result.chars_to_delete == 0
    assert result.text_to_type == ""
    assert result.operation_type == UpdateType.NO_CHANGE

  def test_no_common_prefix_replace_all(self):
    result = calculate_text_diff("hello", "world")
    assert result.chars_to_delete == 5
    assert result.text_to_type == "world"
    assert result.operation_type == UpdateType.REPLACE_ALL

  def test_suffix_replacement_extension(self):
    result = calculate_text_diff("hello", "hello world")
    assert result.chars_to_delete == 0
    assert result.text_to_type == " world"
    assert result.operation_type == UpdateType.REPLACE_SUFFIX

  def test_suffix_replacement_modification(self):
    result = calculate_text_diff("hello world", "hello there")
    assert result.chars_to_delete == 5  # delete "world" (common prefix is "hello ")
    assert result.text_to_type == "there"  # type "there" (space already in common prefix)
    assert result.operation_type == UpdateType.REPLACE_SUFFIX

  def test_suffix_replacement_truncation(self):
    result = calculate_text_diff("hello world", "hello")
    assert result.chars_to_delete == 6  # delete " world"
    assert result.text_to_type == ""
    assert result.operation_type == UpdateType.REPLACE_SUFFIX

  def test_complex_transcription_correction(self):
    # Common scenario: transcription gets corrected during speech
    result = calculate_text_diff("I think the answer", "I think the answer is")
    assert result.chars_to_delete == 0
    assert result.text_to_type == " is"
    assert result.operation_type == UpdateType.REPLACE_SUFFIX

  def test_word_correction_in_middle(self):
    # Transcription corrects a word in the middle
    result = calculate_text_diff("the cat sat on the mat", "the dog sat on the mat")
    assert result.chars_to_delete == 18  # delete "cat sat on the mat"
    assert result.text_to_type == "dog sat on the mat"
    assert result.operation_type == UpdateType.REPLACE_SUFFIX


class TestTextUpdate:
  """Test TextUpdate data class validation."""

  def test_valid_text_update(self):
    update = TextUpdate(5, "hello", UpdateType.REPLACE_SUFFIX)
    assert update.chars_to_delete == 5
    assert update.text_to_type == "hello"
    assert update.operation_type == UpdateType.REPLACE_SUFFIX

  def test_negative_chars_to_delete_raises(self):
    with pytest.raises(ValueError):
      TextUpdate(-1, "hello", UpdateType.REPLACE_SUFFIX)

  def test_non_string_text_raises(self):
    with pytest.raises(ValueError):
      TextUpdate(0, 123, UpdateType.REPLACE_SUFFIX)  # type: ignore


class TestTextState:
  """Test TextState class."""

  def test_initial_state(self):
    state = TextState()
    assert len(state.completed_segment_ids) == 0
    assert state.current_segment is None
    assert state.get_complete_text() == ""

  def test_get_complete_text_with_completed_only(self):
    state = TextState()
    segment1 = Segment(
      id=1,
      seek=0,
      start=0.0,
      end=1.0,
      text="Hello",
      tokens=[],
      avg_logprob=0.0,
      compression_ratio=1.0,
      words=None,
      temperature=0.0,
      completed=True,
    )
    segment2 = Segment(
      id=2,
      seek=10,
      start=1.0,
      end=2.0,
      text="world",
      tokens=[],
      avg_logprob=0.0,
      compression_ratio=1.0,
      words=None,
      temperature=0.0,
      completed=True,
    )
    state.completed_segment_ids[1] = segment1
    state.completed_segment_ids[2] = segment2
    assert state.get_complete_text() == "Hello world"

  def test_get_complete_text_with_in_progress(self):
    state = TextState()
    completed_segment = Segment(
      id=1,
      seek=0,
      start=0.0,
      end=1.0,
      text="Hello",
      tokens=[],
      avg_logprob=0.0,
      compression_ratio=1.0,
      words=None,
      temperature=0.0,
      completed=True,
    )
    state.completed_segment_ids[1] = completed_segment
    state.current_segment = Segment(
      id=2,
      seek=10,
      start=1.0,
      end=2.0,
      text="world",
      tokens=[],
      avg_logprob=0.0,
      compression_ratio=1.0,
      words=None,
      temperature=0.0,
      completed=False,
    )
    assert state.get_complete_text() == "Hello world"

  def test_segment_completion(self):
    state = TextState()
    # Set up initial in-progress segment
    in_progress_segment = Segment(
      id=1,
      seek=0,
      start=0.0,
      end=1.0,
      text="hello",
      tokens=[],
      avg_logprob=0.0,
      compression_ratio=1.0,
      words=None,
      temperature=0.0,
      completed=False,
    )
    state.current_segment = in_progress_segment

    # Complete the segment
    completed_segment = Segment(
      id=1,
      seek=0,
      start=0.0,
      end=1.0,
      text="hello world",
      tokens=[],
      avg_logprob=0.0,
      compression_ratio=1.0,
      words=None,
      temperature=0.0,
      completed=True,
    )
    state.process_segment(completed_segment)

    assert 1 in state.completed_segment_ids
    assert state.completed_segment_ids[1].text == "hello world"
    assert state.current_segment is None

  def test_new_in_progress_segment(self):
    state = TextState()
    segment = Segment(
      id=1,
      seek=0,
      start=0.0,
      end=1.0,
      text="hello",
      tokens=[],
      avg_logprob=0.0,
      compression_ratio=1.0,
      words=None,
      temperature=0.0,
      completed=False,
    )

    state.process_segment(segment)

    assert state.current_segment is not None
    assert state.current_segment.id == 1
    assert state.current_segment.text == "hello"

  def test_process_segment_new_segment(self):
    state = TextState()
    segment = Segment(
      id=1,
      seek=0,
      start=0.0,
      end=1.0,
      text="hello",
      tokens=[],
      avg_logprob=0.0,
      compression_ratio=1.0,
      words=None,
      temperature=0.0,
      completed=False,
    )

    update = state.process_segment(segment)

    assert update is not None
    assert update.chars_to_delete == 0
    assert update.text_to_type == "hello"
    assert update.operation_type == UpdateType.NEW_SEGMENT
    assert state.current_segment is not None
    assert state.current_segment.id == 1
    assert state.current_segment.text == "hello"

  def test_process_segment_in_progress_no_change(self):
    state = TextState()
    # Set initial segment
    initial_segment = Segment(
      id=1,
      seek=0,
      start=0.0,
      end=1.0,
      text="hello",
      tokens=[],
      avg_logprob=0.0,
      compression_ratio=1.0,
      words=None,
      temperature=0.0,
      completed=False,
    )
    state.current_segment = initial_segment

    segment = Segment(
      id=1,
      seek=0,
      start=0.0,
      end=1.0,
      text="hello",
      tokens=[],
      avg_logprob=0.0,
      compression_ratio=1.0,
      words=None,
      temperature=0.0,
      completed=False,
    )
    update = state.process_segment(segment)

    assert update is not None
    assert update.operation_type == UpdateType.NO_CHANGE
    assert update.chars_to_delete == 0
    assert update.text_to_type == ""

  def test_process_segment_in_progress_extension(self):
    state = TextState()
    # Set initial segment
    initial_segment = Segment(
      id=1,
      seek=0,
      start=0.0,
      end=1.0,
      text="hello",
      tokens=[],
      avg_logprob=0.0,
      compression_ratio=1.0,
      words=None,
      temperature=0.0,
      completed=False,
    )
    state.current_segment = initial_segment

    segment = Segment(
      id=1,
      seek=0,
      start=0.0,
      end=1.0,
      text="hello world",
      tokens=[],
      avg_logprob=0.0,
      compression_ratio=1.0,
      words=None,
      temperature=0.0,
      completed=False,
    )
    update = state.process_segment(segment)

    assert update is not None
    assert update.operation_type == UpdateType.REPLACE_SUFFIX
    assert update.chars_to_delete == 0
    assert update.text_to_type == " world"
    assert state.current_segment is not None
    assert state.current_segment.text == "hello world"

  def test_process_segment_different_segment_id(self):
    state = TextState()
    # Set initial segment
    initial_segment = Segment(
      id=1,
      seek=0,
      start=0.0,
      end=1.0,
      text="hello",
      tokens=[],
      avg_logprob=0.0,
      compression_ratio=1.0,
      words=None,
      temperature=0.0,
      completed=False,
    )
    state.current_segment = initial_segment

    # Complete first segment first
    completed_segment = Segment(
      id=1,
      seek=0,
      start=0.0,
      end=1.0,
      text="hello",
      tokens=[],
      avg_logprob=0.0,
      compression_ratio=1.0,
      words=None,
      temperature=0.0,
      completed=True,
    )
    state.process_segment(completed_segment)

    # Now process new segment
    segment = Segment(
      id=2,
      seek=0,
      start=1.0,
      end=2.0,
      text="world",
      tokens=[],
      avg_logprob=0.0,
      compression_ratio=1.0,
      words=None,
      temperature=0.0,
      completed=False,
    )
    update = state.process_segment(segment)

    assert update is not None
    assert update.operation_type == UpdateType.NEW_SEGMENT
    assert update.chars_to_delete == 0
    assert update.text_to_type == "world"
    assert state.current_segment is not None
    assert state.current_segment.id == 2
    assert state.current_segment.text == "world"
