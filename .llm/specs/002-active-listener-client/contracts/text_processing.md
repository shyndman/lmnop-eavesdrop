# Text Processing Contract

## Text State Management Interface

### TextState Class Contract

```python
class TextState:
    def get_complete_text(self) -> str:
        """Returns the complete text as currently typed to desktop.

        Returns:
            Combined text of all completed segments plus current in-progress segment
        """

    def calculate_update(self, message: TranscriptionMessage) -> TextUpdate:
        """Calculate minimal typing operations needed to update desktop text.

        Args:
            message: New transcription message with segments

        Returns:
            TextUpdate with deletion count and text to type

        Raises:
            ValueError: If message contains invalid segment data
        """

    def apply_update(self, update: TextUpdate) -> None:
        """Apply calculated update to internal text state.

        Args:
            update: Update operations to apply

        Post-conditions:
            - Internal state matches expected result of update operations
            - completed_segments updated if segments were completed
            - current_in_progress_text reflects new in-progress segment
        """
```

### TextUpdate Class Contract

```python
class TextUpdate:
    chars_to_delete: int
    text_to_type: str
    operation_type: UpdateType

    def is_no_op(self) -> bool:
        """Returns True if no typing operations are needed."""

    def validate(self) -> None:
        """Validate update operations are coherent.

        Raises:
            ValueError: If chars_to_delete < 0 or text contains invalid characters
        """
```

## Text Diffing Algorithm Contract

### Prefix Matching Function

```python
def find_common_prefix(old_text: str, new_text: str) -> int:
    """Find length of common prefix between two text strings.

    Args:
        old_text: Previously typed text
        new_text: New text to be typed

    Returns:
        Length of common prefix in characters

    Performance:
        O(min(len(old_text), len(new_text))) time complexity
        O(1) space complexity
    """
```

### Update Calculation Function

```python
def calculate_text_diff(current_text: str, target_text: str) -> TextUpdate:
    """Calculate minimal operations to transform current text to target text.

    Args:
        current_text: Text currently on desktop
        target_text: Desired final text state

    Returns:
        TextUpdate with minimal character operations

    Algorithm:
        1. Find longest common prefix
        2. Calculate characters to delete (current_length - prefix_length)
        3. Calculate text to type (target_text[prefix_length:])

    Edge Cases:
        - Empty strings: Handle gracefully
        - Unicode characters: Proper multi-byte handling
        - Same text: Return no-op update
    """
```

## Segment Processing Contract

### Message Validation

```python
def validate_transcription_message(message: TranscriptionMessage) -> None:
    """Validate transcription message structure and content.

    Args:
        message: Message to validate

    Raises:
        ValueError: If message structure is invalid

    Validation Rules:
        - Must contain at least one segment
        - At most one in-progress segment (completed=False)
        - In-progress segment must be last in segments list
        - Segment IDs must be unique within message
        - All segment text must be valid UTF-8
    """
```

### Segment State Transitions

```python
def process_segments(
    current_state: TextState,
    new_segments: list[Segment]
) -> tuple[list[str], Segment | None]:
    """Process segment list to extract completed and in-progress segments.

    Args:
        current_state: Current text state for comparison
        new_segments: New segments from transcription message

    Returns:
        Tuple of (completed_segment_texts, in_progress_segment)

    Processing Rules:
        - Completed segments (completed=True) are finalized
        - In-progress segment (completed=False) may change in future messages
        - Segment completion detected by comparing IDs with current state
        - New in-progress segment starts when ID differs from current
    """
```

## Unicode and Character Handling

### Character Deletion Contract

```python
def safe_character_deletion(text: str, chars_to_delete: int) -> str:
    """Safely delete characters from end of text, respecting Unicode boundaries.

    Args:
        text: Text to delete characters from
        chars_to_delete: Number of characters to delete

    Returns:
        Text with characters removed from end

    Raises:
        ValueError: If chars_to_delete > len(text)

    Unicode Safety:
        - Properly handles multi-byte UTF-8 characters
        - Never breaks Unicode character boundaries
        - Handles combining characters and emoji correctly
    """
```

### Text Validation

```python
def validate_text_content(text: str) -> None:
    """Validate text content is suitable for desktop typing.

    Args:
        text: Text to validate

    Raises:
        ValueError: If text contains invalid or problematic characters

    Validation Rules:
        - Must be valid UTF-8
        - No control characters except common whitespace
        - No characters that could interfere with ydotool operation
    """
```

## Error Recovery Contract

### State Consistency

```python
def verify_state_consistency(state: TextState, expected_text: str) -> bool:
    """Verify text state matches expected desktop text.

    Args:
        state: Current text state object
        expected_text: Text that should be on desktop

    Returns:
        True if state is consistent with expected text

    Used for:
        - Error detection after typing operations
        - State validation during reconnection
        - Debug verification in development
    """
```

### Recovery Operations

```python
def recover_from_inconsistent_state(
    current_state: TextState,
    actual_desktop_text: str
) -> TextUpdate:
    """Calculate recovery operations when state becomes inconsistent.

    Args:
        current_state: What we think the text state is
        actual_desktop_text: What text is actually on desktop

    Returns:
        Update operations to restore consistency

    Recovery Strategy:
        - Clear all text and retype from scratch (safe but visible)
        - Used when incremental updates have failed
        - Logs inconsistency for debugging
    """
```

## Performance Requirements

### Time Complexity
- `find_common_prefix`: O(min(old_length, new_length))
- `calculate_text_diff`: O(min(current_length, target_length))
- `process_segments`: O(number_of_segments)
- All operations must complete within 10ms for real-time responsiveness

### Memory Usage
- Text state storage: O(total_text_length)
- Update calculations: O(1) additional memory
- No unbounded memory growth with long transcription sessions

### Accuracy Requirements
- Zero text loss during segment updates
- Exact character-by-character consistency
- Proper Unicode handling in all operations
- Deterministic results for same input