# Phase 1: Data Model Design

## Core Entities

### TextState
**Purpose**: Manages the complete state of text that has been typed to the desktop

**Fields**:
- `completed_segments: list[str]` - List of finalized transcription segments that have been typed
- `current_in_progress_text: str` - Text of the current in-progress segment being typed
- `current_segment_id: int | None` - ID of the current in-progress segment for update tracking
- `total_typed_length: int` - Total character count of all typed text for efficient calculations

**Key Operations**:
- `get_complete_text() -> str` - Returns full text as currently typed (completed + in-progress)
- `calculate_update(new_segment: Segment) -> TextUpdate` - Determines typing actions needed
- `apply_segment_completion(completed_segment: str)` - Moves in-progress to completed
- `reset_in_progress(new_segment: Segment)` - Starts tracking new in-progress segment

**State Transitions**:
- Initial: Empty state, no segments
- In-progress update: Same segment ID, text changes → calculate diff
- Segment completion: Move current to completed list, clear in-progress
- New segment: Different segment ID → reset in-progress tracking

### TextUpdate
**Purpose**: Represents the typing actions needed to update desktop text from old to new state

**Fields**:
- `chars_to_delete: int` - Number of characters to backspace from current position
- `text_to_type: str` - New text content to type after deletions
- `operation_type: UpdateType` - Classification of update for logging/debugging

**Update Types**:
- `REPLACE_SUFFIX`: Update end of in-progress segment (common case)
- `REPLACE_ALL`: Complete replacement of in-progress segment
- `NEW_SEGMENT`: Start typing new segment after completion
- `NO_CHANGE`: No typing action needed

**Validation Rules**:
- `chars_to_delete >= 0` - Cannot delete negative characters
- `text_to_type` must be valid UTF-8 string
- Total operation should result in coherent text state

### ConnectionState
**Purpose**: Tracks the health and status of the eavesdrop server connection

**Fields**:
- `is_connected: bool` - Current WebSocket connection status
- `is_streaming: bool` - Whether audio streaming is active
- `last_message_time: float` - Timestamp of last received message for health monitoring
- `reconnection_attempts: int` - Count of connection retry attempts
- `error_message: str | None` - Last error encountered, if any

**State Transitions**:
- Disconnected → Connecting → Connected → Streaming
- Any state → Error (with reconnection attempts)
- Error → Connecting (retry logic)

**Health Monitoring**:
- Connection timeout detection (>30s without messages)
- Audio device availability tracking
- Server response latency measurement

### TypingOperation
**Purpose**: Encapsulates a single desktop typing action with error recovery

**Fields**:
- `operation_id: str` - Unique identifier for operation tracking
- `chars_to_delete: int` - Backspace operations to perform
- `text_to_type: str` - Text content to type
- `timestamp: float` - When operation was initiated
- `completed: bool` - Whether operation finished successfully

**Error Handling**:
- `ydotool` availability verification before operations
- Retry logic for failed typing operations
- Operation atomicity (all-or-nothing for complex updates)

## Data Relationships

### Message Flow
```
TranscriptionMessage → TextState.calculate_update() → TextUpdate → TypingOperation → ydotool
```

### State Dependencies
- `TextState` depends on `TranscriptionMessage.segments` for updates
- `TextUpdate` is calculated from current `TextState` and new segment data
- `TypingOperation` executes `TextUpdate` instructions via ydotool
- `ConnectionState` influences error recovery and retry behavior

### Error Recovery Chain
1. Connection failure → `ConnectionState.error` → reconnection attempts
2. ydotool failure → `TypingOperation.retry` → fallback strategies
3. Segment processing error → `TextState` preservation → user notification

## Validation Rules

### TextState Validation
- `completed_segments` must contain only non-empty strings
- `current_segment_id` must match active segment from server when present
- `total_typed_length` must equal sum of all segment lengths
- Text state must be recoverable from connection interruptions

### Transcription Message Validation
- Must contain at least one segment (completed or in-progress)
- In-progress segment (completed=False) must be last in segments list
- Segment IDs must be unique and monotonically increasing
- Segment text must be valid UTF-8 strings

### Typing Operation Validation
- `chars_to_delete` cannot exceed current text length
- Combined delete + type operations must result in coherent text
- Operations must be idempotent (safe to retry)
- Unicode handling for multi-byte character deletions

## Performance Considerations

### Memory Efficiency
- Completed segments stored as immutable strings
- Text diffing uses string slicing rather than character arrays
- Limited history retention (configurable segment limit)

### Computational Efficiency
- O(1) prefix matching for common in-progress updates
- Minimal string allocations during diff calculations
- Batch typing operations when possible via ydotool

### Network Efficiency
- Single previous segment context reduces message size
- WebSocket connection reuse with keepalive
- Async message processing prevents blocking

## Thread Safety

### Async Context
- All operations designed for single async event loop
- No shared mutable state between coroutines
- Message queue handling is naturally serialized

### External System Integration
- ydotool operations are synchronous but isolated
- Connection state updates are atomic
- Error recovery maintains consistency across async boundaries

## Testing Data Patterns

### Unit Test Scenarios
- Text state transitions with various segment patterns
- Edge cases: empty segments, unicode characters, very long text
- Error conditions: invalid segment IDs, malformed messages

### Integration Test Data
- Real TranscriptionMessage samples from eavesdrop server
- Connection failure and recovery scenarios
- Audio device availability simulation

### End-to-End Test Cases
- Complete typing workflow from audio to desktop
- Keyboard interrupt and graceful shutdown
- Multiple rapid segment updates (stress testing)