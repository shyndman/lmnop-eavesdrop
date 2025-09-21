# UI State Machine Specification

## Overview

The transcription UI operates as an always-running overlay that transitions between visible and invisible states based on content availability. The application maintains two independent content areas (TRANSCRIBE mode and COMMAND mode) and only becomes visible when at least one contains displayable text.

## Visibility States

### INVISIBLE State
- `body` element has `opacity: 0`
- Both `#transcription` and `#commands` elements are empty of content
- Default state when no transcription activity is occurring
- Window remains positioned but provides no visual feedback

### VISIBLE State
- `body` element has `opacity: 1`
- At least one of `#transcription` or `#commands` contains non-empty content
- Active during transcription sessions and command recognition

## State Transitions

### INVISIBLE → VISIBLE Triggers
Content-adding messages that result in non-empty text display:

- **`AppendSegmentsMessage`**: When `in_progress_segment` contains non-empty text OR `completed_segments` array contains segments
- **`SetSegmentsMessage`**: When `segments` array contains non-empty segments
- **`SetStringMessage`**: When `content` field contains non-empty string

### VISIBLE → INVISIBLE Triggers
Content-clearing messages that result in both modes being empty:

- **`SetStringMessage`**: When `content` is empty string, clearing the target mode, AND this leaves both transcription and command modes empty
- **`CommitOperationMessage`**: Always clears content from both modes regardless of `cancelled` flag value

### No Visibility Change
Messages that don't affect the empty/non-empty content state:

- **`CommandExecutedMessage`**: Provides visual feedback but doesn't modify content
- **`ChangeModeMessage`**: Switches active mode but doesn't clear content from either mode
- **Content-adding messages when already VISIBLE**: Stay visible, no re-trigger of fade-in
- **`AppendSegmentsMessage` with empty `in_progress_segment`**: Wait for next message (invalid state - there's always content)

## Content Management Rules

### Independent Mode Content
- `#transcription` (TRANSCRIBE mode) and `#commands` (COMMAND mode) maintain separate content
- Mode switching via `ChangeModeMessage` does not clear content from either element
- Visibility depends on WHETHER ANY mode has content, not which mode is currently active

### Content Clearing Behavior
- **`SetStringMessage`**: Clears and replaces content in the specified `target_mode` only
- **`CommitOperationMessage`**: Clears content from BOTH modes (complete session reset)
- **Mode switching**: Never clears content

### Content Addition Behavior
- **`AppendSegmentsMessage`**: Adds to existing content in specified `target_mode`
- **`SetSegmentsMessage`**: Completely replaces content in specified `target_mode`
- **`SetStringMessage`**: Completely replaces content in specified `target_mode`

## Message-Specific Implementation

### AppendSegmentsMessage
1. Remove any existing in-progress segment elements from `target_mode` DOM element
2. Append `completed_segments` as permanent content
3. Append `in_progress_segment` as temporary content (if non-empty)
4. **Visibility**: Trigger fade-in if transitioning from both-modes-empty to any-content

### SetSegmentsMessage
1. Clear all existing content from `target_mode` DOM element
2. Render all `segments` as permanent content
3. **Visibility**: Trigger fade-in if segments non-empty and transitioning from both-modes-empty

### SetStringMessage
1. Clear all existing content from `target_mode` DOM element
2. Process `content` string (markdown → HTML preprocessing)
3. Render processed content
4. **Visibility**:
   - Trigger fade-in if content non-empty and transitioning from both-modes-empty
   - Trigger fade-out if content empty and this leaves both modes empty

### ChangeModeMessage
1. Update visual focus indicators (`.has-focus` class)
2. Redirect subsequent transcription messages to new `target_mode` element
3. **Visibility**: No change (content remains in both modes)

### CommandExecutedMessage
1. Provide visual feedback that command recognition has begun
2. No content modification
3. **Visibility**: No change

### CommitOperationMessage
1. Clear content from BOTH `#transcription` and `#commands` elements
2. Reset application to TRANSCRIBE mode
3. **Visibility**: Always trigger fade-out (both modes now empty)

## Technical Implementation

### CSS Transitions
```css
body {
  transition: opacity 240ms ease-out;
}
```

### Visibility Management
- Use `opacity` property for fade transitions (not `visibility` or `display`)
- Transition duration: `240ms`
- Easing function: `ease-out`

### State Tracking
The renderer must track:
- Current content state of `#transcription` element (empty/non-empty)
- Current content state of `#commands` element (empty/non-empty)
- Current visibility state (visible/invisible)

### Edge Cases

#### Empty In-Progress Segment
- `AppendSegmentsMessage` with empty `in_progress_segment.text` is treated as no content addition
- Wait for subsequent message with actual content before triggering visibility change
- This handles the rapid message updates during initial transcription startup

#### Rapid Mode Switching
- Multiple `ChangeModeMessage` calls don't affect visibility
- Content remains preserved in both modes during switches
- Only content-modifying messages can trigger visibility changes

#### Multiple Content Messages When Visible
- Additional content messages while already visible don't retrigger fade-in
- Smooth content updates without flickering transitions
- Only empty ↔ content boundaries trigger visibility transitions