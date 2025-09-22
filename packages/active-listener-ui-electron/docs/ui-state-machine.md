# UI State Machine Specification

## Overview

The transcription UI operates as an always-running overlay that transitions between visible and invisible states based on content availability. The application maintains two independent content areas (TRANSCRIBE mode and COMMAND mode) with different visibility behaviors, and indicates the active mode through body classes.

## Visibility States

### Overall UI Visibility

#### Default State (No Content)
- Body has no visibility class
- CSS sets `body { opacity: 0 }`
- Both `#transcription` and `#command` elements are empty of content
- Default state when no transcription activity is occurring
- Window remains positioned but provides no visual feedback

#### Active State (Has Content)
- Body class: `active`
- CSS sets `body.active { opacity: 1 }`
- At least one of `#transcription` or `#command` contains non-empty content
- Active during transcription sessions and command recognition

### Element-Specific Visibility

#### Transcription Element (`#transcription`)
- **Always visible** when overall UI is visible
- **Always renders content**: minimum of `&nbsp;` when empty to maintain shape and visual presence
- **Never fades in/out independently** - follows overall UI visibility only
- Represents the primary composition area users are always aware of

#### Command Element (`#command`)
- **Conditionally visible** with independent fade transitions
- **VISIBLE when**:
  - Current mode is COMMAND (regardless of content), OR
  - Element contains non-empty text (regardless of current mode)
- **HIDDEN when**:
  - Current mode is TRANSCRIBE AND element is empty
- **Fade duration**: Same as overall UI transitions

### Active Mode Indication
- Body element class indicates current mode: `transcribe-active`, `command-active`, or `command-executing`
- **Default state**: No mode class (null) when UI is in default state (no content)
- **Initial mode**: Set by first content-adding message (`SetStringMessage`, `AppendSegmentsMessage`, or `SetSegmentsMessage`) when transitioning from default to active
- **Updates immediately** on `ChangeModeMessage` receipt
- **Command execution state**: `command-executing` when processing commands with waiting feedback
- **CommitOperationMessage exception**: Mode change to default state (no class) occurs after overall UI fade-out completes

### Command Execution Feedback
- **Overlay visibility**: `#overlay-layer` becomes visible during command execution
- **Waiting messages**: Displayed in `#command-waiting-messages` as cycling `<li>` elements
- **Message rotation**: Each message shows for 2 seconds, looping indefinitely
- **Default message**: "Generating..." when `waiting_messages` array is empty
- **Exit condition**: Cleared when `SetSegmentsMessage` or `SetStringMessage` arrives with results

## State Transitions

### Default → Active Triggers
Content-adding messages that result in non-empty text display:

- **`AppendSegmentsMessage`**: When `in_progress_segment` contains non-empty text OR `completed_segments` array contains segments
- **`SetSegmentsMessage`**: When `segments` array contains non-empty segments
- **`SetStringMessage`**: When `content` field contains non-empty string

### Active → Default Triggers
Content-clearing messages that result in both modes being empty:

- **`SetStringMessage`**: When `content` is empty string, clearing the target mode, AND this leaves both transcription and command modes empty
- **`CommitOperationMessage`**: Always clears content from both modes regardless of `cancelled` flag value

### No Visibility Change
Messages that don't affect the empty/non-empty content state:

- **`CommandExecutingMessage`**: Provides visual feedback but doesn't modify content
- **`ChangeModeMessage`**: Switches active mode but doesn't clear content from either mode
- **Content-adding messages when already active**: Stay active, no re-trigger of fade-in
- **`AppendSegmentsMessage` with empty `in_progress_segment`**: Wait for next message (invalid state - there's always content)

## Content Management Rules

### Independent Mode Content
- `#transcription` (TRANSCRIBE mode) and `#command` (COMMAND mode) maintain separate content
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
2. Create `<span>` elements for each `completed_segment`:
   - ID: `segment-${segment.id}`
   - CSS class: `segment-prob-{rounded_avg_probability}` (avg_probability * 100, rounded to nearest 5)
   - Staggered fade-in animation with 50ms delays between segments
3. Create `<span>` for `in_progress_segment` (if non-empty):
   - CSS class: `in-progress-segment segment-prob-{rounded_avg_probability}`
   - Fade-in animation with appropriate delay after completed segments
4. **Visibility**: Trigger fade-in if transitioning from both-modes-empty to any-content

### SetSegmentsMessage
1. **If currently in command-executing state**: Clear body class and hide `#overlay-layer`
2. Clear all existing content from `target_mode` DOM element
3. Create `<span>` elements for all `segments`:
   - ID: `segment-${segment.id}`
   - CSS class: `segment-prob-{rounded_avg_probability}` (avg_probability * 100, rounded to nearest 5)
   - Block fade-in animation (all segments fade in simultaneously as a group)
4. **Visibility**: Trigger fade-in if segments non-empty and transitioning from both-modes-empty

### SetStringMessage
1. **If currently in command-executing state**: Clear body class and hide `#overlay-layer`
2. Clear all existing content from `target_mode` DOM element
3. Process `content` string (markdown → HTML preprocessing)
4. Render processed content with block fade-in animation
5. **Mode transition**: If transitioning from default state (no content) to active state with non-empty content, set mode to `target_mode` (e.g., `transcribe-active` or `command-active`)
6. **Visibility**:
   - Trigger fade-in if content non-empty and transitioning from both-modes-empty
   - Trigger fade-out if content empty and this leaves both modes empty

### ChangeModeMessage
1. **Immediately** update body class to `transcribe-active` or `command-active`
2. **Command element visibility**: Apply conditional visibility logic based on mode and content state
   - If switching TO COMMAND: Immediately fade in command element (even if empty)
   - If switching FROM COMMAND: Fade out command element only if it's empty
3. Redirect subsequent transcription messages to new `target_mode` element
4. **Overall UI visibility**: No change (content remains in both modes)

### CommandExecutingMessage
1. Clear existing `<li>` elements from `#command-waiting-messages`
2. Create new `<li>` elements for each string in `waiting_messages` (or "Generating..." if empty)
3. Set body class to `command-executing`
4. Start cycling through waiting messages (2 seconds each, looping indefinitely)
5. **Visibility**: `#overlay-layer` becomes visible, no change to overall UI visibility

### CommitOperationMessage
TODO: You know, we should do something special here, because it's a state transition that needs some acknowledgement.

I think we should set a class on the body (commit-active?), which shows some visual representation of "committing", wait for 1 second for the user to notice, then allow the fade out transitions to take place.

1. Clear content from BOTH `#transcription` and `#command` elements
2. **Immediately** fade out command element (if visible)
3. **Immediately** trigger overall UI fade-out (both modes now empty)
4. **After fade-out completes**: Reset body class to `transcribe-active` mode

## Technical Implementation

### CSS Transitions
```css
body {
  opacity: 0;
  transition: opacity 240ms ease-out;
}

body.active {
  opacity: 1;
}
```

### Timing Constants
Define these as constants for easy tweaking:

```typescript
const TRANSITION_DURATION_MS = 240;
const COMMIT_FEEDBACK_DURATION_MS = 1000;
const WAITING_MESSAGE_DURATION_MS = 2000;
const SEGMENT_STAGGER_DELAY_MS = 50;
```

### Animation System
- **Segment animations**: Use enhanced `AnimatedValue` class with delay support
- **Staggered timing**: Each segment in `AppendSegments` delays by `index * SEGMENT_STAGGER_DELAY_MS`
- **Simultaneous timing**: All segments in `SetSegments/SetString` start immediately (delay = 0)
- **RAF integration**: Animations use `requestAnimationFrame` timestamp for frame-perfect timing
- **State tracking**: `AnimatedValue.isRunning()` returns true during both delay and animation phases

### Visibility Management
- Use body `active` class to control overall UI visibility
- CSS handles opacity transitions with `transition: opacity 240ms ease-out`
- Transition duration: `TRANSITION_DURATION_MS` (240ms)
- Easing function: `ease-out`
- All elements use the same transition duration for consistency

### State Tracking
The UIStateManager must track:
- Current content state of `#transcription` element (empty/non-empty)
- Current content state of `#command` element (empty/non-empty)
- Current overall visibility state (default/active)

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

## Implementation Status

### ✅ Completed Features

#### Core Infrastructure
- **UIStateManager class**: Central state management with content tracking outside DOM
- **Timing constants**: All magic numbers replaced with named constants (`TRANSITION_DURATION_MS`, etc.)
- **Easing constants**: Centralized easing function definitions (`FADE_OUT_EASING`, `FADE_IN_EASING`)
- **MessageHandler class**: Type-safe message processing with enum-based exhaustive checking
- **Mock API**: Development testing via `window._mock.setString()` for SetStringMessage

#### Content State Management
- **Independent mode tracking**: Separate boolean flags for transcription and command content state
- **Mode state management**: Tracks current active mode (`transcribe-active`, `command-active`, or null)
- **Overall visibility control**: Body `active` class based on content presence in any mode
- **Initial mode setting**: First content-adding message sets the active mode when transitioning from default state

#### Animation System
- **Unified opacity animation**: Single `animateOpacity()` method handling fade-in/fade-out logic
- **Paragraph-based animations**: Targets `<p>` elements for future segment/word support
- **Smooth content transitions**: Fade-out → fade-in sequences for content replacement
- **Animation assertions**: Fatal errors if animations overlap (concurrency protection)

#### Concurrency Protection
- **Explicit serialization tracking**: `contentSettingInProgress` Set prevents overlapping calls
- **Fatal error on violations**: Throws descriptive errors if serialization assumption is violated
- **Try/finally cleanup**: Ensures progress tracking is always cleared

#### SetStringMessage Implementation
- **Complete SetString handling**: All animation cases (empty→content, content→content, content→empty)
- **Paragraph tag wrapping**: Ensures all content is wrapped in `<p>` tags for future extensibility
- **Content preprocessing pipeline**: Ready for markdown → HTML transformation

### 🚧 Partial Implementation

#### Message Types
- **SetStringMessage**: ✅ Fully implemented with animations and state management
- **AppendSegmentsMessage**: ❌ Not implemented (needs segment span creation with staggered animations)
- **SetSegmentsMessage**: ❌ Not implemented (needs segment span creation with block animations)
- **ChangeModeMessage**: ❌ Not implemented (needs focus class management and command element visibility)
- **CommandExecutingMessage**: ❌ Not implemented (needs overlay layer and waiting message cycling)
- **CommitOperationMessage**: ❌ Not implemented (needs session reset and commit feedback)

### ❌ Missing Features

#### Command Element Conditional Visibility
- Command element independent fade transitions based on mode and content state

#### Segment-Based Content Rendering
- `<span>` element creation with segment IDs and probability classes
- Staggered fade-in animations for AppendSegments (50ms delays)
- Simultaneous fade-in animations for SetSegments

#### Command Execution Feedback
- `#overlay-layer` visibility during command execution
- `#command-waiting-messages` cycling with 2-second rotation
- Integration with CommandExecutingMessage

#### Content Preprocessing
- Markdown → HTML transformation for SetStringMessage
- Content validation and error handling

#### Commit Operation Handling
- Session reset clearing both modes
- Commit feedback visual state with timing
- Post-commit mode reset to `transcribe-active`

### Next Implementation Priorities
1. **AppendSegmentsMessage**: Segment span creation with staggered animations
2. **Command element conditional visibility**: Independent fade behavior
3. **ChangeModeMessage**: Focus management and mode switching
4. **SetSegmentsMessage**: Segment replacement with block animations
