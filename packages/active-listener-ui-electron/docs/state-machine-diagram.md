# UI State Machine Diagram

```mermaid
---
config:
  theme: redux
  layout: elk
---
stateDiagram-v2
    direction TB
    classDef activeState fill:#1976d2,color:#fff;
    classDef executingState fill:#f57c00,color:#fff;
    classDef inactiveState fill:#616161,color:#fff;

    [*] --> inactive
    inactive --> transcribe_active: Content Added
    transcribe_active --> command_active: Switch to Command
    command_active --> transcribe_active: Switch to Transcribe
    command_active --> command_executing: Command Executing
    command_executing --> transcribe_active: Results to Transcribe
    command_executing --> command_active: Results to Command

    transcribe_active --> inactive: Session End
    command_active --> inactive: Session End
    command_executing --> inactive: Session End

    inactive: Inactive
    transcribe_active: Transcribe Active
    command_active: Command Active
    command_executing: Command Executing

    note right of inactive
      AppendSegments/SetString triggers fade-in
    end note
    note left of inactive
      CommitOperation or content cleared
      1s feedback delay
    end note
    note left of transcribe_active
      #command fades based on content
    end note
    note right of command_active
      #command always visible
    end note
    note right of command_executing
      Shows waiting messages
      2s cycling loop
    end note

    class transcribe_active,command_active activeState
    class command_executing executingState
    class inactive inactiveState
```

## State Annotations

### inactive

- **Body class**: None (or `inactive`)
- **Body opacity**: 0
- **Content**: Both #transcription and #command empty
- **#command visibility**: Hidden
- **#overlay-layer**: Hidden

### transcribe_active

- **Body class**: `transcribe-active`
- **Body opacity**: 1
- **Content**: #transcription has content (min &nbsp;), #command may/may not
- **#command visibility**: Visible only if has content
- **#overlay-layer**: Hidden

### command_active

- **Body class**: `command-active`
- **Body opacity**: 1
- **Content**: #transcription has content, #command visible regardless of content
- **#command visibility**: Always visible
- **#overlay-layer**: Hidden

### command_executing

- **Body class**: `command-executing`
- **Body opacity**: 1
- **Content**: Previous content preserved
- **#command visibility**: Always visible
- **#overlay-layer**: Visible with cycling waiting messages (2s each)

## Transition Details

### Content Addition Transitions

- **inactive → transcribe_active**: Any content-adding message triggers fade-in
- **Within active states**: Content messages update DOM but don't change state

### Mode Switching

- **transcribe_active ↔ command_active**: Immediate body class change + #command fade
- **No content clearing**: Mode switches preserve all existing content

### Command Execution

- **command_active → command_executing**: Only possible from command mode
- **Exit**: Only via SetString with command results
- **Determines return state**: Based on target_mode in result message

### Session End

- **Any state → inactive**: CommitOperation OR SetString that empties all content
- **Special commit feedback**: 1-second visual indication before fade-out
- **Cleanup**: All overlays hidden, body class reset after fade completes
