# Active Listener UI Integration Architecture

## Overview

This document details the architectural refactoring needed to integrate Active Listener with its UI application. The system transforms from a simple transcription-to-typing tool into a voice-driven text workspace where users can iteratively build and refine text content through both transcription and voice commands.

## Goals

1. **UI Integration**: Launch and manage a UI subprocess that displays transcription state
2. **Text Workspace**: Create a REPL-like environment for iterative text editing via voice
3. **Dual Mode Operation**: Support both transcription mode and command mode for text manipulation
4. **Clean Architecture**: Separate CLI concerns from business logic
5. **Process Management**: Robust subprocess lifecycle management with proper error handling

## Current Architecture Problems

### CLI God Object
The `ActiveListener` class in `__main__.py` violates single responsibility by handling:
- Argument parsing (appropriate)
- Client initialization (belongs in App)
- Signal handler setup (belongs in App)
- Application lifecycle management (belongs in App)
- Component orchestration (belongs in App)

### Missing UI Communication
- No subprocess management for UI application
- No message protocol for state synchronization
- No architecture for bidirectional state management

### Tight Coupling
- App directly owns text manager and typist
- No abstraction for different output modes (typing vs UI display)
- Hard to extend for new interaction modes

## New Architecture Design

### Component Hierarchy

```
ActiveListener (CLI)
  └── App (Application Controller)
      ├── EavesdropClientWrapper (Existing)
      ├── UIChannel (New - UI Process Communication)
      ├── TextTranscriptionWorkspace (New - Core Logic)
      └── YdoToolTypist (Existing - Desktop Typing)
```

### Key Principles

1. **Single Responsibility**: Each component has one clear purpose
2. **Dependency Injection**: App creates and wires components
3. **Message-Driven**: Communication through well-defined message protocols
4. **Fail-Fast**: Fatal errors on critical component failures
5. **Process Ownership**: Clear subprocess lifecycle management

## Component Specifications

### ActiveListener (CLI - Refactored)

**Purpose**: Thin argument parser and application launcher

**Responsibilities**:
- Parse and validate command line arguments
- Create App instance with validated configuration
- Delegate to App for all business logic

**Interface**:
```python
class ActiveListener(Command):
    # CLI argument definitions (existing)
    server: ServerHostPort
    audio_device: str
    ui_bin: Path

    async def run(self) -> None:
        app = App.create(
            server_host=self.server.host,
            server_port=self.server.port,
            audio_device=self.audio_device,
            ui_bin_path=self.ui_bin
        )
        await app.run()
```

**Changes Required**:
- Remove all business logic from `__init__` and `run`
- Remove signal handler setup
- Remove component initialization
- Become pure argument parser + delegator

### App (Application Controller - Enhanced)

**Purpose**: Application lifecycle and component orchestration

**Responsibilities**:
- Create and configure all components
- Manage UI subprocess lifecycle
- Setup signal handlers for graceful shutdown
- Coordinate message flow between components
- Handle fatal errors and cleanup

**Interface**:
```python
class App:
    @classmethod
    def create(cls, server_host: str, server_port: int,
               audio_device: str, ui_bin_path: Path) -> "App":
        # Factory method that creates and wires all components

    async def run(self) -> None:
        # Main application entry point - replaces ActiveListener.run()

    def shutdown(self) -> None:
        # Graceful shutdown - replaces ActiveListener signal handling
```

**New Responsibilities**:
- UI subprocess management (launch, monitor, cleanup)
- Signal handler registration and management
- Component lifecycle coordination
- Error isolation and fatal error handling

### UIChannel (New - UI Communication)

**Purpose**: Subprocess communication wrapper

**Responsibilities**:
- Launch UI subprocess with validated executable path
- Monitor UI ready signal ("ACTIVE_LISTENER_UI_READY" on stdout)
- Send JSON-line messages to UI stdin
- Monitor UI stderr and log errors
- Handle broken pipe detection and fatal error signaling
- Clean subprocess termination with SIGTERM

**Interface**:
```python
class UIChannel:
    def __init__(self, ui_bin_path: Path, working_dir: Path):
        # Initialize with UI executable path and working directory

    async def start(self) -> None:
        # Launch subprocess and wait for ready signal

    def send_message(self, message: dict[str, Any]) -> None:
        # Send JSON-line message to UI stdin (fire-and-forget)

    async def shutdown(self) -> None:
        # Terminate subprocess with SIGTERM

    def is_healthy(self) -> bool:
        # Check if subprocess is still running
```

**Error Handling**:
- Process launch failure → Fatal error
- Ready signal timeout → Fatal error (wait indefinitely but log warnings)
- Broken pipe on stdin → Fatal error
- Subprocess crash → Fatal error

**Message Format**:
- JSON Lines: One JSON object per line, terminated by newline
- Fire-and-forget: No acknowledgment expected
- Low volume: ~1 message per second expected

### TextTranscriptionWorkspace (New - Core Logic)

**Purpose**: Voice-driven text workspace management

**Responsibilities**:
- Track current text content state
- Manage transcription vs command mode
- Process incoming transcription messages
- Generate UI update messages
- Maintain segment completion tracking for UI

**Interface**:
```python
class TextTranscriptionWorkspace:
    def __init__(self, ui_channel: UIChannel):
        # Initialize with UI communication channel

    def process_transcription_message(self, message: TranscriptionMessage) -> None:
        # Main message processing pipeline entry point

    def set_mode(self, mode: Mode) -> None:
        # Switch between TRANSCRIBE and COMMAND modes

    def get_current_text(self) -> str:
        # Return current workspace text content

    def get_current_mode(self) -> Mode:
        # Return current workspace mode
```

**State Management**:
- Current text content (actual text buffer)
- Current mode (TRANSCRIBE or COMMAND)
- Segment tracking for UI updates (completed vs in-progress)
- No persistence - starts fresh on each run

**Message Generation Logic**:
- Send `AppendSegmentsMessage` when transcription segments change
- Send `ChangeModeMessage` when mode switches
- Track which segments already sent to avoid duplication
- Follow UI protocol: only recently completed segments + current in-progress

## Message Flow Design

### Server to Workspace Flow

```
Eavesdrop Server
  → TranscriptionMessage
  → App._handle_transcription_message()
  → TextTranscriptionWorkspace.process_transcription_message()
  → UIChannel.send_message()
  → UI Process stdin
```

### Mode Change Flow

```
External Trigger (future: hotkey, voice command, etc.)
  → App.set_mode() or TextTranscriptionWorkspace.set_mode()
  → UIChannel.send_message(ChangeModeMessage)
  → UI Process stdin
```

### Error Flow

```
UIChannel detects process failure
  → Fatal error logged
  → App.shutdown() called
  → ActiveListener exits immediately
```

## Process Management

### UI Subprocess Lifecycle

1. **Launch**:
   - Spawn UI process with working directory set to ActiveListener's working directory
   - Pass ui_bin_path as executable
   - Inherit environment from ActiveListener process
   - Capture stdin, stdout, stderr

2. **Ready Detection**:
   - Monitor stdout for exact string "ACTIVE_LISTENER_UI_READY"
   - Wait indefinitely but log warnings after reasonable timeout
   - Fatal error if process exits before ready signal

3. **Communication**:
   - Send JSON-line messages to stdin
   - Monitor stderr and log any output (don't parse)
   - Fire-and-forget message sending

4. **Health Monitoring**:
   - Check process.poll() to detect crashes
   - Fatal error on unexpected termination

5. **Shutdown**:
   - Send SIGTERM to process
   - Wait reasonable time for graceful exit
   - Force kill if necessary during cleanup

### Error Conditions

All subprocess-related errors are **fatal** and cause immediate ActiveListener exit:
- UI executable not found or not executable
- Process launch failure
- Process crash during operation
- Broken pipe on stdin communication

## UI Message Protocol

### Message Format
JSON Lines format: `{"type": "message_type", ...}\n`

### Message Types

#### AppendSegmentsMessage
```json
{
  "type": "append_segments",
  "completed_segments": [/* only newly completed segments */],
  "in_progress_segment": {/* current incomplete segment */}
}
```

#### ChangeModeMessage
```json
{
  "type": "change_mode",
  "target_mode": "TRANSCRIBE" | "COMMAND"
}
```

### State Synchronization Rules

1. **Segment Tracking**: Only send newly completed segments, not historical data
2. **In-Progress Updates**: Always include current in-progress segment
3. **Mode Changes**: UI initializes in TRANSCRIBE mode, only send mode changes when switching
4. **Message Ordering**: Guaranteed in-order delivery (single-threaded processing)

## Implementation Plan

### Phase 1: Skeleton Structure ✅ COMPLETED
1. ✅ Create skeleton `UIChannel` class with process management
2. ✅ Create skeleton `TextTranscriptionWorkspace` class with state tracking
3. ✅ Create `App.create()` factory method
4. ✅ Refactor `ActiveListener` to be thin CLI parser

**Status**: All skeleton classes created with comprehensive Sphinx-style docstrings. Type checking and linting validation completed. System is broken as expected but has clean architecture foundation.

**Files Created/Modified**:
- `src/eavesdrop/active_listener/ui_channel.py` - New skeleton class
- `src/eavesdrop/active_listener/workspace.py` - New skeleton class
- `src/eavesdrop/active_listener/__main__.py` - Gutted to thin CLI parser
- `src/eavesdrop/active_listener/app.py` - Complete refactor with factory method

**Known Issues**:
- Import resolution issues for `eavesdrop.*` packages (expected - dependencies not installed in test environment)
- `App.create()` method returns `pass` instead of actual implementation (skeleton only)
- All skeleton methods contain only `pass` statements

### Phase 2: Process Management
1. Implement UI subprocess launch and ready signal detection
2. Implement JSON-line message sending
3. Implement error detection and fatal error handling
4. Add signal handler migration from ActiveListener to App

### Phase 3: Message Processing
1. Implement transcription message processing in workspace
2. Implement UI message generation and sending
3. Implement segment state tracking
4. Add mode switching API

### Phase 4: Integration
1. Wire workspace into App's message processing flow
2. Remove existing text manager integration (temporarily)
3. Test end-to-end message flow
4. Add error handling and edge cases

## Required Reading for Implementation

### Core Architecture Files
- `packages/active-listener/src/eavesdrop/active_listener/__main__.py` - Current CLI implementation to refactor
- `packages/active-listener/src/eavesdrop/active_listener/app.py` - Current App class to enhance
- `packages/active-listener/src/eavesdrop/active_listener/client.py` - EavesdropClientWrapper interface

### Message Protocol Files
- `packages/wire/src/eavesdrop/wire/messages.py` - Server message types (TranscriptionMessage)
- `packages/wire/src/eavesdrop/wire/transcription.py` - Segment and transcription data structures
- `packages/active-listener/src/eavesdrop/active_listener/messages.py` - UI message types to implement

### State Management Files
- `packages/active-listener/src/eavesdrop/active_listener/text_manager.py` - Current text state management (being replaced)

### Support Files
- `packages/active-listener/src/eavesdrop/active_listener/typist.py` - Desktop typing implementation (remains)
- `packages/common/src/eavesdrop/common/__init__.py` - Logging utilities

### Configuration Files
- `packages/active-listener/pyproject.toml` - Package structure and dependencies
- `CODE_STYLE.md` - Type safety and documentation requirements
- `CLAUDE.md` - Project-specific guidance

## Critical Design Decisions

### State Management
- **Text content tracking**: Workspace maintains actual text content, not just metadata
- **Mode persistence**: No persistence across runs - always start in TRANSCRIBE mode
- **Segment tracking**: Track completion state for UI messaging, not for text processing

### Error Handling
- **Fatal error philosophy**: Any UI process failure causes immediate exit
- **No fallback modes**: Don't attempt to continue without UI
- **Fast failure**: Better to crash quickly than operate in degraded state

### Process Communication
- **Fire-and-forget messaging**: No acknowledgment protocol needed
- **JSON Lines format**: Simple, human-readable, line-delimited
- **Single-threaded processing**: Guarantees message ordering without complex synchronization

### Architecture Principles
- **Separation of concerns**: CLI, App, UI communication, and text processing as distinct responsibilities
- **Dependency injection**: App creates and wires components rather than components self-configuring
- **Message-driven design**: Communication through well-defined message types rather than direct method calls
- **Resource ownership**: Clear ownership of subprocesses and lifecycle management

## Future Extension Points

### Mode Switching Triggers
- Voice command detection ("computer, switch to command mode")
- Hotkey integration
- External API endpoints
- Time-based automatic switching

### Additional Message Types
- Buffer clear commands
- Undo/redo operations
- Text transformation commands
- Export/save operations

### Enhanced Error Recovery
- UI process restart capability
- Graceful degradation modes
- State persistence across restarts

### Performance Optimization
- Message batching for high-frequency updates
- Differential segment updates
- Background UI process pre-warming