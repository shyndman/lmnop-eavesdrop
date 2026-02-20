# Research: Refactor Client Project as Importable Library

## Overview
This document consolidates research findings for transforming the CLI-based eavesdrop client into a clean, importable Python library with factory methods and async iterator interfaces.

## Research Areas

### 1. Python Async Iterator Design Patterns

**Decision**: Use `__aiter__()` and `__anext__()` protocol with internal async queue for transcription messages

**Rationale**: 
- Provides natural streaming interface that integrates well with `async for` loops
- Allows clean separation between WebSocket message handling and user consumption
- Enables backpressure handling through queue management
- Standard Python async iteration protocol ensures compatibility

**Alternatives Considered**:
- AsyncGenerator functions: Rejected due to lifecycle management complexity
- Observer/callback-only pattern: Rejected as less Pythonic than iteration
- Sync iterators with threading: Rejected due to complexity and GIL issues

### 2. Factory Method Pattern vs Constructor Parameters

**Decision**: Use class methods `EavesdropClient.transcriber()` and `EavesdropClient.subscriber()` 

**Rationale**:
- Type safety: Each factory method can have mode-specific parameter signatures
- Clear intent: Method names make usage obvious without reading documentation
- Validation: Can validate mode-specific parameters at creation time
- Extensibility: Easy to add new modes without breaking existing constructors

**Alternatives Considered**:
- Single constructor with mode parameter: Rejected due to weaker type safety
- Separate classes: Rejected due to code duplication concerns
- Builder pattern: Rejected as unnecessarily complex for this use case

### 3. WebSocket Lifecycle Management with Async Context Managers

**Decision**: Implement `__aenter__()` and `__aexit__()` for automatic connection management

**Rationale**:
- Guarantees proper cleanup even if exceptions occur during usage
- Follows Python resource management best practices
- Reduces boilerplate code for users
- Compatible with `async with` statements

**Alternatives Considered**:
- Manual connect/disconnect only: Rejected due to resource leak potential
- Automatic connection on first use: Rejected due to implicit behavior
- Finalizer-based cleanup: Rejected as unreliable in async contexts

### 4. Audio Device Selection Strategy

**Decision**: Support device selection by index, exact name, or substring matching via sounddevice integration

**Rationale**:
- Index selection: Fast and unambiguous for known device configurations
- Name matching: User-friendly for interactive usage
- Substring matching: Flexible for partial device names
- Sounddevice integration: Leverages existing robust device enumeration

**Alternatives Considered**:
- Device ID/UUID only: Rejected as less user-friendly
- Configuration file-based: Rejected as adding unnecessary complexity
- Auto-selection heuristics: Rejected due to unpredictability

### 5. Error Handling Strategy (Fail-Fast vs Graceful Degradation)

**Decision**: Fail-fast with immediate exceptions for invalid configurations, runtime errors via callbacks

**Rationale**:
- Configuration errors: Immediate feedback prevents silent failures
- Runtime errors: Callbacks allow application-specific handling without stopping iteration
- Clear separation: Setup vs runtime error handling use different mechanisms
- Debugging: Explicit failures are easier to diagnose than silent degradation

**Alternatives Considered**:
- Graceful degradation: Rejected per user requirements for clear failure modes
- Exception-only error handling: Rejected as stopping iteration for transient errors
- Logging-only errors: Rejected as potentially hiding critical issues

### 6. Thread Safety via Instance Isolation

**Decision**: Each client instance maintains completely isolated state without shared resources

**Rationale**:
- No shared state eliminates need for locks or synchronization primitives
- Independent WebSocket connections per instance
- Independent audio capture sessions per instance
- Simple mental model: one client = one connection + one audio session

**Alternatives Considered**:
- Shared connection pools: Rejected due to complexity and thread safety requirements
- Global state with locks: Rejected per user requirements against synchronization
- Singleton patterns: Rejected as preventing concurrent usage

### 7. Message Protocol Integration

**Decision**: Reuse existing `eavesdrop.wire` message types with enhanced callback signatures

**Rationale**:
- Maintains compatibility with existing server implementations
- Leverages existing Pydantic validation and serialization
- Preserves all metadata (segments, timestamps, confidence scores)
- Enhanced callbacks provide rich `TranscriptionMessage` objects instead of text-only

**Implementation Details**:
- Use existing types: `TranscriptionMessage`, `Segment`, `UserTranscriptionOptions`
- Support both `ClientType.TRANSCRIBER` and `ClientType.RTSP_SUBSCRIBER`
- Pass stream names via `WebSocketHeaders.STREAM_NAMES` for subscriber mode
- Remove stream filtering in message processing for multi-stream subscriber mode

**Alternatives Considered**:
- Text-only callbacks: Rejected as losing valuable metadata
- Custom message formats: Rejected due to compatibility concerns
- Message transformation layers: Rejected as unnecessary complexity

### 8. Configuration Options Exposure

**Decision**: Expose all transcription options (beam_size, word_timestamps, initial_prompt, hotwords) as factory method parameters

**Rationale**:
- Complete feature parity with existing CLI client
- Direct mapping to underlying Whisper/transcription engine capabilities
- Type-safe parameter validation at client creation
- Clear documentation of available options

**Alternatives Considered**:
- Configuration objects: Rejected as adding abstraction layer
- Post-creation configuration: Rejected due to state management complexity
- Subset of options: Rejected as reducing functionality

## Implementation Dependencies

### Required Libraries
- `websockets`: Mature, async-native WebSocket client implementation
- `sounddevice`: Cross-platform audio I/O with device enumeration
- `numpy`: Audio data manipulation and format conversion
- `pydantic`: Message validation and serialization (via eavesdrop-wire)
- `structlog`: Structured logging for observability
- `eavesdrop-wire`: Existing message protocol definitions

### Testing Strategy
- `pytest-asyncio`: Async test support
- Real WebSocket connections: Integration tests with actual server
- Real audio devices: Hardware integration tests with dummy/test audio devices
- Contract tests: Validate API surface and message protocols

## Architecture Decisions Summary

1. **Factory Methods**: `EavesdropClient.transcriber()` and `EavesdropClient.subscriber()`
2. **Async Iteration**: Native `__aiter__()/__anext__()` protocol implementation
3. **Context Management**: `async with` support for automatic lifecycle
4. **Instance Isolation**: Thread safety through independent instances
5. **Fail-Fast**: Immediate exceptions for configuration errors
6. **Protocol Preservation**: Direct reuse of existing message formats
7. **Full Configuration**: Expose all transcription options in factory methods

All research areas have been resolved with no remaining NEEDS CLARIFICATION items.