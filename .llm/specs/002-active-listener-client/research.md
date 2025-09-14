# Phase 0: Research & Unknowns Resolution

## PyYdotool Integration Research

**Decision**: Use python-ydotool for desktop automation typing
**Rationale**:
- PyYdotool provides Python bindings for ydotool, which is a Linux uinput-based automation tool
- Supports typing text and simulating keyboard actions without requiring X11 (works with Wayland)
- Lightweight and reliable for desktop automation tasks
- Active maintenance and good compatibility with modern Linux desktop environments

**Key API Patterns**:
- `pydotool.init()` - Initialize pydotool (call once before use)
- `pydotool.type_string("text to type")` - Types text at current cursor position
- `pydotool.key_combination([KEY_LEFTCTRL, KEY_A])` - Sends key combinations for selection
- `pydotool.key_sequence([(KEY_BACKSPACE, DOWN), (KEY_BACKSPACE, UP)] * count)` - Multiple backspace operations

**Integration Approach**:
- Import python-ydotool in typer.py module
- Call `pydotool.init()` during application startup
- Handle deletion using repeated backspace key sequences
- Type new content using `type_string()` method
- Error handling for ydotool daemon availability and permissions

**Alternatives considered**:
- Direct uinput manipulation: Too low-level, PyYdotool provides better abstraction
- xdotool: X11 only, not compatible with Wayland environments
- pynput: May have permission issues on modern Linux systems

## Clypi CLI Framework Integration

**Decision**: Use Clypi for command-line interface
**Rationale**:
- Type-safe argument parsing using Python type annotations
- Async-first design matches our async client requirements
- Built-in UI features (spinners, colored output) for better user experience
- Class-based command definition is cleaner than decorator-based approaches
- Automatic --help and --version flag generation

**Key Patterns for Active Listener**:
```python
from clypi import Command, Positional, arg
from clypi.ui import spinner, cprint

class ActiveListenerCli(Command):
    host: str = arg(default="localhost")
    port: int = arg(default=9090)
    audio_device: str = arg(default="default")

    async def run(self):
        # Main async application logic
        pass
```

**Error Handling Strategy**:
- Use clypi.ui.cprint for colored error messages
- Structured error context with connection failures, audio device issues
- Graceful shutdown on keyboard interrupt

**Alternatives considered**:
- Click: Decorator-based, not async-native, requires more boilerplate
- argparse: Built-in but lacks modern features and async support
- Typer: Good but not as feature-rich for async use cases

## Text Diffing and State Management

**Decision**: Implement custom text diffing algorithm for segment updates
**Rationale**:
- TranscriptionMessage contains segments with completion status and IDs
- Need to track complete typed text state to calculate minimal character changes
- Must handle prefix matching between old and new in-progress segments efficiently

**Algorithm Approach**:
1. Maintain complete current text state (completed segments + current in-progress)
2. On segment update, find longest common prefix between old and new text
3. Calculate characters to delete (old text length - prefix length)
4. Type remaining characters (new text after prefix)
5. Update internal state to reflect new complete text

**Key Data Structures**:
```python
@dataclass
class TextState:
    completed_segments: list[str]
    current_segment: str | None
    current_segment_id: int | None

    def get_complete_text(self) -> str:
        # Returns full text as currently typed
        pass
```

**Edge Cases Handled**:
- New in-progress segment (different ID): Clear previous, type new
- Updated in-progress segment (same ID): Diff and update
- Segment completion: Move to completed list, start new in-progress
- Connection loss during typing: State preservation

**Alternatives considered**:
- Full text replacement: Inefficient, causes visual flickering
- External diff libraries: Overkill for simple prefix matching needs
- Server-side diffing: Not available, must be handled client-side

## Eavesdrop Client Integration Patterns

**Decision**: Use EavesdropClient.transcriber() factory with specific configuration
**Rationale**:
- Existing client API provides exactly what we need
- Async iterator pattern matches our real-time processing requirements
- Built-in WebSocket handling and error recovery
- Message queuing prevents loss of transcription updates

**Configuration for Active Listener**:
```python
client = EavesdropClient.transcriber(
    host=host,
    port=port,
    audio_device=audio_device,
    hotwords=["com"],  # Required hotword configuration
    # Key setting: only get 1 previous segment for context
    transcription_options=UserTranscriptionOptions(
        send_last_n_segments=1,
        word_timestamps=False,  # Not needed for typing
        beam_size=5,  # Default is fine
    )
)
```

**Message Handling Pattern**:
```python
async with client as conn:
    await conn.start_streaming()
    async for message in conn:
        if message.type == "transcription":
            await process_transcription_message(message)
```

**Error Recovery Strategy**:
- Connection loss: Attempt reconnection with exponential backoff
- Audio device unavailable: Clear error message and graceful exit
- Server errors: Display error, attempt to continue or exit cleanly

**Alternatives considered**:
- Direct WebSocket handling: Client library provides better abstraction
- Subscriber mode: Not appropriate, we need to send audio
- Multiple segment context: Unnecessary complexity for typing use case

## Testing Strategy Research

**Decision**: Multi-level testing approach with real dependencies
**Rationale**:
- Integration tests ensure real client connectivity and message handling
- Unit tests verify text diffing logic in isolation
- End-to-end tests validate complete typing workflow
- Real dependencies provide confidence in actual usage scenarios

**Test Structure**:
1. **Integration Tests**:
   - Real eavesdrop server connection testing
   - Audio device availability testing
   - WebSocket message handling verification

2. **Unit Tests**:
   - Text diffing algorithm correctness
   - State management edge cases
   - Error condition handling

3. **End-to-End Tests**:
   - Mock ydotool for automated testing
   - Simulate complete transcription workflow
   - Keyboard interrupt handling

**Test Dependencies**:
- pytest-asyncio for async test support
- pytest-mock for ydotool mocking in E2E tests
- Real eavesdrop server instance for integration tests

**Alternatives considered**:
- Mocked client library: Would miss real WebSocket integration issues
- No E2E testing: Would miss complete workflow validation
- Manual testing only: Not sustainable for continuous development

## Performance and Reliability Considerations

**Decision**: Optimize for responsiveness and error recovery
**Rationale**:
- Real-time typing requires minimal latency (<100ms perceived delay)
- Network instability must not break typing state
- Audio device changes should be handled gracefully

**Performance Strategies**:
- Minimal text diffing computation (prefix matching only)
- Async message processing to avoid blocking
- Efficient ydotool key sending (batch operations when possible)
- Connection pooling and keepalive for WebSocket reliability

**Reliability Patterns**:
- State persistence through connection interruptions
- Graceful degradation on system service unavailability
- Clear user feedback for error conditions
- Automatic reconnection with user notification

**Monitoring Points**:
- Connection status visibility
- Audio streaming health
- Transcription message processing latency
- ydotool operation success/failure rates

**Alternatives considered**:
- Complex state persistence: Overkill for desktop application
- Background service architecture: Unnecessary complexity
- GUI application: Requirements specify console application

## Progress Tracking

**Phase 0 Status**: ✅ **COMPLETE**
- [x] PyYdotool integration approach defined
- [x] Clypi CLI framework patterns established
- [x] Text diffing algorithm designed
- [x] Eavesdrop client integration strategy confirmed
- [x] Testing approach outlined
- [x] Performance and reliability considerations documented

**Readiness for Phase 1**: ✅ **READY**
All technical unknowns resolved, ready to proceed with design and contracts.