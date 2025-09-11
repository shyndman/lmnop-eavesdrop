# Eavesdrop Client API Refactor

Refactor the client project, transforming the existing CLI-based eavesdrop client into a clean, importable library while preserving all existing functionality. Rearrange the current WebSocket connection, audio capture, and message handling components into a unified programmatic API that supports both transcriber mode (sending audio for transcription) and subscriber mode (receiving transcriptions from RTSP streams). Keep the core technical implementation the same—simply remove the terminal interface and CLI aspects while exposing the underlying capabilities through a streaming async iterator API.

## API Design

### Unified Client Interface

```python
from eavesdrop.client import EavesdropClient
from eavesdrop.wire import TranscriptionMessage

class EavesdropClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 9090,
        mode: Literal["transcriber", "subscriber"] = "transcriber",

        # Common callbacks (optional)
        on_ready: Callable[[str], None] | None = None,              # backend_name
        on_transcription: Callable[[TranscriptionMessage], None] | None = None,  # rich message data
        on_error: Callable[[str], None] | None = None,              # error_message

        # Transcriber mode specific
        audio_device: int | str | None = None,                      # sounddevice selection
        on_streaming_started: Callable[[], None] | None = None,     # optional
        on_streaming_stopped: Callable[[], None] | None = None,     # optional

        # Subscriber mode specific
        stream_names: list[str] | None = None,                      # RTSP stream names
        on_stream_status: Callable[[str, str, str | None], None] | None = None,  # stream, status, message
    ):
        ...

    # Async iterator interface
    def __aiter__(self) -> AsyncIterator[TranscriptionMessage]: ...
    async def __anext__(self) -> TranscriptionMessage: ...

    # Context manager (auto connect/disconnect)
    async def __aenter__(self) -> 'EavesdropClient': ...
    async def __aexit__(self, *args) -> None: ...

    # Connection management
    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...

    # Transcriber mode control (raises error on subscriber mode)
    async def start_streaming(self) -> None: ...
    async def stop_streaming(self) -> None: ...

    # Status checks
    def is_connected(self) -> bool: ...
    def is_streaming(self) -> bool: ...  # transcriber mode only
```

## Usage Examples

### Transcriber Mode (Audio → Transcription)

```python
# Basic usage with context manager
async with EavesdropClient(mode="transcriber", audio_device="USB Audio") as client:
    await client.start_streaming()

    async for transcription in client:
        print(f"Stream: {transcription.stream}")
        print(f"Language: {transcription.language}")
        for segment in transcription.segments:
            print(f"  {segment.start:.2f}s: {segment.text}")

        # Control streaming based on content
        if "stop recording" in segment.text.lower():
            await client.stop_streaming()

# Manual connection management
client = EavesdropClient(
    mode="transcriber",
    on_ready=lambda backend: print(f"Ready with {backend}"),
    on_error=lambda msg: print(f"Error: {msg}"),
    on_streaming_started=lambda: print("Started streaming audio"),
    on_streaming_stopped=lambda: print("Stopped streaming audio")
)

await client.connect()
await client.start_streaming()

async for transcription in client:
    # Process rich transcription data
    pass

await client.disconnect()
```

### Subscriber Mode (RTSP Stream Transcriptions)

```python
# Subscribe to multiple RTSP streams
async with EavesdropClient(
    mode="subscriber",
    stream_names=["office_audio", "meeting_room_1"],
    on_stream_status=lambda stream, status, msg: print(f"[{stream}] {status}: {msg}")
) as client:

    async for transcription in client:
        print(f"[{transcription.stream}] {transcription.language}")
        for segment in transcription.segments:
            print(f"  {segment.text}")
```

### Audio Device Selection

```python
import sounddevice as sd

# List available devices
print(sd.query_devices())
# Output:
#   0 Built-in Microphone, Core Audio (2 in, 0 out)
# > 1 USB Audio Device, Core Audio (1 in, 0 out)
#   2 BlackHole 2ch, Core Audio (2 in, 2 out)

# Use device by index, name, or substring
client1 = EavesdropClient(audio_device=1)              # By index
client2 = EavesdropClient(audio_device="USB Audio")    # By name
client3 = EavesdropClient(audio_device="BlackHole")    # Partial match
client4 = EavesdropClient(audio_device=None)           # Default device
```

## Lifecycle & Behavior

### Transcriber Mode Lifecycle
1. `connect()` → WebSocket connection + session setup + microphone starts (no audio transmission)
2. `start_streaming()` → Begin sending captured audio packets to server
3. `stop_streaming()` → Stop sending audio (microphone stays "warm"/capturing)
4. `disconnect()` → Close WebSocket + stop microphone capture

### Subscriber Mode Lifecycle
1. `connect()` → WebSocket connection + immediately begins receiving transcriptions
2. `disconnect()` → Close WebSocket connection

### Error Handling
- All `ErrorMessage` objects from server → Call `on_error()` callback for logging
- WebSocket disconnection → Automatically stops async iteration (`StopAsyncIteration`)
- Server determines error fatality by closing connection or not
- Setup errors (invalid streams, missing headers) → Server closes connection immediately
- Runtime errors → Logged via callback, iteration continues unless server closes connection

### Rich Transcription Data
The client passes through complete `TranscriptionMessage` objects containing:
- `stream: str` - Stream identifier
- `language: str | None` - Detected/specified language
- `segments: list[Segment]` - Rich segment data with timestamps, confidence scores
- `timestamp: float` - Message timestamp

No data simplification or text extraction—users receive the full wire protocol data.

## Implementation Notes

### Components to Keep
- `WebSocketConnection` - Refactor to support both client types via headers
- `AudioCapture` - Keep for transcriber mode with device selection support
- Message protocol handling from `eavesdrop.wire`

### Components to Remove
- `TerminalInterface` - All terminal/keyboard interaction
- `MicrophoneClient` - CLI orchestration class
- `__main__.py` - CLI entry point
- `app.py` - CLI application logic

### Components to Add
- Stream subscription logic for subscriber mode
- Unified `EavesdropClient` class with mode parameter
- AsyncIterator implementation for streaming results
- Context manager support for automatic connection management

### Dependencies
- Keep existing: `sounddevice`, `websockets`, `pydantic`, `numpy`, `structlog`, `eavesdrop-wire`
- Remove CLI script entry point from `pyproject.toml`

## Future Enhancements

A higher-level robust client could wrap this core client to provide:
- Automatic reconnection with exponential backoff
- Connection pooling
- Retry logic and error recovery
- Health monitoring

```python
# Future robust wrapper example
class RobustEavesdropClient:
    async def __aiter__(self):
        while self.should_retry:
            try:
                async with EavesdropClient(...) as client:
                    async for transcription in client:
                        yield transcription
            except StopAsyncIteration:
                await self._handle_reconnection()
```

This keeps the core client simple and predictable while enabling advanced reliability features as opt-in higher-level abstractions.
