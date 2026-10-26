# Data Model: Refactor Client Project as Importable Library

## Overview
This document defines the data entities and their relationships for the refactored eavesdrop client library. The design reuses existing message types from `eavesdrop.wire` while defining new library-specific entities.

## Core Entities

### EavesdropClient
**Purpose**: Primary interface providing live transcription and independent connection, capture, and transmission lifecycles

**Fields**:
- `host: str` - WebSocket server hostname/IP
- `port: int` - WebSocket server port 
- `_ws_connection: WebSocketConnection | None` - Active WebSocket connection
- `_audio_session: AudioCapture | None` - Live audio capture session
- `_message_queue: asyncio.Queue[TranscriptionMessage]` - Internal message queue for async iteration
- `_callbacks: ClientCallbacks` - User-provided callback functions
- `_connected: bool` - Connection status flag
- `_streaming: bool` - Audio streaming status flag (transcriber mode)

**State Transitions**:
- Created → Connected (via `connect()` or `__aenter__()`)
- Connected → Streaming (via `start_streaming()` in transcriber mode)
- Streaming → Connected (via `stop_streaming()` in transcriber mode) 
- Connected → Disconnected (via `disconnect()` or `__aexit__()`)

**Validation Rules**:
- `host` must be non-empty string
- `port` must be valid TCP port (1-65535)
- Audio device (transcriber mode) must exist and be accessible

### ClientCallbacks
**Purpose**: Container for user-provided callback functions

**Fields**:
- `on_ready: Callable[[str], None] | None` - Server ready notification (backend name)
- `on_transcription: Callable[[TranscriptionMessage], None] | None` - Rich transcription message received
- `on_error: Callable[[str], None] | None` - Error occurred
- `on_streaming_started: Callable[[], None] | None` - Audio streaming started (transcriber only)
- `on_streaming_stopped: Callable[[], None] | None` - Audio streaming stopped (transcriber only)

**Breaking Change from Existing Client**:
- Current `WebSocketConnection.on_transcription` receives extracted text string
- New library callbacks receive full `TranscriptionMessage` objects with all metadata

### TranscriberOptions
**Purpose**: Configuration options for transcriber mode

**Fields**:
- `audio_device: int | str | None` - Audio device selection (index, name, or substring)
- `language: str` - Fixed English recognition language
- `send_last_n_segments: int` - Live transcript segment window

**Validation Rules**:
- `audio_device` if provided must match available device

## Existing Entities (Reused from eavesdrop.wire)

### TranscriptionMessage
**Purpose**: Rich data structure containing transcription results

**Fields** (from existing schema):
- `stream: str` - Stream identifier  
- `language: str | None` - Detected/specified language
- `segments: list[Segment]` - Segment data with mandatory native segment and individual word timestamps
- `timestamp: float` - Message timestamp

### Segment
**Purpose**: Individual transcription segment with metadata

**Fields** (from existing schema):
- `text: str` - Transcribed text content
- `start: float` - Start timestamp in seconds
- `end: float` - End timestamp in seconds  
- `confidence: float | None` - Recognition confidence score
- `words: list[Word]` - Required native individual word timestamps on the recording timeline

### UserTranscriptionOptions
**Purpose**: Server-side transcription configuration

**Fields** (from existing schema):
- `language: str`
- `send_last_n_segments: int`

Native segment and individual word timings are mandatory output, not a user-selectable configuration field.

## Entity Relationships

```
EavesdropClient
├── ClientCallbacks (composition)
├── TranscriberOptions (composition)
├── WebSocketConnection (aggregation, lifecycle managed)
├── AudioCapture (aggregation, lifecycle managed)
└── Queue[TranscriptionMessage] (composition, internal)

TranscriptionMessage
├── stream: str
├── language: str | None
├── segments: list[Segment]
└── timestamp: float

Segment
├── text: str
├── start: float
├── end: float
└── confidence: float | None
```

## Data Flow

### Transcriber Mode Flow
1. `EavesdropClient.transcriber()` creates instance with `TranscriberOptions`
2. `connect()` establishes WebSocket and starts audio capture (no transmission)
3. `start_streaming()` begins audio data transmission to server
4. Server sends `TranscriptionMessage` objects via WebSocket
5. Messages queued internally and yielded via async iteration
6. `stop_streaming()` stops transmission while maintaining capture and connection

### Recording Lifecycle
One connection supports multiple recordings. Each recording boundary resets transcript state and recording-relative timestamp zero after accepted work for the preceding recording has completed. Stop recording A, force-complete its flush, and deliver its final transcript and flush-complete response before processing B. Flush covers audio accepted at its boundary and acknowledges completion even when no audio is pending.

Cancellation discards the active utterance, clears its utterance-local history, and suppresses outstanding results; another recording can start on the same connection. With VAD disabled, every pending segment remains incomplete until an explicit force-completing flush. An ordinary update or non-force-completing flush cannot complete any segment. With VAD enabled, preserve silence-based completion and the trailing incomplete segment. Native word times retain original recording offsets across silence.

## State Management

### Connection States
- `DISCONNECTED`: No WebSocket connection
- `CONNECTING`: WebSocket connection in progress
- `CONNECTED`: WebSocket established, ready for operations
- `DISCONNECTING`: WebSocket closing in progress

### Streaming States (Transcriber Mode Only)
- `NOT_STREAMING`: Audio captured but not transmitted
- `STREAMING`: Audio captured and transmitted to server

### Error Handling
- Configuration errors: Immediate exceptions during factory method calls
- Connection errors: Callbacks + automatic iteration termination  
- Runtime errors: Callbacks without stopping iteration (unless server closes connection)
- Device errors: Immediate exceptions with callback notification

## Thread Safety

Each `EavesdropClient` instance maintains completely isolated state:
- Independent WebSocket connections
- Independent audio capture sessions  
- Independent message queues
- No shared static/class variables
- No synchronization primitives required

Concurrent usage achieved through multiple independent client instances.