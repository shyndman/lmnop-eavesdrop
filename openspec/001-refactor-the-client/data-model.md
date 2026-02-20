# Data Model: Refactor Client Project as Importable Library

## Overview
This document defines the data entities and their relationships for the refactored eavesdrop client library. The design reuses existing message types from `eavesdrop.wire` while defining new library-specific entities.

## Core Entities

### EavesdropClient
**Purpose**: Primary interface providing unified access to transcription services with mode-specific behavior

**Fields**:
- `host: str` - WebSocket server hostname/IP
- `port: int` - WebSocket server port 
- `_mode: ClientMode` - Internal mode (transcriber/subscriber)
- `_ws_connection: WebSocketConnection | None` - Active WebSocket connection
- `_audio_session: AudioCapture | None` - Audio capture session (transcriber mode only)
- `_stream_subscriptions: list[str] | None` - RTSP stream names (subscriber mode only)
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
- Stream names (subscriber mode) must be non-empty list

### ClientMode
**Purpose**: Enumeration defining client operation modes

**Values**:
- `TRANSCRIBER` - Send audio for transcription
- `SUBSCRIBER` - Receive transcriptions from RTSP streams

### ClientCallbacks
**Purpose**: Container for user-provided callback functions

**Fields**:
- `on_ready: Callable[[str], None] | None` - Server ready notification (backend name)
- `on_transcription: Callable[[TranscriptionMessage], None] | None` - Rich transcription message received
- `on_error: Callable[[str], None] | None` - Error occurred
- `on_streaming_started: Callable[[], None] | None` - Audio streaming started (transcriber only)
- `on_streaming_stopped: Callable[[], None] | None` - Audio streaming stopped (transcriber only)
- `on_stream_status: Callable[[str, str, str | None], None] | None` - Stream status change (subscriber only)

**Breaking Change from Existing Client**:
- Current `WebSocketConnection.on_transcription` receives extracted text string
- New library callbacks receive full `TranscriptionMessage` objects with all metadata

### TranscriberOptions
**Purpose**: Configuration options for transcriber mode

**Fields**:
- `audio_device: int | str | None` - Audio device selection (index, name, or substring)
- `beam_size: int` - Whisper beam search size (default: 5)
- `word_timestamps: bool` - Include word-level timestamps (default: False)
- `initial_prompt: str | None` - Initial prompt for transcription context
- `hotwords: list[str] | None` - Priority words for recognition

**Validation Rules**:
- `beam_size` must be positive integer
- `audio_device` if provided must match available device

### SubscriberOptions  
**Purpose**: Configuration options for subscriber mode

**Fields**:
- `stream_names: list[str]` - RTSP stream names to subscribe to

**Validation Rules**:
- `stream_names` must be non-empty list
- Each stream name must be non-empty string

## Existing Entities (Reused from eavesdrop.wire)

### TranscriptionMessage
**Purpose**: Rich data structure containing transcription results

**Fields** (from existing schema):
- `stream: str` - Stream identifier  
- `language: str | None` - Detected/specified language
- `segments: list[Segment]` - Rich segment data with timestamps, confidence scores
- `timestamp: float` - Message timestamp

### Segment
**Purpose**: Individual transcription segment with metadata

**Fields** (from existing schema):
- `text: str` - Transcribed text content
- `start: float` - Start timestamp in seconds
- `end: float` - End timestamp in seconds  
- `confidence: float | None` - Recognition confidence score

### UserTranscriptionOptions
**Purpose**: Server-side transcription configuration

**Fields** (from existing schema):
- `initial_prompt: str | None`
- `hotwords: list[str] | None`
- `beam_size: int`
- `word_timestamps: bool`

## Entity Relationships

```
EavesdropClient
├── ClientMode (composition)
├── ClientCallbacks (composition)
├── TranscriberOptions | SubscriberOptions (composition, mode-dependent)
├── WebSocketConnection (aggregation, lifecycle managed)
├── AudioCapture (aggregation, transcriber mode only)
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

### Subscriber Mode Flow  
1. `EavesdropClient.subscriber()` creates instance with `SubscriberOptions`
2. `connect()` establishes WebSocket with `ClientType.RTSP_SUBSCRIBER` and stream names in `X-Stream-Names` header
3. Server immediately begins sending `TranscriptionMessage` objects for all subscribed streams
4. Messages queued internally and yielded via async iteration (no stream filtering)
5. `disconnect()` closes WebSocket connection

**Implementation Notes**:
- Uses `WebSocketHeaders.STREAM_NAMES` to pass comma-separated stream names during connection
- Message processing removes stream filtering to receive transcriptions from all subscribed streams
- `StreamStatusMessage` objects handled via `on_stream_status` callback for stream health monitoring

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