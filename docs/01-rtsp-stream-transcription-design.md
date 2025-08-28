# RTSP Stream Transcription Design

## Overview

This design document outlines the addition of RTSP stream transcription capabilities to Eavesdrop, enabling the server to automatically ingest and transcribe audio from configured RTSP streams concurrently with existing WebSocket client support.

## Architecture Goals

- **Concurrent Operation**: RTSP streams operate alongside existing WebSocket clients without interference
- **Asyncio-Native**: Leverage Python's asyncio for efficient concurrent stream processing
- **Configuration-Driven**: TOML-based stream definitions loaded at startup
- **Resilient**: Aggressive retry logic with exponential backoff for stream failures
- **Unified Output**: Single endpoint streams transcriptions from all sources with stream identification
- **Resource Unconstrained**: Remove arbitrary client limits to allow full resource utilization

## High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ RTSP Streams    │    │ WebSocket       │    │ Transcription   │
│ (configured)    │────│ Clients         │────│ Results         │
│                 │    │ (interactive)   │    │ Endpoint        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Eavesdrop Server Core                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ RTSP Stream │  │ WebSocket   │  │ Unified Transcription   │  │
│  │ Manager     │  │ Handler     │  │ Result Broadcaster      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│         │                 │                       │             │
│         └─────────────────┼───────────────────────┘             │
│                           ▼                                     │
│              ┌─────────────────────────┐                        │
│              │ Shared Whisper Model    │                        │
│              │ Pool & Audio Pipeline   │                        │
│              └─────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

## Component Design

### 1. TOML Configuration Format

**File**: `streams.toml` (configurable path)

```toml
[streams.office_camera]
name = "Office Camera"
url = "rtsp://192.168.1.100:8554/office?audio=aac"
language = "en"
initial_prompt = "Office meeting audio transcription"

[streams.lobby_monitor]  
name = "Lobby Monitor"
url = "rtsp://192.168.1.101:8554/lobby"
language = "en"
vad_parameters = { threshold = 0.3 }

[output]
endpoint_type = "websocket"  # or "tcp"
endpoint_port = 9091
include_timestamps = true
include_confidence = true
```

**Configuration Schema**:
- `streams.<id>`: Stream identifier (used in transcription output)
- `name`: Human-readable name for the stream
- `url`: RTSP stream URL
- `language`: Language code for Whisper model
- `initial_prompt`: Optional Whisper initial prompt
- `vad_parameters`: Optional VAD configuration override
- `output`: Global output endpoint configuration

### 2. RTSP Stream Manager

**Core Component**: `RTSPStreamManager`

Responsibilities:
- Parse TOML configuration at startup
- Initialize and manage asyncio tasks for each configured stream
- Handle stream failures with exponential backoff retry logic
- Route audio data to shared transcription pipeline

**Key Implementation Details**:
- Based on your asyncio subprocess approach with FFmpeg
- Each stream runs as an independent asyncio task
- Shared audio queue feeding into existing Eavesdrop audio pipeline
- Stream identification tags preserved through the entire pipeline

```python
class RTSPStreamManager:
    async def start_stream(self, stream_config: StreamConfig) -> None:
        """Start processing for a single RTSP stream with retry logic"""
        
    async def stream_rtsp_audio(self, stream_config: StreamConfig) -> AsyncIterator[AudioChunk]:
        """Async generator yielding tagged audio chunks from RTSP stream"""
        
    async def handle_stream_failure(self, stream_id: str, error: Exception) -> None:
        """Handle stream failures with exponential backoff retry"""
```

### 3. Unified Transcription Pipeline

**Integration Point**: Extend existing `ServeClientBase` architecture

**Key Changes**:
- Modify `ServeClientFasterWhisper` to accept stream-tagged audio sources
- Add `RTSPStreamClient` class that mimics WebSocket client interface
- Stream results route to both WebSocket clients (existing) and new results endpoint

**Audio Flow**:
1. RTSP streams provide tagged audio chunks: `(stream_id, audio_data)`
2. Existing VAD and audio processing applied per stream
3. Transcription results tagged with `stream_id` and `stream_name`
4. Results broadcast to unified output endpoint

### 4. Transcription Results Streaming

**New Component**: `TranscriptionResultsBroadcaster`

**Output Format**:
```json
{
  "stream_id": "office_camera",
  "stream_name": "Office Camera", 
  "timestamp": "2024-01-15T10:30:45.123Z",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Good morning everyone",
      "confidence": 0.89
    }
  ],
  "language": "en",
  "source_type": "rtsp_stream"  # vs "websocket_client"
}
```

**Endpoint Options**:
- **WebSocket Endpoint**: `ws://host:9091/transcriptions` - JSON messages
- **TCP Endpoint**: `host:9091` - Newline-delimited JSON

## Integration with Existing Architecture

### Changes to Core Components

**`TranscriptionServer`**:
- Remove `max_clients` limitation entirely
- Add `RTSPStreamManager` initialization
- Start RTSP stream tasks alongside WebSocket server
- Integrate results broadcaster

**`ClientManager`**:
- Extend to track both WebSocket clients and RTSP streams
- Remove client count restrictions
- Maintain existing timeout behavior for WebSocket clients only

**Audio Pipeline**:
- Minimal changes - existing VAD, transcription, and model sharing work unchanged  
- Add stream tagging throughout pipeline
- Results routing enhanced to support multiple output destinations

### Startup Sequence

1. Parse TOML configuration file
2. Initialize shared Whisper models and audio pipeline (existing)
3. Start RTSP stream manager and configured streams
4. Start transcription results broadcaster endpoint
5. Start WebSocket server (existing)
6. All components run concurrently via asyncio

## Retry and Resilience Strategy

### RTSP Stream Reconnection
- **Immediate retry**: 0 seconds
- **Exponential backoff**: 1s, 2s, 4s, 8s, 16s, 32s, 60s (cap at 1 minute)
- **Endless retries**: Continue attempting reconnection indefinitely
- **Backoff reset**: Reset to immediate retry after successful 5-minute connection

### Error Handling
- **Network failures**: Retry with backoff
- **FFmpeg process crashes**: Restart process, retry stream
- **Invalid RTSP URLs**: Log error, disable stream (no retry)
- **Authentication failures**: Log error, disable stream (no retry)

## Performance Considerations

### Concurrency
- Each RTSP stream runs as independent asyncio task
- Shared model pool prevents resource duplication
- WebSocket clients unaffected by RTSP stream processing

### Memory Management  
- Audio buffering per stream (limited queue sizes)
- Transcription results streamed immediately (no accumulation)
- FFmpeg process memory isolated per stream

### Resource Scaling
- No arbitrary limits on concurrent streams
- Resource consumption scales linearly with active streams
- User responsible for hardware resource management

## Implementation Plan

### Phase 1: Core RTSP Integration
1. Implement `RTSPStreamManager` using your asyncio approach
2. Create TOML configuration parsing
3. Integrate RTSP audio into existing transcription pipeline
4. Test with single stream

### Phase 2: Results Broadcasting  
1. Implement `TranscriptionResultsBroadcaster`
2. Add stream tagging throughout pipeline
3. WebSocket and TCP endpoint options
4. Test unified output from mixed sources

### Phase 3: Production Hardening
1. Comprehensive retry logic with exponential backoff
2. Configuration validation and error handling
3. Performance testing with multiple concurrent streams
4. Documentation and deployment guides

## Configuration Examples

**Minimal Configuration** (`streams.toml`):
```toml
[streams.camera1]
name = "Main Camera"
url = "rtsp://192.168.1.100:8554/stream"

[output]
endpoint_type = "websocket"
endpoint_port = 9091
```

**Advanced Configuration**:
```toml
[streams.office]
name = "Office Meeting Room"
url = "rtsp://192.168.1.100:8554/office?audio=aac"
language = "en" 
initial_prompt = "Meeting discussion about quarterly results"
vad_parameters = { threshold = 0.3, min_speech_duration_ms = 250 }

[streams.lobby]
name = "Lobby Security"  
url = "rtsp://192.168.1.101:8554/lobby"
language = "en"
vad_parameters = { threshold = 0.5 }

[output]
endpoint_type = "tcp"
endpoint_port = 9091
include_timestamps = true
include_confidence = true
```

## CLI Integration

**New Command Line Options**:
```bash
eavesdrop --streams-config /path/to/streams.toml --results-port 9091
```

**Environment Variables**:
- `EAVESDROP_STREAMS_CONFIG`: Path to streams configuration file  
- `EAVESDROP_RESULTS_PORT`: Transcription results endpoint port
- `EAVESDROP_RESULTS_TYPE`: "websocket" or "tcp"

This design maintains Eavesdrop's existing strengths while adding powerful RTSP stream processing capabilities that scale with your hardware resources.

## Open Questions & Future Enhancements

### Q: RTSP Stream Pause Detection and Optimization

**Context**: RTSP streams from indoor cameras could be paused when rooms are unoccupied (using radar people detectors). This presents an opportunity to optimize compute resources by detecting stream pause state and suspending unnecessary processing.

**Technical Details**:
- RTSP carries control messages (DESCRIBE, SETUP, PLAY, PAUSE, TEARDOWN) while RTP carries media data
- When RTSP stream is paused, RTP data flow stops but FFmpeg process remains alive
- Our `await process.stdout.read()` will block indefinitely during pause state

**Potential Implementation Approaches**:

1. **Timeout-based detection**: Use `asyncio.wait_for()` with timeout to detect when no data arrives
2. **FFmpeg stderr parsing**: Monitor FFmpeg verbose logs for RTSP state change messages  
3. **External RTSP control monitoring**: Separate RTSP client to monitor stream state independently

**Optimization Opportunities**:
- Skip VAD processing when stream paused (no audio data to analyze)
- Suspend transcription pipeline to save GPU/CPU cycles
- Send "stream_paused" status to results endpoint  
- Distinguish intentional pause from connection failure in retry logic

**Configuration Enhancement**:
```toml
[streams.office]
name = "Office Camera"
url = "rtsp://192.168.1.100:8554/office"
pause_detection = true
pause_timeout = 5.0  # seconds before considering paused
vad_skip_when_paused = true
```

**Data Structure Impact**:
```python
@dataclass
class AudioChunk:
    stream_id: str
    data: bytes | None  # None when paused
    status: Literal["active", "paused", "error"]
    timestamp: float
```

**Decision Required**: Should pause detection be implemented in Phase 1 or deferred to Phase 3? Implementation complexity vs. immediate resource savings trade-off.