# WebSocket Audio Buffer Interface Extraction

## Executive Summary
Extract the sophisticated audio buffer management and streaming transcription logic from the WebSocket implementation into reusable abstractions. This enables the RTSP implementation to benefit from the same adaptive, context-aware transcription approach.

## Current State Analysis

### Core Components to Extract (from `ServeClientBase`)
1. **Buffer Management State**:
   - `frames_np: np.ndarray | None` - Raw audio buffer
   - `frames_offset: float` - Time offset for buffer start
   - `timestamp_offset: float` - Processed audio boundary
   - `lock: threading.Lock` - Thread synchronization

2. **Buffer Operations** (`base.py:147-213`):
   - `add_frames()` - Smart buffer growth with 45s/30s cleanup
   - `get_audio_chunk_for_processing()` - Variable-length chunk extraction
   - `clip_audio_if_no_valid_segment()` - Stall detection and recovery

3. **Transcription Loop** (`base.py:66-115`):
   - `speech_to_text()` - Continuous processing with ≥1s minimum
   - Segment handling and timestamp management
   - VAD integration and result processing

4. **Segment Management** (`base.py:307-403`):
   - `update_segments()` - Smart segment processing with overlap handling
   - `prepare_segments()` - Client result preparation
   - Repeated output detection and completion logic

## Interface Design

### 1. AudioStreamBuffer
```python
@dataclass
class BufferConfig:
    sample_rate: int = 16000
    max_buffer_duration: float = 45.0
    cleanup_duration: float = 30.0
    min_chunk_duration: float = 1.0
    clip_audio: bool = False
    max_stall_duration: float = 25.0

class AudioStreamBuffer:
    def __init__(self, config: BufferConfig)
    def add_frames(self, frame_np: np.ndarray) -> None
    def get_chunk_for_processing(self) -> tuple[np.ndarray, float]
    def advance_processed_boundary(self, offset: float) -> None
    def clip_if_stalled(self) -> None
    def reset(self) -> None
    
    # Properties
    @property
    def available_duration(self) -> float
    @property
    def total_duration(self) -> float
    @property
    def processed_duration(self) -> float
```

### 2. AudioSource Interface
```python
class AudioSource(ABC):
    @abstractmethod
    async def read_audio(self) -> np.ndarray | None:
        """Returns None for end-of-stream"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        pass

class WebSocketAudioSource(AudioSource):
    def __init__(self, websocket: ServerConnection)
    async def read_audio(self) -> np.ndarray | None
    def close(self) -> None
```

### 3. TranscriptionSink Interface  
```python
class TranscriptionResult:
    segments: list[dict]
    language: str | None
    language_probability: float | None

class TranscriptionSink(ABC):
    @abstractmethod
    async def send_result(self, result: TranscriptionResult) -> None:
        pass
    
    @abstractmethod
    async def send_error(self, error: str) -> None:
        pass

class WebSocketTranscriptionSink(TranscriptionSink):
    def __init__(self, websocket: ServerConnection, client_uid: str)
    async def send_result(self, result: TranscriptionResult) -> None
    async def send_error(self, error: str) -> None
```

### 4. StreamingTranscriptionProcessor
```python
@dataclass  
class TranscriptionConfig:
    send_last_n_segments: int = 10
    no_speech_thresh: float = 0.45
    same_output_threshold: int = 10
    use_vad: bool = True
    clip_audio: bool = False

class StreamingTranscriptionProcessor:
    def __init__(
        self,
        transcriber,  # ServeClientFasterWhisper or similar
        buffer: AudioStreamBuffer,
        sink: TranscriptionSink,
        config: TranscriptionConfig,
        logger_name: str = "transcription_processor"
    )
    
    async def start_processing(self) -> None
    async def stop_processing(self) -> None
    def add_audio_frames(self, frames: np.ndarray) -> None
    
    # Internal methods (extracted from speech_to_text and segment handling)
    async def _transcription_loop(self) -> None  
    def _update_segments(self, segments, duration) -> dict | None
    def _prepare_segments(self, last_segment=None) -> list
```

## Dependency Relationships

### Core Dependencies
```
AudioSource → StreamingTranscriptionProcessor → TranscriptionSink
                           ↓
                    AudioStreamBuffer
```

### WebSocket Integration
```
WebSocketAudioSource → StreamingTranscriptionProcessor → WebSocketTranscriptionSink
                                    ↓
                           AudioStreamBuffer
                                    ↑
                         WebSocketAudioClient (facade)
```

## Extraction Strategy

### Phase 1: Interface Definition
1. Create new module `src/eavesdrop/streaming/` with:
   - `buffer.py` - AudioStreamBuffer + BufferConfig
   - `interfaces.py` - AudioSource, TranscriptionSink, TranscriptionResult
   - `processor.py` - StreamingTranscriptionProcessor + TranscriptionConfig
   - `websocket_adapters.py` - WebSocket implementations

### Phase 2: Buffer Extraction  
1. Move buffer logic from `ServeClientBase.add_frames()` → `AudioStreamBuffer.add_frames()`
2. Move `get_audio_chunk_for_processing()` → `AudioStreamBuffer.get_chunk_for_processing()` 
3. Move `clip_audio_if_no_valid_segment()` → `AudioStreamBuffer.clip_if_stalled()`
4. Extract buffer configuration from constructor parameters

### Phase 3: Processor Extraction
1. Move `speech_to_text()` → `StreamingTranscriptionProcessor._transcription_loop()`
2. Move `update_segments()` → `StreamingTranscriptionProcessor._update_segments()`
3. Move segment state management into processor
4. Extract transcription configuration

### Phase 4: WebSocket Adapter Creation
1. Create `WebSocketAudioSource` wrapping `get_audio_from_websocket()`
2. Create `WebSocketTranscriptionSink` wrapping WebSocket send operations
3. Create facade `WebSocketStreamingClient` that combines components

### Phase 5: Integration and Migration
1. Replace `ServeClientFasterWhisper` entirely with new abstracted architecture
2. Replace `TranscriptionServer.initialize_client()` with new component creation
3. Redesign WebSocket API and behavior to use new abstractions
4. Add comprehensive testing for each abstraction

## Greenfield Development Approach

**Note**: This is greenfield development. Backwards compatibility is NOT a consideration. We will perform complete migrations and breaking changes all at once.

### Complete Replacement Strategy
- Replace `ServeClientBase` entirely with new abstractions
- Remove legacy buffer management code completely
- Redesign WebSocket protocol handlers to use new architecture
- Clean up all deprecated patterns and technical debt

## Error Handling and Logging

### Modern Error Handling
- Implement structured logging with proper context propagation
- Design clean error message formats for new architecture
- Implement debug audio capture as first-class feature in new abstractions

### Error Recovery
- Buffer state recovery after exceptions
- Processor restart capability
- Graceful degradation on component failures

## Testing Strategy

### Unit Testing
- `AudioStreamBuffer`: Buffer management, cleanup, offset tracking
- `StreamingTranscriptionProcessor`: Segment handling, transcription flow
- WebSocket adapters: Protocol compliance, error handling

### Integration Testing  
- End-to-end WebSocket client functionality with new architecture
- Performance optimization validation
- Memory usage validation

### Implementation Testing
- Comprehensive validation of new abstracted components
- WebSocket protocol functionality with new design
- Client integration with redesigned interfaces