# RTSP Integration with Abstracted Audio Buffer

## Executive Summary
Replace the rigid 1-second AudioBuffer in RTSP implementation with the sophisticated AudioStreamBuffer abstractions extracted from WebSocket, enabling adaptive transcription with context preservation, better latency, and VAD integration.

## Current RTSP Problems Analysis

### Limitations of Current Implementation
1. **Fixed Chunking** (`rtsp.py:381-443`):
   - `AudioBuffer` only releases audio at exactly 1-second boundaries
   - `target_duration = 1.0` hardcoded, no flexibility
   - Accumulates to `target_bytes = 32,000` before releasing

2. **Simplistic Processing** (`rtsp.py:601-650`):
   - `_transcription_worker()` pulls from queue with 1-second timeout
   - No overlap between transcription windows
   - No sophisticated segment handling

3. **No Context Preservation**:
   - Each 1-second chunk processed in isolation
   - No segment boundary intelligence
   - No repeated output detection

4. **Limited Integration**:
   - Results only logged, no sophisticated output handling
   - No VAD integration at buffer level
   - No memory management or stall detection

## Integration Architecture

### Component Mapping
```
Current RTSP                →    New Abstracted Architecture
═══════════════════════════════════════════════════════════════
AudioBuffer                 →    AudioStreamBuffer
_transcription_worker()     →    StreamingTranscriptionProcessor
Queue-based audio flow      →    RTSPAudioSource
Simple logging             →    RTSPTranscriptionSink
RTSPTranscriptionClient    →    Enhanced with new components
```

### New Component Structure
```
RTSPClient (FFmpeg management)
    ↓ (audio bytes)
RTSPAudioSource (conversion & buffering)
    ↓ (numpy arrays)  
AudioStreamBuffer (sophisticated buffering)
    ↓ (processed chunks)
StreamingTranscriptionProcessor (adaptive transcription)
    ↓ (results)
RTSPTranscriptionSink (logging & future extensibility)
```

## RTSP-Specific Implementations

### 1. RTSPAudioSource
```python
class RTSPAudioSource(AudioSource):
    def __init__(self, audio_queue: asyncio.Queue[bytes]):
        self.audio_queue = audio_queue
        self.sample_rate = 16000
        self.bytes_per_sample = 2  # 16-bit PCM
        self.closed = False
        
    async def read_audio(self) -> np.ndarray | None:
        """
        Read from FFmpeg queue and convert to numpy array.
        Returns None when stream ends.
        """
        try:
            # Get audio chunk from FFmpeg (via RTSPClient)
            chunk = await asyncio.wait_for(
                self.audio_queue.get(), 
                timeout=1.0
            )
            
            if not chunk or self.closed:
                return None
                
            # Convert FFmpeg bytes to numpy array (like WebSocket does)  
            # FFmpeg outputs 16-bit signed PCM little-endian (s16le)
            audio_array = np.frombuffer(chunk, dtype=np.int16)
            
            # Convert to float32 normalized to [-1.0, 1.0] for Whisper
            audio_array = audio_array.astype(np.float32) / 32768.0
            
            return audio_array
            
        except asyncio.TimeoutError:
            # No audio available - return empty array to keep processor alive
            return np.array([], dtype=np.float32)
            
    def close(self) -> None:
        self.closed = True
```

### 2. RTSPTranscriptionSink  
```python
class RTSPTranscriptionSink(TranscriptionSink):
    def __init__(self, stream_name: str, logger_name: str = "rtsp_transcription"):
        self.stream_name = stream_name
        self.logger = get_logger(logger_name).bind(stream=stream_name)
        self.transcription_count = 0
        
    async def send_result(self, result: TranscriptionResult) -> None:
        """Log transcription results with structured data"""
        self.transcription_count += 1
        
        for segment in result.segments:
            if segment.get("text", "").strip():
                self.logger.info(
                    "Transcription result",
                    text=segment["text"].strip(),
                    start=segment["start"], 
                    end=segment["end"],
                    completed=segment.get("completed", False),
                    transcription_number=self.transcription_count
                )
                
    async def send_error(self, error: str) -> None:
        """Log transcription errors"""
        self.logger.error("Transcription error", error=error)
        
    async def send_language_detection(self, language: str, probability: float) -> None:
        """Log language detection results"""
        self.logger.info(
            "Language detected", 
            language=language, 
            probability=probability
        )
```

### 3. Enhanced RTSPTranscriptionClient
```python
class RTSPTranscriptionClient(RTSPClient):
    def __init__(self, stream_name: str, rtsp_url: str, transcriber):
        # Initialize parent with internal queue
        super().__init__(stream_name, rtsp_url, asyncio.Queue(maxsize=100))
        
        # Create new abstracted components
        self.transcriber = transcriber
        
        # Configuration (extracted from current defaults + new options)
        buffer_config = BufferConfig(
            sample_rate=16000,
            max_buffer_duration=45.0,      # New: memory management
            cleanup_duration=30.0,         # New: automatic cleanup
            min_chunk_duration=1.0,        # Keep existing 1s minimum
            clip_audio=False,              # New: optional stall handling
            max_stall_duration=25.0        # New: stall detection
        )
        
        transcription_config = TranscriptionConfig(
            send_last_n_segments=10,       # Not used for RTSP but required
            no_speech_thresh=0.45,         # Match WebSocket default
            same_output_threshold=10,      # New: repeated output handling
            use_vad=True,                  # Enable VAD integration
            clip_audio=False               # Optional stall recovery
        )
        
        # Create abstracted components
        self.audio_source = RTSPAudioSource(self.audio_queue)
        self.stream_buffer = AudioStreamBuffer(buffer_config)  
        self.transcription_sink = RTSPTranscriptionSink(stream_name)
        self.processor = StreamingTranscriptionProcessor(
            transcriber=transcriber,
            buffer=self.stream_buffer,
            sink=self.transcription_sink,
            config=transcription_config,
            logger_name=f"rtsp_processor_{stream_name}"
        )
        
        # Statistics (preserved from current implementation)
        self.transcriptions_completed = 0
        self.transcription_errors = 0
        
    async def run(self) -> None:
        """Enhanced run method with new processing architecture"""
        with structlog.contextvars.bound_contextvars(stream=self.stream_name):
            self.logger.info("Starting RTSP transcription client", url=self.rtsp_url)

            while not self.stopped:
                try:
                    # ... existing FFmpeg process creation logic ...
                    process = await self._create_ffmpeg_process()
                    
                    # Start all concurrent tasks
                    audio_task = asyncio.create_task(self._read_audio_stream())
                    error_task = asyncio.create_task(self._monitor_process_errors()) 
                    
                    # NEW: Use streaming processor instead of simple worker
                    streaming_task = asyncio.create_task(self._streaming_processor_task())
                    audio_feeding_task = asyncio.create_task(self._feed_audio_to_buffer())
                    
                    # Wait for any task to complete
                    done, pending = await asyncio.wait([
                        audio_task, error_task, streaming_task, audio_feeding_task
                    ], return_when=asyncio.FIRST_COMPLETED)
                    
                    # ... existing cleanup logic ...
                    
                except Exception:
                    # ... existing error handling ...
                    
    async def _streaming_processor_task(self) -> None:
        """New task to run the streaming transcription processor"""
        try:
            await self.processor.start_processing()
        except Exception:
            self.logger.exception("Streaming processor failed")
            
    async def _feed_audio_to_buffer(self) -> None:
        """New task to feed audio from source to buffer"""
        try:
            while not self.stopped:
                # Read from RTSP audio source (converts bytes to numpy)
                audio_array = await self.audio_source.read_audio()
                
                if audio_array is None:  # End of stream
                    break
                    
                if len(audio_array) > 0:  # Skip empty arrays from timeouts
                    # Add to sophisticated buffer (replaces AudioBuffer)
                    self.processor.add_audio_frames(audio_array)
                    
        except Exception:
            self.logger.exception("Audio feeding task failed")
            
    # Remove old _transcription_worker - replaced by streaming processor
    
    async def stop(self) -> None:
        """Enhanced stop with new component cleanup"""
        if self.stopped:
            return
            
        self.logger.info("Stopping RTSP transcription client")
        self.stopped = True
        
        # Stop new components
        await self.processor.stop_processing()
        self.audio_source.close()
        
        # ... existing FFmpeg cleanup ...
```

## Configuration Migration

### Parameter Mapping
```python
# Current AudioBuffer parameters → New BufferConfig
OLD_AUDIO_BUFFER = {
    'sample_rate': 16000,           → buffer_config.sample_rate: 16000
    'target_duration': 1.0,        → buffer_config.min_chunk_duration: 1.0
    'bytes_per_sample': 2           → (handled internally)
}

# New sophisticated buffer parameters (not available in old system)
NEW_BUFFER_FEATURES = {
    'max_buffer_duration': 45.0,    # Memory management
    'cleanup_duration': 30.0,       # Automatic cleanup
    'clip_audio': False,            # Stall detection
    'max_stall_duration': 25.0      # Stall recovery
}

# Transcription configuration alignment with WebSocket
TRANSCRIPTION_CONFIG = {
    'no_speech_thresh': 0.45,       # Match WebSocket default  
    'same_output_threshold': 10,    # New: repeated output handling
    'use_vad': True,                # Enable VAD integration
    'send_last_n_segments': 10      # Unused for RTSP but required
}
```

### Environment Variable Extensions
```bash
# Existing RTSP environment variables (preserved)
EAVESDROP_FW_MODEL_PATH=...
EAVESDROP_MAX_CLIENTS=4
EAVESDROP_MAX_CONNECTION_TIME=300

# New buffer management options
EAVESDROP_RTSP_BUFFER_DURATION=45.0      # Max buffer before cleanup
EAVESDROP_RTSP_CLEANUP_DURATION=30.0     # Amount to cleanup
EAVESDROP_RTSP_MIN_CHUNK_DURATION=1.0    # Minimum before transcription
EAVESDROP_RTSP_CLIP_AUDIO=false          # Enable stall detection
EAVESDROP_RTSP_MAX_STALL_DURATION=25.0   # Stall detection threshold

# Transcription behavior options
EAVESDROP_RTSP_NO_SPEECH_THRESH=0.45     # VAD threshold
EAVESDROP_RTSP_SAME_OUTPUT_THRESH=10     # Repeated output threshold
EAVESDROP_RTSP_USE_VAD=true              # Enable VAD filtering
```

## Integration with Existing Systems

### RTSPClientManager Modifications
```python
class RTSPClientManager:
    async def add_stream(self, stream_name: str, rtsp_url: str) -> bool:
        try:
            # Get shared transcriber (unchanged)
            transcriber = await self.model_manager.get_transcriber()
            
            # Create NEW enhanced RTSP client with abstractions
            client = RTSPTranscriptionClient(stream_name, rtsp_url, transcriber)
            
            # ... rest unchanged ...
            
    def get_stream_status(self) -> dict[str, dict]:
        """Enhanced status with new metrics"""
        # ... existing status logic ...
        
        for stream_name, client in self.clients.items():
            status["streams"][stream_name].update({
                # Existing metrics (preserved)
                "transcriptions_completed": client.transcriptions_completed,
                "transcription_errors": client.transcription_errors,
                
                # NEW: Enhanced buffer metrics
                "buffer_duration": client.stream_buffer.total_duration,
                "processed_duration": client.stream_buffer.processed_duration, 
                "available_duration": client.stream_buffer.available_duration,
                
                # NEW: Processor metrics
                "processor_active": not client.processor.stopped,
                "segments_processed": getattr(client.processor, 'segments_processed', 0)
            })
```

### Greenfield Development Approach

**Note**: This is greenfield development. Backwards compatibility is NOT a consideration. We will perform complete migrations and breaking changes all at once.

### Complete Replacement Strategy
1. **RTSP configuration** - May require updates for new buffer parameters
2. **Log format and structure** - Completely redesigned for new architecture
3. **Docker container behavior** - May have breaking changes to environment variables
4. **Environment variable names** - New variables added, some may be renamed
5. **Performance characteristics** - Significant improvements expected, but breaking API changes allowed

## Testing and Validation Strategy

### Unit Testing
```python
# New component testing
test_rtsp_audio_source.py          # FFmpeg byte conversion
test_rtsp_transcription_sink.py    # Logging behavior  
test_enhanced_rtsp_client.py       # Integration behavior

# Regression testing  
test_rtsp_compatibility.py         # Existing behavior preservation
test_rtsp_performance.py           # Latency and accuracy comparison
```

### Integration Testing  
1. **New system validation**: Complete testing of AudioStreamBuffer functionality
2. **Performance measurement**: Validate adaptive chunking performance
3. **Memory usage validation**: Buffer cleanup behavior
4. **Long-running stability**: 24+ hour stream processing
5. **FFmpeg integration**: Various stream formats and qualities

### Deployment Strategy
1. **Complete replacement**: Remove old AudioBuffer system entirely
2. **All-at-once migration**: No gradual rollout, complete system replacement
3. **Enhanced monitoring**: New metrics for buffer and processor health
4. **Clean architecture**: No legacy fallback mechanisms, simplified codebase

## Expected Benefits

### Performance Improvements
- **Reduced latency**: Adaptive chunking vs fixed 1-second delays
- **Better accuracy**: Context preservation across chunk boundaries  
- **Memory efficiency**: Automatic buffer cleanup and management
- **CPU optimization**: VAD integration reduces processing of silence

### Operational Benefits
- **Unified codebase**: Same buffer logic as proven WebSocket implementation
- **Enhanced monitoring**: Rich metrics for buffer and transcription health
- **Better error recovery**: Sophisticated stall detection and recovery
- **Future extensibility**: Easy integration of new features from WebSocket side

### Code Quality Benefits  
- **Reduced duplication**: Shared abstractions between WebSocket and RTSP
- **Better testability**: Components can be unit tested independently
- **Cleaner architecture**: Clear separation of concerns
- **Maintainability**: Single source of truth for buffer management logic