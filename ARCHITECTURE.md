# Eavesdrop Architecture Documentation

## Overview

Eavesdrop is a real-time audio transcription server built on a modern, streaming-first architecture. It provides WebSocket-based speech-to-text services using Whisper models, designed for high performance in AMD ROCm GPU environments with support for both containerized and native deployments.

The architecture emphasizes modularity, scalability, and resource efficiency through a protocol-based design that cleanly separates concerns between audio input, processing, and output delivery.

## High-Level Architecture

```mermaid
graph TD
    subgraph CLIENT ["🌐 Client Layer"]
        WS_CLIENT[WebSocket Clients]
        RTSP_STREAM[RTSP Streams]
    end

    subgraph SERVER ["🖥️ Server Layer"]
        WS_SERVER[WebSocketServer]
        RTSP_MANAGER[RTSPClientManager]  
        CLIENT_MGR[WebSocketClientManager]
    end

    subgraph PROCESSING ["⚙️ Processing Layer"]
        WS_STREAMING_CLIENT[WebSocketStreamingClient]
        RTSP_TRANSCRIPTION_CLIENT[RTSPTranscriptionClient]
        AUDIO_BUFFER[AudioStreamBuffer]
        TRANSCRIPTION_PROC[StreamingTranscriptionProcessor]
    end

    subgraph MODEL ["🤖 Model Layer"]
        WHISPER_BACKEND[ServeClientFasterWhisper]
        WHISPER_MODEL[WhisperModel]
        RTSP_MODEL_MGR[RTSPModelManager]
    end

    subgraph OUTPUT ["📤 Output Layer"]
        WS_SINK[WebSocketTranscriptionSink]
        RTSP_SINK[RTSPAudioSource]
        TRANSLATION_QUEUE[Translation Queue]
    end

    %% %% Layer-to-layer connections (vertical flow)
    CLIENT --> SERVER
    SERVER --> PROCESSING  
    PROCESSING --> MODEL
    MODEL --> OUTPUT
```

## Core Components

### 1. TranscriptionServer (`src/eavesdrop/server.py`)

The main orchestrator that coordinates all components and manages the server lifecycle.

**Responsibilities:**
- WebSocket connection handling
- Client initialization and lifecycle management
- RTSP stream coordination (when configured)
- Configuration management and validation
- Resource cleanup and graceful shutdown

**Key Features:**
- Max client limit enforcement (default: 4 clients)
- Connection timeout management (default: 300s)
- Debug audio capture capability
- Single model mode support for resource efficiency

### 2. Streaming Transcription System

#### WebSocketStreamingClient (`src/eavesdrop/streaming/websocket_client.py`)

High-level facade that combines all streaming components for WebSocket clients.

```mermaid
graph LR
    subgraph "WebSocketStreamingClient"
        AUDIO_SOURCE[WebSocketAudioSource]
        AUDIO_BUFFER[AudioStreamBuffer]
        TRANSCRIPTION_PROC[StreamingTranscriptionProcessor]
        WS_SINK[WebSocketTranscriptionSink]
    end
    
    WEBSOCKET[WebSocket] --> AUDIO_SOURCE
    AUDIO_SOURCE --> AUDIO_BUFFER
    AUDIO_BUFFER --> TRANSCRIPTION_PROC
    TRANSCRIPTION_PROC --> WS_SINK
    WS_SINK --> WEBSOCKET
```

#### AudioStreamBuffer (`src/eavesdrop/streaming/buffer.py`)

Manages audio frame buffering with intelligent cleanup and processing coordination.

**Key Features:**
- Automatic buffer cleanup (removes old audio after 45s, keeps 30s)
- Thread-safe audio frame addition
- Configurable stall detection and clipping
- Precise timestamp management for seamless transcription

#### StreamingTranscriptionProcessor (`src/eavesdrop/streaming/processor.py`)

The core transcription engine that integrates Faster Whisper model management with streaming processing.

**Responsibilities:**
- Model loading and initialization (with CTranslate2 conversion)
- Continuous transcription loop with VAD integration
- Segment processing and completion detection
- Single model mode support with thread-safe access

### 3. Protocol-Based Architecture

The system uses Protocol interfaces for clean separation and extensibility:

```mermaid
classDiagram
    class AudioSource {
        <<protocol>>
        +read_audio() -> ndarray | None
        +close() -> None
    }
    
    class TranscriptionSink {
        <<protocol>>
        +send_result(result: TranscriptionResult) -> None
        +send_error(error: str) -> None  
        +send_language_detection(lang: str, prob: float) -> None
        +send_server_ready(backend: str) -> None
        +disconnect() -> None
    }
    
    class WebSocketAudioSource {
        +websocket: ServerConnection
        +get_audio_func: Callable
        +read_audio() -> ndarray | None
        +close() -> None
    }
    
    class WebSocketTranscriptionSink {
        +websocket: ServerConnection
        +client_uid: str
        +send_result(result: TranscriptionResult) -> None
        +send_error(error: str) -> None
        +disconnect() -> None
    }
    
    AudioSource <|.. WebSocketAudioSource
    TranscriptionSink <|.. WebSocketTranscriptionSink
```

### 4. RTSP Support System

#### RTSPClientManager (`src/eavesdrop/rtsp_manager.py`)

Manages multiple RTSP streams with centralized model sharing and health monitoring.

**Features:**
- Concurrent RTSP stream processing
- Stream failure detection and restart capability
- Centralized statistics and monitoring
- Graceful shutdown coordination

#### RTSPModelManager (`src/eavesdrop/rtsp_models.py`)

Provides shared Whisper model instances for RTSP streams to optimize resource usage.

### 5. Model Management and Backend

#### Faster Whisper Backend (`src/eavesdrop/backend.py`)

Legacy backend implementation, being superseded by the streaming processor architecture.

#### WhisperModel (`src/eavesdrop/transcription/whisper_model.py`)

Wrapper around Faster Whisper with CTranslate2 optimization and GPU selection.

## Data Flow Patterns

### WebSocket Client Flow

```mermaid
sequenceDiagram
    participant Client
    participant WSServer as WebSocket Server
    participant ClientMgr as Client Manager
    participant StreamClient as Streaming Client
    participant Buffer as Audio Buffer
    participant Processor as Transcription Processor
    participant Model as Whisper Model
    participant Sink as WebSocket Sink

    Client->>WSServer: Connect + Options
    WSServer->>ClientMgr: Check capacity
    WSServer->>StreamClient: Initialize client
    StreamClient->>Processor: Initialize model
    Processor->>Model: Load Whisper model
    Processor->>Sink: Send SERVER_READY
    
    loop Audio Streaming
        Client->>WSServer: Audio frames
        WSServer->>Buffer: Add frames
        Buffer->>Processor: Audio chunks
        Processor->>Model: Transcribe
        Model->>Processor: Segments
        Processor->>Sink: Transcription result
        Sink->>Client: JSON segments
    end
    
    Client->>WSServer: END_OF_AUDIO | Disconnect
    StreamClient->>Processor: Stop processing
    StreamClient->>Sink: Disconnect
    ClientMgr->>StreamClient: Cleanup
```

### RTSP Stream Flow

```mermaid
sequenceDiagram
    participant Config as RTSP Config
    participant Manager as RTSP Manager
    participant Client as RTSP Client
    participant ModelMgr as Model Manager
    participant Processor as Stream Processor
    participant Buffer as Audio Buffer

    Config->>Manager: Load stream configuration
    Manager->>ModelMgr: Get shared transcriber
    Manager->>Client: Create RTSP client
    Client->>Client: Connect to RTSP stream
    
    loop Stream Processing
        Client->>Buffer: Audio from RTSP
        Buffer->>Processor: Audio chunks  
        Processor->>ModelMgr: Transcribe (shared model)
        Processor->>Client: Transcription results
        Client->>Client: Process/store results
    end
    
    Manager->>Client: Stop stream
    Client->>Client: Cleanup connection
    ModelMgr->>ModelMgr: Cleanup shared resources
```

### Model Resource Management

```mermaid
graph TD
    subgraph "Single Model Mode"
        SM_LOCK[Single Model Lock]
        SHARED_MODEL[Shared Whisper Model]
        CLIENT1[Client 1]
        CLIENT2[Client 2]
        CLIENT3[Client 3]
    end
    
    subgraph "Per-Client Mode"
        MODEL1[Whisper Model 1]
        MODEL2[Whisper Model 2]
        MODEL3[Whisper Model 3]
        PC_CLIENT1[Client 1]
        PC_CLIENT2[Client 2] 
        PC_CLIENT3[Client 3]
    end
    
    CLIENT1 --> SM_LOCK
    CLIENT2 --> SM_LOCK
    CLIENT3 --> SM_LOCK
    SM_LOCK --> SHARED_MODEL
    
    PC_CLIENT1 --> MODEL1
    PC_CLIENT2 --> MODEL2
    PC_CLIENT3 --> MODEL3
```

## Critical Types and Data Structures

### Configuration Types

```python
@dataclass
class BufferConfig:
    sample_rate: int = 16000
    max_buffer_duration: float = 45.0
    cleanup_duration: float = 30.0
    min_chunk_duration: float = 1.0
    clip_audio: bool = False
    max_stall_duration: float = 25.0

@dataclass  
class TranscriptionConfig:
    send_last_n_segments: int = 10
    no_speech_thresh: float = 0.45
    same_output_threshold: int = 10
    use_vad: bool = True
    model: str = "distil-small.en"
    task: str = "transcribe"
    language: str | None = None
    single_model: bool = False
    cache_path: str = "~/.cache/eavesdrop/"
    device_index: int = 0
```

### Core Data Types

```python
@dataclass
class TranscriptionResult:
    segments: list[dict]
    language: str | None = None
    language_probability: float | None = None

@dataclass
class Segment:
    id: int
    start: float
    end: float  
    text: str
    no_speech_prob: float
    # Additional transcription metadata...

@dataclass
class TranscriptionInfo:
    language: str
    language_probability: float
    duration: float
    # Model and processing metadata...
```

## Resource Management

### GPU and Device Selection

The system automatically detects and optimizes for available hardware:

- **CUDA Detection**: Automatic GPU capability detection with precision optimization
- **ROCm Support**: AMD GPU support with configurable architecture targeting
- **CPU Fallback**: Automatic fallback to CPU with int8 quantization
- **Device Index**: Configurable GPU device selection for multi-GPU systems

### Memory Management

- **Buffer Limits**: Automatic audio buffer cleanup to prevent memory exhaustion
- **Model Sharing**: Single model mode reduces memory footprint for multiple clients
- **Resource Cleanup**: Comprehensive cleanup on client disconnection

### Concurrency Model

- **Async/Await**: Full async implementation for non-blocking I/O
- **Thread Safety**: Critical sections protected with locks for model access
- **Task Management**: Proper task lifecycle management with graceful cancellation
- **Connection Pools**: Efficient WebSocket connection management

## Configuration and Deployment

### Environment Variables

All command-line arguments have environment variable equivalents:

- `EAVESDROP_PORT` (default: 9090)
- `EAVESDROP_BACKEND` (default: faster_whisper)  
- `EAVESDROP_MAX_CLIENTS` (default: 4)
- `EAVESDROP_MAX_CONNECTION_TIME` (default: 300)
- `EAVESDROP_CACHE_PATH` (default: /app/.cache/eavesdrop/)
- `JSON_LOGS` - Enable structured JSON logging
- `LOG_LEVEL` (default: INFO)

### Docker Architecture

```mermaid
graph LR
    subgraph "Docker Container"
        subgraph "ROCm Runtime"
            ROCM[ROCm 6.4.2]
            PYTORCH[PyTorch with ROCm]
        end
        
        subgraph "Python Environment"  
            WHISPER[Faster Whisper]
            CTRANSLATE2[CTranslate2]
            SERVER[Eavesdrop Server]
        end
        
        subgraph "GPU Access"
            KFD[/dev/kfd]
            DRI[/dev/dri]
        end
    end
    
    ROCM --> PYTORCH
    PYTORCH --> WHISPER
    WHISPER --> CTRANSLATE2
    CTRANSLATE2 --> SERVER
    KFD --> ROCM
    DRI --> ROCM
```

### Model Caching Strategy

- **Local Cache**: Models cached in `~/.cache/eavesdrop/whisper-ct2-models/`
- **Auto-conversion**: HuggingFace models automatically converted to CTranslate2
- **Version Management**: Safe model name transformation for filesystem storage
- **Quantization**: Automatic precision selection based on device capability

## Performance Characteristics

### Latency Optimization

- **Streaming Processing**: Sub-second latency for real-time transcription
- **VAD Integration**: Voice Activity Detection reduces unnecessary processing
- **Efficient Buffering**: Minimal memory copy operations in audio pipeline
- **Model Reuse**: Single model mode eliminates initialization overhead

### Scalability Features

- **Connection Limits**: Configurable client limits prevent resource exhaustion
- **Timeout Management**: Automatic cleanup of stale connections
- **RTSP Scaling**: Multiple concurrent RTSP streams with shared resources
- **Error Recovery**: Robust error handling with graceful degradation

### Resource Efficiency

- **Memory Bounds**: Automatic buffer cleanup prevents memory leaks
- **GPU Utilization**: Efficient GPU memory usage with precision optimization
- **Thread Management**: Minimal thread usage through async architecture
- **Model Sharing**: Reduced memory footprint in multi-client scenarios

## Extensibility Points

### Protocol System

The protocol-based architecture enables easy extension:

- **AudioSource**: Custom audio input sources (file, network, etc.)
- **TranscriptionSink**: Custom output destinations (files, databases, etc.)
- **Backend Interfaces**: Support for different transcription engines

### Configuration System

- **Modular Config**: Separate configuration classes for different components
- **Environment Integration**: Comprehensive environment variable support
- **YAML Configuration**: File-based configuration for RTSP streams

### Processing Pipeline

- **Translation Integration**: Built-in translation queue support
- **Custom Processing**: Pluggable segment processing
- **Output Filtering**: Configurable segment filtering and validation

This architecture provides a robust, scalable foundation for real-time audio transcription with clear separation of concerns, efficient resource management, and extensive configurability.
