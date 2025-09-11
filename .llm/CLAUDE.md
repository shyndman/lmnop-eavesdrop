# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Eavesdrop is a real-time audio transcription system with WebSocket-based speech-to-text services using Whisper models. It's designed as a monorepo with three interconnected packages:

- **eavesdrop-server** - Core transcription server with WebSocket handling and model management
- **eavesdrop-client** - Python client library for streaming transcription 
- **eavesdrop-wire** - Shared message types and protocol definitions

## Commands

### Development Setup
```bash
# Install dependencies for each package
cd packages/server && uv sync
cd packages/client && uv sync  
cd packages/wire && uv sync
```

### Code Quality (run from any package directory)
```bash
# Linting and formatting
ruff check            # Check for linting issues
ruff check --fix      # Auto-fix linting issues
ruff format          # Format code

# Type checking
pyright              # Type checking with strict settings
```

### Running the Server
```bash
cd packages/server
# Copy and edit config file first
cp config.sample.yaml config.yaml
# Run server (requires config file)
eavesdrop-server --config config.yaml --port 9090
```

### Testing
```bash
# Run tests (from package directories with test suites)
cd packages/server && pytest
```

## Architecture

### Package Structure
- **packages/server/src/eavesdrop/server/** - Main server implementation
  - `server.py` - TranscriptionServer orchestrator
  - `streaming/` - Audio streaming and transcription processing
  - `rtsp/` - RTSP stream management and caching
  - `transcription/` - Whisper model integration
  - `config.py` - Pydantic-based configuration system
  
- **packages/client/src/eavesdrop/client/** - Client library
  - `core.py` - Main client components
  - `connection.py` - WebSocket connection handling
  - `audio.py` - Audio capture and processing
  
- **packages/wire/src/eavesdrop/wire/** - Protocol definitions
  - `messages.py` - Pydantic message types for WebSocket communication
  - `transcription.py` - Transcription data structures
  - `codec.py` - Message serialization utilities

### Core Components
- **TranscriptionServer** - Main orchestrator managing WebSocket connections and RTSP streams
- **WebSocketStreamingClient** - Protocol-based streaming client combining audio input, buffering, and transcription
- **StreamingTranscriptionProcessor** - Core transcription engine with Faster Whisper integration
- **AudioStreamBuffer** - Intelligent audio frame buffering with configurable cleanup
- **RTSPClientManager** - Manages multiple RTSP streams with centralized model sharing

### Protocol-Based Design
Uses Protocol interfaces for extensibility:
- **AudioSource** - Audio input abstraction (WebSocket, RTSP, etc.)
- **TranscriptionSink** - Output destination abstraction (WebSocket, file, etc.)

## Type Safety Requirements

This codebase enforces strict typing standards:

- **NEVER use `Any` type** - All code must be fully typed
- Use modern union syntax: `str | None` instead of `Optional[str]`
- All class attributes and method parameters must have explicit type annotations
- Protocol implementations must be explicit when implementing interfaces
- Handle `Iterable` to `list` conversions explicitly (critical for Whisper transcriber results)

## Configuration

- **YAML-based configuration** via EavesdropConfig (Pydantic models)
- **Required**: `--config` parameter or `EAVESDROP_CONFIG` environment variable
- **Environment variables**: `EAVESDROP_PORT`, `JSON_LOGS`, `LOG_LEVEL`, `CORRELATION_ID`
- Configuration includes transcription, RTSP, buffer management, and VAD settings

## Docker Support

```bash
# Build with GPU support (AMD ROCm)
docker build --build-arg GFX_ARCH=gfx1030 -f docker/Dockerfile .

# Run with GPU access
docker run --device /dev/kfd --device /dev/dri --publish 9090:9090 eavesdrop
```

## Key Development Practices

- **Protocol-based architecture** - Use AudioSource and TranscriptionSink protocols for new components
- **Resource management** - Always cancel and await async tasks during cleanup
- **Error handling** - Use `.exception()` for proper stack traces, fail fast on critical errors
- **Thread safety** - Use appropriate lock types (asyncio.Lock for async contexts)
- **Memory management** - Implement proper buffer cleanup and model sharing
- **Single model mode** - Shared Whisper model resources across clients for efficiency

## Model and GPU Support

- **Whisper models** - All standard models supported (tiny to large-v3, distil variants, turbo)
- **GPU acceleration** - CUDA/ROCm with automatic precision selection, CPU fallback
- **Model caching** - Automatic HuggingFace to CTranslate2 conversion and local caching
- **Voice Activity Detection** - ONNX-based VAD with configurable parameters

## Testing and Validation

- Configure test framework by examining existing test files and dependencies
- Never assume specific test commands - check package configuration
- Run type checking (`pyright`) and linting (`ruff check`) before testing