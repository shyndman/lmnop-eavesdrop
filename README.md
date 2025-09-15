# Eavesdrop

A real-time audio transcription system with WebSocket-based speech-to-text services using Whisper models. Designed for AMD ROCm GPU environments with support for both containerized and native deployments.

## Architecture

Eavesdrop is built around a central, reusable transcription pipeline that's agnostic to audio sources and transcription destinations. The system uses protocol-based adapters to integrate with different transport layers (WebSockets, RTSP streams).

### Packages

- **[eavesdrop-server](./packages/server/)** - WebSocket server with core transcription pipeline, RTSP stream processing, and client connection management
- **[eavesdrop-client](./packages/client/)** - Python client library for streaming transcription and RTSP subscription
- **[eavesdrop-wire](./packages/wire/)** - Shared message types and protocol definitions
- **[eavesdrop-common](./packages/common/)** - Shared utilities, logging, and data structures
- **[active-listener](./packages/active-listener/)** - Desktop application for voice-to-text input using system audio capture

### Core Design

The architecture separates audio sources from transcription logic through two key protocols:

- **`AudioSource`** - Defines how audio enters the system (WebSocket, RTSP, etc.)
- **`TranscriptionSink`** - Defines how results are delivered (WebSocket responses, broadcast to subscribers, etc.)

This allows the same `StreamingTranscriptionProcessor` to handle both real-time client transcription and background RTSP stream processing.

For detailed architecture information, see [ARCHITECTURE.md](./ARCHITECTURE.md).

## Quick Start

### Server

```bash
cd packages/server
uv sync
cp config.sample.yaml config.yaml
# Edit config.yaml for your needs
eavesdrop --config config.yaml
```

### Client

```bash
cd packages/client
uv sync
# Use the client library in your Python code
```

## Features

- **Real-time transcription** - Stream audio and get transcription results in real-time
- **RTSP stream support** - Subscribe to transcription results from IP cameras and audio streams
- **Multiple Whisper models** - Support for all standard Whisper models (tiny to large-v3, distil variants, turbo)
- **GPU acceleration** - CUDA/ROCm and CPU inference with automatic precision selection
- **WebSocket protocol** - Simple WebSocket-based API for easy integration
- **Voice Activity Detection** - Intelligent audio filtering with configurable VAD parameters
- **Hotwords support** - Improve recognition accuracy for specific terms
- **Multi-client support** - No limits on concurrent connections or connection duration

## Development

This is a monorepo using `uv` for dependency management. Each package has its own development environment:

```bash
# Install all dependencies across packages
cd packages/server && uv sync
cd packages/client && uv sync  
cd packages/wire && uv sync

# Code quality (run from any package directory)
ruff check && ruff format
pyright
```

### Development Tools

- **watchexec** - Used for file watching and auto-restarting during development

## Docker

```bash
# Build with GPU support
docker build --build-arg GFX_ARCH=gfx1030 -f docker/Dockerfile .

# Run with GPU access
docker run --device /dev/kfd --device /dev/dri --publish 9090:9090 eavesdrop
```

## License

MIT