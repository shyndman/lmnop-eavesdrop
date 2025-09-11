# Eavesdrop

A real-time audio transcription system with WebSocket-based speech-to-text services using Whisper models. Designed for AMD ROCm GPU environments with support for both containerized and native deployments.

## Architecture

Eavesdrop is split into three packages:

- **[eavesdrop-server](./packages/server/)** - WebSocket server handling client connections, audio processing, and transcription coordination
- **[eavesdrop-client](./packages/client/)** - Python client library for streaming transcription
- **[eavesdrop-wire](./packages/wire/)** - Shared message types and protocol definitions

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