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
# Edit config.yaml for your needs (model, language, RTSP streams, etc.)
uv run eavesdrop-server --config config.yaml
```

### Active Listener (Desktop Voice Input)

```bash
# Install or update the user service bound to graphical-session.target
task install-active-listener-service

# Inspect service health and logs
systemctl --user status active-listener.service
journalctl --user -u active-listener.service -f

# Remove the user service
task uninstall-active-listener-service
```

The active-listener runtime config now lives at `~/.config/eavesdrop/active-listener.yaml`.
The optional rewrite prompt override lives at `~/.config/eavesdrop/active-listener.rewrite.system.md`.

When rewrite is enabled, active-listener now expects a local LiteRT `.litertlm` bundle at `llm_rewrite.model_path`. That setting, along with `llm_rewrite.prompt_path`, resolves relative to the config file directory. Prompt files are markdown only now — no hidden endpoint settings, routing metadata, or template rendering in the rewrite path.

GNOME prefs still edits the same override file, and the service reloads that markdown prompt on every rewrite request.

For manual development runs, use the same CLI entrypoint the service uses:

```bash
cd packages/active-listener
uv sync
uv run active-listener --config-path ~/.config/eavesdrop/active-listener.yaml
```

### Client Library

```bash
cd packages/client
uv sync

# Use the client library in your Python code
# See examples in packages/client/examples/
```

## Features

### Core Transcription
- **Real-time transcription** - Stream audio and get transcription results in real-time with configurable latency
- **Multiple Whisper models** - Support for all standard Whisper models (tiny to large-v3, distil variants, turbo)
- **Custom model support** - Load fine-tuned or custom Whisper models for specialized domains
- **GPU acceleration** - CUDA/ROCm and CPU inference with automatic precision selection
- **Voice Activity Detection** - Intelligent audio filtering with configurable VAD parameters
- **Hotwords support** - Improve recognition accuracy for specific terms and domains
- **Language support** - Configurable language settings for optimal transcription accuracy

### Connectivity & Streaming
- **WebSocket protocol** - Simple WebSocket-based API for easy integration
- **RTSP stream support** - Subscribe to transcription results from IP cameras and audio streams
- **Multi-client support** - No limits on concurrent connections or connection duration
- **Historical transcription** - New RTSP subscribers receive recent transcription history
- **Configurable caching** - Smart cache management based on active listeners

### Applications
- **Active Listener** - Desktop application for voice-to-text input using system audio capture and automatic typing
- **Parallel processing** - Configurable worker count for improved throughput
- **Automatic reconnection** - Robust connection handling for RTSP streams

## Development

This is a monorepo using `uv` for dependency management. Each package has its own development environment:

```bash
# Install all dependencies across packages
cd packages/server && uv sync
cd packages/client && uv sync
cd packages/wire && uv sync
cd packages/common && uv sync
cd packages/active-listener && uv sync

# Code quality (run from each package directory)
ruff check && ruff format
basedpyright

# Server host type checking needs the opt-in transcription dependency group.
cd packages/server && uv run --group type_checkable basedpyright

# Testing (where applicable)
pytest
```

### Development Tools

- **uv** - Fast Python package manager and environment management
- **ruff** - Fast Python linter and formatter
- **basedpyright** - Static type checker for Python
- **pytest** - Testing framework
- **watchexec** - File watching and auto-restarting during development

## Docker

```bash
# Build with GPU support
docker build --build-arg GFX_ARCH=gfx1030 -f docker/Dockerfile .

# Run with GPU access
docker run --device /dev/kfd --device /dev/dri --publish 9090:9090 eavesdrop
```

## License

MIT