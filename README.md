# Eavesdrop

A real-time English audio transcription system using Orukeet through CPU sherpa-onnx (INT8 ONNX) or Vulkan NeMo-Speech.cpp (Q8 GGUF). Run directly on the host or use the unified CPU/Vulkan Docker image.

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
# From the repository root; Python 3.12, FFmpeg, and libsndfile required.
uv sync --package eavesdrop-server --extra transcription
cp packages/server/config.sample.yaml packages/server/config.yaml
# The sample defaults to CPU; select device: vulkan for a Linux x86_64 Vulkan host.
uv run --package eavesdrop-server --extra transcription eavesdrop-server \
  --config packages/server/config.yaml
```

The selected backend's pinned model assets download on first use and are cached for reuse. Vulkan uses GPU 0 on Linux x86_64. See the [server guide](./packages/server/README.md) for system dependencies, model caching, and migration from the previous backend.

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
- **Orukeet backends** - Fixed `oruk/orukeet` model with CPU INT8 ONNX or Vulkan Q8 GGUF recognition
- **Native word timestamps** - CPU token onsets and durations at 80 ms granularity, or Vulkan native word timestamps converted from milliseconds, preserved on the original recording timeline
- **Host inference** - CPU sherpa-onnx by default, or pinned NeMo-Speech.cpp on Vulkan GPU 0 for Linux x86_64
- **Voice Activity Detection** - Standalone Silero VAD with configurable speech/silence thresholds

### Connectivity & Streaming
- **WebSocket protocol** - Simple WebSocket-based API for easy integration
- **RTSP stream support** - Subscribe to transcription results from IP cameras and audio streams
- **Multi-client support** - No limits on concurrent connections or connection duration
- **Historical transcription** - New RTSP subscribers receive recent transcription history
- **Configurable caching** - Smart cache management based on active listeners

### Applications
- **Active Listener** - Desktop application for voice-to-text input using system audio capture and automatic typing
- **Automatic reconnection** - Robust connection handling for RTSP streams

## Development

This is a `uv` workspace. Run development commands from the repository root:

```bash
# Lightweight development and unit tests, without downloading the ASR runtime.
uv sync --package eavesdrop-server
uv run --package eavesdrop-server pytest packages/server/tests
uvx ruff check
uv run --package eavesdrop-server basedpyright

# Opt into sherpa-onnx and standalone Silero when running transcription.
uv sync --package eavesdrop-server --extra transcription
```

Unit tests inject recognizer/VAD fakes; they do not download a checkpoint or establish real-model accuracy or performance. Use the server guide's audio smoke check for runtime validation.

### Development Tools

- **uv** - Fast Python package manager and environment management
- **ruff** - Fast Python linter and formatter
- **basedpyright** - Static type checker for Python
- **pytest** - Testing framework
- **watchexec** - File watching and auto-restarting during development

## Docker

```bash
# Unified CPU/Vulkan image. Run from the repository root.
task build-server-image
task run-server-image
```

The run tasks select CPU (default) or Vulkan from `transcription.device` in the mounted config. Override its path with `SERVER_CONFIG=/path/to/config.yaml`; use `task run-server-image-persistent` for a background container with `--restart always`. Vulkan containers require Linux x86_64 with a Mesa-supported GPU, its host kernel driver, and `/dev/dri`; Mesa userspace drivers are included in the image.

See [server Docker instructions](./packages/server/README.md#docker-deployment) for prerequisites, cache volumes, manual commands, and verification limits.

## License

MIT
