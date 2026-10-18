This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

The rules in @CODE_STYLE.md are an imperative. Failure to follow them will represent a failure in your ability to do your job.

## Project Overview

Eavesdrop is a real-time audio transcription system using NVIDIA Parakeet TDT 0.6B v2 through NeMo. Monorepo with six packages:

| Package | Description |
|---------|-------------|
| `server` | Core transcription server (WebSocket, RTSP, Parakeet integration) |
| `client` | Python client library for streaming transcription |
| `wire` | Shared Pydantic message types and protocol definitions |
| `common` | Shared Python utilities and data structures |
| `active-listener` | Keyboard-driven workstation dictation service |
| `active-listener-ui-gnome` | GNOME Shell extension for status, controls, transcript display, and preferences |

Use package-level README files and package configuration for detailed architecture and commands.

## Type Safety

- **NEVER use `Any` type** — all code must be fully typed
- Use `str | None` not `Optional[str]`
- Keep native word start/end times on the original audio timeline, including across VAD gaps

## Documentation

Use **reStructuredText** docstrings:

```python
def process_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> AudioResult:
    """Process audio data with optional VAD.

    :param audio: Input audio waveform.
    :type audio: np.ndarray
    :param sample_rate: Audio sample rate in Hz.
    :type sample_rate: int
    :returns: Processed audio result.
    :rtype: AudioResult
    :raises ValueError: If audio format is invalid.
    """
```

## Key Practices

- **Protocol-based architecture** — AudioSource, TranscriptionSink interfaces
- **Resource management** — Cancel and await async tasks during cleanup
- **Error handling** — Use `.exception()` for stack traces, fail fast on critical errors
- **Thread safety** — asyncio.Lock for async contexts

## GNOME Extension

When `packages/active-listener-ui-gnome` is modified, install the extension through the repo task:

```bash
task install-active-listener-ui-gnome
```

Scott wants his local GNOME Shell extension to reflect the repository's local state whenever that state is believed to work. Do not install mid-change or from a known-broken state, but once the extension change is coherent and verified, install it before yielding.

Do not rely on the package-local `install:extension` script for repository-level verification.

## Transcription Algorithm

Silence-based segment completion:
- `silence_completion_threshold` (default 0.8s) controls when segments complete
- VAD's `min_silence_duration_ms` auto-derived from threshold
- All segments except last marked complete; last completes on silence detection
- Always maintains incomplete segment at tail (client state machine invariant)

## Dependency Management

- **MUST use `uv add` to add dependencies to Python projects** — do not edit dependency lists by hand.

## Testing

```bash
# Type checking
task typecheck

# Optional runtime type check with NeMo/Silero dependencies
task typecheck-server

# GNOME extension build and install
task install-active-listener-ui-gnome

# Linting
uv run ruff check
```

Check package configuration for test commands — don't assume.

## Transcription Runtime

- Fixed English-only model: `nvidia/parakeet-tdt-0.6b-v2`; no translation, prompts, hotwords, or alternate checkpoints
- Heavy dependencies are opt-in: `uv sync --package eavesdrop-server --extra transcription`
- Keep normal unit tests independent of NeMo, model downloads, and GPU availability
- Docker build context is the repository root; use its `uv.lock`, not package-local historical locks
- All image variants use `/opt/venv`; protect vendor Torch/audio/vision and GPU libraries during dependency synchronization
- `task build-server-image` defaults to CPU; set `ACCELERATOR=cuda` or `ACCELERATOR=rocm` explicitly
- ROCm is experimental; never treat import/unit checks as real GPU inference validation
