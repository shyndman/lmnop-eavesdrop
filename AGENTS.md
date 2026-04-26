This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

The rules in @CODE_STYLE.md are an imperative. Failure to follow them will represent a failure in your ability to do your job.

## Project Overview

Eavesdrop is a real-time audio transcription system using Whisper models. Monorepo with five packages:

| Package | Description |
|---------|-------------|
| `server` | Core transcription server (WebSocket, RTSP, Whisper integration) |
| `client` | Python client library for streaming transcription |
| `wire` | Shared Pydantic message types and protocol definitions |
| `active-listener` | Python orchestrator spawning Electron UI for voice transcription |
| `active-listener-ui-electron` | Transparent floating overlay displaying transcription |

See package-level CLAUDE.md files for detailed architecture.

## Type Safety

- **NEVER use `Any` type** — all code must be fully typed
- Use `str | None` not `Optional[str]`
- Handle `Iterable` to `list` conversions explicitly (critical for Whisper results)

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

## Transcription Algorithm

Silence-based segment completion:
- `silence_completion_threshold` (default 0.8s) controls when segments complete
- VAD's `min_silence_duration_ms` auto-derived from threshold
- All segments except last marked complete; last completes on silence detection
- Always maintains incomplete segment at tail (client state machine invariant)

## Testing

```bash
# Type checking
uv run basedpyright

# packages/server on a host machine needs its opt-in transcription deps
cd packages/server && uv run --group type_checkable basedpyright

# Linting
uv run ruff check
```

Check package configuration for test commands — don't assume.
