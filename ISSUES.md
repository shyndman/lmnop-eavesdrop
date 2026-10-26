# Architectural Issues

### Inconsistent Error Handling Strategy

**Location:** Multiple files

Different modules handle errors differently:
- `processor.py:340`  logs exception, sleeps, continues
- `audio_flow.py:98-100`  silent return on `ConnectionClosedError`
- `server.py:128`  re-raises

Consider establishing explicit categories:
| Category | Examples | Response |
|----------|----------|----------|
| Recoverable | Transient network, audio decode failure | Log, continue |
| Fatal | Model load failure, config invalid | Log, propagate |
| Expected | Client disconnect | Log at INFO, clean shutdown |

### Task Management Could Use `TaskGroup`

**Location:** `packages/server/src/eavesdrop/server/streaming/client.py:89-112`

The manual `asyncio.wait()` + cancellation pattern is correct but verbose. Python 3.11's `asyncio.TaskGroup` or `anyio.create_task_group()` would be cleaner and handle edge cases better.

---

## Low Priority

### `SINGLE_MODEL` Hidden in Constants

**Location:** `packages/server/src/eavesdrop/server/constants.py:10`

This critical architectural decision (`SINGLE_MODEL = True`) controls whether all clients share a single Orukeet model for the configured CPU or Vulkan backend. It's buried in a constants file with no documentation.

Future improvement: make configurable via `EavesdropConfig` and inject the model instance into `StreamingTranscriptionProcessor` rather than having it reach up to a class variable.

### Buffer Exposes Internal Timing Semantics

**Location:** `packages/server/src/eavesdrop/server/streaming/buffer.py:37-244`

Callers must understand `buffer_start_time` and `processed_up_to_time` invariants. The TODOs on lines 29-34 note missing invariant checks. A higher-level API like `get_next_unprocessed_chunk()` would be safer.

## Architectural Strengths (Preserve These)

1. **Protocol-based abstractions**  `AudioSource` and `TranscriptionSink` in `streaming/interfaces.py` cleanly decouple I/O from processing
2. **Configuration validation**  Pydantic validators enforce invariants and derive VAD parameters from `silence_completion_threshold`
3. **Session tracing**  `TranscriptionSession` provides consistent observability with stage-specific tracers
4. **Single-model sharing**  Efficient CPU utilization across concurrent clients
5. **Pipeline pattern**  `AudioSource StreamingTranscriptionProcessor TranscriptionSink` is sound and extensible
