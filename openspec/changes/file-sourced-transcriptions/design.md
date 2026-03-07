## Context

The current transcriber flow is optimized for live streams (microphone/RTSP) and uses a shared buffering/processing lifecycle. Finite file input introduces different invariants:

1. input ends deterministically,
2. all buffered audio must be drained before terminal signaling, and
3. callers want a single awaitable result rather than an open-ended stream API.

This change spans `packages/server`, `packages/client`, and `packages/wire`. It must preserve existing live streaming behavior while adding a file-oriented path with strong completion guarantees. The user preference is a single-shot OOP client API, server-side decode/normalization, bounded in-memory queueing (900 seconds), and periodic queue/throughput observability.

## What a Junior Engineer Should Keep in Their Head

This change is easier if you think of it as **adding one new transcriber mode**, not as rewriting the whole server.

1. Live mode already exists. Do not break it.
2. File mode is a second path that starts the same way, but diverges after setup.
3. File mode has two jobs:
   1. receive and decode file data,
   2. keep processing until all decoded audio is consumed.
4. The client-side `transcribe_file(...)` method is just a convenience wrapper over the existing websocket protocol.

## Goals / Non-Goals

**Goals:**
1. Add file-sourced transcription semantics that guarantee EOF drain and deterministic finalization.
2. Add `EavesdropClient` one-shot file transcription API over existing transcriber protocol.
3. Decode/normalize file audio on server side before queue residency (mono, 16kHz, float32 canonical format).
4. Enforce bounded in-memory queueing for file ingestion (900s canonical audio) with blocking backpressure.
5. Emit periodic operational metrics for file ingest/processing queue health.
6. Preserve existing live transcriber and RTSP behavior.

**Non-Goals:**
1. Introducing disk-backed spooling for file audio.
2. Replacing existing live-stream `AudioStreamBuffer` implementation for all modes.
3. Adding offline batched inference in this change.
4. Implementing multi-file job orchestration or concurrent file sessions beyond a single active file session policy.

## Verified Current-State Map

These are the files a junior engineer should read before touching code:

1. **Wire setup and messages**
   1. `packages/wire/src/eavesdrop/wire/messages.py`
   2. `packages/wire/src/eavesdrop/wire/transcription.py`
   3. `packages/wire/src/eavesdrop/wire/codec.py`
2. **Client public API and websocket handling**
   1. `packages/client/src/eavesdrop/client/core.py`
   2. `packages/client/src/eavesdrop/client/connection.py`
3. **Server transcriber routing and lifecycle**
   1. `packages/server/src/eavesdrop/server/connection_handler.py`
   2. `packages/server/src/eavesdrop/server/server.py`
   3. `packages/server/src/eavesdrop/server/streaming/client.py`
   4. `packages/server/src/eavesdrop/server/streaming/processor.py`
   5. `packages/server/src/eavesdrop/server/streaming/audio_flow.py`
   6. `packages/server/src/eavesdrop/server/streaming/buffer.py`
   7. `packages/server/src/eavesdrop/server/config.py`
4. **Existing FFmpeg subprocess pattern**
   1. `packages/server/src/eavesdrop/server/rtsp/client.py`
5. **Tests that define current streaming behavior**
   1. `packages/server/tests/test_streaming_lifecycle_contracts.py`
   2. `packages/server/tests/test_transcription_output_contracts.py`
   3. `packages/server/tests/test_audio_stream_buffer_invariants.py`
   4. `packages/client/tests/test_client_mode_contracts.py`
   5. `packages/wire/tests/test_wire_codec_contracts.py`

## End-to-End Target Flow

The target flow should look like this:

```text
Client transcribe_file(path)
  -> connect as normal transcriber
  -> send setup(options.source_mode="file")
  -> upload file bytes
  -> send END_OF_AUDIO sentinel
  -> receive TranscriptionMessage windows
  -> reduce completed segments
  -> receive DisconnectMessage or socket close
  -> return FileTranscriptionResult

Server file mode
  -> accept transcriber setup
  -> detect source_mode="file"
  -> receive uploaded file bytes
  -> decode with ffmpeg
  -> normalize to mono 16kHz float32
  -> enqueue canonical audio into bounded file queue
  -> processor consumes canonical audio
  -> after EOF, keep draining until queue empty
  -> emit terminal disconnect only after drain completes
```

## File-by-File Implementation Plan

### 1) Wire changes

**Files likely to change:**
1. `packages/wire/src/eavesdrop/wire/transcription.py`
2. `packages/wire/src/eavesdrop/wire/messages.py`
3. `packages/wire/tests/test_wire_codec_contracts.py`

**What to add:**
1. Extend `UserTranscriptionOptions` with a field that declares source mode.
2. Use a simple value shape that is easy to serialize and inspect, for example:
   1. string enum-like field,
   2. default live behavior,
   3. explicit file mode value.
3. Keep defaults backward compatible so old clients still mean “live”.

**Junior pitfall to avoid:**
Do not add a whole new websocket message type for setup if the current setup message can already carry one more field.

### 2) Client websocket handling changes

**Files likely to change:**
1. `packages/client/src/eavesdrop/client/connection.py`
2. `packages/client/src/eavesdrop/client/core.py`
3. `packages/client/tests/test_client_mode_contracts.py`

**What to add:**
1. Explicit handling for `DisconnectMessage`.
2. A terminal condition that marks the one-shot operation complete.
3. A non-reentrancy guard so one client instance cannot run two operations at once.

**Junior pitfall to avoid:**
Do not rely on “the socket might close eventually” as the only terminal signal. Handle wire disconnect explicitly and treat close as fallback.

### 3) Server routing changes

**Files likely to change:**
1. `packages/server/src/eavesdrop/server/connection_handler.py`
2. `packages/server/src/eavesdrop/server/server.py`
3. possibly a new helper module under `packages/server/src/eavesdrop/server/streaming/`

**What to add:**
1. Read `source_mode` from setup.
2. Keep current live path unchanged.
3. Route file sessions into a file-specific ingest path.

**Junior pitfall to avoid:**
Do not bury `if source_mode == ...` checks all over unrelated code. Route early, then keep live/file behaviors separated.

### 4) Decoder path

**Decision:** use the existing FFmpeg runtime pattern.

**Why:**
1. `soundfile` alone cannot satisfy AAC.
2. Server images already install `ffmpeg`.
3. RTSP code already demonstrates subprocess-based FFmpeg usage.

**Files likely to change:**
1. a new helper module under `packages/server/src/eavesdrop/server/streaming/` or nearby,
2. `packages/server/src/eavesdrop/server/rtsp/client.py` only if extracting shared helper logic makes sense,
3. `packages/server/pyproject.toml` only if implementation unexpectedly requires a new Python package.

**Expected call order:**
1. receive raw uploaded file bytes,
2. start ffmpeg subprocess,
3. write file bytes to ffmpeg stdin,
4. read canonical audio bytes from ffmpeg stdout,
5. convert canonical bytes to NumPy float32,
6. enqueue canonical frames,
7. close stdin,
8. drain remaining stdout,
9. wait for process exit,
10. fail clearly if exit code is non-zero.

**Important detail:**
Canonical output format should be chosen so downstream server code can continue using `np.frombuffer(..., dtype=np.float32)` or another equally direct conversion.

**Junior pitfall to avoid:**
Do not store the original compressed file in the queue. Decode first, then queue canonical audio.

### 5) File queue design

**Decision:** dedicated bounded in-memory queue with 900 seconds of canonical audio.

**Recommended shape:**
1. queue stores canonical audio chunks only,
2. queue tracks total buffered samples or seconds,
3. enqueue blocks when adding a chunk would exceed the limit,
4. dequeue reduces the tracked total,
5. queue supports “producer is done” state.

**What “900 seconds” means:**
It means 900 seconds of **mono 16kHz float32**, not 900 seconds of source-format bytes.

**Junior pitfall to avoid:**
Do not use `AudioStreamBuffer` as the queue itself. The new queue is upstream of the processor; `AudioStreamBuffer` stays the processor’s working buffer unless a later refactor replaces it.

### 6) Lifecycle state machine

Use explicit states. Keep them simple.

```text
INGESTING   = still receiving file bytes / decoded canonical audio may still arrive
DRAINING    = no more input will arrive, but queued/buffered audio still exists
FINALIZING  = all audio processed, emit final disconnect / cleanup
TERMINAL    = session fully closed
```

**Transition rules:**
1. Start in `INGESTING`.
2. Move to `DRAINING` after upload EOF + decoder EOF.
3. Stay in `DRAINING` until:
   1. file queue empty,
   2. processor buffer fully consumed,
   3. final completed output emitted.
4. Move to `FINALIZING`, send disconnect, release tasks/resources.
5. Move to `TERMINAL` only after cleanup finishes.

**Junior pitfall to avoid:**
Do not keep using current `FIRST_COMPLETED` semantics for file mode. That is exactly the bug shape this change is trying to avoid.

### 6.1) Required in-code intent capture (single doc comment)

To prevent design drift, implementation MUST include exactly one structural doc comment in the primary server function that owns file-mode lifecycle transitions.

**Required comment text (copy exactly):**

```text
<intent>
Accept a finite audio file upload and produce the complete transcription result for that file.
This function coordinates ingest, drain, and finalization so the caller gets one deterministic terminal result.
</intent>
```

**Proposed location:**
`packages/server/src/eavesdrop/server/streaming/client.py` inside the new file-mode lifecycle orchestrator function (recommended name: `_run_file_mode_lifecycle`) at the top of the function body, immediately after the function signature.

If implementation chooses a different function name, place this SAME single comment in whichever single function is the lifecycle owner; do not duplicate this intent comment elsewhere.

### 7) Client one-shot API

**Decision:** add `EavesdropClient.transcribe_file(..., timeout_s=...)`.

**Likely return shape:**
1. `segments: list[Segment]`
2. `text: str`
3. `language: str | None`
4. `warnings: list[...]`

**Expected method steps:**
1. acquire operation lock,
2. connect,
3. send transcriber setup with file mode,
4. upload file bytes,
5. send `END_OF_AUDIO`,
6. read messages until terminal signal,
7. reduce completed segments,
8. cleanup connection,
9. release lock,
10. return result.

**Timeout/cancellation rules:**
1. per-call timeout only,
2. cancellation means cleanup then re-raise,
3. timeout means cleanup then raise timeout,
4. internal layers should not spam duplicate exception logs.

### 8) Reducer algorithm

This is the most subtle client-side piece. Keep it mechanical.

**Message shape assumption from current server:**
1. completed segments window first,
2. one incomplete tail segment last.

**Algorithm:**
1. if no segments, do nothing,
2. ignore the last segment if it is the tail/incomplete segment,
3. scan the remaining completed segments in reverse order,
4. stop when `last_committed_id` is found,
5. everything newer than that is appended,
6. if `last_committed_id` is not found, warn and append the received completed window in chronological order,
7. update `last_committed_id` to the newest committed segment.

**Junior pitfall to avoid:**
Do not dedupe by text. Dedupe by segment ID only.

### 9) Logging and exception ownership

Use one owner for exception logs.

1. inner helper catches only to add cleanup, then re-raises,
2. outer operation boundary logs handled errors if needed,
3. expected control-flow exceptions (`CancelledError`, timeout) should not create duplicate stack traces at every layer,
4. non-exception anomalies such as sentinel misses should log warning where detected.

### 10) Observability

Log file-session metrics periodically while a file session is active.

**Metrics to log:**
1. queue fill ratio,
2. queued canonical audio seconds,
3. ingest audio seconds per real second,
4. processed audio seconds per real second,
5. cumulative or recent enqueue block time.

**Recommended cadence for first implementation:**
fixed interval, simple and predictable. Do not invent dynamic cadence logic.

## Recommended Implementation Order

A junior engineer should implement in this exact order:

1. Wire option changes and wire tests.
2. Client connection handling for disconnect.
3. Server routing switch between live and file mode.
4. File decoder helper.
5. File queue.
6. File-mode lifecycle/drain logic.
7. Client `transcribe_file(...)` method.
8. Client reducer.
9. Metrics logging.
10. Full focused tests.

This order matters because each later step depends on earlier contracts being in place.

## Validation Plan

At minimum, implementation must prove these cases:

1. live transcriber behavior is unchanged,
2. file mode setup serializes/deserializes correctly,
3. WAV input works,
4. MP3 input works,
5. AAC input works,
6. EOF with short remaining tail still produces final output,
7. queue blocks instead of dropping data when full,
8. reducer appends only new completed segments,
9. sentinel miss logs warning and continues,
10. concurrent `transcribe_file(...)` on one client fails fast,
11. timeout cleans up,
12. cancellation cleans up.

## Risks / Trade-offs

1. **Queue saturation under slow inference** → Mitigation: bounded queue + enqueue blocking + periodic fill-ratio logging + warn threshold.
2. **Message window gaps hide missing completed segments** → Mitigation: reducer warns on sentinel miss and continues forward progress.
3. **File mode regresses live mode behavior** → Mitigation: isolate finite-source ingestion path and add contract tests per mode.
4. **Duplicate terminal signaling (disconnect vs socket close)** → Mitigation: idempotent terminal handling in reducer/client lifecycle.
5. **Memory-only queue over disk spool** → Mitigation: conservative capacity bound (900s) and single file-session policy.
6. **FFmpeg CLI dependency in non-container environments** → Mitigation: document `ffmpeg` as a server runtime prerequisite anywhere the server is run outside the provided Docker images, and fail fast with a clear startup/runtime error if unavailable.

## Migration Plan

1. Add wire/setup option support for file-mode intent while preserving defaults for live transcriber mode.
2. Add server finite-source ingestion/decode path using the existing FFmpeg runtime pattern, plus bounded queue integration.
3. Add lifecycle finalization behavior for finite sources (drain then terminal signal).
4. Add client one-shot API with per-call timeout and non-reentrant guard.
5. Add contract tests (server/wire/client) for EOF drain, queue backpressure, reducer correctness, and terminal behavior.
6. Rollout behind default live behavior; no config migration required for existing transcriber users.

Rollback strategy: remove/disable file-mode option handling and one-shot API entrypoint while retaining existing transcriber path unchanged.

## Open Questions

1. Should file-mode observability cadence be globally configurable or fixed for first release?
2. Should sentinel-miss warnings include structured counters/telemetry fields for alerting in addition to logs?
3. Should future offline batched inference reuse the same queue contract or use a separate execution mode?
