## Why

Push-to-talk transcription needs an explicit end-of-utterance boundary. `EavesdropClient.stop_streaming()` stops local microphone capture while keeping the WebSocket session open; flush must bypass avoidable processing waits and finalize the captured boundary without closing the session.

## Verified Baseline (Current Code)

- **Live client streaming already has separate stop vs disconnect semantics**: `EavesdropClient` exposes `connect()`, `disconnect()`, `start_streaming()`, and `stop_streaming()`. `stop_streaming()` only clears `_streaming` and stops the recorder; it does not close the socket (`packages/client/src/eavesdrop/client/core.py`).
- **The client uses an awaitable request/response pattern**: use `_message_queue` plus `_disconnect_event` to wait for a terminal flush result.
- **The wire protocol has a text setup phase**: transcriber setup is sent as `TranscriptionSetupMessage` JSON before live streaming begins.
- **The wire codec already has the necessary third-party support**: wire messages are Pydantic dataclasses collected into a discriminated union codec, so adding a new message variant does not require a new serialization package (`packages/wire/src/eavesdrop/wire/messages.py`, `packages/wire/src/eavesdrop/wire/codec.py`, `packages/wire/pyproject.toml`).
- **There is no existing flush primitive**: no `CONTROL_FLUSH` message or `flush(...)` client API exists in `packages/` today.
- **Post-setup live ingest needs explicit control routing**: keep one WebSocket reader and decode text control frames separately from binary float32 audio.
- **Dependency drift already exists around WebSocket transport**: the client declares `websockets>=13.0`, while the server leaves `websockets` unbounded despite importing `websockets.asyncio.server.ServerConnection` and using `recv(decode=True)` from that API (`packages/client/pyproject.toml`, `packages/server/pyproject.toml`, `packages/server/src/eavesdrop/server/connection_handler.py`).
- **Current latency comes from two explicit waits**: the processor waits for `min_chunk_duration` (default `1.0s`) before processing short live chunks and then waits out `transcription_interval` (default `2.0s`) after a normal pass (`packages/server/src/eavesdrop/server/config.py`, `packages/server/src/eavesdrop/server/streaming/processor.py`).
- **The output contract preserves an in-progress tail**: with VAD disabled every segment remains provisional until force-complete flush; with VAD enabled silence may complete segments. Preserve the incomplete tail or synthetic placeholder expected by consumers.
- **Relevant contract tests cover adjacent behavior**: wire codec contracts and live streaming lifecycle/output contracts.

## What Changes

- Add a session-level `CONTROL_FLUSH` command on the existing transcriber WebSocket protocol.
- Define flush as a request/response boundary: every accepted flush produces exactly one `TranscriptionMessage` while keeping the session connected.
- Allow flush callers to request `force_complete`, which treats the flush as an explicit end-of-utterance boundary and completes all remaining provisional segments before returning the response.
- Reject a second flush while one is already pending and return an explicit error message instead of coalescing requests.
- Add a single-source-of-truth pending-flush state on the server, captured as a stream boundary sample index.
- Wake the processor early from minimum-chunk and interval waits when a flush is pending.
- Add cooperative pre-commit interrupt checkpoints so stale in-flight work can be abandoned before VAD or Whisper inference when a flush supersedes it.
- Expose a high-level client-library `flush(...)` request/response API that awaits and returns the flush-correlated transcription result.

## Capabilities

### New Capabilities
- `session-flush-command`: Real-time session flush semantics for live transcription, including wire command handling, processor wakeup/interrupt behavior, flush completion guarantees, illegal concurrent flush rejection, and the client-library awaitable flush API.

### Modified Capabilities
- None.

## Impact

- **Wire (`packages/wire`)**: add the flush control message shape and reuse the existing Pydantic dataclass/discriminated-union codec plus existing error messaging for illegal concurrent flush rejection.
- **Server (`packages/server`)**: extend the live transcriber path to demultiplex post-setup control frames from binary audio, track pending flush boundaries, wake early from existing waits, and force completion when a flush-satisfying response requires it.
- **Client (`packages/client`)**: add an awaitable `flush(...)` API with local misuse guards and response correlation behavior, building on the client’s existing queued-message and disconnect-event patterns.
- **Dependencies**: no new third-party package is expected for this change; the only dependency change likely needed is making the server’s `websockets` version floor explicit and aligning it with the client’s existing `>=13.0` floor.
- **Live transcriber callers**: can finish an utterance and receive a transcription response promptly without disconnecting or cold-starting the session.
- **Tests**: extend the repo’s existing wire codec, live lifecycle/output, and client contract suites with flush-specific coverage.
