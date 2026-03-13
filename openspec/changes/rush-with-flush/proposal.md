## Why

Push-to-talk style transcription is a planned primary user flow. Verified current behavior is close, but not explicit enough: `EavesdropClient.stop_streaming()` stops local microphone capture while keeping the WebSocket session open, and the live server path keeps running on its normal processing cadence until more audio arrives, `END_OF_AUDIO` is sent, or the connection closes. That makes end-of-utterance responsiveness depend on existing pacing rather than an explicit user boundary.

## Verified Baseline (Current Code)

- **Live client streaming already has separate stop vs disconnect semantics**: `EavesdropClient` exposes `connect()`, `disconnect()`, `start_streaming()`, and `stop_streaming()`. `stop_streaming()` only clears `_streaming` and stops the recorder; it does not close the socket (`packages/client/src/eavesdrop/client/core.py`).
- **The client already has an awaitable request/response pattern we can borrow**: `transcribe_file(...)` is non-reentrant, guarded by `_operation_lock`, and uses `_message_queue` plus `_disconnect_event` to wait for a terminal result (`packages/client/src/eavesdrop/client/core.py`).
- **The wire protocol already has a text setup phase and explicit source-mode routing**: `UserTranscriptionOptions` already carries `source_mode` (`LIVE` / `FILE`), and transcriber setup is sent as `TranscriptionSetupMessage` JSON before live streaming begins (`packages/wire/src/eavesdrop/wire/transcription.py`, `packages/client/src/eavesdrop/client/connection.py`, `packages/server/src/eavesdrop/server/connection_handler.py`).
- **The wire codec already has the necessary third-party support**: wire messages are Pydantic dataclasses collected into a discriminated union codec, so adding a new message variant does not require a new serialization package (`packages/wire/src/eavesdrop/wire/messages.py`, `packages/wire/src/eavesdrop/wire/codec.py`, `packages/wire/pyproject.toml`).
- **There is no existing flush primitive**: no `CONTROL_FLUSH` message or `flush(...)` client API exists in `packages/` today.
- **Post-setup live ingest is currently audio-only at the server read boundary**: the live transcriber path has a single `websocket.recv()` reader that treats `b"END_OF_AUDIO"` as EOF and otherwise coerces text frames to bytes before `np.frombuffer(..., dtype=np.float32)`. A text control frame sent during live ingest would not be interpreted as a protocol message today (`packages/server/src/eavesdrop/server/server.py`).
- **Dependency drift already exists around WebSocket transport**: the client declares `websockets>=13.0`, while the server leaves `websockets` unbounded despite importing `websockets.asyncio.server.ServerConnection` and using `recv(decode=True)` from that API (`packages/client/pyproject.toml`, `packages/server/pyproject.toml`, `packages/server/src/eavesdrop/server/connection_handler.py`).
- **Current latency comes from two explicit waits**: the processor waits for `min_chunk_duration` (default `1.0s`) before processing short live chunks and then waits out `transcription_interval` (default `2.0s`) after a normal pass (`packages/server/src/eavesdrop/server/config.py`, `packages/server/src/eavesdrop/server/streaming/processor.py`).
- **The current output contract assumes an in-progress tail**: the processor marks all but the last segment complete, may complete the last segment by silence analysis, and always sends an incomplete tail or synthetic placeholder. The active-listener workspace currently reads `message.segments[-1]` as the in-progress segment (`packages/server/src/eavesdrop/server/streaming/processor.py`, `packages/active-listener/src/eavesdrop/active_listener/workspace.py`).
- **Relevant contract tests already exist for adjacent behavior**: the repo already has wire codec contract tests, live streaming lifecycle/output contract tests, and client file-transcription contract tests (`packages/wire/tests/test_wire_codec_contracts.py`, `packages/server/tests/test_streaming_lifecycle_contracts.py`, `packages/server/tests/test_transcription_output_contracts.py`, `packages/client/tests/test_file_transcription_contracts.py`).

## What Changes

- Add a session-level `CONTROL_FLUSH` command on the existing transcriber WebSocket protocol.
- Define flush as a request/response boundary: every accepted flush produces exactly one `TranscriptionMessage` while keeping the session connected.
- Allow flush callers to request `force_complete`, which treats the flush as an explicit end-of-utterance boundary and completes the current tentative segment before returning the response.
- Reject a second flush while one is already pending and return an explicit error message instead of coalescing requests.
- Add a single-source-of-truth pending-flush state on the server, captured as a stream boundary sample index.
- Wake the processor early from minimum-chunk and interval waits when a flush is pending.
- Add cooperative pre-commit interrupt checkpoints so stale in-flight work can be abandoned before VAD or Whisper inference when a flush supersedes it.
- Expose a high-level client-library `flush(...)` API that awaits and returns the flush-correlated transcription result, following the same request/response style the client already uses for `transcribe_file(...)`.

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
