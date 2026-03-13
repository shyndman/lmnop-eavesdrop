## 1. Wire message and dependency groundwork

- [ ] 1.1 Add a new `FlushControlMessage` dataclass to `packages/wire/src/eavesdrop/wire/messages.py` using the same pattern as the other wire messages:
  - `@dataclass(kw_only=True)`
  - `type: Literal["control_flush"] = "control_flush"`
  - `stream: str`
  - `force_complete: bool = True`
- [ ] 1.2 Add `FlushControlMessage` to the `Message` union in `packages/wire/src/eavesdrop/wire/codec.py` so `serialize_message(...)` and `deserialize_message(...)` handle it.
- [ ] 1.3 Export `FlushControlMessage` from `packages/wire/src/eavesdrop/wire/__init__.py` so server code, client code, and tests can import it from `eavesdrop.wire`.
- [ ] 1.4 Align the server WebSocket dependency floor with the client by declaring `websockets>=13.0` in `packages/server/pyproject.toml`.
- [ ] 1.5 Extend `packages/wire/tests/test_wire_codec_contracts.py` with one round-trip test for `FlushControlMessage` and one negative test that proves unknown discriminators still fail validation.

## 2. Server live-session flush state and receive path

- [ ] 2.1 Add one canonical server-side pending-flush record to the live session owner in `packages/server/src/eavesdrop/server/streaming/client.py`:
  - no more than one pending flush at a time,
  - fields for `boundary_sample` and `force_complete`,
  - wakeup primitives for async waits and worker-thread checkpoints.
- [ ] 2.2 Add one helper to `packages/server/src/eavesdrop/server/streaming/buffer.py` that returns the current buffer-end sample index under the buffer lock. This helper must be the only place that computes the flush boundary from buffer state.
- [ ] 2.3 Refactor the live receive path so there is exactly one post-setup `websocket.recv()` owner in live mode. The implementation must not introduce a second socket reader because `websockets` rejects concurrent `recv()` calls.
- [ ] 2.4 In that single live receive path, branch by frame type:
  - binary audio frame -> continue existing audio decoding path,
  - `b"END_OF_AUDIO"` -> continue existing end-of-stream behavior,
  - text frame -> deserialize with `deserialize_message(...)` and handle `FlushControlMessage`.
- [ ] 2.5 On the first accepted flush:
  - capture `boundary_sample` using the new buffer helper,
  - store the pending flush record,
  - set the wakeup/interrupt doorbells,
  - keep the connection open.
- [ ] 2.6 On an illegal second flush while one is already pending:
  - send the agreed `ErrorMessage`,
  - keep the connection open,
  - leave the original pending flush unchanged.
- [ ] 2.7 Add or extend live-session tests in `packages/server/tests/test_streaming_lifecycle_contracts.py` to prove:
  - a second flush is rejected,
  - the connection remains open after rejection,
  - the first flush remains active.

## 3. Server processor wakeup behavior

- [ ] 3.1 Update `packages/server/src/eavesdrop/server/streaming/processor.py::_get_next_audio_chunk()` so the current minimum-chunk wait becomes a wakeable wait:
  - if no flush is pending, wait up to the normal timeout,
  - if a flush is already pending, skip the wait,
  - if a flush arrives during the wait, wake immediately.
- [ ] 3.2 Update `packages/server/src/eavesdrop/server/streaming/processor.py::_wait_for_next_interval()` with the same wakeable-wait rules for the normal interval sleep.
- [ ] 3.3 Add focused tests showing both waits are interrupted by a pending flush. Put these tests in `packages/server/tests/test_streaming_lifecycle_contracts.py` unless another existing live-lifecycle contract file is a better fit.

## 4. Server processor interruption and boundary satisfaction

- [ ] 4.1 Introduce an explicit “interrupted before commit” outcome in `packages/server/src/eavesdrop/server/streaming/processor.py`. Do not reuse exceptions and do not reuse the empty-segments path.
- [ ] 4.2 Add safe pre-commit checkpoints that can return that interrupted outcome:
  - before VAD preprocessing,
  - before Whisper inference,
  - and anywhere else needed before authoritative state mutation begins.
- [ ] 4.3 Ensure the worker-thread checkpoint path uses a thread-safe stdlib primitive rather than `asyncio.Event`, because `_transcribe_chunk()` crosses the `asyncio.to_thread(...)` boundary.
- [ ] 4.4 In the outer transcription loop, if a pass returns the interrupted outcome:
  - do not advance the processed boundary,
  - do not emit a client response,
  - do not clear the pending flush,
  - immediately continue the loop.
- [ ] 4.5 For each real `AudioChunk`, compute pass coverage explicitly:
  - `pass_start_sample = int(chunk.start_time * sample_rate)`
  - `pass_end_sample = pass_start_sample + chunk.data.shape[0]`
  - compare `pass_end_sample` to `pending_flush.boundary_sample`.
- [ ] 4.6 If the pass does not yet cover the flush boundary:
  - keep the pending flush,
  - skip the normal interval sleep,
  - immediately start another outer loop iteration so the next chunk snapshot includes more audio.
- [ ] 4.7 If the pass does cover the flush boundary:
  - build the one flush-satisfying response,
  - apply `force_complete` if requested,
  - emit exactly one `TranscriptionMessage`,
  - clear the pending flush only after `send_result(...)` returns.

## 5. Server response shaping and segment invariants

- [ ] 5.1 Update the transcription-output path in `packages/server/src/eavesdrop/server/streaming/processor.py` so `force_complete=true` completes the current tentative tail segment even if silence would not have completed it.
- [ ] 5.2 After force-completing that segment, append a fresh incomplete tail segment so the emitted response still ends with exactly one incomplete tail.
- [ ] 5.3 Preserve ordinary behavior when `force_complete=false`; the change must not force completion in that case.
- [ ] 5.4 Extend `packages/server/tests/test_transcription_output_contracts.py` with flush-specific assertions covering:
  - completed prior tail when `force_complete=true`,
  - preserved incomplete tail when `force_complete=false`,
  - always exactly one incomplete segment at the end of the response.

## 6. Client connection and awaitable flush API

- [ ] 6.1 Add a small helper such as `send_flush(...)` to `packages/client/src/eavesdrop/client/connection.py` that serializes and sends `FlushControlMessage` as a text frame.
- [ ] 6.2 Keep the existing socket-reader model in the client: `handle_messages()` remains the only incoming-message reader.
- [ ] 6.3 Add minimal client-side flush state in `packages/client/src/eavesdrop/client/core.py` so a second local `flush()` call can fail fast before another command is sent.
- [ ] 6.4 Add public `async def flush(self, *, force_complete: bool = True) -> TranscriptionMessage` to `packages/client/src/eavesdrop/client/core.py`.
- [ ] 6.5 In `flush(...)`, enforce these preconditions explicitly:
  - client type must be transcriber,
  - client must already be connected in live mode,
  - no other local flush call may already be waiting.
- [ ] 6.6 Build `flush(...)` on existing client patterns rather than inventing a second socket reader:
  - reuse `_message_queue` for transcription messages,
  - reuse `_disconnect_event` patterns for connection loss,
  - clear local flush state in a `finally` block.
- [ ] 6.7 Make `flush(...)` resolve when the flush-satisfying `TranscriptionMessage` is received.
- [ ] 6.8 Make `flush(...)` fail if the server sends the concurrent-flush `ErrorMessage`, and surface that failure as a `RuntimeError` carrying the server’s message text.
- [ ] 6.9 Add a strong docstring to `flush(...)` explaining:
  - it is a request/response operation,
  - only one flush may be in flight,
  - calling it again before the first flush completes is illegal.

## 7. Client contract tests

- [ ] 7.1 Extend `packages/client/tests/test_client_mode_contracts.py` with a success-path test for `await client.flush(force_complete=True)`.
- [ ] 7.2 Add a local fast-fail test showing a second `flush()` call raises before sending another command.
- [ ] 7.3 Add a server-rejection test showing the concurrent-flush `ErrorMessage` becomes a failed `flush()` call.
- [ ] 7.4 Reuse the existing fake/recording connection test style already present in the client contract suites rather than building a new test harness.

## 8. Focused verification

- [ ] 8.1 Run the affected wire contract tests.
- [ ] 8.2 Run the affected server lifecycle/output contract tests.
- [ ] 8.3 Run the affected client contract tests.
- [ ] 8.4 Run package-local type checking / linting commands only for the touched packages if those commands are part of the normal package workflow.
- [ ] 8.5 Record the exact commands and observed results in implementation notes so the next engineer can reproduce verification easily.
