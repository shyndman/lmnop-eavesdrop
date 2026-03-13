## Context

The current live transcriber protocol uses a single WebSocket connection with a setup message followed by raw float32 audio frames. On the server side, `TranscriptionServer.get_audio_from_websocket(...)` reads each post-setup frame and treats it as audio bytes unless it matches the `END_OF_AUDIO` sentinel. On the processing side, `StreamingTranscriptionProcessor` snapshots all unprocessed audio from `AudioStreamBuffer`, runs transcription, commits completed segments, and then sleeps to satisfy `min_chunk_duration` and `transcription_interval` pacing.

That shape is fine for always-on microphone streaming, but it is a poor fit for push-to-talk. A user who releases the key is communicating an end-of-utterance boundary, yet the current system may still wait for more audio, wait for the next interval, or leave the current segment tentative until later speech arrives.

This change spans `packages/wire`, `packages/server`, and `packages/client`. It must preserve the existing live session model: one WebSocket, warm session state, binary audio streaming, and the invariant that transcription responses end with an incomplete tail segment.

Third-party dependency baseline, verified against current manifests and official docs:
1. No new external package is required for the flush wire codec; the repo already uses `pydantic>=2`, and the existing dataclass + discriminated-union codec pattern supports adding another literal-tagged message variant.
2. No new external package is required for wakeable waits or cross-thread interrupt signaling; Python 3.12 stdlib `asyncio` and `threading` primitives are sufficient.
3. The only dependency requirement that should be made explicit is `websockets>=13.0` on the server package, because both packages rely on the documented `websockets.asyncio.*` API and the client already declares that floor while the server currently does not.

## Goals / Non-Goals

**Goals:**
1. Add an explicit flush session command for live transcriber sessions without requiring disconnect/reconnect.
2. Guarantee that every accepted flush produces exactly one `TranscriptionMessage` while keeping the session connected.
3. Let callers request `force_complete` so the server can treat flush as an explicit end-of-utterance boundary.
4. Short-circuit avoidable latency by waking the processor out of minimum-chunk and interval waits.
5. Allow low-touch cooperative abandonment of stale pre-commit work when a pending flush supersedes it.
6. Expose a client-library `flush(...)` API that nudges callers into the legal request/response pattern.

**Non-Goals:**
1. Introducing a general transport-agnostic command bus.
2. Adding session IDs or flush IDs to the wire protocol.
3. Interrupting library calls inside Silero VAD, feature extraction, or Whisper generation.
4. Changing file-mode or RTSP control semantics in this change.
5. Removing the invariant incomplete tail segment from transcription responses.

## File Map and Ownership

This section exists to remove guesswork for implementation. Each file below has one job in this change.

| File | Job in this change |
| --- | --- |
| `packages/wire/src/eavesdrop/wire/messages.py` | Define the new flush control message dataclass. |
| `packages/wire/src/eavesdrop/wire/codec.py` | Add the new message to the discriminated union so JSON text frames can round-trip. |
| `packages/wire/src/eavesdrop/wire/__init__.py` | Export the new message type for callers/tests. |
| `packages/server/pyproject.toml` | Declare `websockets>=13.0` explicitly. |
| `packages/server/src/eavesdrop/server/streaming/client.py` | Own live-session flush state, accept/reject flush commands, capture boundary samples, and wake the processor. This is the transport edge for live flush commands. |
| `packages/server/src/eavesdrop/server/server.py` | Stop treating live text frames as audio. After this change it should only decode binary audio frames, or delegate live frame reading entirely to `WebSocketStreamingClient`. |
| `packages/server/src/eavesdrop/server/streaming/buffer.py` | Provide a safe helper for reading the current buffer-end sample/time when a flush is accepted. |
| `packages/server/src/eavesdrop/server/streaming/processor.py` | React to flush state: wakeable waits, interrupted outcomes, boundary satisfaction checks, and `force_complete` commit behavior. |
| `packages/client/src/eavesdrop/client/connection.py` | Send the flush control message as a text frame and continue routing `TranscriptionMessage` / `ErrorMessage` callbacks. |
| `packages/client/src/eavesdrop/client/core.py` | Public `flush(...)` API, local legality guard, and waiting for the corresponding response. |
| `packages/wire/tests/test_wire_codec_contracts.py` | Round-trip tests for the new wire message. |
| `packages/server/tests/test_streaming_lifecycle_contracts.py` | Live flush lifecycle tests: rejection, wake-from-wait, and keep-connection-open behavior. |
| `packages/server/tests/test_transcription_output_contracts.py` | Segment completion / tail invariant tests under `force_complete`. |
| `packages/client/tests/test_client_mode_contracts.py` | Local second-flush rejection and server-rejection propagation tests. |

## Concrete Protocol Shape

### New wire message

Use the same pattern as the existing messages in `packages/wire/src/eavesdrop/wire/messages.py`: Pydantic dataclass, `kw_only=True`, inherited `timestamp`, and a literal `type` discriminator.

Recommended message class:

```python
@dataclass(kw_only=True)
class FlushControlMessage(BaseMessage):
  type: Literal["control_flush"] = "control_flush"
  stream: str
  force_complete: bool = True
```

Expected JSON text frame on the wire:

```json
{
  "type": "control_flush",
  "stream": "mic-123",
  "force_complete": true,
  "timestamp": 1710000000.0
}
```

Notes for the implementer:
1. `timestamp` already comes from `BaseMessage`; do not invent a second clock field.
2. `stream` must remain present because existing message shapes use it and client/server routing already expects it.
3. This command must travel in a **text** WebSocket frame, not a binary frame.
4. Audio remains binary float32 frames exactly as it works today.

### Rejection message

Reuse the existing `ErrorMessage` shape. Do not invent a new rejection envelope.

```json
{
  "type": "error",
  "stream": "mic-123",
  "message": "Flush rejected: another flush is already pending",
  "timestamp": 1710000001.0
}
```

Connection behavior on rejection:
1. keep the WebSocket open,
2. keep the original flush pending,
3. do not clear any existing flush state.

## Runtime Terms

These terms are used precisely throughout the design.

1. **Tentative tail segment**
   - The last segment in an emitted `TranscriptionMessage`.
   - It has `completed=False`.
   - Current clients, including active-listener, treat `message.segments[-1]` as this in-progress tail.

2. **Flush boundary**
   - The buffer end, expressed as a sample index, at the instant the server accepts a flush.
   - This is the audio coverage target for the flush response.

3. **Flush-satisfying response**
   - The one `TranscriptionMessage` emitted after a processing pass covers the accepted flush boundary.
   - Because only one flush may be in flight at a time, this response is correlated by sequence, not by an explicit flush ID.

## Decision 1: Use existing WebSocket frame multiplexing

Live transcriber sessions continue to use one WebSocket. Binary frames remain audio payloads. Text frames become protocol/control frames, including `CONTROL_FLUSH`.

This avoids inventing a custom binary header and keeps the audio hot path raw. It also matches the existing protocol shape, where setup is already JSON/text and live audio is already binary.

Dependency note: this design relies on the documented `websockets.asyncio` connection API. The client already requires `websockets>=13.0`; the server package should declare the same floor explicitly rather than relying on an unbounded install. Current PyPI latest is 16.0, so a `>=13.0` floor leaves room for minor and patch updates.

**Alternatives considered:**
- **Custom frame header:** rejected as unnecessary protocol surface for a single control command.
- **Second control connection:** rejected as extra lifecycle complexity and worse ergonomics for callers.

## Decision 2: Put live command receipt at the `WebSocketStreamingClient` edge

The live session already has one server-side object that owns the session (`WebSocketStreamingClient`). That object should also own acceptance/rejection of flush commands.

Concretely:
1. `WebSocketStreamingClient` remains the live session owner.
2. Exactly one code path may call `websocket.recv()` in live mode.
3. That code path must branch on frame type:
   1. binary frame -> decode to float32 audio and append to the buffer,
   2. `b"END_OF_AUDIO"` -> end the live audio source,
   3. text frame -> deserialize and handle session commands.

This matters because the `websockets` library forbids concurrent `recv()` calls on the same connection. A junior engineer should not introduce a second background task that also reads the socket.

**Implementation note:**
The current code spreads this responsibility across `TranscriptionServer.get_audio_from_websocket(...)`, `WebSocketAudioSource.read_audio()`, and `WebSocketStreamingClient._audio_ingestion_loop()`. During implementation, prefer the smallest refactor that leaves only one `recv()` owner in live mode. The easiest shape is:
1. move live frame reading into `WebSocketStreamingClient`,
2. keep `server.py` as a binary-audio decoder helper only,
3. leave file-mode reading untouched.

## Decision 3: Keep one canonical pending-flush record on the server

The server stores at most one pending flush per live session:

```text
pending_flush = None
            or {
                 boundary_sample: int,
                 force_complete: bool
               }
```

`boundary_sample` is the canonical stream fence: the end of buffered audio at the instant the server accepts the flush. Sample indices are preferred over floating-point seconds to avoid fencepost ambiguity.

This is the **single source of truth**. Wakeup primitives are allowed, but they are only doorbells. They must never be used to answer “is a flush pending?” or “what boundary must be covered?” Those answers come only from `pending_flush`.

### Recommended server-side state placement

Put these fields on `WebSocketStreamingClient` (or on one tiny helper object owned by it):

```text
_pending_flush: PendingFlush | None
_flush_wakeup: asyncio.Event
_flush_interrupt: threading.Event
```

Rules:
1. `_pending_flush` is authoritative.
2. `_flush_wakeup` only wakes async waits on the event-loop side.
3. `_flush_interrupt` only signals worker-thread checkpoints running inside `asyncio.to_thread(...)`.
4. Clearing `_flush_wakeup` or `_flush_interrupt` does **not** mean the flush is satisfied.
5. A flush is only satisfied when the flush-satisfying response is emitted and `_pending_flush` is cleared.

### Recommended helper in `AudioStreamBuffer`

Do not make callers manually inspect `frames_np`, `buffer_start_time`, and `lock` in several places. Add one helper to `AudioStreamBuffer` so boundary capture stays correct and centralized.

Recommended helper:

```python
def get_buffer_end_sample(self) -> int:
    ...
```

Its value should be computed from the current locked buffer state:

```text
buffer_end_sample = int(buffer_start_time * sample_rate) + buffered_sample_count
```

Where:
1. `buffer_start_time` already tracks how much audio has been clipped from the front,
2. `sample_rate` comes from `buffer.config.sample_rate`,
3. `buffered_sample_count` is `0` if `frames_np is None`, else `frames_np.shape[0]`.

This helper is important because it gives the junior implementer one safe way to capture the flush boundary.

## Decision 4: Reject concurrent flushes instead of coalescing

The server accepts at most one in-flight flush per session. If another flush arrives before the first one has been satisfied, the server rejects it with the existing `ErrorMessage` shape.

A flush is complete only after the flush-satisfying `TranscriptionMessage` has been emitted.

### Acceptance algorithm

When a live text frame deserializes to `FlushControlMessage`:

1. If `_pending_flush is not None`:
   1. send `ErrorMessage(stream=self.stream_name, message="Flush rejected: another flush is already pending")`,
   2. leave `_pending_flush` unchanged,
   3. do not wake the processor because the original flush is already the active one,
   4. continue reading frames.
2. Otherwise:
   1. compute `boundary_sample = buffer.get_buffer_end_sample()`,
   2. store `_pending_flush = PendingFlush(boundary_sample=..., force_complete=...)`,
   3. set `_flush_wakeup`,
   4. set `_flush_interrupt`,
   5. continue reading frames.

This keeps the state machine simple and matches the intended client usage: `await flush(...)` before sending another flush.

## Decision 5: Wake sleeps directly instead of polling

Two existing waits matter to flush latency:
1. waiting for `min_chunk_duration` before processing undersized audio,
2. waiting out the remainder of `transcription_interval` after a normal pass.

Both waits should become wakeable waits that return early when a flush is pending. This avoids turning latency into a polling-interval tuning problem.

### Concrete waits to replace

The junior engineer should touch exactly these wait sites in `packages/server/src/eavesdrop/server/streaming/processor.py`:
1. `_get_next_audio_chunk()` -> the `await asyncio.sleep(remaining_wait)` branch,
2. `_wait_for_next_interval()` -> the `await asyncio.sleep(remaining_wait)` branch.

### Recommended behavior for wakeable waits

For both sites, follow this rule:
1. If no flush is pending, wait up to the normal timeout.
2. If a flush is already pending, skip the wait immediately.
3. If a flush arrives during the wait, wake immediately.
4. After waking, re-check `_pending_flush` before deciding what to do next.

Pseudo-shape for the event-loop side:

```text
if pending_flush exists:
    return immediately
clear wake_event
wait for either:
    - wake_event.set()
    - timeout expires
re-check pending_flush
```

The important point is not the exact helper signature. The important point is: no polling loop, no arbitrary 50ms sleeps, and always re-check the authoritative state after waking.

## Decision 6: Support cooperative pre-commit interruption only at safe checkpoints

This change will not attempt to kill VAD or Whisper mid-call. Instead, the processor checks flush state only at safe pre-commit checkpoints:
1. before entering minimum-chunk sleep,
2. before entering interval sleep,
3. before VAD preprocessing,
4. before Whisper inference.

If a checkpoint decides the current pass is stale relative to a pending flush boundary, it returns an explicit interrupted outcome rather than raising an exception.

This is intentionally low touch and measurable. It abandons stale work only before authoritative session/buffer/client state has been mutated.

### Explicit interrupted outcome

Do **not** encode interruption as either:
1. an exception,
2. `segments=[]`,
3. `segments=None` on a normal result object.

Those paths already mean other things in the current code.

Recommended approach:
1. add a dedicated result type such as `InterruptedChunkResult`, or
2. add a dedicated enum/state discriminator on the existing result type.

The important rule is:
- `_transcription_loop()` must be able to tell “this pass was intentionally abandoned before commit” apart from “the pass completed and found no segments.”

### Why this matters

Current code advances the processed boundary when `result.segments` is empty. That behavior is correct for silence, but wrong for interruption. If a junior engineer reuses the empty-segments path, the server can silently discard audio that was never committed.

### Where to check the interrupt flag

The worker-thread side lives inside `_transcribe_chunk()` -> `asyncio.to_thread(self._transcribe_audio, chunk)`.

Recommended checks:
1. at the top of `_transcribe_audio(...)`, before any expensive preprocessing,
2. inside the transcription pipeline immediately before VAD begins,
3. immediately before the Whisper inference call.

Use `_flush_interrupt` on the worker-thread side because `asyncio.Event` is not thread-safe.

### Clearing the interrupt doorbell

When a pre-commit interrupt is consumed:
1. clear `_flush_interrupt`,
2. leave `_pending_flush` untouched,
3. do **not** clear `_flush_wakeup` just because one pass was interrupted,
4. continue the outer loop immediately.

This avoids losing the active flush request.

## Decision 7: Define flush satisfaction by boundary coverage, not by loop count

A flush is satisfied when a processing pass covers the recorded boundary sample and emits its response.

Conceptually:

```text
pass_end_sample >= pending_flush.boundary_sample
```

This matters because one outer processor iteration grabs a chunk snapshot of all currently unprocessed audio, but more audio may arrive before the flush frame is received. Whisper’s internal ~30s windowing does not require multiple outer loop iterations by itself; it already handles long chunks inside one transcription call. A second outer iteration is only required when the currently running snapshot does not yet extend to the flush boundary.

### Exact way to compute pass coverage

Use the chunk snapshot that the processor already has.

For one `AudioChunk`:
1. `pass_start_sample = int(chunk.start_time * sample_rate)`
2. `pass_end_sample = pass_start_sample + chunk.data.shape[0]`
3. `sample_rate = buffer.config.sample_rate`

Then compare:

```text
covers_flush = pass_end_sample >= pending_flush.boundary_sample
```

### Outer loop behavior

After a pass finishes its pre-commit work:
1. If there is no `_pending_flush`, behave exactly as the processor does today.
2. If `_pending_flush` exists and `covers_flush` is false:
   1. do not clear `_pending_flush`,
   2. do not emit the flush response yet,
   3. skip interval sleep,
   4. immediately start the next outer loop iteration so it snapshots the larger buffer.
3. If `_pending_flush` exists and `covers_flush` is true:
   1. this pass is now responsible for producing the flush response,
   2. apply `force_complete` rules if requested,
   3. emit exactly one `TranscriptionMessage`,
   4. clear `_pending_flush` only **after** `send_result(...)` returns successfully,
   5. clear `_flush_wakeup` and `_flush_interrupt` after `_pending_flush` is cleared.

This “clear after send” rule is important. It prevents a second flush from sneaking in before the first one has actually produced its promised response.

## Decision 8: Keep the invariant incomplete tail in every flush response

Accepted flushes still return ordinary `TranscriptionMessage` payloads. The response SHALL still end with an incomplete tail segment.

### Current invariant to preserve

The processor currently does this:
1. mark all non-tail completed segments as `completed=True`,
2. optionally mark the last segment complete via silence,
3. append either the current incomplete segment or a synthetic empty incomplete segment,
4. send the result.

The flush change must preserve step 3 for **every** emitted response.

### Force-complete behavior

If `force_complete=true` and the pass that covers the boundary still has a tentative tail segment:
1. mark that tentative tail complete even if silence threshold has not been met,
2. add it to the completed chain,
3. append a fresh incomplete tail segment after it,
4. send the response.

### Non-force-complete behavior

If `force_complete=false`:
1. do not invent extra completion,
2. preserve the ordinary incomplete-tail logic,
3. still ensure the emitted response ends with one incomplete tail segment.

## Decision 9: Expose flush as an awaitable client-library API

The client library exposes a high-level method shaped like:

```python
await client.flush(force_complete=True) -> TranscriptionMessage
```

This is intentionally stronger than “send a command.” It makes flush a request/response operation in the public API and discourages callers from issuing a second flush before the first one finishes.

### Recommended public behavior

1. `flush()` is only valid for transcriber clients.
2. `flush()` requires an already connected live session.
3. `flush()` must fail fast locally if another local flush call is already waiting.
4. `flush()` sends exactly one `FlushControlMessage` text frame.
5. `flush()` waits for one of two outcomes:
   1. the flush-satisfying `TranscriptionMessage` -> return it,
   2. an `ErrorMessage` rejecting the flush -> raise `RuntimeError(error.message)`.

### Recommended client-side implementation strategy

Build on patterns that already exist in `EavesdropClient`:
1. reuse `_operation_lock` style local legality checks for non-reentrant behavior,
2. reuse `_message_queue` and `_disconnect_event` style waiting patterns,
3. do not introduce a second socket reader.

A junior engineer should **not** invent a separate message queue for flush unless it is necessary. The existing message callback path already pushes `TranscriptionMessage` instances into `_message_queue`.

### Recommended local state

Keep the client-side state minimal:

```text
_flush_waiting: bool
_flush_error: str | None
```

Rules:
1. `_flush_waiting` exists only to reject a second local `flush()` call early.
2. The server remains authoritative; even if local state gets it wrong, server rejection must still be surfaced.
3. Clear local flush-waiting state in a `finally` block inside `flush()`.

### Response correlation rule

Because only one flush may be in flight at a time, the client does **not** need a flush ID. The returned response is simply:
- the first `TranscriptionMessage` received after the server accepted the flush and satisfied its boundary.

This rule must be written into the implementation comments so future maintainers do not “helpfully” add ad-hoc correlation logic that conflicts with the one-flush-at-a-time contract.

## End-to-End Sequence

This section is intentionally concrete.

### Happy path: `force_complete=true`

```text
client.flush(force_complete=True)
    |
    | send FlushControlMessage text frame
    v
server live receive loop
    |
    | if no pending flush:
    |   boundary_sample = buffer.get_buffer_end_sample()
    |   pending_flush = {boundary_sample, force_complete=True}
    |   set wake + interrupt doorbells
    v
processor
    |
    | wake immediately if sleeping
    | interrupt stale pre-commit work if safe
    | keep looping until pass_end_sample >= boundary_sample
    | mark tail complete
    | append fresh incomplete tail
    | send one TranscriptionMessage
    | clear pending_flush after send_result
    v
client
    |
    | receive TranscriptionMessage
    | resolve flush() awaiter
    v
connection stays open
```

### Illegal second flush

```text
flush #1 accepted -> pending_flush exists
flush #2 arrives -> send ErrorMessage("another flush is already pending")
                  -> keep flush #1 active
                  -> keep connection open
```

## Risks / Trade-offs

- **[Server and client install different `websockets` majors]** → The repo already imports `websockets.asyncio.client` / `websockets.asyncio.server`, which assumes the new asyncio API. Mitigation: declare `websockets>=13.0` explicitly in the server package and keep the floor aligned with the client package.
- **[Flush arrives while a stale pass is already running]** → The current pass may not cover the recorded boundary. Mitigation: keep `pending_flush` until a pass proves boundary coverage and skip interval waiting until satisfied.
- **[Library calls remain non-interruptible mid-execution]** → Worst-case latency is still bounded by the remainder of the current VAD/feature/Whisper call. Mitigation: interrupt only at coarse checkpoints, then measure the end-to-end win before considering deeper changes.
- **[Live-path receive logic becomes more complex]** → Post-setup WebSocket reads must demux text control from binary audio without adding a second socket reader. Mitigation: keep one read owner in the live transcriber path and forward decoded commands into session state.
- **[Client-local guard and server guard disagree]** → The client may think a flush is legal while the server rejects it. Mitigation: treat server rejection as authoritative and document that a flush is complete only after the corresponding transcription response arrives.
- **[Force-complete may finalize a shorter-than-silence-threshold utterance]** → This is intentional product behavior, but it can change transcript segmentation compared with passive silence detection. Mitigation: make `force_complete` explicit on the wire and in the client API.

## Safe Implementation Order

A junior engineer should follow this order exactly. It keeps the change testable in small pieces.

1. Add the new wire message and exports first.
2. Add or adjust wire tests so the new message round-trips before touching server/client behavior.
3. Add the explicit `websockets>=13.0` server dependency floor.
4. Add server-side flush state and the buffer-end helper.
5. Refactor the live receive path so there is exactly one post-setup `recv()` owner and it can demux text vs binary.
6. Add server-side concurrent-flush rejection.
7. Replace the two processor sleeps with wakeable waits.
8. Add the explicit interrupted outcome and safe checkpoints.
9. Add boundary coverage checks and “loop again immediately” behavior.
10. Add `force_complete` handling while preserving the incomplete-tail invariant.
11. Add client `send_flush(...)` and public `flush(...)` API.
12. Add client-side local legality guard and rejection propagation.
13. Add focused contract tests for wire, server, and client.
14. Run the focused verification commands for the touched packages.

## Open Questions

- None at artifact time. The remaining work is implementation and measurement rather than unresolved design direction.
