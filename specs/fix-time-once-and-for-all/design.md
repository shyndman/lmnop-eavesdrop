## Context

Active-listener has one foreground recording at a time. For that recording, the user intent is explicit: the first sample captured for the recording is sample 0, and every meaningful time inside the recording is derived from the number of samples elapsed since that point.

Current code does not represent that boundary in the live transcriber protocol:

- `ActiveListenerService.handle_action()` calls `client.start_streaming()` when recording starts.
- `EavesdropClient.start_streaming()` starts microphone capture and sends raw audio bytes over an already-open WebSocket.
- The server owns one long-lived `AudioStreamBuffer`, `TranscriptionSession`, completed segment chain, flush state, and transcription loop for that WebSocket.
- `AudioStreamBuffer.get_chunk_for_processing()` returns `start_time` from `processed_up_to_time`; the processor passes that into transcription as `absolute_stream_start`.
- Active-listener command spans are recorded as foreground-recording-relative values, but incoming segment/word timestamps may be server-stream-relative.

That mismatch explains the observed bug class: stale completed segments, stale incomplete tails, and stale buffer timelines can survive across foreground recordings because the server sees the WebSocket as the timeline owner. The design fixes that by making the foreground recording the timeline owner and demoting the WebSocket to transport.

Locked decisions from prior discussion:

- Recording sample time is the canonical timebase.
- A new recording must be announced by a control message immediately before audio bytes for that recording.
- The server must reset the live recording buffer entirely at the new recording boundary. Keeping old audio and translating offsets is out of scope.
- Windowing remains internal; it must not leak as caller-visible time.

## Goals / Non-Goals

**Goals:**

- Define one authoritative live-recording timebase:

  ```text
  recording_sample_index = number of samples elapsed since this recording began
  time_s = recording_sample_index / sample_rate_hz
  recording sample 0 == recording start == t=0
  ```

- Represent foreground recording start in the wire protocol.
- Reset all server state that can carry old recording data before accepting audio for a new recording.
- Scope live transcription results, flush responses, cancellations, and stale in-flight inference to a recording epoch.
- Ensure all live active-listener `Segment.start`, `Segment.end`, `Word.start`, and `Word.end` values are recording-relative before leaving the server.
- Keep active-listener reducer state, command spans, live overlay updates, finalization, and captured audio on the same recording-relative sample timeline.
- Make stale data from canceled or finished recordings structurally impossible to accept as current recording data.
- Add regression coverage for canceled recording leakage, command text flips after caps release, unexpected initial segment history, and long-recording prefix retention.

**Non-Goals:**

- No client-side late timestamp translation as the primary fix.
- No preserving pre-boundary server audio across active-listener recordings.
- No change to finite file transcription semantics.
- No change to RTSP subscriber timeline semantics unless shared types require optional fields.
- No attempt to make wall-clock/log timestamps part of app semantics.
- No multi-recording concurrency inside one live transcriber connection. One live connection has at most one active recording epoch.

## Source-Verified API Surface

These are the existing APIs and files the implementation is expected to use. They are part of the design, not background context the implementor should rediscover.

### Wire package

- `packages/wire/src/eavesdrop/wire/messages.py`
  - Messages are Pydantic dataclasses using `@dataclass(kw_only=True)`.
  - Existing live control messages are named with the `control_*` prefix: `FlushControlMessage.type == "control_flush"` and `UtteranceCancelledMessage.type == "control_utterance_cancelled"`.
  - Add `RecordingStartedMessage` here:

    ```python
    @dataclass(kw_only=True)
    class RecordingStartedMessage(BaseMessage):
      type: Literal["control_recording_started"] = "control_recording_started"
      stream: str
      recording_id: str
      sample_rate_hz: int
    ```

  - Add `recording_id: str | None = None` to `TranscriptionMessage`, `FlushControlMessage`, and `UtteranceCancelledMessage`. Live active-listener paths must set it; file and RTSP compatibility may leave it `None`.
- `packages/wire/src/eavesdrop/wire/codec.py`
  - `Message` is a type union used by `_MessageCodec` with `Field(discriminator="type")`.
  - Add `RecordingStartedMessage` to the `Message` union.
  - `serialize_message()` currently omits `TranscriptionMessage.flush_complete` when it is `None`; optional `recording_id=None` is not automatically omitted unless serialization is updated explicitly.
- `packages/wire/src/eavesdrop/wire/__init__.py`
  - Exports are managed with `__all__`; export `RecordingStartedMessage` there.

### Client package

- `packages/client/src/eavesdrop/client/connection.py`
  - `WebSocketConnection` owns low-level WebSocket serialization and send helpers.
  - Add `send_recording_started(recording_id: str, sample_rate_hz: int) -> None` that serializes `RecordingStartedMessage`.
  - Update `send_flush_control(..., recording_id: str)` and `send_utterance_cancelled(recording_id: str)` to include the active recording id.
  - Existing `send_audio_data(audio_data: bytes)` sends binary audio frames; no audio frame may be sent before `send_recording_started()` has completed.
- `packages/client/src/eavesdrop/client/core.py`
  - `EavesdropClient.start_streaming()` is the high-level API that transitions into microphone capture and starts `_audio_streaming_loop()`.
  - Change it to `start_streaming(recording_id: str) -> None` for this clean cutover. The caller owns the foreground recording id so active-listener can start local recording state before capture begins.
  - `start_streaming(recording_id)` must drain any stale `AudioCapture.audio_queue` bytes before sending the boundary.
  - It must call `WebSocketConnection.send_recording_started(recording_id, SAMPLE_RATE)` and await it before `AudioCapture.start_recording()` and before creating `_audio_streaming_loop()`.
  - `_audio_streaming_loop()` sends bytes through `WebSocketConnection.send_audio_data()` and then calls `_on_capture(audio_data)` with the same bytes. That callback is the active-listener local capture mirror.
  - `flush(recording_id: str, *, force_complete: bool = True)` must wait only for `TranscriptionMessage(recording_id=recording_id, flush_complete=True)`.
  - `cancel_utterance(recording_id: str)` must stop streaming and send `UtteranceCancelledMessage(recording_id=recording_id)`.
  - `_on_transcription_message()` is the shared callback feeding the internal flush queue and live event queue. It must drop stale live messages before they reach either queue. File-mode collection is exempt because file transcription has no active-listener recording epoch.
  - `_clear_message_queue()` only drains the internal flush/file queue today; stale live events can also sit in `_event_queue`, so a new recording boundary must drain or filter both paths.
- `packages/client/src/eavesdrop/client/audio.py`
  - `AudioCapture.audio_callback()` queues `float32` microphone bytes with `audio_data.tobytes()`.
  - `AudioCapture.audio_queue` is not currently cleared between recordings; the new start path must clear it before capture begins.

### Server package

- `packages/server/src/eavesdrop/server/connection_handler.py`
  - `WebSocketConnectionHandler.handle_connection()` receives `TranscriptionSetupMessage` before creating `WebSocketStreamingClient`.
  - Pre-setup flush/cancel messages are rejected there; pre-setup `RecordingStartedMessage` should be rejected the same way.
- `packages/server/src/eavesdrop/server/streaming/client.py`
  - `_audio_ingestion_loop()` is the single post-setup WebSocket receive owner in live mode.
  - Text frames are routed to `_handle_live_text_frame()`; binary frames go to `add_frames()`.
  - `_handle_file_text_frame()` rejects live controls in file mode; it must reject `RecordingStartedMessage` in file mode too.
  - `_handle_live_text_frame()` must handle `RecordingStartedMessage` by coordinating a full recording reset before any following binary frame is accepted into the buffer.
- `packages/server/src/eavesdrop/server/streaming/buffer.py`
  - `AudioStreamBuffer.reset()` sets `frames_np = None`, `buffer_start_time = 0.0`, and `processed_up_to_time = 0.0`; this is the required new-recording buffer operation.
  - `discard_unprocessed_audio()` clears buffered audio but preserves the current timeline cursor by leaving `processed_up_to_time` unchanged; it must not be used for a new recording boundary.
  - `get_chunk_for_processing()` returns `(audio_data, duration, start_time)` where `start_time == processed_up_to_time`.
- `packages/server/src/eavesdrop/server/streaming/flush_state.py`
  - `LiveSessionFlushState` owns pending flush state, interrupt/wakeup events, and the existing active utterance generation.
  - Add a recording reset operation that clears pending flush, advances the generation/epoch used for stale pass rejection, and wakes interruptible waits.
- `packages/server/src/eavesdrop/server/transcription/session.py`
  - `TranscriptionSession.completed_segments` is the emitted completed history and segment ID chain source.
  - `reset_utterance()` clears `completed_segments` only. It does not reset buffer time, language, or processor text accumulation.
  - New recording reset must call `reset_utterance()` or recreate the session, and must also reset every other recording-local owner separately.
- `packages/server/src/eavesdrop/server/streaming/processor.py`
  - `StreamingTranscriptionProcessor._get_next_audio_chunk()` receives `start_time` from the buffer and constructs `AudioChunk` with that value.
  - `_run_transcription_pass()` already records `chunk_start_sample`, `chunk_sample_count`, and an utterance generation in `ChunkTranscriptionResult`; that generation should become the recording epoch guard.
  - `_process_transcription_result()` is the first async-side point after inference where stale results can be dropped before language/session/buffer/sink mutation.
  - `_set_language()`, `_update_segments()`, `session.add_completed_segment()`, `_advance_buffer_by_completed_segments()`, and `sink.send_result()` are recording-local mutation/emission points and must be epoch-gated.
  - `_create_synthetic_incomplete_segment()` creates chunk-local `start`/`end` with `time_offset=self.buffer.processed_up_to_time`; the resulting `Segment.absolute_*` properties depend on `time_offset`.
- `packages/server/src/eavesdrop/server/transcription/request_runner.py` and `packages/server/src/eavesdrop/server/transcription/utils.py`
  - `RequestRunner.run()` calls `finalize_recording_timestamps(...)`.
  - `finalize_recording_timestamps()` offsets segment and word timestamps and sets `segment.time_offset`. It must receive the current recording epoch offset only, not a WebSocket-lifetime offset.

### Active-listener package

- `packages/active-listener/src/active_listener/app/ports.py`
  - `ActiveListenerClient` currently exposes `start_streaming()`, `stop_streaming()`, `cancel_utterance()`, and `flush()` without recording ids.
  - Change it to `start_streaming(recording_id: str)`, `cancel_utterance(recording_id: str)`, and `flush(recording_id: str, *, force_complete: bool = True)`. `stop_streaming()` does not need a recording id because it sends no lifecycle message; flush/cancel close the epoch semantics.
  - Add `recording_id: str` to `FinishedRecording`.
- `packages/active-listener/src/active_listener/app/service.py`
  - `ActiveListenerService.handle_action()` currently calls `client.start_streaming()` before `RecordingSession.start_recording()`. That order is wrong for sample-zero alignment.
  - New start order: generate `recording_id`, call `RecordingSession.start_recording(recording_id)`, then call `client.start_streaming(recording_id)` so the local capture buffer is active before `_on_capture` can deliver the first upstream bytes.
  - Finish currently transitions to `ForegroundPhase.IDLE` and starts finalization in a background task; new starts must be serialized with that background flush or must explicitly abandon it before sending a new recording boundary.
- `packages/active-listener/src/active_listener/recording/session.py`
  - `RecordingSession.start_recording()` must accept/store `recording_id` and start `RecordingAudioBuffer` before client capture can deliver bytes.
  - `finish_recording()` must return `FinishedRecording(recording_id=..., reducer_state=..., captured_audio=...)`.
  - `RecordingAudioBuffer.append()` ignores chunks while inactive; that is why local session start must happen before client capture starts.
- `packages/active-listener/src/active_listener/recording/finalizer.py`
  - `RecordingFinalizer.finalize_recording()` currently calls `client.flush(force_complete=True)` using only `FinishedRecording.reducer_state`; it must call `client.flush(finished_recording.recording_id, force_complete=True)`.
- `packages/active-listener/src/active_listener/recording/reducer.py`
  - `classify_word()` uses midpoint classification: `(word.start_s + word.end_s) / 2`.
  - `RecordingReducerState` stores `completed_words`, `incomplete_words`, `closed_command_spans`, and `open_command_start_s` in seconds today.
  - `apply_segment_reduction()` appends committed words and replaces incomplete words.
  - `reduce_new_segments()` returns `missing_last_id=True` and the whole committed prefix when the previous sentinel is absent. That must not erase the local prefix; after epoch scoping, missing sentinels inside one recording should log a data-integrity warning and avoid duplicate committed words by segment id.
  - If command text is active and any segment lacks `words`, `CommandTextWordTimestampError` is raised; word timestamps remain mandatory for command classification.

### Validation commands

- Root type check: `uv run basedpyright`.
- Server type check with opt-in transcription dependencies: `task typecheck-server` from the repository root. This runs the server package with the `type_checkable` dependency group.
- Package tests: run `uv run pytest` from each touched package directory.
- Package lint/format checks: run `uv run ruff check` and `uv run ruff format --check` from each touched package directory.

## Third-Party Dependency API Contract

No new third-party dependency is required for this spec. Implementation uses dependencies already declared by the workspace and locked in `uv.lock`.

### Pydantic

- Version in use: `pydantic==2.12.5` from `uv.lock`; package constraints are `pydantic>=2` in `packages/wire`, `packages/client`, `packages/server`, and `packages/active-listener`.
- Documentation used: Pydantic 2.12 union and serialization docs.
  - Discriminated unions: https://docs.pydantic.dev/2.12/concepts/unions/#discriminated-unions
  - Serialization and `exclude_none`: https://docs.pydantic.dev/2.12/concepts/serialization/#excluding-and-including-fields-based-on-their-value
- Required APIs:
  - Keep using `pydantic.dataclasses.dataclass(kw_only=True)` for wire messages.
  - Keep using `Field(discriminator="type")` on the private codec wrapper. Pydantic 2.12 validates only the union member selected by the discriminator field, so every message class in the `Message` union must have a unique `type: Literal[...]` value.
  - Keep using `TypeAdapter(type(message)).dump_json(message)` for message serialization.
  - If the wire format should omit `recording_id: None`, call `dump_json(..., exclude_none=True)` or add explicit exclusion logic. Pydantic 2.12 documents `exclude_none` as the serialization parameter that excludes fields whose value is `None`.
  - Do not use Pydantic 2.13-only APIs such as `polymorphic_serialization`; the lock is 2.12.5.

### websockets

- Version in use: `websockets==16.0` from `uv.lock`; client and server constraints are `websockets>=13.0`.
- Documentation used: websockets 16.0 asyncio common API.
  - https://websockets.readthedocs.io/en/stable/reference/asyncio/common.html
- Required APIs and constraints:
  - `await websocket.send(str_message)` sends a text frame.
  - `await websocket.send(bytes_message)` sends a binary frame.
  - `await websocket.recv(decode=True)` returns the next received message as text when used by the existing server setup path.
  - The docs state that two coroutines may not call `recv()`/`recv_streaming()` concurrently; the implementation must preserve the existing single receive owner after setup (`_audio_ingestion_loop()`).
  - The recording-start boundary depends on explicit application sequencing: await the text-frame send of `RecordingStartedMessage` before starting audio capture or creating any task that can send audio bytes.
  - Do not cancel an in-progress `send()` to enforce lifecycle transitions; websockets 16.0 discourages canceling `send()`. Close the connection on send failure and let reconnect/lifecycle code handle the interruption.
  - If implementation introduces multiple producer coroutines that can send controls/audio on one connection, serialize them with one writer path or an `asyncio.Lock`; do not rely on incidental scheduling for `control_recording_started` before binary audio frames.

### python-sounddevice

- Version in use: `sounddevice==0.5.5` from `uv.lock`; client constraint is `sounddevice>=0.5.2`.
- Documentation/source used:
  - sounddevice 0.5.5 stream callback API: https://python-sounddevice.readthedocs.io/en/0.5.5/api/streams.html
  - upstream asyncio generator example: https://github.com/spatialaudio/python-sounddevice/blob/master/examples/asyncio_generators.py
- Required APIs and constraints:
  - `sounddevice.InputStream(..., callback=callback)` invokes `callback(indata, frames, time, status)` for input-only streams.
  - `indata` is a two-dimensional `numpy.ndarray` with shape `(frames, channels)` and dtype configured by the stream.
  - `blocksize=0` requests host-selected block sizes; this means audio callback frame counts may vary. Sample time must be counted from `frames`/byte length, not from callback count.
  - sounddevice documents that the PortAudio callback runs at high/real-time priority and must not block or call unpredictable code.
  - The implementation must change the callback-to-asyncio handoff to the documented pattern: capture the event loop when starting the stream and use `loop.call_soon_threadsafe(queue.put_nowait, audio_data)` from the callback, rather than directly mutating an `asyncio.Queue` from the audio callback context.
  - The callback must copy input data before handing it to the asyncio loop because the callback buffer is reused after return. The current `indata.copy().astype(DTYPE)` satisfies the copy requirement; preserve that or use an equivalent copy/cast before queuing.

### NumPy

- Version in use: `numpy==1.26.4` from `uv.lock`; package constraints are `numpy<2`.
- Documentation used: NumPy 1.26 `ndarray.tobytes` docs.
  - https://numpy.org/doc/1.26/reference/generated/numpy.ndarray.tobytes.html
- Required APIs and constraints:
  - `ndarray.tobytes(order="C")` constructs Python `bytes` containing a copy of the raw array data in C order by default.
  - `tobytes()` does not change dtype. Audio bytes must be produced after converting to `np.float32`.
  - Preserve the existing audio wire format as mono `float32` PCM bytes at `SAMPLE_RATE == 16000`; sample count is `len(audio_bytes) // np.dtype(np.float32).itemsize` for one channel.
  - If channel count changes in the future, sample-frame count must divide by `channels * np.dtype(np.float32).itemsize`; this spec keeps `CHANNELS == 1`.

## Decisions

### 1. Canonical time is recording sample time

All foreground-recording domain logic uses recording sample time.

Use this model everywhere in the live active-listener path:

```text
RecordingTime:
  sample_index: int
  sample_rate_hz: int

seconds is a serialization/display projection:
  seconds = sample_index / sample_rate_hz
```

`float` seconds may still appear in existing public wire models, logs, DBus messages, and UI DTOs, but those floats must be projections from sample indices on the recording timeline. They must not be derived from WebSocket age, process monotonic age, buffer lifetime, or transcription window age.

Rationale:

- Sample count is the only clock both client and server can share exactly for audio-derived events.
- It matches the user's mental model and the actual captured audio artifact.
- It avoids trying to reconcile wall-clock scheduling, WebSocket lifetime, and inference timing.

Alternatives considered:

- **Use monotonic time from active-listener start.** Rejected because it does not bind server timestamps to the same origin.
- **Use server buffer time and translate on the client.** Rejected because it preserves the wrong abstraction and leaves stale state alive.
- **Use wall/log time.** Rejected because logs are observability only.

### 2. A live WebSocket is transport, not a recording timeline

A live transcriber WebSocket can stay connected across foreground recordings, but it must not imply one continuous recording timeline.

Add a protocol message:

```python
@dataclass(kw_only=True)
class RecordingStartedMessage(BaseMessage):
  type: Literal["control_recording_started"] = "control_recording_started"
  stream: str
  recording_id: str
  sample_rate_hz: int
```

`recording_id` is an opaque client-generated epoch identifier. It is not a timestamp. It exists to scope messages and stale work.

`EavesdropClient.start_streaming(recording_id)` sends `RecordingStartedMessage` before starting microphone capture or sending any audio bytes for that recording.

Required ordering:

```text
connect/setup once per WebSocket

for each foreground recording:
  active-listener starts local RecordingSession(recording_id)
  client sends control_recording_started(recording_id, sample_rate_hz)
  start audio capture
  send audio bytes for that recording
  stop audio capture
  flush(recording_id) OR cancel(recording_id)
```

Because WebSocket message ordering is preserved, the server can treat the first binary frame after `control_recording_started` as containing recording sample 0. No acknowledgement message is required for this ordering; the client must await the send before starting capture.

Rationale:

- This cleanly models the domain boundary the system currently lacks.
- It keeps the existing long-lived connection/reconnect model without inheriting a long-lived timeline.
- It lets the server reset state before any new audio can enter the buffer.

Alternatives considered:

- **Reconnect the WebSocket for every recording.** Simpler boundary, but slower and unnecessary; the user specifically raised a preceding control message and the transport can remain long-lived.
- **Send only cancel/flush controls and infer new recordings from audio gaps.** Rejected because silence/gaps are audio content, not lifecycle.

### 3. New recording boundary fully resets server recording state

When the server accepts `control_recording_started`, it starts a new recording epoch for that WebSocket.

The reset must include:

- `AudioStreamBuffer.reset()` so `frames_np is None`, `buffer_start_time == 0`, and `processed_up_to_time == 0`.
- A fresh `TranscriptionSession` or equivalent cleared completed segment chain.
- No current incomplete tail.
- Cleared pending flush state.
- Advanced recording generation so in-flight transcription from the old epoch cannot commit.
- Cleared language/session state if it is recording-local.
- Cleared processor text accumulation used for older transcript helpers.

The reset must not use `discard_unprocessed_audio()` because that preserves `processed_up_to_time` and therefore preserves the old timeline origin. The reset must be total.

Postcondition:

```text
server current recording_id == message.recording_id
buffer_start_sample == 0
processed_sample == 0
completed_segments == []
pending_flush == None
first following audio byte belongs to sample index 0
```

Rationale:

- Any retained audio or processed pointer can reintroduce old recording samples into current windows.
- Any retained completed segment chain can reintroduce old sentinels and explain segment history appearing immediately after start.
- Any retained generation can allow stale in-flight results to emit into the new recording.

Alternatives considered:

- **Keep processed audio and subtract an offset.** Rejected by locked decision: reset buffer entirely.
- **Discard only unprocessed tail on cancel.** Insufficient for new recording because completed history and processed time still survive.

### 4. Results and controls are scoped by recording id

Extend live control/result messages to carry `recording_id` for active live transcriber recordings:

```python
@dataclass(kw_only=True)
class TranscriptionMessage(BaseMessage):
  type: Literal["transcription"] = "transcription"
  stream: str
  recording_id: str | None = None
  segments: list[Segment]
  language: str | None = None
  flush_complete: bool | None = None

@dataclass(kw_only=True)
class FlushControlMessage(BaseMessage):
  type: Literal["control_flush"] = "control_flush"
  stream: str
  recording_id: str | None = None
  force_complete: bool = True

@dataclass(kw_only=True)
class UtteranceCancelledMessage(BaseMessage):
  type: Literal["control_utterance_cancelled"] = "control_utterance_cancelled"
  stream: str
  recording_id: str | None = None
```

Rules:

- For live active-listener recordings, `recording_id` is required after the new protocol is in use.
- The server rejects or ignores controls whose `recording_id` does not match the active recording epoch.
- The server includes `recording_id` on every transcription result produced for that epoch.
- The client drops transcription messages whose `recording_id` does not match the recording it is currently streaming/finalizing.
- Active-listener ignores live transcription events whose `recording_id` does not match the active `RecordingSession`.
- Flush waits only for `flush_complete=True` on the requested `recording_id`.
- Cancel applies only to the requested `recording_id`.

Rationale:

- Resetting state handles future work; epoch scoping handles stale work that was already in flight.
- This makes leakage from canceled recordings testable and structurally visible.

Alternatives considered:

- **Use only stream name.** Rejected because the stream name is intentionally stable across recordings.
- **Use a numeric generation only on the server.** Rejected because the client also needs to reject stale queued messages.

### 5. In-flight transcription is generation-checked before emit

The processor must capture the active recording generation/id when it starts a transcription pass. Before updating session state or sending results, it must compare the captured epoch against the current epoch.

If they differ:

```text
old transcription result -> drop without mutating completed_segments, buffer, language, or sink
```

The recording boundary should also wake/interrupt worker waits just like cancel/flush does today, so the loop can observe the new epoch quickly.

Rationale:

- `asyncio.to_thread()` inference may finish after a new recording starts.
- Buffer reset alone cannot stop a result object already produced by the old pass.

Alternatives considered:

- **Cancel the worker thread.** Not reliable for blocking model inference.
- **Let stale result commit then clear on next loop.** Rejected because it can emit externally before being cleared.

### 6. Windowing is internal; emitted timestamps are recording-relative

The server may still process overlapping or partial windows internally. That does not change the public timebase.

Rules:

- `AudioChunk.start_time` is recording-relative because the buffer resets to zero at recording start.
- `absolute_stream_start` should become recording-relative start, not WebSocket-stream-relative start.
- `finalize_recording_timestamps()` may remain as the point where chunk/window-local timestamps become recording-relative timestamps, but its offset must be from the current recording epoch only.
- Synthetic incomplete segments must also be placed on the recording timeline.
- Segment completion and buffer advancement must compare values on the same timeline.

Rationale:

- Whisper and VAD can work in chunk/window-local coordinates internally.
- Callers must never need to know which chunk/window produced a word.

Alternatives considered:

- **Expose chunk-relative times and let active-listener combine them.** Rejected because it spreads windowing details into app policy.

### 7. Active-listener stores command spans on the same sample timeline

Active-listener command spans should be stored as recording-relative sample spans internally, with seconds only projected for existing display/API boundaries.

Target model:

```python
@dataclass(frozen=True)
class RecordingSampleSpan:
  start_sample: int
  end_sample: int
  sample_rate_hz: int
```

Keyboard events are external events, so they need conversion onto the recording sample timeline at the active-listener boundary. The conversion must use the same recording start boundary as audio capture. After conversion, reducer classification compares word sample midpoints to command sample spans; it does not compare wall times or connection times.

For an incremental implementation, existing `TimeSpan(start_s, end_s)` and `TimedWord(start_s, end_s)` may remain if every value is guaranteed to be a projection from recording sample time. The design preference is to move internal reducer comparisons to integer samples to remove ambiguity.

Rationale:

- Command classification is exactly where timeline mismatch showed up.
- Integer sample indices make the invariant visible to readers and tests.

Alternatives considered:

- **Keep command spans as monotonic elapsed seconds forever.** Rejected because it hides that reducer comparisons are supposed to be sample-timeline comparisons.

### 8. Active-listener start order must align local and server sample zero

Current active-listener start order starts upstream streaming before `RecordingSession.start_recording()` initializes local reducer/audio state. That order must change so local recording state and server epoch state agree about sample zero.

Required intent:

```text
generate recording_id
start local RecordingSession(recording_id)
send server recording epoch boundary from client.start_streaming(recording_id)
drain stale AudioCapture queue bytes
start one audio capture stream whose bytes feed both server and local captured-audio buffer
```

The method split is intentional: active-listener owns the foreground recording id and starts its local recording state before the client can start microphone capture. The invariant is mandatory:

```text
The bytes stored in FinishedRecording.captured_audio are the same bytes sent to the server for that recording, in the same order, starting at recording sample 0.
```

No audio bytes may be sent to the server before the recording boundary send has completed. No captured bytes may be omitted from the local recording because local state started after capture.

Rationale:

- The user's guarantee depends on both sides seeing the same sample sequence.
- Local finalization and server transcription must describe the same audio artifact.

Alternatives considered:

- **Keep the current start order and rely on fast scheduling.** Rejected because a race at sample zero breaks the whole model.

### 9. Starting a new recording is serialized with finalization flush

A new recording boundary resets the server buffer entirely. Therefore active-listener must not send a new recording boundary while the previous recording still needs a flush result for finalization.

Rules:

- Cancel may reset immediately because the recording is intentionally discarded.
- Finish must preserve the just-recorded audio until its flush either succeeds or fails.
- If finalization flush is still in progress, a new foreground recording start must wait for that flush path to release the live transcriber epoch, or the service must explicitly abandon the prior recording before starting the next epoch.
- The implementation must make this policy visible in state, not rely on accidental timing.

Rationale:

- Full reset is destructive by design.
- Without serialization, a fast next recording could erase the previous finished recording before finalization consumes it.

Alternatives considered:

- **Allow multiple server epochs concurrently on one WebSocket.** Rejected as unnecessary complexity.
- **Let new start cancel prior finalization implicitly.** Rejected because finish and cancel are distinct user actions.

### 10. Reducer windows are current-epoch windows only

The active-listener reducer continues to accumulate the recording prefix locally. Server emissions may still include a limited recent completed window plus one incomplete tail, but every segment in that message must belong to the same `recording_id`.

Rules:

- The first transcription event after a new recording must not include completed segments from a prior recording.
- Missing sentinel inside the same recording should be treated as a data-integrity warning. It must not erase already accumulated local prefix.
- Live overlay state is built from accumulated current-recording reducer state, not from the server window alone.
- The incomplete tail may be replaced on every update; completed prefix must be append-only inside a recording except when the entire recording is canceled/discarded.

Rationale:

- This protects long recordings where the server sends only `send_last_n_segments` history.
- It separates transport windows from the user's transcript state.

Alternatives considered:

- **Make the server send the entire recording on every update.** Simpler reducer, but unnecessary and expensive for long recordings.

## Risks / Trade-offs

- [Risk] New protocol fields/messages break older clients or servers. → Mitigation: make the implementation a clean cutover inside this monorepo, update all call sites/tests together, and reject unsupported live control ordering clearly.
- [Risk] Recording start ordering can still race if capture begins before the boundary send completes. → Mitigation: send and await `control_recording_started` before `AudioCapture.start_recording()` and before creating the audio streaming loop.
- [Risk] In-flight inference from an old epoch finishes after reset. → Mitigation: epoch-check before mutating session/buffer/sink, and wake interruptible waits on boundary acceptance.
- [Risk] Finish followed immediately by new start can destroy the previous epoch before flush. → Mitigation: serialize new starts behind finalization flush or introduce an explicit visible abandoning path.
- [Risk] Keyboard events are not audio samples. → Mitigation: convert external event times to recording sample indices once at the active-listener boundary and store the resulting sample spans for reducer comparisons.
- [Risk] Float seconds can still hide timeline bugs. → Mitigation: prefer integer sample indices internally; use seconds only at wire/UI/log boundaries and test exact sample-derived values.
- [Risk] Existing helper names like `absolute_stream_start` encode the old model. → Mitigation: rename or constrain them during implementation so names tell the truth about recording-relative time.

## Migration Plan

1. Add wire protocol support for `RecordingStartedMessage` and optional `recording_id` on live result/control messages.
2. Update client connection APIs to send `control_recording_started` and carry active/last recording id through start, flush, cancel, reconnect, and message filtering. `stop_streaming()` remains a local capture stop and sends no recording lifecycle message.
3. Update server live client handling to accept the new boundary, reject illegal ordering, reset recording state fully, and epoch-tag results.
4. Update streaming processor state so recording epoch changes interrupt waits and stale transcription passes cannot commit.
5. Update timestamp handling so all live active-listener segments/words are recording-relative before emission.
6. Update active-listener recording lifecycle so local state starts before client capture, server boundary, audio capture, command spans, captured audio, and finalization flush share one recording epoch.
7. Update reducer tests and service tests for the observed bugs and new invariants.
8. Run targeted tests for `wire`, `client`, `server`, and `active-listener` packages touched by the change: `uv run pytest` from each touched package directory.
9. Run type checks: `uv run basedpyright` from the repository root, plus `task typecheck-server` from the repository root for the server opt-in transcription dependency group when server code is touched.
10. Run lint/format checks from touched package directories: `uv run ruff check` and `uv run ruff format --check`.

Rollback strategy: revert the protocol and lifecycle change as one unit. Partial rollback is unsafe because old clients and new servers would disagree about whether WebSocket lifetime or recording epoch owns time.

## Open Questions

The key behavioral decisions are locked: recording sample time is canonical, new recordings are explicit protocol epochs, and server buffer reset at a new recording boundary is total.

- Reconnect during an active recording: should the active recording be failed/abandoned on disconnect, or should reconnect attempt to reuse the same `recording_id` and continue the epoch? Reuse is only correct if the implementation can prove no captured/sent audio gap exists.
