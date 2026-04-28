## Why

Active-listener currently treats foreground recording time as if every participant already agrees on the same zero point, but the protocol does not explicitly represent that zero point. That allows server-side live-stream history, buffered audio, in-flight inference, completed segment history, and client-side command spans to drift into different timelines even though the foreground recording itself has one simple truth: sample 0 is the start of that recording. This spec decides the treatment of time once and for all so command classification, live overlays, cancellation, flush, finalization, and recorded audio all use one recording-relative timebase. The change is needed now because stale transcription from canceled recordings, command text reclassification, unexpected segment history, and disappearing prefix text all point to the same missing recording boundary.

## What Changes

- Establish recording sample time as the canonical foreground-recording timebase: `time_s = recording_sample_index / sample_rate_hz`, with sample index 0 at recording start.
- Add an explicit live recording boundary to the transcriber protocol before any audio bytes for a new active-listener recording.
- **BREAKING**: A live WebSocket transcriber connection is no longer itself a continuous transcription timeline for active-listener recordings; it is a transport that can contain multiple isolated recording epochs.
- Reset server live recording state completely when a new recording boundary is accepted, including the audio buffer, processed pointer, completed segment chain, incomplete tail, pending flush state, and stale in-flight generation ownership.
- Scope live transcription messages, flush responses, and cancellation effects to the current recording epoch so prior recordings cannot appear in later recordings.
- Require all segment and word timestamps emitted for active-listener live recordings to be recording-relative before they leave the server.
- Preserve internal windowing as an implementation detail only; window offsets must not leak into protocol timestamps or app policy.
- Update reducer and app behavior so command spans, transcript words, captured audio duration, live overlay state, and finalization all share the same recording-relative timeline.

## Scope

### New Capabilities
- `live-recording-epoch`: Introduces an explicit protocol-level recording epoch for long-lived live transcriber connections.
- `canonical-recording-timebase`: Defines recording-relative sample time as the authoritative clock for active-listener foreground recordings.
- `epoch-scoped-live-results`: Ensures live transcription, flush, and cancel messages belong to one recording epoch and stale epoch results are dropped.

### Modified Capabilities
- `live-transcription-streaming`: Changes live transcriber semantics from one connection-level audio timeline to isolated recording epochs inside a long-lived transport.
- `active-listener-recording`: Requires foreground recording state, command spans, live overlay updates, captured audio, and final transcript reduction to use the same recording-relative sample time.
- `transcription-window-reduction`: Requires reducer windows to contain only current-epoch segments and prevents previous recording sentinels or completed history from affecting new recordings.
- `server-buffer-management`: Requires a new active-listener recording to reset the server audio buffer entirely rather than preserving or offsetting old buffered samples.

## Impact

- Wire protocol: add a new live recording control message and route it through serialization/deserialization.
- Client package: send the new recording boundary from `EavesdropClient.start_streaming()` before audio bytes, keep flush/cancel scoped to the active recording, and handle epoch-tagged results.
- Server package: handle the new recording boundary in live transcriber sessions, reset buffer/session/flush/generation state, and emit recording-relative segment and word timestamps.
- Active-listener package: keep recording lifecycle, command spans, reducer state, overlay updates, finalization, and captured audio aligned to the canonical recording timebase.
- Tests: add protocol, server lifecycle, client ordering, active-listener reducer, command classification, cancel/restart, flush, and long-recording regression coverage.
- Non-goals: changing finite file transcription semantics, changing RTSP subscriber timeline semantics, or applying late timestamp translation in active-listener as the primary fix.
