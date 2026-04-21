## Why

Active Listener needs live spectrum data for overlay visualization, but today no part of system exposes audio frames to anything outside websocket transport. Existing D-Bus contract only carries recording state and transcription text, so UI cannot render bars without inventing its own audio path. We now have concrete visualization requirements: 50 equal log-spaced bars from 60 Hz to 8 kHz, measured every 16 ms from a 512-sample window, using dB compression. This change is needed now because design work has already narrowed projection and transport choices, and implementation needs one canonical contract instead of ad hoc side channels.

## What Changes

- Add optional `on_capture` callback to transcriber-mode `EavesdropClient` construction so callers can observe canonical audio payload bytes from live capture in send order.
- Add Active Listener spectrum analysis pipeline that ingests captured audio bytes, maintains a rolling buffer, computes 50 log-spaced spectrum bars every 16 ms from a 512-sample Hann-windowed FFT, interpolates at log-spaced band centers, compresses to dB, and quantizes normalized bar heights to `0..255` bytes for D-Bus/UI transport.
- Extend Active Listener D-Bus interface to publish live spectrum frames alongside existing recording/transcription state.
- Update GNOME overlay consumer to render bar frames received over D-Bus as real-time visualization.
- Keep existing transcription behavior unchanged; spectrum data is visualization-only and does not affect audio transport, transcription, or rewrite logic.

## Scope

### New Capabilities
- `spectrum-over-dbus`: Produce live visualization-ready spectrum frames from Active Listener capture audio and publish them over D-Bus for UI rendering.

### Modified Capabilities
- `eavesdrop-client-transcriber`: Allow live transcriber clients to expose canonical captured audio chunks through an optional constructor callback without changing websocket payload format.
- `active-listener-dbus-state`: Expand Active Listener session D-Bus surface beyond state/transcription updates to include live spectrum frame publication for recording UI using quantized integer bar payloads.
- `active-listener-ui-overlay`: Update overlay rendering requirements to consume externally produced spectrum bars instead of relying on placeholder overlay content.

## Impact

Affected systems: `packages/client` audio capture and streaming loop, `packages/active-listener` runtime orchestration and D-Bus service, and `packages/active-listener-ui-gnome` overlay rendering. Public API impact is additive through optional capture callback on the transcriber client factory/constructor and a new D-Bus signal carrying 50 quantized bar bytes per frame. No brand-new analysis library is required beyond NumPy already used elsewhere in the repo, but `packages/active-listener` must declare `numpy<2` directly because the workspace currently resolves NumPy 1.26.4 and the client/common packages already target the NumPy 1.x line.
