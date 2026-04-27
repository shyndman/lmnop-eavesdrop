## Why

`active-listener` already persists finalized transcript metadata into a local SQLite database, but it throws away the microphone audio that produced each recording. That makes the history useful for counting and inspection, but not for replaying what was actually said, auditing rewrite behavior, or reusing past captures for debugging and model evaluation. The user wants each finished recording to carry its full captured audio alongside the transcript history, using AAC in an `m4a` container, without adding latency to the text-emission path.

This change is needed now because the capture seam already exists locally: the client forwards each raw float32 PCM chunk to the server and also exposes those same bytes to `active-listener` through the `on_capture` callback. The remaining gap is to preserve those chunks per recording, archive them after text emission, and make that archival truthful about failures without blocking dictation.

## What Changes

- Add full-recording audio capture retention inside `active-listener` for each active recording, preserving the existing float32 PCM bytes received through the local capture callback.
- Extend transcript history persistence so one finalized recording can archive both transcript metadata and an optional 1:1 audio artifact stored in SQLite.
- Introduce AAC-in-`m4a` encoding via an external `ffmpeg` binary, with startup validation of a configured `ffmpeg_path` and fallback lookup from `PATH`.
- Preserve the current fast text-emission behavior: transcript emission remains primary, while audio archival happens afterward and may fail independently.
- Surface post-emission audio archival failures through the existing D-Bus notification path while keeping the transcript history row even when the audio row is absent.

## Scope

### New Capabilities
- `historical-recording-audio`: Archive the full microphone capture for each finalized recording as `m4a` bytes in SQLite, keyed 1:1 to the transcript history row.

### Modified Capabilities
- `active-listener-transcript-history`: finalized history persistence expands from transcript-only rows to transcript rows plus optional audio rows, with startup validation for the encoder dependency.
- `active-listener-recording-finalization`: finalization retains per-recording capture bytes and hands them to the history store after successful text emission, without adding encode latency before emission.
- `active-listener-startup-validation`: startup must resolve the `ffmpeg` binary from config or `PATH` and fail fast when archival support cannot run.
- `active-listener-failure-notification`: post-emission archival failures become user-visible through D-Bus notifications instead of remaining silent log-only history failures.

## Impact

Affected code lives primarily in `packages/active-listener`: the capture wiring in `bootstrap.py`, the recording-session/finalizer path, the transcript-history store, config models/loaders, README/runtime docs, and D-Bus notification integration. The SQLite schema gains a new audio table related 1:1 to `transcript_history`, and the active-listener runtime gains a hard dependency on a resolvable `ffmpeg` binary for startup. No new PyPI or npm package dependency is required for this feature; the implementation targets the existing system `ffmpeg` CLI. Existing transcript rows, totals scripts, and transcript-only workflows remain valid; absence of an audio row simply means that archival did not complete for that recording.
