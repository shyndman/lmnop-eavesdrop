## 1. Encoder configuration and startup
Files likely touched: `packages/active-listener/src/active_listener/config/models.py`, `packages/active-listener/src/active_listener/config/loader.py`, `packages/active-listener/config.sample.yaml`, `packages/active-listener/README.md`, `packages/active-listener/src/active_listener/bootstrap.py`, and active-listener config/startup tests.

- [x] 1.1 Add a top-level optional `ffmpeg_path` config field, normalize it with the existing config-loader path rules, rewrite the stale sample config to the current provider-based model shape, and document the new hard runtime `ffmpeg` dependency in the active-listener README.
- [x] 1.1a Verify config-loading tests cover explicit absolute paths, relative-path normalization from the config directory, and omission of `ffmpeg_path` without breaking existing configs.
- [x] 1.2 Change history-store construction so startup resolves FFmpeg once from `config.ffmpeg_path` first and `PATH` second, logs the resolved binary, and fails fast when neither source provides a usable executable while keeping the failure inside the existing startup error-reporting path.
- [x] 1.2a Verify service-startup tests cover configured-path resolution, `PATH` fallback resolution, and startup failure when FFmpeg cannot be resolved.

## 2. Recording-time audio retention and finalizer handoff
Files likely touched: `packages/active-listener/src/active_listener/bootstrap.py`, `packages/active-listener/src/active_listener/app/service.py`, `packages/active-listener/src/active_listener/app/ports.py`, `packages/active-listener/src/active_listener/recording/session.py`, `packages/active-listener/src/active_listener/recording/finalizer.py`, and active-listener recording/finalizer tests.

- [x] 2.1 Introduce a recording-scoped in-memory audio buffer that receives the existing `float32` capture bytes alongside spectrum analysis, with lifecycle owned by `RecordingSession` across start, finish, cancel, and disconnect-driven cleanup paths; return an immutable `FinishedRecording` snapshot instead of making the finalizer read mutable session state.
- [x] 2.1a Verify recording-path tests cover chunk accumulation across one recording, idle-state chunk dropping, cancel/disconnect discard behavior, and immutable finish snapshots.
- [x] 2.2 Expand the transcript-history protocol from transcript-only persistence to one synchronous finalized-recording archival call so finalization passes transcript metadata plus captured audio to the store after successful text emission and the store schedules D-Bus failure notification back onto the running event loop.
- [x] 2.2a Verify finalizer tests cover emit-before-archive ordering, successful handoff of captured audio to the store, and unchanged behavior when text emission fails before archival begins.

## 3. SQLite audio archival and FFmpeg encoding
Files likely touched: `packages/active-listener/src/active_listener/infra/transcript_history.py`, `packages/active-listener/scripts/transcript_history_totals.py`, and transcript-history tests.

- [x] 3.1 Extend the SQLite schema with a `transcript_audio` table keyed 1:1 to `transcript_history`, while keeping transcript-row persistence and legacy-database migration behavior intact in both the main store helper and the duplicated totals-script schema helper.
- [x] 3.1a Verify transcript-history and totals-script tests cover fresh databases, legacy transcript-only databases, and transcript rows that legitimately have no related audio row.
- [x] 3.2 Implement FFmpeg-based raw-`f32le` to `m4a` encoding inside the synchronous SQLite store using the native `aac` encoder, stdin for PCM input, and a temporary seekable output file created from `TemporaryDirectory()` for the MOV-family container; insert the audio row only after the transcript row exists, preserving transcript-only rows when encoding or audio insertion fails.
- [x] 3.2a Verify store tests cover successful transcript+audio persistence, encode failure with transcript-only fallback, and audio-row insert failure with transcript-only fallback plus logging/notification.

## 4. Archive-failure D-Bus notification
Files likely touched: `packages/active-listener/src/active_listener/infra/dbus.py`, `packages/active-listener-ui-gnome/src/active-listener-service-client.ts`, `packages/active-listener-ui-gnome/src/extension.ts`, and any active-listener / GNOME DBus client tests.

- [x] 4.1 Add a dedicated `AudioArchiveFailed(reason: str)` D-Bus signal from the active-listener service and map it to the existing GNOME error-notification bridge without changing `PipelineFailed` semantics.
- [x] 4.1a Verify Python D-Bus tests and TypeScript client tests cover archive-failure signal emission/decoding and confirm rewrite `PipelineFailed` handling remains unchanged.

## 5. Targeted validation
This task is intentionally last. Do not run it until config/startup resolution, recording capture, SQLite archival, and D-Bus notification work are all complete.

- [x] 5.1 Run targeted automated validation for affected packages (`packages/active-listener` plus any touched GNOME tests) and capture the outputs as artifacts.
- [ ] 5.1a (HUMAN_REQUIRED) Verify on a workstation that a normal recording still emits text immediately and that an induced archive failure raises a GNOME notification while leaving the transcript row present without an audio row.

```mermaid
graph TD
  "1.1" --> "1.1a"
  "1.1" --> "1.2"
  "1.2" --> "1.2a"
  "2.1" --> "2.1a"
  "2.1" --> "2.2"
  "2.2" --> "2.2a"
  "1.2" --> "3.1"
  "3.1" --> "3.1a"
  "2.2" --> "3.2"
  "3.1" --> "3.2"
  "3.2" --> "3.2a"
  "3.2" --> "4.1"
  "4.1" --> "4.1a"
  "1.2a" --> "5.1"
  "2.2a" --> "5.1"
  "3.2a" --> "5.1"
  "4.1a" --> "5.1"
  "5.1" --> "5.1a"
```
