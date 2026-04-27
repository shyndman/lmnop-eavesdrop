## Context

`active-listener` already has the core seams this feature needs, but a few of them are thinner or rougher than the original spec assumed.

Observed current behavior in the repository:
- `packages/client/src/eavesdrop/client/audio.py` captures mono `float32` PCM at 16 kHz.
- `packages/client/src/eavesdrop/client/core.py` sends each captured chunk to the websocket server and then invokes the local `on_capture` callback with the same bytes.
- `packages/active-listener/src/active_listener/bootstrap.py` currently wires `on_capture` only to `SpectrumAnalyzer.ingest(...)`.
- `packages/active-listener/src/active_listener/recording/finalizer.py` emits text first and only then calls the history store.
- `packages/active-listener/src/active_listener/app/ports.py` exposes a single synchronous transcript-history store method today: `record_finalized_transcript(record: FinalizedTranscriptRecord) -> None`.
- `packages/active-listener/src/active_listener/infra/transcript_history.py` owns the SQLite schema for transcript history, while `packages/active-listener/scripts/transcript_history_totals.py` duplicates that schema/path logic in its standalone script.
- `packages/active-listener/src/active_listener/config/loader.py` normalizes only specific path-bearing config fields today; this is not automatic across all string fields.
- `packages/active-listener/config.sample.yaml` is currently stale and does not match the active config model shape, so touching it for `ffmpeg_path` requires bringing it back in sync rather than appending one line.
- `packages/active-listener/src/active_listener/infra/dbus.py` defines several failure-related signals, but `packages/active-listener-ui-gnome/src/active-listener-service-client.ts` currently only turns `PipelineFailed` into a UI error notification.

User decisions locked during design:
- Archive the **entire capture**, not only the committed transcript window.
- Store archived audio in a separate SQLite table with a **1:1** relationship to `transcript_history`.
- Persist raw audio as **AAC in an `m4a` container**.
- Keep text emission on the hot path; audio archival happens **after emit**.
- If archival fails, that is acceptable: **text still emits**, the **transcript history row still persists**, and the user sees a **D-Bus notification**.
- `ffmpeg` is the encoder, with an explicit config path override and fallback `PATH` lookup.
- Recording audio stays **in memory**; no temp-file spool is required for the expected recording lengths.
- Let **FFmpeg** do sample-format conversion instead of reimplementing it in app code.
- One persistence boundary should own both transcript-row persistence and optional audio archival.

## Goals / Non-Goals

**Goals:**
- Preserve the full microphone capture for each finalized recording in memory until post-emission archival completes or fails.
- Expand the history store boundary so one finalized recording can persist transcript metadata plus an optional archived audio artifact.
- Encode the captured `float32` PCM bytes to AAC-in-`m4a` by invoking a validated `ffmpeg` binary.
- Keep text emission latency unchanged by making audio archival a post-emission side effect.
- Preserve transcript history rows even when audio archival fails.
- Surface archival failures truthfully through D-Bus notifications.
- Fail fast at service startup if the configured-or-discovered `ffmpeg` binary cannot be resolved.
- Document the exact current APIs and test seams the implementation must update.

**Non-Goals:**
- No change to server-side audio handling or wire protocol.
- No attempt to backfill audio for preexisting transcript history rows.
- No new audio export UI, playback UI, or query/reporting workflow in this feature.
- No temp-file spooling path unless later measurement proves the in-memory model insufficient.
- No extra metadata columns in the audio table beyond the transcript foreign key and blob payload.
- No guarantee that every transcript row has audio; missing audio remains a valid steady-state outcome after archival failure.

## Implementation-facing APIs

The implementor should not need to rediscover the current active-listener seams.

### 1. Capture and client APIs

`packages/client/src/eavesdrop/client/audio.py`

```python
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.float32
BLOCKSIZE = 0
```

```python
class AudioCapture:
    def audio_callback(self, indata: Float32Audio, _frames: int, _time_info: object, status: object) -> None
    async def get_audio_data(self, timeout: float = 0.1) -> bytes | None
```

Observed behavior:
- `audio_callback(...)` copies `indata`, coerces it to `np.float32`, and enqueues `audio_data.tobytes()`.
- The queued bytes are raw mono `f32le` PCM at 16 kHz.

`packages/client/src/eavesdrop/client/core.py`

```python
@classmethod
def transcriber(
    cls,
    host: str = "localhost",
    port: int = 9090,
    audio_device: str = "default",
    on_capture: Callable[[bytes], None] | None = None,
    word_timestamps: bool | None = None,
    initial_prompt: str | None = None,
    hotwords: list[str] | None = None,
    send_last_n_segments: int | None = None,
    beam_size: int | None = None,
    model: str | None = None,
) -> "EavesdropClient"
```

```python
async def _audio_streaming_loop(self) -> None:
    while self._streaming and self._connected:
        audio_data = await self._audio_capture.get_audio_data(timeout=0.1)
        if audio_data:
            await self._connection.send_audio_data(audio_data)
            if self._on_capture is not None:
                self._on_capture(audio_data)
```

Observed behavior:
- Local `on_capture` runs **after** websocket send inside the same loop iteration.
- The callback receives the exact same raw byte payload sent upstream.

`packages/active-listener/src/active_listener/bootstrap.py`

```python
def build_client(
    config: ActiveListenerConfig,
    on_capture: Callable[[Float32PcmChunk], None],
) -> ActiveListenerClient
```

```python
def build_capture_callback(
    *,
    spectrum_analyzer: SpectrumCaptureSink,
    logger: BoundLogger,
) -> Callable[[Float32PcmChunk], None]
```

Observed behavior:
- `build_client(...)` currently enables `word_timestamps=True` and passes `on_capture` through to `EavesdropClient.transcriber(...)`.
- `build_capture_callback(...)` currently only calls `spectrum_analyzer.ingest(chunk)` inside a protective `try/except`.

### 2. Finalization and persistence APIs

`packages/active-listener/src/active_listener/app/ports.py`

```python
@dataclass(frozen=True)
class FinalizedTranscriptRecord:
    pre_finalization_text: str
    post_finalization_text: str
    llm_model: str | None
    tokens_in: int | None
    tokens_out: int | None
    cost: Decimal | None
    word_count: int
    duration_seconds: float | None
```

```python
class ActiveListenerTranscriptHistoryStore(Protocol):
    def record_finalized_transcript(self, record: FinalizedTranscriptRecord) -> None: ...
```

Observed behavior:
- The current store protocol is **synchronous**.
- `FinalizedTranscriptRecord` does not currently carry audio.

`packages/active-listener/src/active_listener/recording/finalizer.py`

```python
async def finalize_recording(
    self,
    *,
    disconnect_generation: int,
    reducer_state: RecordingReducerState,
) -> None
```

Observed behavior:
- Finalizer flushes committed text from the server.
- It runs the rewrite/local text pipeline.
- It calls `self.emitter.emit_text(final_text)`.
- Only after a successful emit does it call `self.history_store.record_finalized_transcript(...)`.
- If `emit_text(...)` fails, history persistence does not run.
- The store call is not wrapped in a local `try/except`, so store exceptions currently escape the finalization task.

Implication for this feature:
- The new archival store method should remain a single call site from finalizer.
- Because text must emit even when archival fails, the archival path itself must catch encode/insert failures, log them, and signal D-Bus without re-raising.

### 3. Startup and config APIs

`packages/active-listener/src/active_listener/config/models.py`

```python
class ActiveListenerConfig(BaseModel):
    keyboard_name: str
    host: str
    port: int
    audio_device: str
    llm_rewrite: LlmRewriteConfig | None = None
```

Observed behavior:
- `ffmpeg_path` does **not** exist today.

`packages/active-listener/src/active_listener/config/loader.py`

```python
def load_active_listener_config(
    *,
    config_path: str | None = None,
    overrides: Mapping[str, object | None],
) -> ActiveListenerConfig
```

```python
def normalize_active_listener_config_paths(
    config_data: Mapping[str, object],
    *,
    config_dir: Path,
) -> dict[str, object]
```

Observed behavior:
- Path normalization is hand-written.
- Today it normalizes `llm_rewrite.prompt_path` and `llm_rewrite.provider.model_path` when the provider type is `litert`.
- Adding `ffmpeg_path` requires explicitly extending this normalization logic.

`packages/active-listener/config.sample.yaml`

Observed behavior:
- The sample file is stale.
- It still uses:

```yaml
llm_rewrite:
  enabled: true
  model_path: "models/rewrite.litertlm"
  prompt_path: "prompts/rewrite_prompt.md"
```

- The current model actually expects:

```yaml
llm_rewrite:
  prompt_path: "prompts/rewrite_prompt.md"
  provider:
    type: "litert"
    model_path: "models/rewrite.litertlm"
```

Implication for this feature:
- Updating the sample config is not “add one line.”
- The feature must bring the sample config back in sync with the current model while adding `ffmpeg_path`.

`packages/active-listener/src/active_listener/bootstrap.py`

```python
async def create_service(
    config: ActiveListenerConfig,
    *,
    dbus_service: AppStateService | None = None,
    keyboard_resolver: Callable[[str], KeyboardInput] = resolve_keyboard,
    client_factory: Callable[[ActiveListenerConfig, Callable[[Float32PcmChunk], None]], ActiveListenerClient] | None = None,
    emitter_factory: Callable[[], TextEmitter] | None = None,
    rewrite_client_factory: Callable[[LlmRewriteConfig | None], ActiveListenerRewriteClient] | None = None,
    history_store_factory: Callable[[BoundLogger], ActiveListenerTranscriptHistoryStore] | None = None,
) -> ActiveListenerService
```

```python
def build_history_store(logger: BoundLogger) -> ActiveListenerTranscriptHistoryStore
```

Observed behavior:
- `history_store = resolved_history_store_factory(logger)` happens before keyboard resolution and before client connection.
- Store construction is outside the main startup `try/except`, so factory failures do not currently get the same wrapping/logging as later startup dependency failures.
- `run_service(...)` reports startup failures as `ActiveListenerRuntimeError` and emits `FatalError` over D-Bus when possible.

Implication for this feature:
- The history-store factory signature must widen so FFmpeg resolution can happen at startup.
- Store construction should move into the startup error-handling path or perform equivalent wrapping explicitly.

### 4. Recording lifecycle APIs

`packages/active-listener/src/active_listener/app/service.py`

```python
async def handle_action(self, action: AppAction) -> AppActionDecision
```

Observed behavior:
- `START_RECORDING` starts spectrum analysis, calls `client.start_streaming()`, then `recording_session.start_recording()`, then sets phase to `RECORDING`.
- `CANCEL_RECORDING` stops recording, stops spectrum analysis, resets phase to `IDLE`, and calls `client.cancel_utterance()`.
- `FINISH_RECORDING` calls `recording_session.finish_recording()`, stops spectrum analysis, resets phase to `IDLE`, and starts background finalization with `asyncio.create_task(...)`.

`packages/active-listener/src/active_listener/recording/session.py`

```python
async def start_recording(self) -> None
async def stop_recording(self) -> None
async def finish_recording(self) -> RecordingReducerState
```

Observed behavior:
- `RecordingSession` already owns recording-scoped reducer state and recording resource cleanup.
- `finish_recording()` returns the reducer snapshot used by background finalization.
- Disconnect-aware finalization uses `disconnect_generation` checks to skip emission after reconnect/disconnect races.

Implication for this feature:
- `RecordingSession` is the natural lifecycle owner for starting, finishing, and discarding the audio buffer.
- The capture callback should append bytes into a shared buffer object, but session methods should control buffer start/finish/discard semantics.

### 5. DBus and GNOME client APIs

`packages/active-listener/src/active_listener/infra/dbus.py`

```python
DBUS_BUS_NAME = "ca.lmnop.Eavesdrop.ActiveListener"
DBUS_OBJECT_PATH = "/ca/lmnop/Eavesdrop/ActiveListener"
DBUS_INTERFACE_NAME = "ca.lmnop.Eavesdrop.ActiveListener1"
```

```python
class AppStateService(Protocol):
    async def set_state(self, state: ForegroundPhase) -> None: ...
    async def transcription_updated(self, runs: list[TextRun]) -> None: ...
    async def spectrum_updated(self, bars: QuantizedSpectrumFrame) -> None: ...
    async def recording_aborted(self, reason: str) -> None: ...
    async def pipeline_failed(self, step: str, reason: str) -> None: ...
    async def fatal_error(self, reason: str) -> None: ...
    async def reconnecting(self) -> None: ...
    async def reconnected(self) -> None: ...
    async def close(self) -> None: ...
```

Observed behavior:
- Signal names use `UpperCamelCase` nouns/events such as `SpectrumUpdated`, `PipelineFailed`, and `RecordingAborted`.
- `NoopDbusService` is a method-for-method no-op mirror of the protocol.
- `SdbusDbusService` emits raw signal payloads directly.

`packages/active-listener-ui-gnome/src/active-listener-service-client.ts`

Observed behavior:
- The TS client uses `DBUS_*` constants for signal names.
- It defines D-Bus payload aliases such as `type DbusTextRun = [string, boolean, boolean]` before mapping them to domain types.
- `handleProxySignal(...)` currently handles only `TranscriptionUpdated`, `SpectrumUpdated`, and `PipelineFailed`.
- `RecordingAborted` and `FatalError` are currently ignored by the TS client.
- GNOME notifications come from `extension.ts`, where the client’s `onError(title, detail)` callback calls `Main.notifyError(title, detail)`.

Implication for this feature:
- Adding `AudioArchiveFailed` on the Python side alone is insufficient.
- The TS client must add a new constant, a new `handleProxySignal(...)` branch, and the usual `deepUnpack()` tuple destructuring.

### 6. SQLite schema, script coupling, and test seams

`packages/active-listener/src/active_listener/infra/transcript_history.py`

Observed behavior:
- Schema changes currently follow:

```sql
CREATE TABLE IF NOT EXISTS ...
PRAGMA table_info(...)
ALTER TABLE ... ADD COLUMN ...
```

- The store lazily opens SQLite and ensures schema only when inserting a record.

`packages/active-listener/scripts/transcript_history_totals.py`

Observed behavior:
- The totals script duplicates the database path resolution and schema/bootstrap helper logic instead of importing the store module.
- This duplication is deliberate for the script’s standalone shape.

`packages/active-listener/tests/test_app.py`

Observed behavior:
- `FakeHistoryStore` implements the current synchronous protocol and stores records in a list.
- `AssertingHistoryStore` subclasses that fake to assert finalizer ordering.

`packages/active-listener/tests/test_transcript_history.py`

Observed behavior:
- Tests directly instantiate `FinalizedTranscriptRecord` and inspect stored SQLite rows.

`packages/active-listener/tests/test_config.py`

Observed behavior:
- Tests already cover path normalization for rewrite assets.

Implication for this feature:
- If the store protocol changes, the test fake and assertion helper must change with it.
- If the schema helper grows a `transcript_audio` table, the totals script’s duplicated schema logic must be updated in lockstep.

### 7. Verified third-party dependency surface

`packages/active-listener/pyproject.toml`

Observed behavior:
- The active-listener Python package already depends on `numpy`, `pydantic`, `pyyaml`, `sdbus`, `structlog`, and other runtime libraries.
- There is **no** existing PyPI dependency for FFmpeg bindings such as PyAV or python-ffmpeg.

`packages/active-listener-ui-gnome/package.json`

Observed behavior:
- The GNOME extension already depends on GJS/GIR typings, `esbuild`, and `typescript`.
- This feature does **not** require any new npm package.

Verified external dependency:
- The only new third-party runtime dependency required by this feature is the system `ffmpeg` executable.
- The workstation-provided binary already available to the user is:

```text
/home/linuxbrew/.linuxbrew/opt/ffmpeg-full/bin/ffmpeg
```

- Running `ffmpeg -version` against that binary in this session reports `ffmpeg version 8.1`.
- FFmpeg’s official download page also lists `FFmpeg 8.1` as the latest stable release as of 2026-03-16.

Relevant FFmpeg 8.1 documentation facts verified from official docs:
- `ffmpeg` CLI syntax applies options to the next input or output file, so raw-input options must appear before `-i pipe:0`.
- The `pipe` protocol uses `pipe:0` for stdin and `pipe:1` for stdout by file descriptor number.
- The FFmpeg protocol docs warn that some formats, typically MOV-family outputs, require a **seekable** output protocol and fail with `pipe:` output.
- The native `aac` encoder is built into FFmpeg and is the default AAC encoder.
- The native `aac` encoder defaults to `128kbps` CBR when bitrate is unspecified.
- The native `aac` encoder defaults to the `aac_low` profile when profile is unspecified.
- FFmpeg muxer docs say MOV/MP4/ISOBMFF muxers support many muxing switches via `movflags`, but pipe output is the wrong default contract for seekability-sensitive MOV-family containers.

Implication for this feature:
- Do **not** add a Python FFmpeg binding or npm package.
- Do **not** depend on optional FFmpeg encoders such as `libfdk_aac`.
- Implement against the native FFmpeg 8.1 CLI and documentation, allowing 8.x minor updates.
- Feed PCM to FFmpeg through stdin, but write the `m4a` output to a seekable temporary file rather than `pipe:1`.

## Decisions

### 1. Add a standalone `RecordingAudioBuffer`, but let `RecordingSession` own its lifecycle

The feature will introduce a recording-scoped in-memory buffer object created during bootstrap and injected into both the capture callback path and the recording/finalization path.

Target shape:

```python
class RecordingAudioBuffer:
    def start(self) -> None: ...
    def append(self, chunk: bytes) -> None: ...
    def finish(self) -> bytes: ...
    def discard(self) -> None: ...
```

Recommended internal behavior:

```python
@dataclass
class RecordingAudioBuffer:
    _chunks: list[bytes] = field(default_factory=list)
    _active: bool = False

    def start(self) -> None:
        self._chunks.clear()
        self._active = True

    def append(self, chunk: bytes) -> None:
        if not self._active:
            return
        self._chunks.append(chunk)

    def finish(self) -> bytes:
        combined = b"".join(self._chunks)
        self._chunks.clear()
        self._active = False
        return combined

    def discard(self) -> None:
        self._chunks.clear()
        self._active = False
```

This exact behavior matters:
- `append(...)` must be a no-op while idle so the callback never needs to ask the service whether a recording is active.
- `finish()` must return a snapshot and also clear internal state so late chunks cannot contaminate the next recording.
- `discard()` must be safe to call repeatedly.

Ownership rules:
- `build_capture_callback(...)` appends raw chunks into the buffer.
- `RecordingSession.start_recording()` calls `buffer.start()`.
- `RecordingSession.stop_recording()` and other abort paths call `buffer.discard()`.
- `RecordingSession.finish_recording()` captures a final immutable snapshot for finalization.

Use a dedicated session return type so audio ownership is explicit:

```python
@dataclass(frozen=True)
class FinishedRecording:
    reducer_state: RecordingReducerState
    captured_audio: CapturedRecordingAudio
```

Recommended API cutover:

```python
async def finish_recording(self) -> FinishedRecording
```

and then:

```python
async def finalize_recording(
    self,
    *,
    disconnect_generation: int,
    finished_recording: FinishedRecording,
) -> None
```

Do **not** make the finalizer fetch bytes back out of mutable session state after `finish_recording()` returns. That creates race conditions and makes ownership ambiguous.

Why this cut:
- Bootstrap needs the buffer before service construction so the callback can append to it.
- Recording lifecycle semantics still belong with `RecordingSession`.
- The callback stays dumb; it appends bytes but does not decide recording boundaries.

Alternatives considered:
- Temp-file spooling: unnecessary for the expected clip sizes.
- Making the callback the lifecycle owner: wrong abstraction boundary.
- Reconstructing audio later from server outputs: loses the full capture and adds needless complexity.

### 2. Keep the raw capture bytes exactly as received and let FFmpeg do conversion

The archived input artifact is the same raw byte stream already flowing through `on_capture`: mono `f32le`, 16 kHz.

Why:
- It preserves the full capture exactly.
- It avoids duplicating audio-format conversion logic in app code.
- FFmpeg is better suited to sample-format conversion and AAC encoding.

Alternatives considered:
- Converting to `int16` chunks inside app code: smaller in-memory footprint, but duplicates media logic.
- Writing WAV headers or other container wrappers before encoding: unnecessary intermediary work.

### 3. Expand the history-store boundary, but keep it synchronous and single-call

The feature replaces the old protocol method with one synchronous archival call that owns transcript-row persistence plus optional audio archival.

Target shape:

```python
@dataclass(frozen=True)
class CapturedRecordingAudio:
    pcm_f32le: bytes
    sample_rate_hz: int
    channels: int


class ActiveListenerTranscriptHistoryStore(Protocol):
    def record_finalized_recording(
        self,
        record: FinalizedTranscriptRecord,
        captured_audio: CapturedRecordingAudio,
    ) -> None: ...
```

Because the store stays synchronous while the D-Bus notification API is asynchronous, the store must bridge that mismatch explicitly.

Required pattern:

```python
def _schedule_archive_failure_notification(self, reason: str) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        self.logger.warning("audio archive notification skipped", reason=reason)
        return

    task = loop.create_task(self.dbus_service.audio_archive_failed(reason))
    task.add_done_callback(_log_task_exception)
```

Where `_log_task_exception(...)` should inspect `task.exception()` and log any D-Bus emission failure.

Junior-engineer rule: do **not** call an async D-Bus method directly from the synchronous store without scheduling it on the running event loop.

Why keep it synchronous:
- The current finalizer/store seam is synchronous.
- Existing tests and fakes already assume a synchronous store call.
- The user’s latency requirement applies to pre-emit behavior; archival happens after emit.

Required behavior change from today:
- Unlike the current store call, the new archival method **must not** let encode/insert failures escape.
- It must catch failures internally, log them, signal D-Bus, and leave the transcript row intact.

Alternatives considered:
- Separate `history_store` and `audio_archiver` seams: splits one truth across two infrastructure objects.
- Asynchronous archival protocol: possible, but adds more moving parts and is not required to satisfy the user’s stated constraints.

### 4. Preserve emit-first ordering and transcript-first durability

The finalizer ordering remains:

```python
final_text = finalize_text()
emit_text(final_text)
history_store.record_finalized_recording(record, captured_audio)
```

Inside the store, archival ordering becomes:

```python
transcript_id = insert_transcript_history(record)
try:
    m4a_bytes = encode_m4a(ffmpeg_path, captured_audio.pcm_f32le)
    insert_transcript_audio(transcript_id, m4a_bytes)
except Exception as exc:
    log(exc)
    notify_audio_archive_failed(exc)
```

Why:
- Text emission is the primary contract.
- Transcript history is more important than archived audio.
- This preserves transcript rows even when archival fails.

Alternatives considered:
- Encode before emit: rejected by the user.
- One transaction covering transcript row plus audio row: would drop transcript history on audio failure.

### 5. Use a dedicated `transcript_audio` table with one blob column

Target schema:

```sql
CREATE TABLE IF NOT EXISTS transcript_audio (
  transcript_id INTEGER PRIMARY KEY,
  audio_m4a BLOB NOT NULL,
  FOREIGN KEY (transcript_id) REFERENCES transcript_history(id) ON DELETE CASCADE
)
```

SQLite note:
- SQLite only enforces foreign keys when `PRAGMA foreign_keys = ON` has been enabled for the current connection.
- If implementation keeps `ON DELETE CASCADE` in the schema, the connection setup in `transcript_history.py` must execute that pragma immediately after connecting.
- This feature does not delete transcript rows, so inserts do not depend on cascade behavior, but the schema should not pretend cascade works unless the pragma is enabled.

Meaning:
- A `transcript_history` row with no related `transcript_audio` row means “transcript persisted, audio archival missing.”
- No extra status column is required.

Why:
- Keeps `transcript_history` small and query-friendly.
- Matches the user’s 1:1-table preference.
- The missing-row state is already truthful enough.

### 6. Use FFmpeg subprocess encoding from stdin `f32le` to a seekable temporary `m4a` file

Target encoder contract:

```python
def encode_m4a(
    ffmpeg_path: str,
    pcm_f32le: bytes,
    *,
    sample_rate_hz: int = 16_000,
    channels: int = 1,
) -> bytes: ...
```

Required subprocess properties:
- Resolve and validate the FFmpeg binary at startup, not per recording.
- Declare the input format explicitly as raw `f32le` mono 16 kHz.
- Produce playable AAC in an `m4a` container using FFmpeg's native `aac` encoder.
- Use a seekable temporary output file for the MOV-family container, then read the bytes back into memory.

Required stdlib helpers:

```python
import subprocess
import tempfile
from pathlib import Path
```

Required local error type:

```python
class AudioArchiveError(RuntimeError):
    pass
```

Target command shape:

```text
ffmpeg -hide_banner -loglevel error -f f32le -ar 16000 -ac 1 -i pipe:0 -c:a aac -b:a 128k -profile:a aac_low -f ipod /tmp/output.m4a
```

Recommended implementation outline:

```python
with tempfile.TemporaryDirectory(prefix="active-listener-audio-") as temp_dir:
    output_path = Path(temp_dir) / "recording.m4a"

    result = subprocess.run(
        [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "f32le",
            "-ar",
            str(sample_rate_hz),
            "-ac",
            str(channels),
            "-i",
            "pipe:0",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-profile:a",
            "aac_low",
            "-f",
            "ipod",
            str(output_path),
        ],
        input=pcm_f32le,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr_text = result.stderr.decode("utf-8", errors="replace").strip()
        raise AudioArchiveError(stderr_text)

    if not output_path.exists():
        raise AudioArchiveError("ffmpeg completed without creating an output file")

    m4a_bytes = output_path.read_bytes()
    if not m4a_bytes:
        raise AudioArchiveError("ffmpeg created an empty m4a file")

return m4a_bytes
```

Why `TemporaryDirectory` instead of `NamedTemporaryFile`:
- `NamedTemporaryFile(...)` creates the file immediately.
- FFmpeg would then be writing over an already-existing path, which invites overwrite-prompt behavior unless `-y` is added.
- `TemporaryDirectory()` lets implementation choose a path that does not exist yet, which is simpler and easier for a junior engineer to reason about.

Why the temporary output file is required:
- FFmpeg's protocol docs state that `pipe:0` / `pipe:1` map to stdin/stdout.
- Those same docs warn that some formats, typically MOV-family outputs, require a seekable output protocol and fail with `pipe:` output.
- `m4a` is a MOV/MP4-family container, so `pipe:1` is not the safe default contract.
- This is an output-only temp file; it does **not** contradict the user's rejection of disk-spooling the captured recording input.

Junior-engineer rule: keep `-f f32le -ar 16000 -ac 1` **before** `-i pipe:0`. FFmpeg applies options to the next input or output file, so moving those flags after `-i` would describe the wrong side of the pipeline.

Why:
- FFmpeg is already a known repo-side media dependency.
- It handles both conversion and encoding.
- The native `aac` encoder and `aac_low` profile are documented defaults in FFmpeg 8.1 and do not require optional external codec libraries.
- It avoids stale Python AAC wrappers.

Alternatives considered:
- `pyfaac`: rejected during design as stale, GPL-encumbered, and mismatched to the Python 3.12 / float32 capture path.
- PyAV: viable, but larger API surface and more in-process media complexity than this one-shot archive step needs.
- Writing MOV-family output to `pipe:1`: rejected because FFmpeg documents that such outputs typically require a seekable destination.

### 7. Widen history-store construction so FFmpeg resolution happens during startup

Replace the current logger-only factory with a startup-time factory that has access to config and D-Bus.

Target shape:

```python
def build_history_store(
    config: ActiveListenerConfig,
    logger: BoundLogger,
    dbus_service: AppStateService,
) -> ActiveListenerTranscriptHistoryStore: ...
```

Resolution order:
1. `config.ffmpeg_path` when set
2. `shutil.which("ffmpeg")`
3. fail startup

Additional startup rule:
- Store construction should happen inside the main startup error-handling path so FFmpeg-resolution failures become `ActiveListenerRuntimeError` and participate in the existing fatal-error reporting flow.

Validation rule:
- Startup validation should do more than `which()`.
- After resolving a candidate path, run:

```python
subprocess.run(
    [resolved_ffmpeg_path, "-version"],
    capture_output=True,
    check=False,
)
```

- Treat non-zero exit code or `FileNotFoundError` as startup failure.
- Log the resolved path that passed validation.

Why:
- `systemd --user` PATH can differ from an interactive shell.
- The user explicitly wants config override plus PATH fallback.
- The store needs D-Bus access to notify archival failures.

### 8. Add `AudioArchiveFailed(reason: str)` as a first-class D-Bus signal

Target Python-side addition:

```python
class AppStateService(Protocol):
    async def audio_archive_failed(self, reason: str) -> None: ...
```

Target D-Bus signal shape:

```python
@dbus_signal_async(
    signal_signature="s",
    signal_args_names=("reason",),
    signal_name="AudioArchiveFailed",
)
def audio_archive_failed(self) -> str:
    raise NotImplementedError
```

Target TS-side shape:

```ts
const DBUS_AUDIO_ARCHIVE_FAILED_SIGNAL = 'AudioArchiveFailed';
type DbusAudioArchiveFailedPayload = [string];
```

`handleProxySignal(...)` behavior:
- decode `const [reason] = parameters.deepUnpack() as DbusAudioArchiveFailedPayload`
- call `this.events.onError('Active Listener audio archive failed', reason)`

Why:
- `PipelineFailed` is rewrite/finalization-specific and would be a lie here.
- The GNOME client already has a clear `onError -> Main.notifyError` bridge.
- A dedicated signal matches current naming and payload conventions.

### 9. Rewrite the stale sample config while adding `ffmpeg_path`

Target sample shape:

```yaml
keyboard_name: "AT Translated Set 2 keyboard"
host: "home-brainbox"
port: 9090
audio_device: "default"
ffmpeg_path: "/home/linuxbrew/.linuxbrew/opt/ffmpeg-full/bin/ffmpeg"  # optional example

llm_rewrite:
  prompt_path: "prompts/rewrite_prompt.md"
  provider:
    type: "litert"
    model_path: "models/rewrite.litertlm"
```

Why:
- The current sample file is already inconsistent with the real config model.
- Touching it for this feature without fixing the existing mismatch would preserve a known lie in the repo.

### 10. Concrete file-by-file implementation recipe

This section is intentionally procedural. Follow it when the implementation starts.

1. `packages/active-listener/src/active_listener/app/ports.py`
   - Add `CapturedRecordingAudio` dataclass.
   - Add `FinishedRecording` dataclass.
   - Replace `record_finalized_transcript(...)` with `record_finalized_recording(...)` on `ActiveListenerTranscriptHistoryStore`.
   - Keep the store API synchronous.

2. `packages/active-listener/src/active_listener/config/models.py`
   - Add `ffmpeg_path: str | None = None` to `ActiveListenerConfig`.

3. `packages/active-listener/src/active_listener/config/loader.py`
   - Extend `normalize_active_listener_config_paths(...)` to normalize `ffmpeg_path` with the same config-directory-relative semantics used for rewrite paths.

4. `packages/active-listener/src/active_listener/bootstrap.py`
   - Create one `RecordingAudioBuffer` during service construction.
   - Pass it into `build_capture_callback(...)`.
   - Pass it into `RecordingSession`.
   - Widen `history_store_factory` and `build_history_store(...)` so they receive `config`, `logger`, and `dbus_service`.
   - Move history-store construction into the startup error path.

5. `packages/active-listener/src/active_listener/recording/session.py`
   - Add an `audio_buffer` field.
   - Call `audio_buffer.start()` from `start_recording()` only after recording ownership is established.
   - Call `audio_buffer.discard()` from `stop_recording()` / abort cleanup.
   - Call `audio_buffer.finish()` from `finish_recording()` and return a `FinishedRecording` snapshot.

6. `packages/active-listener/src/active_listener/recording/finalizer.py`
   - Change `finalize_recording(...)` to accept `FinishedRecording` instead of bare `RecordingReducerState`.
   - Keep the existing ordering: flush/rewrite -> emit text -> archive.
   - Build `FinalizedTranscriptRecord` exactly as today, then call the new store method with the captured audio snapshot.

7. `packages/active-listener/src/active_listener/infra/transcript_history.py`
   - Add the `transcript_audio` table schema helper.
   - Add FFmpeg resolution/validation during store construction.
   - Add `AudioArchiveError`.
   - Insert transcript row first and capture `cursor.lastrowid`.
   - Run FFmpeg encode second.
   - Insert audio row third.
   - Catch audio-archive failures inside the store so the transcript row still commits.
   - Schedule D-Bus notification from the running loop.

8. `packages/active-listener/scripts/transcript_history_totals.py`
   - Update the duplicated schema helper so it knows about `transcript_audio`.
   - Do **not** change the totals output unless implementation discovers a real requirement to do so.

9. `packages/active-listener/src/active_listener/infra/dbus.py`
   - Add `audio_archive_failed(...)` to `AppStateService`, `NoopDbusService`, and `SdbusDbusService`.
   - Add the new signal declaration to `ActiveListenerDbusInterface`.

10. `packages/active-listener-ui-gnome/src/active-listener-service-client.ts`
    - Add `DBUS_AUDIO_ARCHIVE_FAILED_SIGNAL` constant.
    - Add `DbusAudioArchiveFailedPayload = [string]`.
    - Extend `handleProxySignal(...)` with a branch that decodes `[reason]` and forwards it to `onError(...)`.

11. `packages/active-listener-ui-gnome/src/extension.ts`
    - No new logic should be required if `onError(...)` continues to call `Main.notifyError(...)`.

### 11. Testing recipe

Update and use these existing test seams:

- `packages/active-listener/tests/test_config.py`
  - add `ffmpeg_path` normalization/omission tests.

- `packages/active-listener/tests/test_app.py`
  - update `FakeHistoryStore` to the new synchronous method.
  - update `AssertingHistoryStore` ordering assertions.
  - add tests proving emit happens before archival.
  - add tests proving text-emission failure prevents archival.
  - add tests proving cancel/disconnect discard buffered audio.

- `packages/active-listener/tests/test_transcript_history.py`
  - add fresh-db transcript+audio persistence test.
  - add legacy-db migration test that results in transcript row plus optional audio row behavior.
  - add archive-failure fallback test where transcript persists but no audio row exists.

- `packages/active-listener/tests/test_transcript_history_totals_script.py`
  - verify the standalone script still bootstraps the schema successfully when the audio table exists.

- Any D-Bus / GNOME client tests already covering `PipelineFailed`
  - mirror those for `AudioArchiveFailed`.

Testing rules:
- It is acceptable to fake the FFmpeg subprocess in unit tests because the executable is an external boundary.
- Prefer one focused unit test for command construction and one integration-style test using the real local FFmpeg binary when practical.
- Do not invent audio data formats in tests that contradict the real capture path; the input bytes should represent raw `f32le` PCM.

## Risks / Trade-offs

- [Large recordings increase memory use] → Mitigation: the expected recording length is short; only one active recording buffer exists; a temp-file spool can remain a future escape hatch.
- [MOV-family output is not reliably pipe-safe] → Mitigation: feed raw PCM through stdin but write `m4a` to a temporary seekable file, then read bytes back into memory.
- [Local archival work still runs on the service event loop] → Mitigation: archival runs only after emit; keep the code path synchronous and simple; if real-world measurements show unacceptable pause time, revisit async/off-thread archival in a later change.
- [History-store construction currently sits outside startup wrapping] → Mitigation: move FFmpeg-resolving store construction into the main startup error path as part of this feature.
- [Current store failures escape the finalizer task] → Mitigation: the new archival method must absorb encode/insert failures and signal them explicitly.
- [The totals script duplicates schema logic] → Mitigation: update both schema helpers in the same change and keep tests covering the script.
- [The GNOME client silently ignores unknown failure signals today] → Mitigation: ship the Python signal and TS client branch in the same task.
- [The sample config is already stale] → Mitigation: rewrite it to the current provider-based shape instead of appending one line.

## Migration Plan

1. Add `ffmpeg_path` to `ActiveListenerConfig`, extend loader normalization for it, and rewrite `config.sample.yaml` to the current model shape while adding the new field.
2. Widen `build_history_store(...)` / `history_store_factory` so store construction can resolve FFmpeg from config or PATH, log the resolved binary, and fail through the existing startup error path.
3. Introduce `RecordingAudioBuffer` and wire `build_capture_callback(...)` to append capture bytes to it alongside spectrum ingestion.
4. Let `RecordingSession` own buffer lifecycle across start, cancel, stop, finish, and close-related cleanup.
5. Replace `record_finalized_transcript(...)` with `record_finalized_recording(...)` and pass both transcript metadata and captured audio through the existing single persistence seam.
6. Extend the SQLite store with the `transcript_audio` table, update the duplicated totals-script schema helper, and keep transcript-row persistence intact for legacy databases.
7. Implement FFmpeg-based `m4a` encoding and transcript-first archival fallback inside the store, using stdin for raw PCM input and a temporary seekable output file for the encoded `m4a` bytes.
8. Add the `AudioArchiveFailed` D-Bus signal on the Python side and the matching decode/notification branch on the GNOME side.
9. Update focused tests for config normalization, startup failure, recording-buffer lifecycle, finalizer ordering, transcript-history migration, transcript-only fallback, audio-row persistence, and D-Bus notification handling.

Rollback strategy:
- Code rollback can leave the new `transcript_audio` table in place; older code ignores unknown tables.
- Transcript-only history remains readable throughout.
- No destructive data migration rollback is required.

## Open Questions

None.
