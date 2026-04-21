## Context

Today Active Listener publishes foreground state and transcription updates over D-Bus, and the GNOME extension listens for those signals to drive indicator and overlay behavior. Audio capture lives in `packages/client`, where `AudioCapture` is configured for mono 16 kHz `float32` input and queues `indata.copy().astype(DTYPE).tobytes()` payloads; `EavesdropClient._audio_streaming_loop()` forwards those exact bytes to `WebSocketConnection.send_audio_data()`. Nothing in that path currently exposes audio to Active Listener except the network send itself, so UI-only spectrum rendering would require either a second capture path or a new observation hook.

Design work already narrowed the visualization contract: 50 equal log-spaced bars over `60 Hz..8 kHz`, measured every `16 ms` from a `512`-sample window, Hann-windowed, projected by interpolation at log-spaced band centers, and compressed in dB. Real sample analysis against `tmp/at2020usb_16k.wav` showed that naive equal-log bucket membership leaves eight low-frequency bands structurally empty at this FFT size, so the projection method is no longer open. The remaining design problem is how to surface canonical capture bytes into Active Listener cheaply, analyze them without stalling streaming, and publish bar frames over the existing D-Bus/UI path.

Relevant files and current responsibilities:
- `packages/client/src/eavesdrop/client/core.py`: public `EavesdropClient` API, transcriber factory, live audio send loop
- `packages/client/src/eavesdrop/client/audio.py`: microphone stream setup and callback-to-bytes conversion
- `packages/active-listener/src/active_listener/bootstrap.py`: constructs the transcriber client used by Active Listener
- `packages/active-listener/src/active_listener/app/service.py`: owns foreground lifecycle, background tasks, and D-Bus publication calls
- `packages/active-listener/src/active_listener/infra/dbus.py`: declares D-Bus property/signal contract and service implementations
- `packages/active-listener-ui-gnome/src/extension.ts`: consumes D-Bus signals and renders indicator/overlay UI
- `packages/client/tests/test_client_mode_contracts.py`: current live-client contract tests
- `packages/active-listener/tests/test_app.py`: current Active Listener service orchestration tests
- `packages/active-listener/tests/test_dbus_service.py`: current D-Bus interface contract tests

Third-party dependency and version context locked for implementation:
- NumPy: workspace lockfile currently resolves `numpy==1.26.4` (`uv.lock`), while upstream stable docs are already on NumPy 2.4.4. The APIs needed here (`np.frombuffer`, `np.hanning`, `np.fft.rfft`, `np.fft.rfftfreq`, `np.interp`, `np.maximum`, `np.log10`, `np.clip`, `np.round`, `astype`) are documented in current NumPy stable and available in NumPy 1.26.4. Because `packages/client` and `packages/common` already declare `numpy<2`, `packages/active-listener` must add the same direct dependency line: `numpy<2`.
- python-sounddevice: workspace lockfile resolves `sounddevice==0.5.5`; implementation relies on the documented `InputStream` callback contract that passes a NumPy ndarray to `callback(indata, frames, time, status)`, with `dtype='float32'` honored when the stream is configured that way.
- python-sdbus: workspace lockfile resolves `sdbus==0.14.2`; implementation relies on `DbusInterfaceCommonAsync`, `dbus_signal_async(...)`, `export_to_dbus(...)`, and `.emit(payload)` on signal emitters.
- Gio/GLib / GJS: extension code receives D-Bus signals through `Gio.DBusProxy::g-signal`, whose `parameters` argument is a `GLib.Variant` tuple of signal arguments. For a single `ay` signal argument, the implementation must first extract child `0` from the tuple variant, then convert that `ay` child into a byte buffer using GLib byte-array APIs rather than treating it as text.

## Goals / Non-Goals

**Goals:**
- Expose transcriber capture bytes through an optional `EavesdropClient` constructor/factory callback without changing websocket payload format.
- Keep visualization aligned with audio the client successfully handed to the websocket send path, not a separate microphone tap.
- Compute spectrum frames inside Active Listener from a rolling buffer fed by callback ingestion.
- Publish live spectrum frames over Active Listener D-Bus so the GNOME extension can render bars without direct audio access.
- Keep existing transcription, flush, cancel, reconnect, rewrite, and overlay text behavior intact.

**Non-Goals:**
- Change transcription quality, audio transport, or rewrite behavior.
- Add timestamps, metadata envelopes, or alternate capture payload formats to the callback.
- Build a second UI-specific audio capture stack.
- Introduce adaptive normalization, perceptual filterbanks, mel scaling, or smoothing beyond the agreed FFT + interpolation + dB pipeline.
- Make the spectrum stream semantically meaningful for downstream DSP; it is visualization-only.

## Decisions

### 1. Add `on_capture(bytes)` to transcriber client construction

`EavesdropClient` already centralizes transcriber-mode send flow in `_audio_streaming_loop()` and already accepts mode-specific construction arguments such as `audio_device`. The public client does not currently expose top-level callbacks on `__init__`; callback wiring exists today on the internal `WebSocketConnection`, and `packages/client/docs/api-client.md` describes a broader callback API that is not implemented in `core.py`. This feature therefore introduces a new public constructor/factory callback rather than extending an already-existing public callback surface.

Implementation-facing API shape:
- current constructor: `EavesdropClient.__init__(client_type, host="localhost", port=9090, stream_names=None, audio_device=None, transcription_options=None)`
- current factory used by Active Listener: `EavesdropClient.transcriber(host=config.host, port=config.port, audio_device=config.audio_device)` in `packages/active-listener/src/active_listener/bootstrap.py`
- required additive surface: both `EavesdropClient.__init__` and `EavesdropClient.transcriber(...)` must accept `on_capture: Callable[[bytes], None] | None = None`, because Active Listener constructs its client through the `transcriber()` factory, not by calling `__init__` directly
- exact file touch points for this decision:
  - `packages/client/src/eavesdrop/client/core.py`
  - `packages/active-listener/src/active_listener/bootstrap.py`

The callback contract is intentionally minimal:
- payload type: raw `bytes`
- contents: canonical mono 16 kHz `float32` PCM payload, produced from `indata.copy().astype(np.float32).tobytes()` and identical to websocket audio payload
- ordering: callback receives chunks in send order
- timing: implementation invokes callback after successful local websocket send, but that ordering is not exposed as a public guarantee beyond the fact that callback follows send order

Why this over alternatives:
- Callback injection matches adjacent internal client code, where `WebSocketConnection` already uses callback-driven event delivery even though `EavesdropClient` does not yet expose constructor callbacks publicly.
- Constructor/factory injection keeps configuration at client-construction time, where `audio_device` already lives.
- `bytes` avoids inventing a wrapper type, timestamp model, or ndarray contract the caller does not need.
- Emitting callback from the existing send path keeps visualization coupled to audio the client actually committed to transport, which matches the chosen priority.

Alternatives considered:
- Hooking `AudioCapture` directly: rejected because it exposes raw mic activity instead of send-path audio and would bypass client-level control.
- Returning ndarrays or event objects: rejected because they add conversion or metadata complexity with no current consumer need.
- Firing callback before websocket send: rejected because it weakens the chosen semantic tie to sent audio.

Junior-engineer implementation note: do not add this callback to `WebSocketConnection`. `WebSocketConnection` already sits below the level where Active Listener constructs its client. The smallest coherent public surface is the top-level client plus the transcriber factory that Active Listener already uses.

### 2. Keep callback ingestion tiny; run analysis on a periodic Active Listener task

The callback must not do FFT work inline, because `_audio_streaming_loop()` is on the streaming hot path. Active Listener will therefore own a spectrum analyzer component with two responsibilities:
- append incoming PCM bytes into a rolling sample buffer on each callback invocation
- run a separate periodic analysis loop every `16 ms`

This is a scheduling separation, not a thread requirement. The default implementation should stay in-process and event-loop friendly; the goal is latency isolation for the send loop, not parallel compute. NumPy FFT cost for `512` samples is small enough that a periodic task is the simplest credible choice.

Implementation-facing NumPy API contract for the analyzer:
- Convert chunk bytes to samples with `np.frombuffer(chunk, dtype=np.float32)`.
- Maintain rolling sample storage in `float32` form; when a frame is needed, read the newest 512 samples.
- Build the analysis window with `np.hanning(512)` and multiply pointwise with the frame before FFT.
- Compute positive-frequency bins with `np.fft.rfft(windowed_frame)`; the result length is `257` for a 512-sample real-valued frame.
- Compute matching frequency coordinates with `np.fft.rfftfreq(512, d=1.0 / 16000.0)`.
- Interpolate FFT magnitudes onto 50 log-spaced bar centers with `np.interp(bar_centers_hz, fft_freqs_hz, magnitudes)`. `xp`/frequency coordinates must remain strictly increasing.
- Convert to dB with `20.0 * np.log10(np.maximum(magnitudes, eps))`.
- Normalize with `np.clip(...)`, quantize with `np.round(... * 255.0).astype(np.uint8)`, and publish with `.tobytes()`.

Concrete analyzer behavior:
- Recommended module location: `packages/active-listener/src/active_listener/recording/spectrum.py`, because `packages/active-listener/src/active_listener/recording/` already owns recording-lifetime logic.
- Recommended public surface: one small class that exposes `ingest(chunk: bytes) -> None`, `start() -> asyncio.Task[None] | None`, `stop() -> None`, and an internal periodic loop that publishes frames through an injected callback such as `publish(frame: bytes) -> Awaitable[None] | None`.
- Storage requirement: keep a fixed-size tail buffer containing at least the newest 512 samples. A simple implementation may keep a slightly larger tail (for example 2048 samples) if that makes slice updates easier, but it must always analyze only the newest 512 samples.
- Startup behavior is locked: do not publish anything until at least 512 samples have been ingested. Do not zero-pad the first few frames.
- Failure isolation is locked: the `on_capture` callback passed from Active Listener must not be allowed to tear down live streaming. Wrap the analyzer-ingest callback at the Active Listener boundary so any local visualization error is caught and logged before returning to the client send loop.

Exact lifecycle hooks already present in the codebase:
- `ActiveListenerService.handle_keyboard_action()` is where recording start/cancel/finish transitions happen and where analyzer start/stop behavior must align
- `RecordingSession.start_recording()` and `RecordingSession.stop_recording()` define the lower-level capture boundaries the analyzer must respect indirectly through service orchestration
- `ActiveListenerService.close()` already cancels and awaits `_background_tasks`; analyzer periodic work should be tracked there so shutdown is truthful and coordinated

Concrete lifecycle rules:
- On `KeyboardDecision.START_RECORDING`, create or reset analyzer state before the service begins consuming callback bytes for that recording.
- On `KeyboardDecision.CANCEL_RECORDING`, stop the analyzer loop and clear buffered samples. Do not publish a terminal zero frame.
- On recording finish, stop the analyzer loop and clear buffered samples after recording capture stops. Again, do not publish a terminal zero frame.
- On disconnect-triggered recording abort, stop the analyzer loop the same way recording capture is stopped.
- UI reset is driven by existing `State` transitions (`recording` -> `idle` / `reconnecting`), not by a synthetic all-zero spectrum event.

Alternatives considered:
- Doing FFT inside the callback: rejected because it risks coupling visualization cost to websocket throughput.
- Using a dedicated worker thread by default: rejected because it adds synchronization complexity before evidence says same-loop scheduling is insufficient.

### 3. FFT shape is fixed: `512` samples, `16 ms` cadence, equal log centers, dB compression

The analyzer will maintain enough trailing samples to materialize the latest `512`-sample window at any `16 ms` tick. Each tick:
1. decode recent PCM bytes into `float32` samples
2. read the latest `512` samples from the rolling buffer
3. apply Hann window
4. compute `rfft`
5. take magnitude
6. interpolate magnitude at 50 equal log-spaced center frequencies spanning `60 Hz..8 kHz`
7. convert to dB
8. clamp and normalize
9. quantize for D-Bus publication

Locked constants for junior implementation:
- `sample_rate_hz = 16000`
- `window_size = 512`
- `tick_interval_ms = 16`
- `bar_count = 50`
- `min_frequency_hz = 60.0`
- `max_frequency_hz = 8000.0`
- `floor_db = -60.0`
- `ceil_db = -12.0`
- `eps = 1e-10`

Derived values the implementation should compute once, not every tick:
- `window = np.hanning(window_size).astype(np.float32)`
- `fft_freqs_hz = np.fft.rfftfreq(window_size, d=1.0 / sample_rate_hz)`
- `bar_edges_hz = np.geomspace(min_frequency_hz, max_frequency_hz, bar_count + 1)`
- `bar_centers_hz = np.sqrt(bar_edges_hz[:-1] * bar_edges_hz[1:])`

Reference pseudocode:

```python
def analyze_latest_frame(latest_512: np.ndarray) -> bytes:
    windowed = latest_512 * window
    spectrum = np.abs(np.fft.rfft(windowed))
    bars_linear = np.interp(bar_centers_hz, fft_freqs_hz, spectrum)
    bars_db = 20.0 * np.log10(np.maximum(bars_linear, eps))
    bars_norm = np.clip((bars_db - floor_db) / (ceil_db - floor_db), 0.0, 1.0)
    bars_u8 = np.round(bars_norm * 255.0).astype(np.uint8)
    return bars_u8.tobytes()
```

Why this exact shape:
- `512` samples gives `32 ms` history, which kept the visual responsive while still producing useful shape in testing.
- `16 ms` cadence gives responsive updates without pretending each frame is a new non-overlapping FFT block.
- Equal log spacing was explicitly chosen over mel-like blending.
- Interpolation is required because hard bucket membership creates structurally dead low bands at this FFT size.
- dB compression gives a tunable display scale without adaptive drift.

Alternatives considered:
- `1024`-sample window: rejected as too sluggish for the desired visual feel.
- Naive equal-log bucket averaging: rejected by measured output; it produced empty low bands.
- Mel/filterbank projection: rejected because equal log bars were explicitly preferred.
- Adaptive normalization: rejected because it makes quiet and loud input look misleadingly similar.

Junior-engineer implementation note: the interpolation step is not optional. Earlier experiments against `tmp/at2020usb_16k.wav` showed that hard bucket membership leaves eight low-frequency bars dead because the low bands are narrower than the FFT bin spacing at a 512-sample window. If the implementation replaces interpolation with bucket averaging, it is no longer implementing this spec.

### 4. Quantize at D-Bus boundary; publish bar frames as 50 bytes

Internal analysis can stay in float math until the transport edge, but D-Bus wire output is locked to quantized integers. The canonical wire shape will be 50 normalized bar bytes in the range `0..255`, emitted as a D-Bus byte array signal (signature `ay`).

Implementation-facing API details:
- Python service side: define the signal with `@dbus_signal_async(signal_signature="ay", signal_name="SpectrumUpdated")`
- Python emission side: emit a `bytes` payload from `SdbusDbusService`, matching how existing signal methods call `.emit(...)`
- GJS receive side: `handleProxySignal(signalName, parameters)` will receive `parameters` as a `GLib.Variant` tuple of signal arguments. Follow the same pattern the extension already uses for `TranscriptionUpdated`: unpack the tuple, then use the first value as the spectrum payload. The implementation should treat that first value as binary byte data (`Uint8Array` or `number[]` depending on runtime exposure). If it is a plain number array, convert it once to `Uint8Array` before rendering.
- The spec treats `ay` as binary transport. Do not model it as text or a generic list-of-int abstraction in the D-Bus contract.

Concrete payload contract:
- frame length is always exactly 50 bytes
- byte `0` maps to the lowest-frequency bar
- byte `49` maps to the highest-frequency bar
- each byte is already normalized display data in `[0, 255]`; the UI must not re-run dB conversion or FFT math

Why this over float transport:
- User locked D-Bus transport to quantized integers.
- Byte payload is compact, cheap to marshal, and exact enough for a visualization-only graph.
- Keeping float math internal preserves implementation flexibility while presenting one small stable UI contract.

Alternatives considered:
- D-Bus doubles: rejected after lock-in decision to quantize before transport.
- Wider integer ranges such as `0..1000`: rejected in favor of simpler byte payload with no demonstrated need for extra precision.

### 5. Extend Active Listener D-Bus with transient spectrum signal, not persisted property

Spectrum frames are high-churn transient data, unlike `State`, which is meaningful to cache and query. Active Listener will therefore add a new D-Bus signal for live spectrum updates rather than a property with `PropertiesChanged` churn. The GNOME extension already listens to raw `g-signal` events, so consuming another signal fits the current integration model.

Exact code surfaces to extend:
- Python: `AppStateService`, `NoopDbusService`, `ActiveListenerDbusInterface`, and `SdbusDbusService` in `packages/active-listener/src/active_listener/infra/dbus.py`
- GJS: `handleProxySignal(...)` in `packages/active-listener-ui-gnome/src/extension.ts`, alongside existing `TranscriptionUpdated` and `PipelineFailed` dispatch

Concrete Python changes expected:
- add `async def spectrum_updated(self, bars: bytes) -> None` to `AppStateService`
- add a no-op implementation to `NoopDbusService`
- add a `@dbus_signal_async(signal_signature="ay", signal_name="SpectrumUpdated")` declaration to `ActiveListenerDbusInterface`
- add `spectrum_updated` emission method to `SdbusDbusService`

Concrete TypeScript changes expected:
- define `const DBUS_SPECTRUM_UPDATED_SIGNAL = 'SpectrumUpdated'`
- extend `handleProxySignal(...)` dispatch to recognize the new signal before the pipeline-failure fallback
- add one small helper dedicated to unpacking spectrum payload and one helper dedicated to rendering/updating the bar UI state

Why signal over property:
- Spectrum frames are event-like and disposable; only latest frame matters.
- Property caching offers little value for a rapidly changing visualization stream.
- Signal handling already exists in the extension for transcription updates and failures.

Alternatives considered:
- D-Bus property carrying latest frame: rejected because high-frequency property churn is a poor fit for ephemeral visualization data.
- Separate side channel outside Active Listener D-Bus: rejected because proposal scope explicitly uses the existing D-Bus path.

### 6. UI remains renderer-only

The GNOME extension will not compute FFTs, infer capture cadence, or smooth bars from raw audio. It receives already prepared bar frames over D-Bus and renders them. Any later visual polish should operate on the bar stream, not reopen audio-analysis responsibilities in the UI layer.

Why this split:
- Keeps one authoritative analysis implementation.
- Avoids duplicating DSP logic in GJS.
- Matches current extension architecture, which already acts as a D-Bus consumer rather than a data producer.

Dependency note: `packages/active-listener` does not currently declare `numpy`, even though `packages/client` does. Because the analyzer will import NumPy directly from Active Listener code, `packages/active-listener/pyproject.toml` must add a direct `numpy` dependency instead of relying on the client package's transitive dependency.

Pinned dependency choice for this spec: add `numpy<2` to `packages/active-listener` to match the existing workspace dependency line and resolved lockfile (`numpy==1.26.4`). Do not widen to NumPy 2.x in this change.

Concrete rendering guidance for a junior engineer:
- Keep rendering state separate from transcript-overlay text state. The extension already has transcript-specific state (`completedTranscriptParts`, `incompleteTranscriptText`); spectrum bars should use their own state.
- Do not block menu/indicator updates on spectrum rendering.
- On any non-recording state (`idle`, `reconnecting`, `absent`), clear local spectrum state immediately so stale bars are not shown after recording stops.

## Concrete Implementation Walkthrough

Use this as the implementation order if you are unfamiliar with the codebase:

1. Update client public surface in `packages/client/src/eavesdrop/client/core.py`.
   - Add `on_capture` storage to the client.
   - Thread `on_capture` through `EavesdropClient.transcriber(...)`.
   - After `await self._connection.send_audio_data(audio_data)`, invoke the callback if present.

2. Update Active Listener bootstrap in `packages/active-listener/src/active_listener/bootstrap.py`.
   - Extend `build_client(...)` so it can pass an `on_capture` callback to `EavesdropClient.transcriber(...)`.
   - This callback should point at Active Listener analyzer ingestion, not at UI code.

3. Implement analyzer module under `packages/active-listener/src/active_listener/recording/`.
   - Keep pure FFT math separate from service orchestration when possible.
   - Give the service an object it can start, stop, and feed bytes into.

4. Wire analyzer lifecycle in `packages/active-listener/src/active_listener/app/service.py`.
   - Start/reset analyzer on recording start.
   - Stop/reset analyzer on cancel, finish, disconnect-abort, and service shutdown.
   - Publish frames with `dbus_service.spectrum_updated(...)` while recording is active.

5. Extend D-Bus contract in `packages/active-listener/src/active_listener/infra/dbus.py`.
   - Add the new protocol method to the service interface.
   - Add signal declaration and emission path.
   - Mirror the change in `NoopDbusService` so `--no-dbus` mode still works.

6. Extend GNOME extension in `packages/active-listener-ui-gnome/src/extension.ts`.
   - Add signal constant.
   - Handle signal dispatch.
   - Render bars from 50 bytes.
   - Clear bars when state is not recording.

7. Add focused tests as each step lands.
   - Do not wait until the end to add coverage.
   - Prefer existing test files when they already cover the right boundary; create `test_spectrum.py` only for analyzer-specific math/state behavior.

## Risks / Trade-offs

- [Visualization tied to websocket send path] → If websocket send stalls, callback ingestion and spectrum updates stall too. This is intentional; spec treats successfully sent audio as more important than raw local mic activity.
- [Periodic analyzer task may still cost enough to affect UI latency] → Keep callback append-only, keep window size fixed at 512, and validate performance before considering threading.
- [Quantization can reduce fine-grained low-level motion] → Byte scale is acceptable for visualization-only output; if bar motion looks too coarse, revisit normalization tuning before widening transport type.
- [D-Bus signal rate can create extra session-bus traffic] → Payload is only 50 bytes per frame plus signal overhead; start at the agreed 16 ms cadence and measure actual shell behavior during implementation.
- [No timestamps in callback or spectrum signal] → Accept because current consumer only needs latest-frame rendering; do not infer replay or synchronization semantics that the contract does not provide.

## Migration Plan

1. Add optional `on_capture` callback to `EavesdropClient.__init__` and `EavesdropClient.transcriber(...)`, then thread it through transcriber-mode setup without affecting existing callers.
2. Add direct `numpy` dependency to `packages/active-listener`, introduce the spectrum analyzer, and integrate callback ingestion plus lifecycle management into service startup/cleanup.
3. Extend D-Bus service and interface with live spectrum signal using 50-byte frame payloads.
4. Update GNOME extension to subscribe to spectrum signal and render bars while preserving existing indicator/transcript behavior.
5. Verify behavior with focused tests and one real-audio manual check using known WAV input and live capture.

Rollback strategy: remove spectrum signal consumption first, then disable analyzer/callback plumbing. Because the new client callback and D-Bus signal are additive, rollback does not require protocol migration for existing transcription paths.

## Open Questions

- No open questions remain for implementation. This spec locks `floor_db = -60.0`, `ceil_db = -12.0`, startup behavior (skip publish until 512 samples exist), and stop behavior (no terminal zero frame; UI clears on non-recording state).
