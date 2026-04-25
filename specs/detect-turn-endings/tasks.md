## 1. Wire contract
Files likely touched: `packages/wire/src/eavesdrop/wire/messages.py`, `packages/wire/src/eavesdrop/wire/codec.py`, `packages/wire/src/eavesdrop/wire/__init__.py`, `packages/wire/tests/test_wire_codec_contracts.py`, `packages/client/tests/test_client_mode_contracts.py`.

- [ ] 1.1 Add `SmartTurn` payload type plus optional `smart_turn` field on `TranscriptionMessage`, and update wire serialization/deserialization so absent `smart_turn` is omitted on the wire and hydrated to `None` in memory.
- [ ] 1.1a Verify wire contract with targeted tests in `packages/wire` and `packages/client` covering present, omitted, and deserialized-`None` `smart_turn` payloads; capture test output as artifact.

## 2. Smart Turn inference support

Files likely touched: `packages/server/src/eavesdrop/server/config.py`, new or existing Smart Turn helper module under `packages/server/src/eavesdrop/server/`, server runtime/dependency manifests, and `packages/server/tests/test_config.py` plus new focused unit tests for preprocessing/inference helper behavior.

- [ ] 2.1 Add server-side Smart Turn runtime support using standalone `onnxruntime` + `transformers` inference APIs (not `pipecat-ai`), `TranscriptionConfig` settings, and model-loading/inference path for buffered turn audio, including configurable context duration capped to supported model input length and an explicit runtime-packaging story for local and Docker environments.
- [ ] 2.1a Verify inference support with targeted `packages/server` tests for configuration, audio-window preparation, and one-run-per-transition scheduling behavior; capture pytest output as artifact.

## 3. Live streaming integration

Files likely touched: `packages/server/src/eavesdrop/server/streaming/interfaces.py`, `packages/server/src/eavesdrop/server/streaming/processor.py`, `packages/server/src/eavesdrop/server/streaming/audio_flow.py`, `packages/server/tests/test_transcription_output_contracts.py`, `packages/server/tests/test_streaming_lifecycle_contracts.py`.

- [ ] 3.1 Integrate Smart Turn into live streaming transcription so any speech-to-silence transition launches evaluation, binds result to same evaluated audio boundary, and includes `smart_turn` only on pause-evaluated `TranscriptionMessage` emissions.
- [ ] 3.1a Verify live streaming behavior with targeted `packages/server` transcription-output and lifecycle tests covering ordinary updates without `smart_turn`, pause-evaluated updates with `smart_turn`, and resumed-speech re-evaluation; capture pytest output as artifact.

## 4. RTSP integration

Files likely touched: `packages/server/src/eavesdrop/server/rtsp/audio_flow.py`, `packages/server/src/eavesdrop/server/rtsp/subscriber.py`, `packages/server/src/eavesdrop/server/rtsp/cache.py`, plus existing RTSP tests or new targeted tests that exercise fanout and replay behavior.

- [ ] 4.1 Extend RTSP transcription emission to use same Smart Turn trigger and payload contract as live streaming, including `RTSPTranscriptionSink`, `RTSPSubscriberManager`, and `RTSPTranscriptionCache` so RTSP no longer strips result-level metadata.
- [ ] 4.1a Verify RTSP behavior with targeted `packages/server` RTSP tests covering Smart Turn-enriched pause updates and unchanged ordinary updates; capture pytest output as artifact.

## 5. Cross-package validation

This task is intentionally last. Do not run it until wire, server, live, and RTSP work is complete.

- [ ] 5.1 Run targeted validation for affected packages (`packages/wire`, `packages/client`, `packages/server`) using exact commands documented in `design.md`: package-local `uv run pytest ...` and `uv run basedpyright`, plus `uv run ruff check` only after confirming/package-provisioning `ruff` for the active environment, and collect logs for review.
- [ ] 5.1a Verify validation artifacts show passing tests, lint, and type checks for all changed packages before implementation is considered complete.

```mermaid
graph TD
  "1.1" --> "1.1a"
  "2.1" --> "2.1a"
  "1.1" --> "3.1"
  "2.1" --> "3.1"
  "3.1" --> "3.1a"
  "1.1" --> "4.1"
  "2.1" --> "4.1"
  "4.1" --> "4.1a"
  "1.1a" --> "5.1"
  "2.1a" --> "5.1"
  "3.1a" --> "5.1"
  "4.1a" --> "5.1"
  "5.1" --> "5.1a"
```
