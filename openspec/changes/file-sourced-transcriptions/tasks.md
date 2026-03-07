## 0. Pre-flight reading checklist (required before coding)

- [ ] 0.1 Read these files end-to-end before editing anything: `packages/wire/src/eavesdrop/wire/messages.py`, `packages/wire/src/eavesdrop/wire/transcription.py`, `packages/client/src/eavesdrop/client/core.py`, `packages/client/src/eavesdrop/client/connection.py`, `packages/server/src/eavesdrop/server/connection_handler.py`, `packages/server/src/eavesdrop/server/server.py`, `packages/server/src/eavesdrop/server/streaming/client.py`, `packages/server/src/eavesdrop/server/streaming/processor.py`, `packages/server/src/eavesdrop/server/streaming/audio_flow.py`, `packages/server/src/eavesdrop/server/rtsp/client.py` [Agent: task-A]
- [ ] 0.2 Write down current behavior in a short engineer note (live mode setup path, `END_OF_AUDIO` handling, disconnect handling, and current FIRST_COMPLETED shutdown behavior) so implementation can be compared against baseline [Agent: task-A]

## 1. Protocol and lifecycle scaffolding

- [ ] 1.1 Add file-source setup option to wire models (`packages/wire/src/eavesdrop/wire/transcription.py`)
  - Add a source-mode field with a default that preserves existing live behavior.
  - Keep schema serialization backward compatible.
  - Avoid introducing a new setup message type.
  [Agent: task-B]
- [ ] 1.2 Ensure setup message uses the updated options model without breaking existing callsites (`packages/wire/src/eavesdrop/wire/messages.py`, `packages/wire/src/eavesdrop/wire/codec.py`) [Agent: task-B]
- [ ] 1.3 Update client message handling to explicitly recognize terminal disconnect signaling (`packages/client/src/eavesdrop/client/connection.py`)
  - Do not treat `DisconnectMessage` as “unknown message”.
  - Keep existing ready/transcription/error behavior unchanged.
  [Agent: task-C]
- [ ] 1.4 Add server session-mode plumbing to route live vs file transcriber paths (`packages/server/src/eavesdrop/server/connection_handler.py`, `packages/server/src/eavesdrop/server/server.py`) [Agent: task-D]
- [ ] 1.5 Verify and document FFmpeg-backed dependency assumptions for implementation and runtime (`packages/server/docker/Dockerfile.cuda-base`, `packages/server/docker/Dockerfile.rocm-base`, `packages/server/src/eavesdrop/server/rtsp/client.py`) [Agent: task-E]

## 2. Wave A (parallel): Server file ingestion core

- [ ] 2.1 Implement decoder helper that converts uploaded file bytes to canonical mono 16kHz float32 audio using FFmpeg subprocess pattern (`packages/server/src/eavesdrop/server/streaming/` new module or equivalent) [Agent: task-F]
- [ ] 2.2 Add bounded in-memory file queue with 900-second canonical capacity and blocking enqueue semantics (`packages/server/src/eavesdrop/server/streaming/` new module or equivalent) [Agent: task-G]
- [ ] 2.3 Integrate decoder output with file queue and existing processing path (`packages/server/src/eavesdrop/server/server.py`, `packages/server/src/eavesdrop/server/streaming/client.py`, and/or dedicated file-mode orchestrator module) [Agent: task-H]
- [ ] 2.4 Implement explicit finite-source lifecycle states and transitions (INGESTING → DRAINING → FINALIZING → TERMINAL) so EOF does not prematurely cancel processing (`packages/server/src/eavesdrop/server/streaming/client.py`, `packages/server/src/eavesdrop/server/streaming/processor.py`) [Agent: task-I]
- [ ] 2.5 Add the SINGLE required `<intent>...</intent>` doc comment to the file-mode lifecycle owner function (`packages/server/src/eavesdrop/server/streaming/client.py`, recommended function `_run_file_mode_lifecycle`) and ensure it appears exactly once in code [Agent: task-J]
- [ ] 2.6 Add periodic file-session observability logs (fill ratio, queued seconds, ingest rate, process rate, enqueue block time) and warning thresholds (`packages/server/src/eavesdrop/server/streaming/` relevant modules) [Agent: task-K]

## 3. Wave B (parallel): Client single-shot API and reducer

- [ ] 3.1 Add non-reentrant `EavesdropClient.transcribe_file(..., timeout_s=...)` API (`packages/client/src/eavesdrop/client/core.py`)
  - Acquire and release per-instance operation guard.
  - Perform connect/setup/upload/EOF/reduce/finalize sequence internally.
  [Agent: task-L]
- [ ] 3.2 Add file upload helper path used by `transcribe_file(...)` (`packages/client/src/eavesdrop/client/connection.py` and/or `packages/client/src/eavesdrop/client/audio.py`) [Agent: task-M]
- [ ] 3.3 Implement reducer logic for windowed outputs (`packages/client/src/eavesdrop/client/core.py`)
  - Skip invariant tail segment.
  - Reverse-scan completed window.
  - Append only new completed segments.
  - Warn (not fail) when last committed sentinel is missing.
  [Agent: task-N]
- [ ] 3.4 Implement timeout/cancellation cleanup behavior with single-owner exception logging (`packages/client/src/eavesdrop/client/core.py`, `packages/client/src/eavesdrop/client/connection.py`) [Agent: task-O]

## 4. Wave C (parallel): Contract tests and verification

- [ ] 4.1 Add wire contract tests for source-mode setup compatibility and codec round-trips (`packages/wire/tests/test_wire_codec_contracts.py`) [Agent: task-P]
- [ ] 4.2 Add server contract tests for decode/canonicalization requirements, bounded queue behavior, and EOF drain/finalization ordering (`packages/server/tests/` new file(s)) [Agent: task-Q]
- [ ] 4.3 Add client contract tests for one-shot API behavior: non-reentrancy, timeout cleanup, cancellation cleanup, disconnect terminal handling, and reducer sentinel-miss warning behavior (`packages/client/tests/` new file(s)) [Agent: task-R]
- [ ] 4.4 Add regression test for “final short tail after EOF still gets transcribed” (`packages/server/tests/` new or existing file) [Agent: task-S]
- [ ] 4.5 Run focused verification commands and record exact results in task notes:
  - `cd packages/wire && uv run pytest -q tests/test_wire_codec_contracts.py`
  - `cd packages/client && uv run pytest -q tests/test_client_mode_contracts.py <new_client_test_files>`
  - `cd packages/server && uv run pytest -q <new_server_test_files> tests/test_streaming_lifecycle_contracts.py tests/test_transcription_output_contracts.py`
  - `cd packages/server && uv run ruff check`
  - `cd packages/client && uv run ruff check`
  - `cd packages/wire && uv run ruff check`
  [Agent: task-T]

## 5. Final definition of done

- [ ] 5.1 Live transcriber mode remains behaviorally unchanged (validated by existing live-mode test coverage) [Agent: task-U]
- [ ] 5.2 File mode successfully transcribes WAV/MP3/AAC and completes with deterministic terminal behavior [Agent: task-U]
- [ ] 5.3 No duplicate exception stack traces are introduced for expected timeout/cancellation control-flow [Agent: task-U]
- [ ] 5.4 All new tests are deterministic and pass repeatedly in local reruns [Agent: task-U]
