## Historical completion record

Checked items below record the original test-hardening work, not verification of issue 20. Coverage targets are updated for the live-only contract; removed ingestion workflows and optional timing formats are not supported test surfaces.

## 1. Orchestration and test-harness prerequisites (sequential)

- [x] 1.1 Agent #0 (type=`task`, high-capability): Enable runnable test harnesses for `packages/client` and `packages/wire` (add/confirm pytest availability and package-local test invocation path), then create minimal test package scaffolding if missing. **Files:** `packages/client/pyproject.toml`, `packages/wire/pyproject.toml`, `packages/client/tests/`, `packages/wire/tests/`. **Acceptance:** `uv run pytest -q` works in each package with at least a smoke test target.
- [x] 1.2 Agent #0 (type=`task`, high-capability): Add a short implementation note in task output (not repo docs) listing exact commands all downstream agents will use for local verification in server/client/wire packages. **Dependency:** 1.1.

## 2. Parallel Wave A — independent high-value test surfaces (all tasks depend only on 1.1, can run concurrently)

- [x] 2.1 Agent #1 (type=`task`, high-capability) — **Live lifecycle/state machine tests**. Cover multiple recordings on one socket, accepted-work-preserving force-completing flush before the next recording, non-force-completing flush, empty-flush responsiveness, cancellation suppression/reset, recording identifiers, and bounded disconnect cleanup. Exercise real WebSocket orchestration with deterministic recognition at the existing transcriber interface. **Primary files:** `packages/server/tests/test_streaming_lifecycle_contracts.py`, with useful behavioral fixtures reused where practical.
- [x] 2.2 Agent #2 (type=`task`, high-capability) — **Output contract invariant tests**. Cover VAD-enabled silence completion and trailing incomplete output, all VAD-disabled pending segments remaining provisional until explicit force completion, segment windows, and mandatory native segment/individual word timestamps across leading and internal silence on recording-relative timelines. **Primary files:** `packages/server/tests/test_transcription_output_contracts.py`, `packages/server/tests/test_timestamp_output.py`. Prefer observable WebSocket behavior over private wiring assertions.
- [x] 2.3 Agent #3 (type=`task`, high-capability) — **Audio buffer invariant tests**. Add boundary and progression tests for minimum chunk thresholds, processed boundary advancement, cleanup/clipping safety, and non-regression of processed positions. **Primary files:** `packages/server/tests/test_audio_stream_buffer_invariants.py` (new). **Reference runtime:** `packages/server/src/eavesdrop/server/streaming/buffer.py`. **Acceptance:** targeted pytest for this file passes.
- [x] 2.4 Agent #4 (type=`task`, high-capability) — **Wire protocol contract tests**. Add serialization/deserialization round-trip tests for active message types and explicit failure tests for malformed/unknown discriminators. **Primary files:** `packages/wire/tests/test_wire_codec_contracts.py` (new). **Reference runtime:** `packages/wire/src/eavesdrop/wire/codec.py`, `packages/wire/src/eavesdrop/wire/messages.py`, `packages/wire/src/eavesdrop/wire/transcription.py`. **Acceptance:** targeted pytest for this file passes.
- [x] 2.5 Agent #5 (type=`task`, high-capability) — **Live client runtime tests**. Cover transcriber option propagation in setup payloads, recording-aware updates and flush responses, and streaming stop/start state transitions. **Primary files:** `packages/client/tests/test_client_mode_contracts.py`. **Reference runtime:** `packages/client/src/eavesdrop/client/core.py`, `packages/client/src/eavesdrop/client/connection.py`. **Acceptance:** targeted pytest for this file passes after consolidation.
- [x] 2.6 Agent #6 (type=`task`, high-capability) — **Failure-injection/recovery tests**. Add deterministic tests for transcription exception handling path, retry/sleep behavior, and bounded teardown when sink errors occur during stop. **Primary files:** `packages/server/tests/test_streaming_error_recovery.py` (new). **Reference runtime:** `packages/server/src/eavesdrop/server/streaming/processor.py`. **Acceptance:** targeted pytest for this file passes.

## 3. Parallel Wave B — optional deepening tasks (start only after corresponding Wave A owner finishes)

- [x] 3.1 Agent #1 (type=`task`, high-capability): If lifecycle tests expose race flakiness, add deterministic synchronization helpers scoped to `test_streaming_lifecycle_contracts.py` only; do not modify production code.
- [x] 3.2 Agent #4 (type=`task`, high-capability): If wire contract tests reveal schema ambiguity, add explicit regression cases for the ambiguous payload shapes without changing protocol definitions.
- [x] 3.3 Agent #5 (type=`task`, high-capability): If live client tests expose implicit assumptions, add fixture-driven cases for supported transcriber initialization and recording-control variants.

## 4. Consolidation and verification (sequential)

- [x] 4.1 Agent #7 (type=`task`, high-capability): Rebase/merge parallel outputs, resolve test overlap if any, and ensure no duplicate coverage with conflicting assertions. **Dependencies:** completion of 2.1–2.6.
- [x] 4.2 Agent #7 (type=`task`, high-capability): Run package-targeted validation commands for all touched packages (`server`, `wire`, `client`) including pytest and ruff checks for changed test files; fix any failures. **Dependencies:** 4.1.
- [x] 4.3 Agent #7 (type=`task`, high-capability): Produce final verification summary listing each requirement from `specs/transcription-test-hardening/spec.md` and the exact test file/case that covers it. **Dependencies:** 4.2.
