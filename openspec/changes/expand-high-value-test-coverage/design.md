## Context

The repository has limited automated coverage in the exact places where regressions are most expensive: asynchronous lifecycle coordination, streaming output invariants, protocol serialization, and buffer/timing edge conditions. Existing tests are concentrated in configuration and connection setup, with sparse direct coverage for `streaming/processor.py`, `streaming/client.py`, `streaming/buffer.py`, `wire/codec.py`, and client runtime behavior.

This change adds a test-hardening layer that establishes executable contracts before broader feature work lands (file ingestion, throughput tuning, batching, and additional transport modes). The design must keep tests deterministic, fast enough for regular CI runs, and isolated from heavyweight model runtime dependencies.

## Goals / Non-Goals

**Goals:**
- Add high-value regression tests for the most failure-prone runtime contracts: lifecycle, ordering, monotonic timestamps, message shape, and recovery behavior.
- Define a parallel implementation plan that can be executed by independent high-capability agents with minimal coordination overhead.
- Preserve production behavior while increasing confidence and reducing regression detection latency.
- Keep tests deterministic via mocks/fakes and contract fixtures rather than model-heavy end-to-end inference.

**Non-Goals:**
- No production feature implementation (no batching/file ingestion behavior changes in this change).
- No broad load/performance benchmarking framework rollout in this iteration.
- No protocol redesign; only test and validate current contracts.

## Decisions

### Decision 1: Organize tests by contract surface, not by module ownership
- **Choice:** Partition the suite into five contract surfaces:
  1. Streaming lifecycle/state machine
  2. Output contract invariants
  3. Wire protocol compatibility/round-trip
  4. Buffer/timing invariants
  5. Client mode/message behavior
- **Rationale:** Contract-oriented tests fail where user-facing behavior breaks, even if internals refactor.
- **Alternatives considered:**
  - *Pure file-by-file unit tests only:* easier to write but weaker at catching cross-module regressions.
  - *Full e2e only:* expensive, flaky, and too coarse for fast diagnosis.

### Decision 2: Keep high-value integration tests deterministic with explicit fakes/mocks
- **Choice:** Use mocked sinks/sources/transcribers and synthetic audio fixtures to avoid model/GPU dependence.
- **Rationale:** Deterministic tests are reliable in CI and easier to parallelize across agents.
- **Alternatives considered:**
  - *Real model invocation:* more realism but high runtime cost and nondeterministic outputs.
  - *Only pure unit tests:* misses ordering/race issues in orchestration loops.

### Decision 3: Treat output shape and progression as normative contracts
- **Choice:** Assert strict invariants for:
  - completed vs incomplete segment semantics,
  - monotonic `absolute_*` timestamps,
  - disconnect and language event ordering,
  - serialization stability for wire message variants.
- **Rationale:** These are external contracts consumed by downstream clients/UI and are high-breakage/high-impact.
- **Alternatives considered:**
  - *Loose assertions (only non-empty output):* inadequate for preventing subtle client breakages.

### Decision 4: Parallelize implementation with independent, bounded workstreams
- **Choice:** Build `tasks.md` around independent work packages with explicit dependencies and numbered sub-agents using high-capability `task` agents.
- **Rationale:** Maximizes throughput while preserving correctness by isolating file scopes and dependency edges.
- **Alternatives considered:**
  - *Single-agent sequential execution:* lower coordination complexity, but slower and increases context-switch errors.

### Decision 5: Add a lightweight quality gate for new tests
- **Choice:** Require targeted pytest run(s) plus ruff check/format checks for changed tests in each task acceptance criteria.
- **Rationale:** Ensures each parallel task is merge-safe and locally verifiable.
- **Alternatives considered:**
  - *Rely only on final global run:* slower feedback and harder fault localization.

## Risks / Trade-offs

- **[Risk] Async lifecycle tests become flaky due to timing assumptions** → **Mitigation:** use deterministic event sequencing with explicit awaits, synthetic fixtures, and timeout-bounded assertions.
- **[Risk] Over-mocking hides integration faults** → **Mitigation:** include a small set of integration-style orchestration tests per surface, not only unit-level tests.
- **[Risk] Parallel agents edit overlapping files and conflict** → **Mitigation:** partition tasks by explicit file ownership and dependency edges; no shared file edits in parallel phases.
- **[Risk] CI time increases materially** → **Mitigation:** keep tests small/deterministic, avoid heavyweight model invocations, and use targeted suites for task acceptance.
- **[Risk] Contract assertions become brittle during legitimate refactors** → **Mitigation:** anchor assertions to documented external behavior (protocol/messages/invariants), not internal implementation details.

## Migration Plan

1. Add new tests in isolated files/workstreams without changing production logic.
2. Run targeted suites per workstream during implementation.
3. Consolidate and run package-level test commands as final verification.
4. If instability appears, quarantine only the unstable test case with a follow-up fix task; do not weaken contractual assertions broadly.

## Open Questions

- Should protocol contract tests include frozen JSON snapshots for message payloads, or only semantic field assertions?
- Do we want a separate “slow/nightly” lane now for future model-backed integration tests, or defer to a later change?
- Should buffer invariant tests adopt property-based generation immediately (Hypothesis) or start with deterministic case tables first?
