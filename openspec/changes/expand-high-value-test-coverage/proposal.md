## Why

Current test coverage is concentrated in configuration and connection setup while the highest-risk behavior lives in streaming lifecycle orchestration, timestamped output contracts, buffer boundaries, and protocol serialization. We need targeted regression coverage now so future feature work (including file ingestion and batching paths) can move quickly without silently breaking production-critical transcription behavior.

## What Changes

- Add a focused, high-value automated test suite across server, wire, and client packages for lifecycle/state-machine behavior, output invariants, protocol contracts, and failure recovery.
- Introduce deterministic integration-style tests (with mocks/fakes) for streaming completion semantics, segment emission contract, and disconnect ordering.
- Add property/invariant-oriented tests for audio buffer boundaries and monotonic timing progression.
- Add wire protocol contract tests (round-trip serialization, discriminator handling, backwards compatibility-sensitive fields).
- Add targeted client tests for setup/options emission, streaming stop/start behavior, and message routing by mode.
- Define parallel implementation tasks with explicit dependency edges and sub-agent assignments using high-capability agents.

## Capabilities

### New Capabilities
- `transcription-test-hardening`: High-confidence regression suite for the core transcription pipeline contracts and failure paths across server, wire, and client packages.

### Modified Capabilities
- None.

## Impact

- Affected code paths under test: `packages/server/src/eavesdrop/server/streaming/*`, `packages/server/src/eavesdrop/server/transcription/*`, `packages/server/src/eavesdrop/server/rtsp/*`, `packages/wire/src/eavesdrop/wire/*`, and `packages/client/src/eavesdrop/client/*`.
- New tests will increase CI runtime moderately but reduce operational risk from regressions in lifecycle/timestamp/protocol behavior.
- No production API behavior changes in this change; this is test infrastructure and contract hardening.
