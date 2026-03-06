## ADDED Requirements

### Requirement: Streaming lifecycle completion semantics SHALL be regression-tested
The test suite MUST validate streaming lifecycle behavior when ingestion ends, processing continues, and shutdown/disconnect paths execute, including cancellation ordering and no premature termination of active processing loops.

#### Scenario: Ingestion ends before processing loop
- **WHEN** audio ingestion reaches end-of-stream while processing still has pending buffered work
- **THEN** tests SHALL verify processing completes expected drain/finalization behavior before terminal disconnect handling

#### Scenario: Client disconnects during active processing
- **WHEN** the websocket transport closes unexpectedly while the processor task is active
- **THEN** tests SHALL verify lifecycle shutdown occurs without uncaught exceptions and emits expected terminal control behavior

### Requirement: Transcription output contract invariants MUST be enforced
The test suite MUST assert externally visible output invariants for segment progression, completion semantics, and timestamp monotonicity across update cycles.

#### Scenario: Completed and incomplete segment envelope
- **WHEN** successive transcription results include both finalized and in-progress segments
- **THEN** tests SHALL verify the emitted payload preserves the documented contract of completed history plus a trailing incomplete segment representation

#### Scenario: Absolute timestamp progression across windows
- **WHEN** segments are emitted from successive processing windows with different offsets
- **THEN** tests SHALL verify absolute timestamp fields are monotonic and consistent with reported offsets

### Requirement: Wire protocol serialization compatibility SHALL be covered
The test suite MUST validate wire message serialization/deserialization behavior for all active message types used by server and client runtime paths.

#### Scenario: Round-trip message fidelity
- **WHEN** a valid message is serialized and then deserialized
- **THEN** tests SHALL verify semantic field fidelity for stream identifiers, segment payloads, and control metadata

#### Scenario: Invalid discriminator handling
- **WHEN** an unknown or malformed message discriminator is provided
- **THEN** tests SHALL verify parsing fails with explicit error behavior rather than silent coercion

### Requirement: Audio buffer and timing boundary invariants MUST be tested
The suite MUST cover buffer boundary behavior and timing progression for chunk extraction, processed boundaries, clipping behavior, and edge-case durations.

#### Scenario: Chunk extraction at boundary thresholds
- **WHEN** buffered audio duration is below and then reaches minimum chunk thresholds
- **THEN** tests SHALL verify chunk availability transitions occur at correct thresholds without skipping or duplicating data windows

#### Scenario: Processed boundary advancement
- **WHEN** completed transcription windows advance processing state
- **THEN** tests SHALL verify processed boundaries never regress and remain within buffered time domain limits

### Requirement: Client runtime mode behavior SHALL be tested
The client package test suite MUST cover transcriber and subscriber mode behaviors for setup options, message routing, and streaming state transitions.

#### Scenario: Transcriber setup option propagation
- **WHEN** a transcriber client is initialized with explicit user transcription options
- **THEN** tests SHALL verify setup payloads include expected option fields and values

#### Scenario: Subscriber routing by stream
- **WHEN** subscriber mode receives messages from configured stream subscriptions
- **THEN** tests SHALL verify message routing preserves stream context and mode-specific handling

### Requirement: Recovery and error-path behavior MUST have deterministic tests
The suite MUST include deterministic failure-injection tests for recoverable and terminal errors across streaming processor and sink interactions.

#### Scenario: Transcription exception recovery path
- **WHEN** a transcription step raises an exception within the processing loop
- **THEN** tests SHALL verify error handling path executes and loop behavior follows defined retry/sleep semantics

#### Scenario: Sink error propagation during shutdown
- **WHEN** sink-side error/disconnect calls occur during stop processing
- **THEN** tests SHALL verify teardown remains bounded and does not leak active tasks
