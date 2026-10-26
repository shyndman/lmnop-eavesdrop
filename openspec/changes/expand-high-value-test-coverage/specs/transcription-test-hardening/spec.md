## ADDED Requirements

### Requirement: Streaming lifecycle completion semantics SHALL be regression-tested
The test suite MUST validate live recording completion, flush, cancellation, and disconnect through the WebSocket protocol with deterministic recognition at the existing transcriber interface.

#### Scenario: A recording completes before the next recording
- **WHEN** recording A stops and requests force completion while recognition remains outstanding, and recording B is requested on the same socket
- **THEN** tests SHALL verify A's accepted work and final flush result are delivered before B processes audio
- **AND** B SHALL start with fresh transcript state and recording-relative timestamp zero

#### Scenario: Client disconnects during active processing
- **WHEN** the websocket transport closes unexpectedly while the processor task is active
- **THEN** tests SHALL verify lifecycle shutdown occurs without uncaught exceptions and emits expected terminal control behavior

### Requirement: Transcription output contract invariants MUST be enforced
The test suite MUST assert externally visible output invariants for segment progression, completion semantics, and timestamp monotonicity across update cycles.

#### Scenario: Completed and incomplete segment envelope
- **WHEN** successive transcription results include both finalized and in-progress segments
- **THEN** tests SHALL verify VAD-enabled output preserves completed history plus a trailing incomplete segment representation

#### Scenario: Disabled VAD preserves provisional output
- **WHEN** VAD is disabled and multiple segments, silence, and revisions occur
- **THEN** every pending segment SHALL remain incomplete across ordinary updates and a non-force-completing flush
- **AND** only an explicit force-completing flush SHALL complete all pending content through its accepted boundary

#### Scenario: Absolute timestamp progression across windows
- **WHEN** segments are emitted from successive processing windows with different offsets
- **THEN** tests SHALL verify mandatory native segment and individual word timestamps retain recording-relative offsets across leading and internal silence

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

### Requirement: Live client runtime behavior SHALL be tested
The client package test suite MUST cover live transcriber setup options, recording-aware message routing, and streaming state transitions.

#### Scenario: Transcriber setup option propagation
- **WHEN** a transcriber client is initialized with explicit user transcription options
- **THEN** tests SHALL verify setup payloads include expected option fields and values

#### Scenario: Recording-aware live routing
- **WHEN** the live client receives updates and flush responses for successive recordings on one connection
- **THEN** tests SHALL verify recording context and completion handling are preserved

#### Scenario: Empty flush remains responsive
- **WHEN** flush is requested before any audio or after the buffer is consumed
- **THEN** tests SHALL observe flush-complete acknowledgement, another responsive connection, and a successful subsequent flush

#### Scenario: Cancellation suppresses outstanding recognition
- **WHEN** an utterance is cancelled while recognition is outstanding
- **THEN** its results SHALL be suppressed and its utterance-local transcript history cleared
- **AND** another recording on the same connection SHALL succeed

### Requirement: Recovery and error-path behavior MUST have deterministic tests
The suite MUST include deterministic failure-injection tests for recoverable and terminal errors across streaming processor and sink interactions.

#### Scenario: Transcription exception recovery path
- **WHEN** a transcription step raises an exception within the processing loop
- **THEN** tests SHALL verify error handling path executes and loop behavior follows defined retry/sleep semantics

#### Scenario: Sink error propagation during shutdown
- **WHEN** sink-side error/disconnect calls occur during stop processing
- **THEN** tests SHALL verify teardown remains bounded and does not leak active tasks
