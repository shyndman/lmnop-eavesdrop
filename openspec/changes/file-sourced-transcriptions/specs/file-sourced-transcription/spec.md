## ADDED Requirements

### Requirement: Server SHALL accept file-sourced transcription sessions
The system SHALL allow transcriber sessions to declare file-sourced input intent at setup time and route the session through finite-source ingestion behavior without changing existing live transcriber defaults.

#### Scenario: File mode setup is requested
- **WHEN** a transcriber client connects and sends setup options indicating file-sourced transcription mode
- **THEN** the server SHALL initialize finite-source ingestion behavior for that session
- **AND** the server SHALL preserve existing live-mode behavior for sessions that do not request file mode

#### Scenario: Existing client omits file mode
- **WHEN** a transcriber client sends setup options that do not include the new file-source field
- **THEN** the server SHALL treat the session as live mode
- **AND** existing live transcriber behavior SHALL remain unchanged

### Requirement: Server SHALL canonicalize file audio before queue residency
The server SHALL decode accepted file formats (WAV, MP3, AAC) and normalize decoded audio to mono, 16kHz, float32 before placing audio into file-ingestion memory queues.

Implementation note for engineers: this requirement is about the format stored in the queue, not necessarily the format uploaded by the client. The queue must hold canonical mono 16kHz float32 audio.

#### Scenario: Multi-channel compressed input is uploaded
- **WHEN** a file-sourced transcription session uploads a valid MP3 or AAC file containing non-canonical channel/rate format
- **THEN** the server SHALL decode and normalize audio to mono 16kHz float32 before enqueueing
- **AND** queue capacity accounting SHALL be based on canonicalized audio seconds

#### Scenario: Decoder cannot read uploaded file
- **WHEN** a file-sourced transcription session uploads bytes that are not a valid supported audio file
- **THEN** the server SHALL fail the transcription operation clearly
- **AND** the failure SHALL occur before invalid bytes are treated as canonical queue audio

### Requirement: File ingestion SHALL be memory-bounded with blocking backpressure
The system SHALL enforce a bounded in-memory file-ingestion queue with capacity equal to 900 seconds of canonicalized audio per active file session.

Implementation note for engineers: the 900-second limit is measured after conversion to mono 16kHz float32. It is not measured in source-file bytes.

#### Scenario: Queue reaches configured capacity
- **WHEN** inbound file ingestion would exceed 900 seconds of canonicalized queued audio
- **THEN** enqueue operations SHALL block until queue capacity is available
- **AND** the system SHALL NOT drop queued canonical audio to admit new file data

#### Scenario: Queue is below configured capacity
- **WHEN** inbound file ingestion does not exceed 900 seconds of canonicalized queued audio
- **THEN** enqueue operations SHALL continue without artificial delay from the queue limit

### Requirement: Finite-source sessions SHALL drain before terminal finalization
The system SHALL separate finite-source EOF detection from processing completion and SHALL finalize sessions only after all queued canonical audio has been processed and terminal output has been emitted.

#### Scenario: EOF arrives with remaining queued audio
- **WHEN** file ingestion reaches EOF while queued canonical audio remains unprocessed
- **THEN** the system SHALL continue processing in draining mode until queued audio is exhausted
- **AND** the system SHALL emit terminal session signaling only after draining completes

#### Scenario: Remaining tail is shorter than normal processing threshold
- **WHEN** EOF is reached and the final remaining canonical audio is shorter than the normal minimum chunk threshold used for live processing
- **THEN** the system SHALL still process that final tail audio before finalization
- **AND** the session SHALL NOT loop forever waiting for additional input

#### Scenario: File session reaches finalization
- **WHEN** queued audio is exhausted and final output has been produced
- **THEN** the server SHALL emit terminal signaling once
- **AND** cleanup SHALL be idempotent if socket close also occurs

### Requirement: Client SHALL provide single-shot OOP file transcription API
The client library SHALL provide a non-reentrant `EavesdropClient.transcribe_file(...)` API that encapsulates connect, upload, EOF signaling, reduction, and completion handling for a single file transcription operation.

Implementation note for engineers: this method is a convenience wrapper over the websocket protocol. It is not a second transport.

#### Scenario: Concurrent file operations are attempted on one client instance
- **WHEN** `transcribe_file(...)` is invoked while another transcription operation is active on the same client instance
- **THEN** the client SHALL fail fast with an explicit operation-in-progress error

#### Scenario: Per-call timeout is configured
- **WHEN** `transcribe_file(..., timeout_s=<value>)` exceeds the configured timeout
- **THEN** the client SHALL perform cleanup and raise a timeout error for the call

#### Scenario: Caller cancels an active file transcription
- **WHEN** the task running `transcribe_file(...)` is cancelled
- **THEN** the client SHALL clean up network and background resources
- **AND** the cancellation SHALL be re-raised to the caller

### Requirement: Client reducer SHALL deterministically handle windowed segment outputs
The client one-shot reducer SHALL treat the final segment in each transcription message as the invariant in-progress tail, and SHALL commit completed segments via reverse scan anchored by last committed segment id.

#### Scenario: Message includes repeated completed history window
- **WHEN** a transcription message contains completed segments that include both old and newly completed entries
- **THEN** the reducer SHALL skip the invariant tail segment
- **AND** the reducer SHALL append only newly completed segments in chronological order

#### Scenario: Message contains no previously unseen completed segments
- **WHEN** a transcription message contains only already-committed completed segments plus the invariant tail segment
- **THEN** the reducer SHALL append nothing new
- **AND** the reducer SHALL preserve previously accumulated results

#### Scenario: Last committed sentinel is missing from completed window
- **WHEN** reverse scan does not find the previously committed segment id in the current completed window
- **THEN** the reducer SHALL log a warning indicating a potential gap
- **AND** the reducer SHALL continue by committing available completed segments in the received window

#### Scenario: Terminal signaling arrives before socket close
- **WHEN** the client receives an explicit `DisconnectMessage`
- **THEN** the one-shot operation SHALL treat that message as terminal completion for the transcription session
- **AND** the client SHALL tolerate the socket closing afterwards without treating it as a second failure

#### Scenario: Socket closes without explicit disconnect message
- **WHEN** the websocket closes after file upload and transcription output without an explicit `DisconnectMessage`
- **THEN** the client SHALL treat the close as terminal fallback behavior
- **AND** cleanup SHALL still complete cleanly

### Requirement: File sessions SHALL emit periodic ingestion and processing observability metrics
The system SHALL periodically log file-session queue and throughput metrics for operational visibility.

#### Scenario: File session remains active
- **WHEN** a file-sourced transcription session is ingesting or draining audio
- **THEN** the server SHALL periodically log queue fill ratio, queued canonical audio seconds, ingestion rate, processing rate, and enqueue blocking time
- **AND** the server SHALL emit warning-level logs when sustained queue saturation thresholds are crossed

#### Scenario: Queue saturation persists
- **WHEN** queue fill remains above the warning threshold across repeated observability intervals
- **THEN** the server SHALL emit warning-level metrics logs
- **AND** the session SHALL continue making forward progress instead of dropping canonical audio
