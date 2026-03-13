## ADDED Requirements

## Glossary

This glossary exists to remove ambiguity for implementation.

- **Live transcriber session**: a transcriber WebSocket connection that already completed `setup` with `source_mode=LIVE` and is actively using the existing streaming path.
- **Flush command**: the new text-frame wire message whose `type` is `control_flush`.
- **Flush boundary**: the buffer-end sample index captured by the server when it accepts a flush command.
- **Tentative tail segment**: the last segment in an emitted `TranscriptionMessage`, where `completed=False`.
- **Flush-satisfying response**: the single `TranscriptionMessage` emitted after processing covers the accepted flush boundary.

### Requirement: Live transcriber sessions SHALL accept flush control frames on the existing WebSocket
The system SHALL allow a live transcriber session to send a flush control frame on the same WebSocket connection that carries binary audio frames.

The wire message SHALL use the existing dataclass/discriminated-union message pattern and SHALL have this payload shape:

```json
{
  "type": "control_flush",
  "stream": "<stream-name>",
  "force_complete": true,
  "timestamp": 1710000000.0
}
```

Where:
1. `type` SHALL be the literal string `control_flush`.
2. `stream` SHALL be the active transcriber stream name.
3. `force_complete` SHALL be a boolean.
4. `timestamp` SHALL continue to use the existing `BaseMessage` behavior.

Implementation note for engineers: this requirement adds protocol multiplexing by frame type, not a second connection and not a custom binary frame header. It relies on the documented `websockets.asyncio` connection API already used by the repo, so implementation should keep server and client on a version floor compatible with `websockets.asyncio.*` imports (`>=13.0` in this repo).

#### Scenario: Flush arrives after live setup
- **WHEN** a transcriber client has completed normal setup for a live session and sends a `control_flush` text frame on the same WebSocket used for audio streaming
- **THEN** the server SHALL treat that frame as a session command instead of audio bytes
- **AND** subsequent binary audio frames SHALL continue to be treated as audio for that session

#### Scenario: Session never sends flush
- **WHEN** a live transcriber session streams audio and never sends `control_flush`
- **THEN** existing live audio behavior SHALL remain unchanged

### Requirement: Accepted flushes SHALL produce exactly one transcription response while keeping the session connected
For each accepted flush, the system SHALL emit exactly one `TranscriptionMessage` associated with that flush and SHALL keep the live session connected after sending the response.

The response SHALL cover audio received by the server before the accepted flush boundary, even if the current processing pass began before the flush was received.

Because the system allows only one accepted in-flight flush at a time, the flush response SHALL be correlated by sequence rather than by a separate flush ID:
1. the server accepts one flush,
2. the server eventually emits one flush-satisfying `TranscriptionMessage`,
3. that message is the response for that flush.

Implementation note for engineers: wakeable waits and cooperative pre-commit interrupt signaling for this requirement do not require a new third-party package. The existing Python 3.12 stdlib primitives are sufficient.

#### Scenario: Flush arrives while processor is waiting for minimum chunk duration
- **WHEN** buffered live audio is shorter than the normal minimum chunk threshold and the client sends an accepted `control_flush`
- **THEN** the server SHALL stop waiting for additional audio to satisfy that flush
- **AND** the server SHALL emit one `TranscriptionMessage` for that flush without requiring the client to send more audio first

#### Scenario: Flush arrives while processor is waiting for the normal interval
- **WHEN** the processor is waiting out the normal transcription interval and the client sends an accepted `control_flush`
- **THEN** the server SHALL stop waiting for the remainder of that normal interval
- **AND** the server SHALL emit one `TranscriptionMessage` for that flush

#### Scenario: Flush boundary extends beyond the current processing snapshot
- **WHEN** the server accepts a flush after additional pre-flush audio has arrived beyond the audio already included in the currently running processing pass
- **THEN** the server SHALL continue processing until the flush response covers audio through the accepted flush boundary
- **AND** the client SHALL NOT need to send a second flush to obtain that response

#### Scenario: Flush response is emitted
- **WHEN** the server emits the `TranscriptionMessage` that satisfies an accepted flush
- **THEN** the live session SHALL remain connected after that message is sent
- **AND** the accepted flush SHALL no longer be considered in flight
- **AND** the response SHALL still end with an incomplete tail segment

### Requirement: Flush force-complete semantics SHALL be explicit and preserved in the response
The flush command SHALL carry an explicit `force_complete` flag. When `force_complete` is true, the system SHALL treat the accepted flush as an end-of-utterance boundary and complete the current tentative segment before sending the flush response.

For this requirement, “current tentative segment” means the last segment that would otherwise have been emitted with `completed=False`.

#### Scenario: Force-complete is true and a tentative segment exists
- **WHEN** the server accepts `control_flush(force_complete=true)` for a session whose current transcription state contains a tentative tail segment
- **THEN** the flush response SHALL mark that prior tentative segment complete
- **AND** the response SHALL append a fresh incomplete tail segment after the completed content

#### Scenario: Force-complete is true and there is no natural in-progress tail to emit
- **WHEN** the server accepts `control_flush(force_complete=true)` and the flush-satisfying pass would otherwise emit only completed content
- **THEN** the server SHALL still append a synthetic incomplete tail segment to preserve the existing response invariant

#### Scenario: Force-complete is false
- **WHEN** the server accepts `control_flush(force_complete=false)`
- **THEN** the flush response SHALL preserve ordinary incomplete-tail behavior for any segment that is not otherwise complete
- **AND** the response SHALL still end with an incomplete tail segment

### Requirement: Concurrent flushes SHALL be rejected explicitly
The system SHALL allow at most one in-flight flush per live transcription session. A flush remains in flight until the `TranscriptionMessage` that satisfies it has been emitted.

If a second flush is received while another flush is in flight, the server SHALL reject the second flush explicitly and SHALL NOT coalesce or overwrite the existing flush.

The rejection SHALL use the existing `ErrorMessage` shape:

```json
{
  "type": "error",
  "stream": "<stream-name>",
  "message": "Flush rejected: another flush is already pending",
  "timestamp": 1710000001.0
}
```

#### Scenario: Second flush arrives before first flush response
- **WHEN** a live session already has an accepted flush in flight and the client sends another `control_flush`
- **THEN** the server SHALL reject the second flush with an `ErrorMessage`
- **AND** the rejection message SHALL state that another flush is already pending
- **AND** the connection SHALL remain open
- **AND** the original in-flight flush SHALL continue toward its own single transcription response

### Requirement: Client library SHALL expose flush as an awaitable request/response API
The client library SHALL provide an awaitable `flush(...)` API that returns the `TranscriptionMessage` corresponding to the accepted flush request.

Implementation note for engineers: the public API should guide callers toward legal sequencing by making flush completion explicit in the method contract. No new client-side dependency is required for this behavior beyond the repo's existing `websockets` and `pydantic` usage.

#### Scenario: Caller awaits a successful flush
- **WHEN** a caller invokes the client-library flush API for a connected live transcriber session
- **THEN** the client SHALL send exactly one `control_flush` command to the server
- **AND** the client SHALL await and return the flush-satisfying `TranscriptionMessage`

#### Scenario: Caller attempts a second local flush before the first completes
- **WHEN** a caller invokes the client-library flush API while a previous flush call is still awaiting its response on the same client instance
- **THEN** the client SHALL fail fast with an explicit operation-in-progress error
- **AND** the client SHALL NOT send the second flush command to the server

#### Scenario: Server rejects a flush that the client sent
- **WHEN** the client has sent `control_flush` and the server responds with the concurrent-flush rejection `ErrorMessage`
- **THEN** the client SHALL surface that rejection as a failed `flush(...)` call
- **AND** the connection SHALL remain usable after that failure
