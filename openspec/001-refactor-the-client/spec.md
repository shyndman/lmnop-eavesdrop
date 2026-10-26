# Feature Specification: Refactor Client Project as Importable Library

**Feature Branch**: `001-refactor-the-client`  
**Created**: 2025-09-10  
**Status**: Draft  
**Input**: Refactor the CLI-based eavesdrop client into an importable live WebSocket transcription library. Preserve microphone capture, independent transmission control, rich async iteration, callbacks, and connection lifecycle management. Issue 20 supersedes the original ingestion and timing choices: live audio is the sole supported input and every transcription includes native segment and individual word timestamps.

## Execution Flow (main)
```
1. Parse user description from Input
   → If empty: ERROR "No feature description provided"
2. Extract key concepts from description
   → Identified: library transformation, live audio API, async streaming, preserved lifecycle
3. For each unclear aspect:
   → Marked with [NEEDS CLARIFICATION: specific question]
4. Fill User Scenarios & Testing section
   → Clear user flows for live transcription and connection lifecycle
5. Generate Functional Requirements
   → Each requirement is testable and specific
6. Identify Key Entities (if data involved)
7. Run Review Checklist
   → No [NEEDS CLARIFICATION] markers remain
   → Implementation details avoided
8. Return: SUCCESS (spec ready for planning)
```

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

---

## User Scenarios & Testing

### Primary User Story
Developers need to integrate live microphone transcription without CLI dependencies through a clean Python library interface. The API provides async iteration, rich callbacks, audio-device selection, and supported recognition configuration while preserving independent capture, transmission, and connection lifecycles.

### Acceptance Scenarios

#### Transcriber Mode Usage
1. **Given** a Python application needs audio transcription, **When** developer calls EavesdropClient.transcriber() with specific audio device and transcription options, **Then** the client connects to server and begins capturing audio without sending it
2. **Given** an active transcriber client connection, **When** developer calls start_streaming(), **Then** audio data flows to server and transcription results are available via async iteration
3. **Given** streaming is active, **When** developer iterates over the client, **Then** each transcription includes native segment and individual word timestamps on the recording timeline
4. **Given** streaming is active, **When** developer calls stop_streaming(), **Then** audio capture continues but transmission stops, allowing restart without reconnection

#### Recording and Completion
1. A connection supports multiple recordings with fresh recording-relative timestamps and transcript state at each boundary.
2. Stop recording A, request a force-completing flush, and deliver its final transcript and flush-complete response before processing recording B.
3. With VAD disabled, all pending segments remain incomplete across ordinary updates and non-force-completing flushes until explicit force completion. With VAD enabled, silence completion remains supported.
4. Cancellation discards the active utterance and suppresses outstanding results without requiring reconnection.

#### Context Manager Usage
1. **Given** developer wants automatic connection lifecycle, **When** using client as async context manager, **Then** connection establishes on entry and cleanly disconnects on exit
2. **Given** an exception occurs during iteration, **When** within context manager, **Then** connection is properly cleaned up without leaving resources open

### Edge Cases
- What happens when specified audio device is unavailable or disconnected during capture?
- How does system handle WebSocket disconnection during active streaming?
- How does client behave when server becomes unavailable mid-session?
- What happens when invalid transcription configuration options are provided?

## Requirements

### Functional Requirements
- **FR-001**: System MUST provide EavesdropClient.transcriber() for live audio client creation
- **FR-002**: System MUST implement async iterator interface returning complete TranscriptionMessage objects with full metadata
- **FR-003**: System MUST support async context manager protocol for automatic connection lifecycle management
- **FR-004**: System MUST preserve supported live WebSocket setup, readiness, errors, transcription, controls, and disconnect behavior
- **FR-005**: System MUST support audio device selection by index, name, or substring matching for transcriber mode
- **FR-006**: System MUST provide callbacks for connection events, rich transcriptions, errors, and streaming status
- **FR-007**: System MUST allow independent control of audio capture and audio transmission in transcriber mode
- **FR-008**: System MUST include native segment and individual word timestamps in every transcription, with no timing-selection option
- **FR-009**: System MUST maintain streaming state through stop/start cycles without requiring reconnection
- **FR-010**: System MUST expose connection and streaming status through queryable properties
- **FR-011**: System MUST handle server errors gracefully and surface them through error callbacks
- **FR-012**: System MUST automatically terminate async iteration when WebSocket connection closes
- **FR-013**: System MUST be importable as standard Python library without CLI dependencies or main execution paths
- **FR-014**: System MUST expose supported recognition configuration in the transcriber factory without a timing-selection option
- **FR-015**: System MUST raise exceptions immediately when invalid audio devices are specified
- **FR-016**: System MUST raise exceptions when audio device becomes unavailable during streaming
- **FR-017**: System MUST provide thread safety through client instance isolation without locks or synchronization primitives

### Key Entities
- **EavesdropClient**: Primary interface for live transcription and its connection and streaming lifecycle
- **TranscriptionMessage**: Rich transcription results with mandatory native segment and individual word timestamps, language, and recording identification
- **Connection Session**: Stateful WebSocket connection managing protocol handshake, message routing, and lifecycle events
- **Audio Capture Session**: Transcriber-mode component managing microphone input, device selection, and audio streaming control

---

## Review & Acceptance Checklist

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous  
- [x] Success criteria are measurable
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

---

## Execution Status

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [x] Review checklist passed

---