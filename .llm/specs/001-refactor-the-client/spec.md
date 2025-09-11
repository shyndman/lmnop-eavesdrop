# Feature Specification: Refactor Client Project as Importable Library

**Feature Branch**: `001-refactor-the-client`  
**Created**: 2025-09-10  
**Status**: Draft  
**Input**: User description: "Refactor the client project, transforming the existing CLI-based eavesdrop client into a clean, importable library while preserving all existing functionality. Rearrange the current WebSocket connection, audio capture, and message handling components into a unified programmatic API that supports both transcriber mode (sending audio for transcription) and subscriber mode (receiving transcriptions from RTSP streams). Keep the core technical implementation the same—simply remove the terminal interface and CLI aspects while exposing the underlying capabilities through a streaming async iterator API. See @packages/client/docs/api-client.md for more details"

## Execution Flow (main)
```
1. Parse user description from Input
   → If empty: ERROR "No feature description provided"
2. Extract key concepts from description
   → Identified: library transformation, dual-mode API, async streaming, preserved functionality
3. For each unclear aspect:
   → Marked with [NEEDS CLARIFICATION: specific question]
4. Fill User Scenarios & Testing section
   → Clear user flows for both transcriber and subscriber modes
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
Developers need to programmatically integrate eavesdrop transcription capabilities into their applications without CLI dependencies. They should be able to either send audio for transcription or receive transcriptions from existing RTSP streams through a clean Python library interface that provides streaming results via async iteration. The API should use factory methods for mode-specific client creation and expose all transcription configuration options.

### Acceptance Scenarios

#### Transcriber Mode Usage
1. **Given** a Python application needs audio transcription, **When** developer calls EavesdropClient.transcriber() with specific audio device and transcription options, **Then** the client connects to server and begins capturing audio without sending it
2. **Given** an active transcriber client connection, **When** developer calls start_streaming(), **Then** audio data flows to server and transcription results are available via async iteration
3. **Given** streaming is active, **When** developer iterates over the client, **Then** each transcription message contains complete metadata including segments, timestamps, and confidence scores
4. **Given** streaming is active, **When** developer calls stop_streaming(), **Then** audio capture continues but transmission stops, allowing restart without reconnection

#### Subscriber Mode Usage
1. **Given** a Python application needs to monitor RTSP stream transcriptions, **When** developer calls EavesdropClient.subscriber() with stream names, **Then** client connects and immediately begins receiving transcriptions from specified streams
2. **Given** an active subscriber client, **When** developer iterates over the client, **Then** transcription messages from all subscribed streams are delivered with stream identification
3. **Given** multiple streams are subscribed, **When** one stream has issues, **Then** other streams continue delivering transcriptions uninterrupted

#### Context Manager Usage
1. **Given** developer wants automatic connection lifecycle, **When** using client as async context manager, **Then** connection establishes on entry and cleanly disconnects on exit
2. **Given** an exception occurs during iteration, **When** within context manager, **Then** connection is properly cleaned up without leaving resources open

### Edge Cases
- What happens when specified audio device is unavailable or disconnected during capture?
- How does system handle WebSocket disconnection during active streaming?
- What occurs when subscriber mode requests non-existent stream names?
- How does client behave when server becomes unavailable mid-session?
- What happens when invalid transcription configuration options are provided?

## Requirements

### Functional Requirements
- **FR-001**: System MUST provide factory methods EavesdropClient.transcriber() and EavesdropClient.subscriber() for mode-specific client creation
- **FR-002**: System MUST implement async iterator interface returning complete TranscriptionMessage objects with full metadata
- **FR-003**: System MUST support async context manager protocol for automatic connection lifecycle management
- **FR-004**: System MUST preserve all existing WebSocket protocol functionality and message handling
- **FR-005**: System MUST support audio device selection by index, name, or substring matching for transcriber mode
- **FR-006**: System MUST provide callback mechanism for connection events, transcriptions, errors, and mode-specific status updates
- **FR-007**: System MUST allow independent control of audio capture and audio transmission in transcriber mode
- **FR-008**: System MUST support subscription to multiple named RTSP streams in subscriber mode
- **FR-009**: System MUST maintain streaming state through stop/start cycles without requiring reconnection
- **FR-010**: System MUST expose connection and streaming status through queryable properties
- **FR-011**: System MUST handle server errors gracefully and surface them through error callbacks
- **FR-012**: System MUST automatically terminate async iteration when WebSocket connection closes
- **FR-013**: System MUST be importable as standard Python library without CLI dependencies or main execution paths
- **FR-014**: System MUST expose transcription configuration options (beam_size, word_timestamps, initial_prompt, hotwords) in transcriber factory method
- **FR-015**: System MUST raise exceptions immediately when invalid audio devices or stream names are specified
- **FR-016**: System MUST raise exceptions when audio device becomes unavailable during streaming
- **FR-017**: System MUST provide thread safety through client instance isolation without locks or synchronization primitives

### Key Entities
- **EavesdropClient**: Primary interface providing unified access to transcription services with mode-specific behavior
- **TranscriptionMessage**: Rich data structure containing transcription results with segments, timestamps, language detection, and stream identification
- **Connection Session**: Stateful WebSocket connection managing protocol handshake, message routing, and lifecycle events
- **Audio Capture Session**: Transcriber-mode component managing microphone input, device selection, and audio streaming control
- **Stream Subscription**: Subscriber-mode component managing RTSP stream selection and multi-stream message routing

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