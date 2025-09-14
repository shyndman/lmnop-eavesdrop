# Feature Specification: Eavesdrop Active Listener Client

**Feature Branch**: `002-active-listener-client`
**Created**: 2025-09-14
**Status**: Draft
**Input**: User description: "active-listener-client: I'd like to introduce a new application package to this project. A client called "eavesdrop-active-listener". It should be placed in the packages/active-listener directory. This will be a console application depending on our client API package "eavesdrop-client".

When launched, it create an Eavesdrop transcription client (not subscriber) and will establish a connection to an eavesdrop server (host:port specified via CLI options), and will begin sending audio from an audio input device (name specified via CLI option). This is basic functionality provided by the API client.

The API client should be configured to expect a special hotword, "com", and should also indicate only a single previous segment should in each transcription result.

Where it gets more interesting is what we do with the received transcription results -- this application will use ydotool to type transcription results. To do this properly, you have to understand how transcription results are received.

A transcription message is composed of multiple segments -- 1 or more "completed" segments that are no longer being considered by the server, and whose contents are now locked in place, and a SINGLE in-progress segment, whose contents might change dramatically from update to update, until it is eventually marked complete and the next in-progress segment begins.

The results of the ydotool typing should always reflect the most recent state of the transcription, which means that as the in-progress segment changes, we should delete characters (until the previous segment and new segment share a prefix), then we should type the remainder of the new segment's characters.

Segments also have an id field that can be used to determine whether the in-progress segment we're handling is brand new, or represents an update to something we typed previously.

The application should store the entirety of the most recent text it has typed in memory. That is, every completed segment, plus whatever the most recent in-progress segment is."

## Execution Flow (main)
```
1. Parse user description from Input
   � If empty: ERROR "No feature description provided"
2. Extract key concepts from description
   � Identify: actors, actions, data, constraints
3. For each unclear aspect:
   � Mark with [NEEDS CLARIFICATION: specific question]
4. Fill User Scenarios & Testing section
   � If no clear user flow: ERROR "Cannot determine user scenarios"
5. Generate Functional Requirements
   � Each requirement must be testable
   � Mark ambiguous requirements
6. Identify Key Entities (if data involved)
7. Run Review Checklist
   � If any [NEEDS CLARIFICATION]: WARN "Spec has uncertainties"
   � If implementation details found: ERROR "Remove tech details"
8. Return: SUCCESS (spec ready for planning)
```

---

## � Quick Guidelines
-  Focus on WHAT users need and WHY
- L Avoid HOW to implement (no tech stack, APIs, code structure)
- =e Written for business stakeholders, not developers

### Section Requirements
- **Mandatory sections**: Must be completed for every feature
- **Optional sections**: Include only when relevant to the feature
- When a section doesn't apply, remove it entirely (don't leave as "N/A")

---

## User Scenarios & Testing *(mandatory)*

### Primary User Story
A user wants to perform voice-to-text dictation that is automatically typed into any active application on their desktop. They launch the active listener client, specify their audio input device and server connection, then begin speaking. The system continuously transcribes their speech and automatically types the results in real-time into whatever application currently has focus.

### Acceptance Scenarios
1. **Given** the active listener client is running and connected to an eavesdrop server, **When** the user speaks, **Then** the spoken words are automatically typed into the currently focused application
2. **Given** the user is dictating text and pauses mid-sentence, **When** they continue speaking, **Then** the system updates the typed text to reflect the complete, corrected transcription without duplicating content
3. **Given** the user has spoken multiple sentences, **When** the transcription server completes processing earlier segments, **Then** only new or changed content is typed, maintaining the integrity of previously completed text
4. **Given** the user launches the application, **When** they provide server host/port and audio device parameters, **Then** the client establishes connection and begins listening for speech
5. **Given** the transcription engine revises an in-progress segment, **When** the new segment shares a prefix with the previously typed content, **Then** only the differing suffix is deleted and retyped

### Edge Cases
- What happens when the network connection to the eavesdrop server is lost during dictation?
- How does the system handle rapid consecutive speech corrections that arrive faster than typing can occur?
- What happens when the user switches to a different application while dictation is in progress?
- How does the system behave when the specified audio input device is unavailable or disconnected?

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: System MUST accept command-line arguments for server host, port, and audio input device name
- **FR-002**: System MUST establish and maintain a connection to the specified eavesdrop transcription server
- **FR-003**: System MUST capture audio from the specified input device and stream it to the server
- **FR-004**: System MUST configure the API client with the hotword "com" as required by the eavesdrop client library
- **FR-005**: System MUST request transcription results that include only one previous segment for context
- **FR-006**: System MUST receive and process real-time transcription messages containing completed and in-progress segments
- **FR-007**: System MUST automatically type transcription results into the currently focused desktop application
- **FR-008**: System MUST track the complete state of all typed text (completed segments plus current in-progress segment)
- **FR-009**: System MUST handle updates to in-progress segments by deleting only the differing portion and typing the new content
- **FR-010**: System MUST use segment IDs to determine whether an in-progress segment is new or an update to existing content
- **FR-011**: System MUST maintain text synchronization by finding the longest common prefix between old and new segment versions
- **FR-012**: Users MUST be able to terminate the application gracefully, ending transcription and cleaning up resources

### Key Entities *(include if feature involves data)*
- **Transcription Session**: Represents an active voice-to-text session, containing current connection state, audio stream, and accumulated text
- **Text Segment**: Individual units of transcribed text with completion status (completed or in-progress) and unique identifiers
- **Typed Text State**: The complete current state of text that has been typed to the desktop, used for calculating necessary corrections
- **Audio Input Source**: The specified microphone or audio device providing input to the transcription system

---

## Review & Acceptance Checklist
*GATE: Automated checks run during main() execution*

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
*Updated by main() during processing*

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [x] Review checklist passed

---