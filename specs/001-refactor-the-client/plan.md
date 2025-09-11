# Implementation Plan: Refactor Client Project as Importable Library

**Branch**: `001-refactor-the-client` | **Date**: 2025-09-10 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/home/shyndman/dev/projects/lmnop/eavesdrop/specs/001-refactor-the-client/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
4. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
5. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, or `GEMINI.md` for Gemini CLI).
6. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
7. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
8. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Transform the existing CLI-based eavesdrop client into a clean, importable Python library while preserving all existing functionality. Create a unified programmatic API with factory methods that supports both transcriber mode (sending audio for transcription) and subscriber mode (receiving transcriptions from RTSP streams) through a streaming async iterator interface.

## Technical Context
**Language/Version**: Python 3.11+  
**Primary Dependencies**: websockets, sounddevice, numpy, pydantic, structlog, eavesdrop-wire  
**Storage**: N/A (streaming only)  
**Testing**: pytest with asyncio support  
**Target Platform**: Cross-platform (Linux, macOS, Windows)
**Project Type**: single (library refactor within existing packages/client)  
**Performance Goals**: Real-time audio streaming (16kHz sample rate), low-latency WebSocket communication  
**Constraints**: Thread-safe through isolation, no shared state, fail-fast error handling  
**Scale/Scope**: Single client instances, concurrent usage through multiple instances

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Simplicity**:
- Projects: 1 (library refactor within existing packages/client)
- Using framework directly? YES (websockets, sounddevice directly)
- Single data model? YES (reuse existing TranscriptionMessage from eavesdrop-wire)
- Avoiding patterns? YES (no Repository/UoW - direct WebSocket and audio device access)

**Architecture**:
- EVERY feature as library? YES (refactoring CLI into importable library)
- Libraries listed: eavesdrop.client (unified transcription client library)
- CLI per library: REMOVING CLI (this feature eliminates CLI interface)
- Library docs: YES (llms.txt format planned)

**Testing (NON-NEGOTIABLE)**:
- RED-GREEN-Refactor cycle enforced? YES (tests written first, must fail)
- Git commits show tests before implementation? YES (will be enforced)
- Order: Contract→Integration→E2E→Unit strictly followed? YES
- Real dependencies used? YES (actual WebSocket connections, audio devices)
- Integration tests for: library API contracts, WebSocket protocol, audio device integration
- FORBIDDEN: Implementation before test, skipping RED phase

**Observability**:
- Structured logging included? YES (existing structlog integration preserved)
- Frontend logs → backend? N/A (library doesn't have frontend)
- Error context sufficient? YES (callback mechanisms for error propagation)

**Versioning**:
- Version number assigned? YES (will increment existing package version)
- BUILD increments on every change? YES
- Breaking changes handled? YES (major refactor, migration from CLI to library API)

## Project Structure

### Documentation (this feature)
```
specs/[###-feature]/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/

# Option 2: Web application (when "frontend" + "backend" detected)
backend/
├── src/
│   ├── models/
│   ├── services/
│   └── api/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/

# Option 3: Mobile + API (when "iOS/Android" detected)
api/
└── [same as backend above]

ios/ or android/
└── [platform-specific structure]
```

**Structure Decision**: Option 1 (Single project) - Refactoring existing packages/client structure

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:
   ```
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - One test file per endpoint
   - Assert request/response schemas
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Each story → integration test scenario
   - Quickstart test = story validation steps

5. **Update agent file incrementally** (O(1) operation):
   - Run `/scripts/update-agent-context.sh [claude|gemini|copilot]` for your AI assistant
   - If exists: Add only NEW tech from current plan
   - Preserve manual additions between markers
   - Update recent changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, agent-specific file

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `/templates/tasks-template.md` as base
- Generate tasks from API contracts (eavesdrop_client_api.py)
- Factory method tests → contract test tasks [P]
- Async iterator protocol tests → contract test tasks [P]
- Context manager tests → contract test tasks [P]
- Connection lifecycle tests → integration test tasks
- Audio device integration tests → integration test tasks
- WebSocket protocol tests → integration test tasks
- Implementation tasks to make tests pass (refactor existing code)

**Ordering Strategy**:
- TDD order: Contract tests → Integration tests → Implementation
- Dependency order: Core client → Factory methods → Protocols
- Mark [P] for parallel execution (independent test files)
- Refactoring tasks: Remove CLI components → Extract library API → Update imports

**Estimated Output**: 20-25 numbered, ordered tasks in tasks.md

**Key Refactoring Tasks**:
1. Extract EavesdropClient class from existing MicrophoneClient
2. Remove TerminalInterface and CLI dependencies (app.py, __main__.py, interface.py)
3. Modify WebSocketConnection to support both ClientType.TRANSCRIBER and ClientType.RTSP_SUBSCRIBER
4. Update WebSocketConnection message processing to handle rich TranscriptionMessage callbacks
5. Remove stream filtering in message processing for subscriber mode multi-stream support
6. Add factory methods and async protocols (transcriber/subscriber, __aiter__/__anext__, context manager)
7. Preserve existing AudioCapture functionality with enhanced device selection
8. Update package exports to expose EavesdropClient as main API
9. Add support for X-Stream-Names header in subscriber mode connections

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |


## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (none required)

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*