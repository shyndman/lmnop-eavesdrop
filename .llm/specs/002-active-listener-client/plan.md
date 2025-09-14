# Implementation Plan: Eavesdrop Active Listener Client


**Branch**: `002-active-listener-client` | **Date**: 2025-09-14 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/002-active-listener-client/spec.md`

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
Create a console application that performs real-time voice-to-text dictation by connecting to an eavesdrop transcription server, capturing audio from a specified input device, and automatically typing the transcription results into the currently focused desktop application using ydotool. The application must intelligently handle incremental transcription updates by managing text state and performing smart insertions/deletions to maintain consistency.

## Technical Context
**Language/Version**: Python 3.12 (matching existing packages)
**Primary Dependencies**: eavesdrop-client, eavesdrop-wire, python-ydotool, clypi (for CLI), asyncio
**Storage**: In-memory text state tracking only
**Testing**: pytest (consistent with existing packages)
**Target Platform**: Linux desktop (ydotool requirement)
**Project Type**: single (console application)
**Performance Goals**: Real-time typing response (<100ms latency), handle rapid transcription updates
**Constraints**: Desktop focus integration, ydotool system dependency, network stability required
**Scale/Scope**: Single-user desktop application, moderate complexity (CLI + async client + text diffing logic)

**Additional Context**:
- Package location: `packages/active-listener`
- Created via: `uv init --name active-listener --app packages/active-listener`
- Uses same lint/type-checking settings as existing packages (pyproject.toml configuration)
- Depends on eavesdrop-client and eavesdrop-wire packages
- Hotword "com" configured for client (passed through, not handled by app logic)
- Single previous segment context (send_last_n_segments=1)

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Simplicity**:
- Projects: 1 (console app only)
- Using framework directly? Yes (clypi, eavesdrop-client directly)
- Single data model? Yes (in-memory text state)
- Avoiding patterns? Yes (no unnecessary abstractions)

**Architecture**:
- EVERY feature as library? N/A (single console application)
- Libraries listed: N/A (using existing eavesdrop-client library)
- CLI per library: Single entry point with clypi
- Library docs: Not applicable for app

**Testing (NON-NEGOTIABLE)**:
- RED-GREEN-Refactor cycle enforced? ✓ Planned
- Git commits show tests before implementation? ✓ Will follow
- Order: Contract→Integration→E2E→Unit strictly followed? ✓ Integration tests for client usage, unit tests for text diffing
- Real dependencies used? ✓ Real ydotool, real eavesdrop server for integration
- Integration tests for: client connection, transcription message handling, ydotool typing
- FORBIDDEN: Implementation before test, skipping RED phase ✓ Acknowledged

**Observability**:
- Structured logging included? ✓ Using structlog (consistent with other packages)
- Frontend logs → backend? N/A (desktop app)
- Error context sufficient? ✓ Connection errors, audio device errors, typing errors

**Versioning**:
- Version number assigned? ✓ 0.0.0 (following package pattern)
- BUILD increments on every change? ✓ Will follow
- Breaking changes handled? N/A (new application)

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

**Structure Decision**: Option 1 (Single project) - Console application structure

```
packages/active-listener/
├── src/
│   └── eavesdrop/
│       └── active_listener/
│           ├── __init__.py
│           ├── __main__.py      # CLI entry point
│           ├── cli.py           # Clypi command class
│           ├── client.py        # Eavesdrop client integration
│           ├── text_manager.py  # Text state and diffing logic
│           └── typer.py         # ydotool integration
├── tests/
│   ├── integration/
│   │   ├── test_client_connection.py
│   │   └── test_end_to_end.py
│   └── unit/
│       ├── test_text_manager.py
│       └── test_typer.py
├── pyproject.toml
└── README.md
```

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
   - Run `/scripts/bash/update-agent-context.sh claude` for your AI assistant
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
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- Package setup → `uv init` and pyproject.toml configuration [P]
- Contract tests → Each interface in contracts/ gets test file [P]
- Data model → Core classes (TextState, TextUpdate, etc.) [P]
- CLI implementation → Clypi command class with argument handling
- Client integration → EavesdropClient wrapper with transcription handling
- Text processing → Diffing algorithms and state management
- ydotool integration → Desktop typing with error recovery
- End-to-end → Complete workflow from speech to typing

**Ordering Strategy**:
- TDD order: Tests written first, then implementation to make them pass
- Dependency order: Package setup → Models → Services → CLI integration
- Parallel execution: Independent components marked [P]
- Sequential execution: Components that depend on previous outputs

**Task Categories**:
1. **Setup Tasks [P]**: Package creation, dependencies, project structure
2. **Contract Test Tasks [P]**: Test files for each contract interface
3. **Core Model Tasks [P]**: Data structures and validation logic
4. **Integration Tasks**: Client, typing, and CLI components (sequential)
5. **End-to-End Tasks**: Complete workflow testing and validation

**Estimated Output**: 20-25 numbered, ordered tasks in tasks.md

**Key Task Dependencies**:
- CLI tasks depend on client and text processing components
- Integration tests depend on all core components
- End-to-end tests depend on complete implementation
- Package configuration tasks can run in parallel with contract tests

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
- [ ] Complexity deviations documented

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*