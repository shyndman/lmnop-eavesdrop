# Tasks: Eavesdrop Active Listener Client

**Input**: Design documents from `/specs/002-active-listener-client/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → If not found: ERROR "No implementation plan found"
   → Extract: tech stack, libraries, structure
2. Load optional design documents:
   → data-model.md: Extract entities → model tasks
   → contracts/: Each file → contract test task
   → research.md: Extract decisions → setup tasks
3. Generate tasks by category:
   → Setup: project init, dependencies, linting
   → Tests: contract tests, integration tests
   → Core: models, services, CLI commands
   → Integration: eavesdrop client, ydotool, text processing
   → Polish: unit tests, performance, docs
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → All contracts have tests?
   → All entities have models?
   → All integration points implemented?
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Single project**: `packages/active-listener/src/`, `packages/active-listener/tests/`
- Paths assume console application structure per plan.md

## Phase 3.1: Setup
- [x] T001 Create packages/active-listener package structure with uv init
- [x] T002 Configure pyproject.toml with dependencies (python-ydotool, clypi, eavesdrop-client, eavesdrop-wire, structlog)
- [x] T003 [P] Configure linting and type-checking tools (ruff, pyright) matching existing packages

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [x] T004 [P] Contract test for CLI interface in tests/contract/test_cli_interface.py
- [x] T005 [P] Contract test for text processing functions in tests/contract/test_text_processing.py
- [x] T006 [P] Contract test for ydotool integration in tests/contract/test_ydotool_integration.py
- [x] T007 [P] Integration test for eavesdrop client connection in tests/integration/test_client_connection.py
- [x] T008 [P] Integration test for complete transcription workflow in tests/integration/test_end_to_end.py

## Phase 3.3: Core Models (ONLY after tests are failing)
- [x] T009 [P] TextState data model in src/eavesdrop/active_listener/text_manager.py
- [x] T010 [P] TextUpdate data model in src/eavesdrop/active_listener/text_manager.py
- [x] T011 [P] ConnectionState data model in src/eavesdrop/active_listener/client.py
- [x] T012 [P] UpdateType enum in src/eavesdrop/active_listener/text_manager.py

## Phase 3.4: Core Implementation
- [x] T013 [P] DesktopTyper class in src/eavesdrop/active_listener/typer.py
- [x] T014 Text diffing algorithms (find_common_prefix, calculate_text_diff) in src/eavesdrop/active_listener/text_manager.py
- [x] T015 Segment processing logic in src/eavesdrop/active_listener/text_manager.py
- [x] T016 EavesdropClient wrapper in src/eavesdrop/active_listener/client.py
- [x] T017 Clypi CLI command class in src/eavesdrop/active_listener/cli.py
- [x] T018 Application main entry point in src/eavesdrop/active_listener/__main__.py

## Phase 3.5: Integration & Error Handling
- [x] T019 python-ydotool initialization and error handling in src/eavesdrop/active_listener/typer.py
- [x] T020 WebSocket message handling and transcription processing in src/eavesdrop/active_listener/client.py
- [x] T021 Audio device validation and error recovery in src/eavesdrop/active_listener/client.py
- [x] T022 Connection state management and reconnection logic in src/eavesdrop/active_listener/client.py
- [ ] T023 Graceful shutdown and cleanup on SIGINT/SIGTERM in src/eavesdrop/active_listener/__main__.py
- [ ] T024 Structured logging throughout application components

## Phase 3.6: Polish & Validation
- [ ] T025 [P] Unit tests for text diffing algorithms in tests/unit/test_text_manager.py
- [ ] T026 [P] Unit tests for typing operations in tests/unit/test_typer.py
- [ ] T027 [P] Mock implementations for testing in tests/unit/test_mocks.py
- [ ] T028 Performance validation (<100ms typing latency)
- [ ] T029 Error handling validation (connection loss, audio device issues)
- [ ] T030 Memory usage validation (no leaks during long sessions)

## Dependencies
- Setup (T001-T003) before all other tasks
- Tests (T004-T008) before implementation (T009-T024)
- Models (T009-T012) before implementation tasks that use them
- Core implementation (T013-T018) before integration (T019-T024)
- Integration complete before polish (T025-T030)

## Parallel Task Groups
### Group 1: Contract Tests (after setup)
```
Task: "Contract test for CLI interface in tests/contract/test_cli_interface.py"
Task: "Contract test for text processing functions in tests/contract/test_text_processing.py"
Task: "Contract test for ydotool integration in tests/contract/test_ydotool_integration.py"
```

### Group 2: Integration Tests (after setup)
```
Task: "Integration test for eavesdrop client connection in tests/integration/test_client_connection.py"
Task: "Integration test for complete transcription workflow in tests/integration/test_end_to_end.py"
```

### Group 3: Core Models (after tests fail)
```
Task: "TextState data model in src/eavesdrop/active_listener/text_manager.py"
Task: "ConnectionState data model in src/eavesdrop/active_listener/client.py"
Task: "UpdateType enum in src/eavesdrop/active_listener/text_manager.py"
```

### Group 4: Unit Tests & Polish (after implementation)
```
Task: "Unit tests for text diffing algorithms in tests/unit/test_text_manager.py"
Task: "Unit tests for typing operations in tests/unit/test_typer.py"
Task: "Mock implementations for testing in tests/unit/test_mocks.py"
```

## Key Implementation Notes
- **python-ydotool**: Use `pydotool.init()` at startup, `type_string()` for typing, `key_sequence()` for backspace
- **Clypi CLI**: Use `Command` class with typed arguments: `host: str`, `port: int`, `audio_device: str`
- **EavesdropClient**: Configure with `send_last_n_segments=1`, `hotwords=["com"]`
- **Text Processing**: Implement prefix-matching diff algorithm for minimal typing operations
- **Error Recovery**: Handle connection loss, audio device unavailability, ydotool system issues

## Validation Checklist
*GATE: Checked by main() before returning*

- [x] All contracts have corresponding tests (T004-T006)
- [x] All entities have model tasks (T009-T012)
- [x] All tests come before implementation (T004-T008 before T009+)
- [x] Parallel tasks truly independent (different files)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task
- [x] Integration points covered (eavesdrop client, ydotool, CLI)
- [x] Error handling and edge cases addressed
- [x] Performance and resource usage validation included
