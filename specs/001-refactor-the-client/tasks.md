# Tasks: Refactor Client Project as Importable Library

**Input**: Design documents from `/home/shyndman/dev/projects/lmnop/eavesdrop/specs/001-refactor-the-client/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/, quickstart.md

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → Tech stack: Python 3.11+, websockets, sounddevice, numpy, pydantic, structlog, eavesdrop-wire
   → Structure: Single project (refactor within existing packages/client)
2. Load design documents:
   → data-model.md: EavesdropClient, ClientCallbacks, TranscriberOptions, SubscriberOptions
   → contracts/: EavesdropClient API contract definition
   → quickstart.md: Integration test scenarios for both modes
3. Generate tasks by category:
   → Setup: Remove CLI components, prepare library structure
   → Tests: Contract tests for API, integration tests for usage scenarios
   → Core: Refactor existing classes, implement new EavesdropClient
   → Integration: WebSocket protocol changes, message handling updates
   → Polish: Package exports, documentation, validation
4. Applied task rules:
   → Different files = mark [P] for parallel execution
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Tasks numbered T001-T027
6. Dependencies validated: Tests → Implementation → Polish
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Phase 3.1: Setup & Preparation
- [ ] T001 Remove CLI entry points from packages/client/src/eavesdrop/client/__main__.py
- [ ] T002 Remove CLI application logic from packages/client/src/eavesdrop/client/app.py  
- [ ] T003 Remove terminal interface from packages/client/src/eavesdrop/client/interface.py
- [ ] T004 [P] Update packages/client/pyproject.toml to remove CLI script entries and update dependencies

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Contract Tests (API Surface)
- [ ] T005 [P] Contract test EavesdropClient.transcriber() factory method in tests/contract/test_transcriber_factory.py
- [ ] T006 [P] Contract test EavesdropClient.subscriber() factory method in tests/contract/test_subscriber_factory.py  
- [ ] T007 [P] Contract test async iterator protocol (__aiter__/__anext__) in tests/contract/test_async_iterator.py
- [ ] T008 [P] Contract test async context manager protocol (__aenter__/__aexit__) in tests/contract/test_context_manager.py
- [ ] T009 [P] Contract test connection management (connect/disconnect) in tests/contract/test_connection_lifecycle.py
- [ ] T010 [P] Contract test transcriber streaming control (start_streaming/stop_streaming) in tests/contract/test_streaming_control.py
- [ ] T011 [P] Contract test status properties (is_connected/is_streaming) in tests/contract/test_status_properties.py

### Integration Tests (Usage Scenarios)
- [ ] T012 [P] Integration test basic transcriber mode with context manager in tests/integration/test_basic_transcriber.py
- [ ] T013 [P] Integration test advanced transcriber configuration in tests/integration/test_advanced_transcriber.py
- [ ] T014 [P] Integration test subscriber mode multi-stream monitoring in tests/integration/test_subscriber_mode.py
- [ ] T015 [P] Integration test audio device selection patterns in tests/integration/test_audio_device_selection.py
- [ ] T016 [P] Integration test error handling scenarios in tests/integration/test_error_handling.py
- [ ] T017 [P] Integration test manual connection lifecycle in tests/integration/test_manual_lifecycle.py

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Refactor Existing Components
- [ ] T018 Modify packages/client/src/eavesdrop/client/connection.py to support both ClientType.TRANSCRIBER and ClientType.RTSP_SUBSCRIBER
- [ ] T019 Update WebSocketConnection message processing to handle rich TranscriptionMessage callbacks instead of text-only
- [ ] T020 Remove stream filtering in WebSocketConnection message processing for multi-stream subscriber support
- [ ] T021 Add X-Stream-Names header support to WebSocketConnection for subscriber mode
- [ ] T022 Update packages/client/src/eavesdrop/client/audio.py to support enhanced device selection (index, name, substring)

### New EavesdropClient Implementation  
- [ ] T023 Create packages/client/src/eavesdrop/client/client.py with EavesdropClient class and factory methods
- [ ] T024 Implement async iterator protocol (__aiter__/__anext__) with internal message queue in EavesdropClient
- [ ] T025 Implement async context manager protocol (__aenter__/__aexit__) in EavesdropClient  
- [ ] T026 Implement connection management and mode-specific logic in EavesdropClient

## Phase 3.4: Integration & Polish
- [ ] T027 Update packages/client/src/eavesdrop/client/__init__.py to export EavesdropClient as main API

## Dependencies
- Setup (T001-T004) before all other tasks
- Contract tests (T005-T011) before implementation (T018-T026)  
- Integration tests (T012-T017) before implementation (T018-T026)
- T018-T022 (existing component modifications) before T023-T026 (new EavesdropClient)
- T023 blocks T024, T025, T026 (EavesdropClient base before protocols)
- All implementation before T027 (package exports)

## Parallel Execution Examples

### Phase 3.1 Parallel Cleanup
```bash
# T001, T002, T003 can run in parallel (different files)
# T004 runs separately (modifies pyproject.toml)
```

### Phase 3.2 Contract Tests (T005-T011)
```python
# Launch all contract tests together (different files):
Task: "Contract test EavesdropClient.transcriber() factory method in tests/contract/test_transcriber_factory.py"
Task: "Contract test EavesdropClient.subscriber() factory method in tests/contract/test_subscriber_factory.py" 
Task: "Contract test async iterator protocol in tests/contract/test_async_iterator.py"
Task: "Contract test async context manager protocol in tests/contract/test_context_manager.py"
Task: "Contract test connection management in tests/contract/test_connection_lifecycle.py"
Task: "Contract test streaming control in tests/contract/test_streaming_control.py"
Task: "Contract test status properties in tests/contract/test_status_properties.py"
```

### Phase 3.2 Integration Tests (T012-T017)  
```python
# Launch all integration tests together (different files):
Task: "Integration test basic transcriber mode in tests/integration/test_basic_transcriber.py"
Task: "Integration test advanced transcriber configuration in tests/integration/test_advanced_transcriber.py"
Task: "Integration test subscriber mode in tests/integration/test_subscriber_mode.py"
Task: "Integration test audio device selection in tests/integration/test_audio_device_selection.py" 
Task: "Integration test error handling in tests/integration/test_error_handling.py"
Task: "Integration test manual lifecycle in tests/integration/test_manual_lifecycle.py"
```

## Implementation Notes

### Key Refactoring Changes
1. **Remove CLI Components**: Tasks T001-T003 eliminate all terminal/CLI dependencies
2. **WebSocket Protocol Enhancement**: Tasks T018-T021 add dual client type support
3. **Message Processing Changes**: Remove text extraction, pass full TranscriptionMessage objects
4. **Audio Device Enhancement**: Task T022 adds flexible device selection
5. **New Library API**: Tasks T023-T026 implement the unified EavesdropClient interface

### Testing Strategy
- **Contract Tests**: Validate API surface matches specification exactly
- **Integration Tests**: Verify real usage scenarios work end-to-end  
- **TDD Enforcement**: All tests must be written and failing before implementation
- **Parallel Execution**: Maximize efficiency by running independent tests simultaneously

### File Structure Impact
```
packages/client/src/eavesdrop/client/
├── __init__.py          # T027: Updated exports
├── client.py           # T023-T026: New EavesdropClient
├── connection.py       # T018-T021: Enhanced WebSocketConnection  
├── audio.py           # T022: Enhanced AudioCapture
├── app.py             # T002: REMOVED
├── interface.py       # T003: REMOVED  
└── __main__.py        # T001: REMOVED

tests/
├── contract/          # T005-T011: API contract tests
└── integration/       # T012-T017: Usage scenario tests
```

## Validation Checklist
*GATE: Checked before execution*

- [x] All API methods from contracts/ have corresponding contract tests
- [x] All usage scenarios from quickstart.md have integration tests  
- [x] All tests come before implementation (TDD enforced)
- [x] Parallel tasks are truly independent (different files)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task
- [x] Dependencies properly sequenced: Setup → Tests → Implementation → Polish

## Success Criteria
1. All CLI components removed while preserving core functionality
2. Unified EavesdropClient API supporting both transcriber and subscriber modes
3. Full async iterator and context manager protocol compliance
4. Enhanced WebSocket connection supporting dual client types  
5. Rich TranscriptionMessage callbacks instead of text-only
6. Thread-safe operation through instance isolation
7. Comprehensive test coverage for all API contracts and usage scenarios