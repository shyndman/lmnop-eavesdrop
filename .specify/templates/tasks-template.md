# Tasks: [FEATURE NAME]

**Input**: Design documents from `/specs/[###-feature-name]/`
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
   → Integration: DB, middleware, logging
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
   → All endpoints implemented?
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Single project**: `src/`, `tests/` at repository root
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume single project - adjust based on plan.md structure

## Phase 3.1: Setup
- [ ] T001 Create project structure per implementation plan
- [ ] T002 Initialize [language] project with [framework] dependencies
- [ ] T003 [P] Configure linting and formatting tools

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

**Type Checker vs Tests**: We use static type checking (Pylance/mypy) which already validates:
- Class/function existence
- Method signatures
- Import correctness
- Type compatibility

**Contract tests should test BEHAVIOR, not existence:**
- ❌ `assert hasattr(obj, 'method')` - Type checker handles this
- ❌ `assert callable(obj.method)` - Type checker handles this
- ❌ `assert isinstance(obj, SomeClass)` - Type checker handles this
- ✅ `assert obj.method(input) == expected_output` - Tests actual behavior
- ✅ `assert obj.validates_correctly(invalid_input) raises ValueError` - Tests contracts
- ✅ `assert complex_workflow() produces correct state changes` - Tests integration

**Tests should fail because functionality is unimplemented, not because classes don't exist.**

- [ ] T004 [P] Contract test POST /api/users in tests/contract/test_users_post.py
- [ ] T005 [P] Contract test GET /api/users/{id} in tests/contract/test_users_get.py
- [ ] T006 [P] Integration test user registration in tests/integration/test_registration.py
- [ ] T007 [P] Integration test auth flow in tests/integration/test_auth.py

## Phase 3.3: Core Implementation (ONLY after tests are failing)

**Python Type Safety Requirements (NON-NEGOTIABLE)**:
- NO `Any` types - type everything explicitly
- Use modern union syntax: `str | None` not `Optional[str]`
- Use `TypedDict` for structured dictionaries
- Use `NamedTuple` for immutable data structures
- Parameterize generics: `dict[str, int]` not `dict`

**Model Creation Examples**:
```python
# ✅ TypedDict for configuration/data structures
class UserConfig(TypedDict):
    name: str
    email: str
    preferences: dict[str, str | int]
    metadata: dict[str, str] | None

# ✅ NamedTuple for immutable records
class UserRecord(NamedTuple):
    id: int
    name: str
    created_at: datetime
    active: bool = True

# ✅ Pydantic/dataclass for mutable models
@dataclass
class User:
    name: str
    email: str
    age: int | None = None
    tags: list[str] = field(default_factory=list)
```

- [ ] T008 [P] User model in src/models/user.py (TypedDict/NamedTuple/dataclass - NO Any types)
- [ ] T009 [P] UserService CRUD in src/services/user_service.py (fully typed methods)
- [ ] T010 [P] CLI --create-user in src/cli/user_commands.py
- [ ] T011 POST /api/users endpoint
- [ ] T012 GET /api/users/{id} endpoint
- [ ] T013 Input validation
- [ ] T014 Error handling and logging

## Phase 3.4: Integration
- [ ] T015 Connect UserService to DB
- [ ] T016 Auth middleware
- [ ] T017 Request/response logging
- [ ] T018 CORS and security headers

## Phase 3.5: Polish
- [ ] T019 [P] Unit tests for validation in tests/unit/test_validation.py
- [ ] T020 Performance tests (<200ms)
- [ ] T021 [P] Update docs/api.md
- [ ] T022 Remove duplication
- [ ] T023 Run manual-testing.md

## Dependencies
- Tests (T004-T007) before implementation (T008-T014)
- T008 blocks T009, T015
- T016 blocks T018
- Implementation before polish (T019-T023)

## Parallel Example
```
# Launch T004-T007 together:
Task: "Contract test POST /api/users in tests/contract/test_users_post.py"
Task: "Contract test GET /api/users/{id} in tests/contract/test_users_get.py"
Task: "Integration test registration in tests/integration/test_registration.py"
Task: "Integration test auth in tests/integration/test_auth.py"
```

## Notes
- [P] tasks = different files, no dependencies
- Verify tests fail before implementing (due to missing functionality, not missing classes)
- Write meaningful behavioral tests, not existence checks
- Type checker validates structure; tests validate behavior
- Commit after each task
- Avoid: vague tasks, same file conflicts, `hasattr()` tests

## Task Generation Rules
*Applied during main() execution*

1. **From Contracts**:
   - Each contract file → contract test task [P]
   - Each endpoint → implementation task
   
2. **From Data Model**:
   - Each entity → model creation task [P]
   - Relationships → service layer tasks
   
3. **From User Stories**:
   - Each story → integration test [P]
   - Quickstart scenarios → validation tasks

4. **Ordering**:
   - Setup → Tests → Models → Services → Endpoints → Polish
   - Dependencies block parallel execution

## Validation Checklist
*GATE: Checked by main() before returning*

- [ ] All contracts have corresponding tests
- [ ] All entities have model tasks
- [ ] All tests come before implementation
- [ ] Parallel tasks truly independent
- [ ] Each task specifies exact file path
- [ ] No task modifies same file as another [P] task