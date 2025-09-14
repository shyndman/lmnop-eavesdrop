# [PROJECT NAME] Development Guidelines

Auto-generated from all feature plans. Last updated: [DATE]

## Active Technologies
[EXTRACTED FROM ALL PLAN.MD FILES]

## Project Structure
```
[ACTUAL STRUCTURE FROM PLANS]
```

## Commands
[ONLY COMMANDS FOR ACTIVE TECHNOLOGIES]

## Code Style
[LANGUAGE-SPECIFIC, ONLY FOR LANGUAGES IN USE]

### Python Type Safety (NON-NEGOTIABLE)
**Strict Typing Requirements**:
- NO `Any` types allowed - type everything explicitly
- Use modern union syntax: `str | None` instead of `Optional[str]`
- Use modern union syntax: `int | str` instead of `Union[int, str]`
- Use `TypedDict` for structured dictionaries with known keys
- Use `NamedTuple` subclasses for immutable data structures
- Always parameterize generics: `dict[str, int]`, `list[User]`, never bare `dict` or `list`

**Preferred Patterns**:
```python
# ✅ TypedDict for configuration/API data
class UserConfig(TypedDict):
    name: str
    email: str
    preferences: dict[str, str | int]
    metadata: dict[str, str] | None

# ✅ NamedTuple for immutable records, and convenient function return values
class UserRecord(NamedTuple):
    id: int
    name: str
    created_at: datetime
    active: bool = True

# ✅ Dataclass for mutable models
@dataclass
class User:
    name: str
    email: str
    age: int | None = None
    tags: list[str] = field(default_factory=list)
```

**Import Style**:
```python
from typing import TypedDict, NamedTuple, TYPE_CHECKING
# Never: from typing import Any, Optional, Union
```

## Recent Changes
[LAST 3 FEATURES AND WHAT THEY ADDED]

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->