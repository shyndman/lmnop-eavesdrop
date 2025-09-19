## Tooling (These run automatically on every commit, so make sure everything passes)

**Static Type Checking**:

```sh
pyright
```

**Formatting**:

```sh
ruff format
```

**Linting**:

```sh
ruff check
ruff check --fix  # for autofixes
```

## Python

### Type Safety (NON-NEGOTIABLE)

#### Strict Typing Requirements

- NO `Any` types allowed - type everything explicitly
- Use type annotations for all class attributes and method parameters
- Use modern union syntax: `str | None` instead of `Optional[str]`
- Use modern union syntax: `int | str` instead of `Union[int, str]`
- Use `TypedDict` for structured dictionaries with known keys
- Use `NamedTuple` subclasses for immutable data structures
- Always parameterize generics: `dict[str, int]`, `list[User]`, never bare `dict` or `list`
- Import `Iterable`, `Awaitable` from `collections.abc`, NOT from `typing`

```python
# ✅ Modern union syntax and proper imports
from collections.abc import Iterable, Awaitable, Mapping
from typing import TypedDict, Literal

def process_items(
    items: list[str | int],  # Modern union syntax
    config: dict[str, bool],  # Parameterized generic
    unique_ids: set[int],  # Built-in set generic
    optional_data: bytes | None = None,  # Modern optional
) -> tuple[str, int]:  # Built-in generic
    # Implementation...

async def fetch_data(sources: Iterable[str]) -> Awaitable[dict[str, int]]:
    # Proper collections.abc imports
    pass

# ❌ Old syntax - FORBIDDEN
from typing import Optional, Union, List, Dict, Tuple, Set, Iterable, Awaitable

def process_items_old(
    items: List[Union[str, int]],  # Old union and generic syntax
    config: Dict[str, bool],
    unique_ids: Set[int],  # Old Set syntax
    optional_data: Optional[bytes] = None,
) -> Tuple[str, int]:
    pass
```

#### Duck Typing is FORBIDDEN

- NO `getattr()`, `hasattr()`, or `setattr()` usage - these bypass type safety
- NO duck typing patterns - always use proper types and the `.` operator
- If you need dynamic attribute access, redesign with proper types (TypedDict, dataclass, etc.)
- If you truly need dynamic attribute access, it requires a discussion and a sign off

#### Testing Type Safety (THESE WILL GET YOUR PR REJECTED)

- If I find a single test that interrogates the shape of an object, heads will roll
- NO tests that check for attribute existence with `hasattr()` or `getattr()`
- NO tests that verify object structure instead of behavior
- NO `isinstance()` checks in tests unless testing error conditions
- NO `type()` comparisons in production or test code

```python
# ❌ Shape interrogation - FORBIDDEN
def test_processor_has_required_methods():
    processor = AudioProcessor()
    assert hasattr(processor, 'process')
    assert hasattr(processor, 'cleanup')
    assert callable(getattr(processor, 'process'))

# ❌ Structure testing - FORBIDDEN
def test_user_object_structure():
    user = create_user()
    assert isinstance(user.id, int)
    assert isinstance(user.name, str)
    assert type(user.created_at) == datetime

# ❌ Duck typing validation - FORBIDDEN
def test_can_process_different_types():
    for obj in [AudioProcessor(), TextProcessor(), VideoProcessor()]:
        if hasattr(obj, 'process'):
            result = obj.process(test_data)
            assert result is not None

# ✅ Behavior testing - type safety enforced by static analysis
def test_audio_processor_transforms_segments():
    processor = AudioProcessor()
    segment = AudioSegment(data=b"test", sample_rate=44100)

    result = processor.process(segment)  # Type-safe call

    assert result.sample_rate == 44100
    assert len(result.data) > 0

# ✅ Protocol compliance - tested through behavior, not shape
def test_processors_implement_cleanup():
    processors: list[Processor] = [
        AudioProcessor(),
        TextProcessor(),
        VideoProcessor()
    ]

    for processor in processors:
        processor.cleanup()  # Type system guarantees this exists
        # Test the behavior/side effects, not the method existence

# ✅ Error condition testing with isinstance - ONLY acceptable use
def test_invalid_input_raises_type_error():
    processor = AudioProcessor()

    with pytest.raises(TypeError) as exc_info:
        processor.process("invalid_input")  # type: ignore

    # Only check isinstance for error validation
    assert isinstance(exc_info.value, TypeError)
```

```python
# ❌ Duck typing - bypasses type safety
if hasattr(obj, 'process'):
    result = getattr(obj, 'process')(data)

# ✅ Proper typing with protocols or unions
from typing import Protocol

class Processor(Protocol):
    def process(self, data: str) -> str: ...

def handle_processor(processor: Processor, data: str) -> str:
    return processor.process(data)  # Type-safe access
```

#### Configuration Handling (FAIL FAST, NO COERCION)

- Configuration must be loaded into Pydantic models with strict validation
- NO silent type coercion - if the input is wrong, fail immediately
- NO default value fallbacks for required configuration
- NO string-to-type conversion beyond what Pydantic provides by default
- Command line arguments and config files must validate to the same strict schema

```python
# ✅ Strict configuration with Pydantic - fails fast on invalid input
from pydantic import BaseModel, Field, ValidationError
from pathlib import Path

class ServerConfig(BaseModel):
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)  # No coercion from strings
    debug: bool  # No coercion from "true"/"false" strings
    workers: int = Field(ge=1)
    timeout: float = Field(gt=0.0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    data_dir: Path

# ❌ Silent coercion and loose validation - FORBIDDEN
def load_config_loose(raw_config: dict[str, str]) -> dict[str, Any]:
    return {
        'port': int(raw_config.get('port', '8000')),  # Silent string->int
        'debug': raw_config.get('debug', 'false').lower() == 'true',  # String->bool
        'workers': max(1, int(raw_config.get('workers', '1'))),  # Coercion with fallback
        'timeout': float(raw_config.get('timeout', '30.0')),  # Silent conversion
    }

# ✅ Strict validation - fail immediately on bad input
def load_config_strict(config_path: Path) -> ServerConfig:
    try:
        with open(config_path) as f:
            return ServerConfig.model_validate_json(f.read(), strict=True)
    except ValidationError:
        logger.exception(f"Invalid configuration in {config_path}")
        raise  # Fail fast - don't start with bad config
    except FileNotFoundError:
        logger.exception(f"Configuration file not found: {config_path}")
        raise  # Required config missing - fail fast

# ✅ Command line parsing with same strict validation
def parse_cli_args() -> ServerConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--workers', type=int, required=True)

    args = parser.parse_args()

    try:
        # Same validation as file config - no special CLI handling
        return ServerConfig.model_validate(vars(args))
    except ValidationError:
        logger.exception("Invalid command line arguments")
        raise SystemExit(1)  # Fail fast on startup
```

### Base classes and when to use them

```python
from datetime import datetime
from typing import NamedTuple, TypedDict, NotRequired
from dataclasses import field
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass

# ✅ BaseModel for API boundaries and data validation
class UserCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(pattern=r'^[^@]+@[^@]+\.[^@]+$')
    age: int = Field(ge=0, le=150)
    preferences: dict[str, str | int] = Field(default_factory=dict)

    # Provides: automatic validation on creation, built-in JSON
    # serialization/deserialization, integration with FastAPI/SQLAlchemy

# ✅ TypedDict for internal data structures - fast, typed, but limited functionality
class UserSegment(TypedDict):
    name: str
    email: str
    preferences: dict[str, str | int]
    metadata: NotRequired[dict[str, str] | None]  # Optional field

# ✅ NamedTuple for immutable records or structured return values
class UserRecord(NamedTuple):
    id: int
    name: str
    created_at: datetime
    active: bool = True

# ✅ Pydantic dataclass for rich data models with validation
@dataclass
class User:
    name: str
    age: int | None = None
    email: NotRequired[str] = None  # Optional field with NotRequired
    tags: list[str] = field(default_factory=list)
```

#### Imports

```python
# ✅ Always use absolute imports
from eavesdrop.common.types import AudioSegment
from eavesdrop.client.core import Client

# ❌ Never use relative imports
from ..common.types import AudioSegment
from .core import Client

# ✅ Allowed typing imports
from typing import TypedDict, NamedTuple, TYPE_CHECKING

# ✅ Import abstract base classes from collections.abc
from collections.abc import Iterable, Awaitable

# ❌ Never import these from typing
from typing import Any, Optional, Union, Iterable, Awaitable
# Any: lazy typing - always type specifically
# Optional[T]: use T | None syntax instead
# Union[A, B]: use A | B syntax instead
# Iterable/Awaitable: import from collections.abc instead
```

#### Errors

**Propagation**:

- CORE TENET: Fail fast on critical initialization failures
- Use proper exception chaining

```python
## Good
try:
    await self.processor.initialize()
except Exception:
    self.logger.exception("Failed to initialize processor")
    raise  # Stop client creation

## Bad
except Exception:
    self.logger.exception("Failed to initialize processor")
    return None  # Swallow critical errors
```

**Logging**:

- COMMON MISTAKE: Use `.exception()` method for proper stack traces
- Don't use `.error()` for exceptions

```python
## Good
try:
    await self.processor.initialize()
except Exception:
    self.logger.exception("Failed to initialize processor")
    raise  # Stop client creation

## Bad
except Exception as e:
    self.logger.error(f"Failed to initialize: {e}")  # Missing stack trace
```

### Documentation

#### Docstrings (Sphinx/reStructuredText Format)

All public functions, classes, and methods must have comprehensive docstrings using Sphinx-style reStructuredText format.

```python
# ✅ Proper Sphinx-style docstring
def process_audio_segment(segment: AudioSegment, sample_rate: int) -> ProcessedSegment:
    """Process an audio segment with noise reduction and normalization.

    :param segment: The audio data to process
    :type segment: AudioSegment
    :param sample_rate: Target sample rate in Hz, must be > 0
    :type sample_rate: int
    :return: Processed audio segment with metadata
    :rtype: ProcessedSegment
    :raises ValueError: If sample_rate is <= 0

    Example:
        >>> result = process_audio_segment(segment, 44100)
    """
    # Implementation...

# ✅ Class docstring
class AudioProcessor:
    """High-performance audio processing pipeline.

    :param buffer_size: Internal buffer size in samples
    :type buffer_size: int
    :param enable_stats: Whether to collect processing statistics
    :type enable_stats: bool
    """
    def __init__(self, buffer_size: int = 1024, enable_stats: bool = False) -> None:
        # Implementation...

# ❌ Insufficient - REJECTED
def process_audio(data, rate):
    """Process audio."""  # Too brief, missing params/types

# ❌ Wrong format - REJECTED
def convert_format(segment: AudioSegment) -> bytes:
    """Convert audio segment to bytes.

    Args:  # Google/NumPy style forbidden
        segment: Audio to convert
    """
```
