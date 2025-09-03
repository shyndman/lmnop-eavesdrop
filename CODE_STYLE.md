# Code Style Guide

## Code Quality Tools

### Formatting
```bash
ruff format
```

### Linting
```bash
ruff check
ruff check --fix  # for autofixes
```

### Type Checking
```bash
pyright
```

## Python Type System Guidelines

### Type Annotations
- **ALWAYS type everything, NEVER use Any**
- Use modern union syntax: `str | None` instead of `Optional[str]`
- Explicit type annotations for all class attributes and method parameters

```python
# Good
class RTSPAudioSource:
    def __init__(self, audio_queue: asyncio.Queue[bytes]) -> None:
        self.audio_queue: asyncio.Queue[bytes] = audio_queue
        self.closed: bool = False

    async def read_audio(self) -> np.ndarray | None:
        # Type guards for None checks
        if audio_data is not None:
            return audio_data
        return None

# Bad
class RTSPAudioSource:
    def __init__(self, audio_queue):  # Missing types
        self.audio_queue = audio_queue  # Missing type annotation
        self.closed = False  # Missing type annotation

    async def read_audio(self):  # Missing return type
        return audio_data  # No type safety
```

### Protocol Implementation
- Protocol inheritance must be explicit when implementing interfaces
- Use structural typing for interface compliance

```python
# Good
class RTSPTranscriptionSink(TranscriptionSink, Protocol):
    async def send_result(self, result: TranscriptionResult) -> None:
        try:
            # transcription work
        except Exception:
            self.logger.exception("Transcription failed")
```

### Type Safety Requirements
- Handle Iterable-to-list conversions explicitly
- Never assume return types without verification

```python
# CRITICAL: Whisper transcriber returns Iterable[Segment], not list
def _transcribe_audio(self, input_sample: np.ndarray) -> tuple[list[Segment] | None, TranscriptionInfo | None]:
    result, info = self.transcriber.transcribe(...)
    # MUST convert Iterable to list immediately
    result_list = list(result) if result else None
    return result_list, info
```

## Error Handling Patterns

### Exception Logging
- Use `.exception()` method for proper stack traces
- Don't use `.error()` for exceptions

```python
# Good
try:
    await self.processor.initialize()
except Exception:
    self.logger.exception("Failed to initialize processor")
    raise  # Stop client creation

# Bad
except Exception as e:
    self.logger.error(f"Failed to initialize: {e}")  # Missing stack trace
```

### Error Propagation
- Fail fast on critical initialization failures
- Use proper exception chaining

```python
# Good
try:
    await self.processor.initialize()
except Exception:
    self.logger.exception("Failed to initialize processor")
    raise  # Stop client creation

# Bad
except Exception:
    self.logger.exception("Failed to initialize processor")
    return None  # Swallow critical errors
```

## Async Patterns

### Task Management
- Always cancel and await tasks during cleanup
- Use proper task lifecycle management

```python
# Good
async def stop(self):
    self._exit = True
    await self.processor.stop_processing()

    # Must cancel and await all tasks
    if self._processing_task and not self._processing_task.done():
        self._processing_task.cancel()
        try:
            await self._processing_task
        except asyncio.CancelledError:
            pass

# Bad
async def stop(self):
    self._processing_task.cancel()  # No await - potential resource leak
```

### Task Coordination
- Use `asyncio.wait()` with `FIRST_COMPLETED` instead of polling
- Avoid `await asyncio.sleep()` loops

```python
# Good
tasks = [ffmpeg_task, audio_task, transcription_task]
done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
for task in pending:
    task.cancel()

# Bad
while not self.stopped:
    await asyncio.sleep(0.1)  # Polling instead of event-driven
```

## Class Design Patterns

### Constructor Type Safety
- All constructor parameters must be explicitly typed
- Use dataclasses for configuration objects

```python
# Good
def __init__(
    self,
    buffer: AudioStreamBuffer,
    sink: TranscriptionSink,
    config: TranscriptionConfig,
    stream_name: str,
    translation_queue: queue.Queue[dict] | None = None,
    logger_name: str = "transcription_processor",
) -> None:

# Bad
def __init__(self, buffer, sink, config, stream_name, **kwargs):  # Missing types
```

### Method Return Types
- Always specify return types, including `-> None`
- Use union types for nullable returns

```python
# Good
async def read_audio(self) -> np.ndarray | None:
    if data is None:
        return None
    return process_data(data)

def get_status(self) -> dict[str, str | int | bool]:
    return {"active": True, "count": 42}

# Bad
async def read_audio():  # Missing return type
    return data

def get_status():  # Missing return type
    return {"active": True}
```

## Threading and Concurrency

### Lock Management
- Use appropriate lock types for context (async vs sync)
- Never mix `threading.Lock()` with async code

```python
# Good - Async context
class AsyncModelManager:
    def __init__(self):
        self.model_lock = asyncio.Lock()

    async def get_model(self):
        async with self.model_lock:
            # async operations

# Bad - Mixed threading models
class MixedManager:
    def __init__(self):
        self.model_lock = threading.Lock()  # Wrong for async context
```

### Resource Management
- Use context managers for resource cleanup
- Ensure proper cleanup in finally blocks

```python
# Good
async with self.model_lock:
    # protected operations

try:
    # work
except Exception:
    self.logger.exception("Operation failed")
finally:
    await self.cleanup()
```
