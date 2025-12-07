# Eavesdrop Issues

Issues identified during code review of recent changes (2025-12-06).

## Summary

| # | Severity | File | Issue | 🗸 |
|---|----------|------|-------|---|
| 1 | **Critical** | `workspace.py` | Missing `self._completed` attribute — will crash at runtime | 🗸 |
| 2 | **Critical** | `ui_channel.py` | stdout/stderr confusion — race condition and wrong stream | 🗸 |
| 3 | **Critical** | `ui_channel.py` | JSON serialization broken — will raise `TypeError` | 🗸 |
| 4 | Medium | `ui_channel.py` | Unused stdout pipe | 🗸 |
| 5 | Low | `ui_messages.py` | `SET_SEGMENTS` enum declared but not implemented | 🗸 |
| 6 | Low | `ui_channel.py` | Type narrowing issue with `UIMessage` union | 🗸 |
| 7 | Low | `workspace.py` | Pyright reports potential `None` access | 🗸 |
| 8 | Low | `client.py` | Optional member access after assignment | 🗸 |
| 9 | Low | `workspace.py` | Pyright doesn't recognize Pydantic dataclass fields | 🗸 |
| 10 | Low | `workspace.py` | `Mapping` type used for mutable dict | 🗸 |
| 11 | Low | `app.py` | Misleading comment about cleanup order | 🗸 |
| 12 | **High** | `text_manager.py` | Potentially dead/orphaned code | 🗸 |
| 13 | Medium | `text_manager.py` | Assert used for control flow | 🗸 |
| 14 | Low | `text_manager.py` | Uses `warn()` for debug logging | 🗸 |
| 15 | Low | `__main__.py` | Redundant assert + if-check pattern | 🗸 |
| 16 | Low | `__main__.py` | Misplaced docstring | 🗸 |
| 17 | **High** | `typist.py` | Typing logic commented out — completely non-functional | 🗸 |
| 18 | Low | `typist.py` | Mutates input parameter | 🗸 |
| 19 | **High** | `typist.py` | Orphaned from new implementation | 🗸 |
| 20 | Medium | `core.py` | Uses `Any` type — violates project rules | |
| 21 | Medium | `core.py` | Recursive `__anext__` — stack overflow risk | 🗸 |
| 22 | Medium | `core.py` | Silent message drops when queue full | |
| 23 | Medium | `audio.py` | Silent audio frame drops | |
| 24 | Low | `core.py` | Google-style docstrings instead of reStructuredText | |
| 25 | Low | `CODE_STYLE.md` | Typos: "all the type" → "time", "Tat" → "That" | 🗸 |
| 26 | Medium | `test_mocks.py` | Uses `Any` type — violates project rules | |
| 27 | Medium | Multiple files | Generic `Exception` instead of specific types | |
| 28 | **High** | `test_text_manager.py` | Tests validate dead code, active code untested | 🗸 |
| 29 | Low | Multiple files | Bare `except Exception:` swallows errors | |
| 30 | Medium | `app.py` | `asyncio.create_task` from signal handler — unreliable | |
| 31 | Low | `ui_messages.py` | Line too long (ruff E501) | 🗸 |
| 32 | **High** | `test_cli_interface.py` | Tests missing required `--ui-bin` arg — will fail | 🗸 |
| 33 | **High** | `test_ydotool_integration.py` | Tests patch commented-out code — false positives | 🗸 |
| 34 | Medium | Multiple packages | Inconsistent default port (9090 vs 8080) | |
| 35 | **High** | Contract tests | All contract tests test dead code architecture | 🗸 |
| 36 | Medium | `__main__.py` | `isinstance` checks violate CODE_STYLE.md | |
| 37 | **Critical** | `audio_flow.py` | Inverted condition logic — sends disconnect when closed | 🗸 |
| 38 | **High** | `server.py` | Dead code — method reference without call | 🗸 |
| 39 | Medium | `websocket.py` | Returns `False` instead of `None` — unconventional | 🗸 |
| 40 | Medium | `server.py`, `rtsp/client.py` | Assert used for runtime validation | |
| 41 | Low | `__main__.py` (server) | Missing return type annotations | 🗸 |
| 42 | Medium | `buffer.py` | Race condition — `processed_duration` property lacks lock | 🗸 |
| 43 | Medium | `processor.py` | Misleading code — `np.frombuffer` on already-ndarray data | 🗸 |
| 44 | Medium | `processor.py` | Bare `list` type used without element type annotation | |
| 45 | Medium | `manager.py` | `getattr` fallbacks suggest missing/uncertain attributes | |
| 46 | Low | `manager.py` | `asyncio.Task` without type parameter | 🗸 |
| 47 | **High** | `rtsp/client.py` | Buffer not reset on reconnection — stale audio state | 🗸 |
| 48 | Medium | `subscriber.py`, `manager.py` | Duplicate logger name "rtsp/mgr" — ambiguous logs | 🗸 |
| 49 | Medium | `subscriber.py` | `# type: ignore` suppressing type error | |
| 50 | Medium | `subscriber.py` | Bare `list` type without element annotation | |
| 51 | Low | `pipeline.py` | Dead code — `_TranscribeContext.anomaly_detector` never used | 🗸 |

### Critical Issues (will crash at runtime)
- ~~**#1, #2, #3**: These will cause the application to fail immediately when exercised~~
- ~~**#37**: Inverted condition sends disconnect message when connection is already closed~~

### High Priority (major functionality problems)
- ~~**#12, #17, #19, #28, #33, #35**: Dead code from incomplete refactor — deleted~~
- ~~**#32**: Tests missing required `--ui-bin` arg~~
- ~~**#38**: Dead code suggests incomplete edit in health check handler~~
- ~~**#47**: RTSP reconnection reuses stale buffer with old timestamps/audio~~

### Medium Priority (latent bugs, policy violations)
- **#20, #26**: `Any` type usage violates CLAUDE.md rules
- ~~**#21**: Stack overflow after ~8 minutes of no messages~~
- **#22, #23**: Silent data loss with no indication
- **#27**: Generic exceptions make debugging harder
- **#30**: Signal handler task creation is unreliable
- **#34**: Default port mismatch causes connection failures
- **#36**: Dynamic type checks violate CODE_STYLE.md
- ~~**#39**: Unconventional `False` return instead of `None` can cause subtle bugs~~
- **#40**: Assert statements removed with `-O` flag causing silent failures
- ~~**#42**: Race condition in buffer property access~~
- ~~**#43**~~, **#44**: Type safety violations in processor
- **#45**: `getattr` usage suggests uncertain object structure
- ~~**#48**: Duplicate logger names make debugging ambiguous~~
- **#49, #50**: Type safety violations in subscriber

---

## 1. `workspace.py` — Missing `self._completed` attribute (Critical)

**Location:** `packages/active-listener/src/eavesdrop/active_listener/workspace.py`

**Problem:** The `TextTranscriptionWorkspace` class defines `self._completed_by_mode` as a per-mode dictionary in `__init__`, but multiple methods reference the non-existent `self._completed` attribute.

**Affected lines:**
- Line 122: `s.id not in self._completed`
- Line 137: `self._completed.update(...)`
- Line 151: `self._completed.values()`

**Fix:** Replace `self._completed` with `self._completed_by_mode[self._current_mode]` in all three locations.

---

## 2. `ui_channel.py` — stdout/stderr confusion (Critical)

**Location:** `packages/active-listener/src/eavesdrop/active_listener/ui_channel.py`

**Problem:** The `_monitor_ready_signal` method's docstring claims it monitors **stdout** for the ready signal, but the implementation reads from **stderr**:
- Line 101: Checks `self._process.stderr`
- Line 116: Reads from `self._process.stderr`

Additionally, `asyncio.gather()` on lines 80-84 runs both `_monitor_ready_signal()` and `_monitor_stderr()` concurrently — both reading from stderr creates a race condition where they steal lines from each other.

**Fix:** Decide which stream the UI sends its ready signal on, then:
- If stdout: Change `_monitor_ready_signal` to read from `self._process.stdout`
- If stderr: Update the docstring and remove `_monitor_stderr` from the gather (or have it start after ready signal received)

---

## 3. `ui_channel.py` — JSON serialization broken (Critical)

**Location:** `packages/active-listener/src/eavesdrop/active_listener/ui_channel.py`, line 182

**Problem:** Uses `json.dumps(message)` directly on a Pydantic dataclass:

```python
json_line = json.dumps(message) + "\n"
```

Pydantic dataclasses are not JSON-serializable via `json.dumps()`. This will raise `TypeError: Object of type AppendSegmentsMessage is not JSON serializable`.

**Fix:** Use the `serialize_ui_message()` function from `ui_messages.py`:

```python
from eavesdrop.active_listener.ui_messages import serialize_ui_message
# ...
json_line = serialize_ui_message(message) + "\n"
```

---

## 4. `ui_channel.py` — Unused stdout pipe

**Location:** `packages/active-listener/src/eavesdrop/active_listener/ui_channel.py`, line 71

**Problem:** The UI subprocess is launched with `stdout=asyncio.subprocess.PIPE` but stdout is never read — all reading happens on stderr. If the UI sends output to stdout, it will eventually block when the pipe buffer fills.

**Fix:** Either:
- Read from stdout if that's where the ready signal comes from
- Use `stdout=asyncio.subprocess.DEVNULL` if stdout isn't needed

---

## 5. `ui_messages.py` — `SetSegmentsMessage` declared but not defined

**Location:** `packages/active-listener/src/eavesdrop/active_listener/ui_messages.py`

**Problem:** `UIMessageType.SET_SEGMENTS = "set_segments"` is declared in the enum (line 36), but there is no corresponding `SetSegmentsMessage` dataclass defined, and it's not included in the `UIMessage` union type.

**Fix:** Either:
- Remove `SET_SEGMENTS` from the enum if not needed
- Implement the `SetSegmentsMessage` dataclass and add it to the union

---

## 6. `ui_channel.py` — Type narrowing issue with `UIMessage` union

**Location:** `packages/active-listener/src/eavesdrop/active_listener/ui_channel.py`, line 188

**Problem:** Pyright reports: `Cannot access attribute "type" for class "UIMessage"`. The `UIMessage` is a union of Pydantic dataclasses, and while each variant has a `type` attribute, pyright doesn't see it as a common attribute across the union without proper type narrowing.

```python
self.logger.debug("Sent message to UI", message_type=message.type)
```

**Fix:** Either:
- Use `getattr(message, "type")`
- Define a `Protocol` with a `type` attribute that all message types implement
- Access it after narrowing the type with `isinstance()` checks

---

## 7. `workspace.py` — Potential `None` access on `in_progress.text`

**Location:** `packages/active-listener/src/eavesdrop/active_listener/workspace.py`, line 225

**Problem:** Pyright reports: `"text" is not a known attribute of "None"`. The code accesses `in_progress.text` in the logger call, but the function signature allows `in_progress: Segment | None`:

```python
def _send_append_segments_message(
    self, newly_completed: list[Segment], in_progress: Segment | None
) -> None:
    # ...
    self.logger.debug(
        "Sent UI update",
        newly_completed_count=len(newly_completed),
        has_in_progress=in_progress.text != "",  # <-- in_progress could be None here
    )
```

While the code does assign a fallback segment when `in_progress is None` (lines 198-211), this happens BEFORE the logger call, so `in_progress` is shadowed and is never `None` at line 225. However, pyright doesn't track this reassignment across the control flow.

**Fix:** Either:
- Rename the fallback to a different variable to avoid confusion
- Add an `assert in_progress is not None` before the logger call
- Use a conditional: `has_in_progress=in_progress.text != "" if in_progress else False`

---

## 8. `client.py` — Optional member access after assignment

**Location:** `packages/active-listener/src/eavesdrop/active_listener/client.py`, line 43

**Problem:** Pyright reports: `"connect" is not a known attribute of "None"`. The code assigns `self._client` on line 42, then immediately calls `await self._client.connect()` on line 43:

```python
self._client = self._create_client()
await self._client.connect()  # pyright thinks self._client could be None
```

Pyright doesn't understand that `_create_client()` always returns a non-None value because `self._client` is typed as `EavesdropClient | None`.

**Fix:** Either:
- Use a local variable: `client = self._create_client(); await client.connect(); self._client = client`
- Add `assert self._client is not None` before the connect call

---

## 9. `workspace.py` — Pyright doesn't recognize Pydantic dataclass field parameters

**Location:** `packages/active-listener/src/eavesdrop/active_listener/workspace.py`, lines 214-216, 277

**Problem:** Pyright reports "No parameter named" errors for `target_mode`, `completed_segments`, `in_progress_segment` when instantiating Pydantic dataclasses. This occurs because pyright doesn't fully understand `pydantic.dataclasses.dataclass` with `Field()` descriptors without the pydantic plugin.

```python
message = AppendSegmentsMessage(
    target_mode=self._current_mode,          # "No parameter named"
    completed_segments=newly_completed,       # "No parameter named"
    in_progress_segment=in_progress,          # "No parameter named"
)
```

**Fix:** Configure pyright to use the Pydantic plugin in `pyrightconfig.json`:

```json
{
  "typeCheckingMode": "strict",
  "plugins": ["pydantic"]
}
```

Or in `pyproject.toml`:

```toml
[tool.pyright]
plugins = ["pydantic"]
```

---

## 10. `workspace.py` — Immutable `Mapping` type used for mutable dict

**Location:** `packages/active-listener/src/eavesdrop/active_listener/workspace.py`, line 59

**Problem:** The `_text_by_mode` attribute is typed as `Mapping[Mode, StringIO]`, but `Mapping` is an abstract read-only type. While this works at runtime (a `dict` is assigned), it signals incorrect intent — the code never mutates the mapping itself, only the `StringIO` values.

```python
self._text_by_mode: Mapping[Mode, StringIO] = {
    Mode.TRANSCRIBE: StringIO(),
    Mode.COMMAND: StringIO(),
}
```

**Recommendation:** This is technically fine but could be clearer. Consider using `dict[Mode, StringIO]` if you ever need to add/remove modes, or keep `Mapping` if read-only access is intentional.

---

## 11. `app.py` — Misleading comment about cleanup order

**Location:** `packages/active-listener/src/eavesdrop/active_listener/app.py`, lines 186-192

**Problem:** The comment on line 190 says "Shutdown UI subprocess first" but the code actually shuts down the **client** first (line 187), then the UI (line 191):

```python
# Shutdown client connection
await self._client.shutdown()       # <-- client first
self.logger.info("Client shutdown complete")

# Shutdown UI subprocess first      # <-- misleading comment
await self._ui_channel.shutdown()   # <-- UI second
```

**Fix:** Either:
- Update the comment to match the code: "Shutdown UI subprocess"
- Reorder operations if UI should actually shutdown first

---

## 12. `text_manager.py` — Potentially dead code

**Location:** `packages/active-listener/src/eavesdrop/active_listener/text_manager.py`

**Problem:** This module implements `TextState` with its own segment tracking and diffing logic, but `workspace.py` has its own parallel implementation with `_completed_by_mode`, `_in_progress`, and text buffering via `StringIO`. Neither module imports from the other.

This suggests either:
1. `text_manager.py` is legacy code that was replaced by `workspace.py`
2. They were intended to work together but integration was never completed

**Recommendation:** Determine which implementation is canonical and remove the other, or integrate them if both are needed.

---

## 13. `text_manager.py` — Assert used for control flow validation

**Location:** `packages/active-listener/src/eavesdrop/active_listener/text_manager.py`, line 89

**Problem:** Uses `assert` to validate a runtime condition:

```python
if not segment.completed and self.current_segment:
    logger.warn("Updated in-progress")
    assert self.current_segment.id == segment.id  # <-- dangerous
```

`assert` statements are removed when Python runs with `-O` (optimize) flag, which would cause this validation to silently disappear in production.

**Fix:** Replace with explicit validation:

```python
if self.current_segment.id != segment.id:
    raise ValueError(f"Segment ID mismatch: expected {self.current_segment.id}, got {segment.id}")
```

---

## 14. `text_manager.py` — Uses `logger.warn()` for debug logging

**Location:** `packages/active-listener/src/eavesdrop/active_listener/text_manager.py`, lines 74, 79, 88, 96

**Problem:** Uses `logger.warn()` for what appears to be debug/trace logging:

```python
logger.warn("Still completed")
logger.warn("New completed", previous_in_progress=self.current_segment)
logger.warn("Updated in-progress")
logger.warn("New in-progress")
```

These aren't warnings — they're normal flow events. This pollutes logs with false warnings.

**Fix:** Use `logger.debug()` or `logger.info()` for normal flow events.

---

## 15. `__main__.py` — Redundant assert + if-check pattern

**Location:** `packages/active-listener/src/eavesdrop/active_listener/__main__.py`

**Problem:** Two functions have redundant validation — an `assert` immediately followed by an `if` check for the same condition:

```python
# parse_ui_binary (lines 49-52):
assert isinstance(value, str), "UI binary path must be a string"

if not isinstance(value, str):  # <-- unreachable if assert passes
    raise ValueError("--ui-bin must be a string")

# parse_server (lines 76-80):
assert isinstance(value, str), "Server must be a string..."

if not isinstance(value, str):  # <-- unreachable if assert passes
    raise ValueError("Server must be a string")
```

The `if` block is unreachable because if the assert fails, the program crashes before reaching it.

**Fix:** Remove the redundant `if` checks since `assert` handles the validation (or remove the `assert` if you want the gentler `ValueError`).

---

## 16. `__main__.py` — Misplaced docstring in `parse_server`

**Location:** `packages/active-listener/src/eavesdrop/active_listener/__main__.py`, lines 75-78

**Problem:** The docstring appears after a statement, making it a no-op string literal instead of an actual docstring:

```python
def parse_server(value: str | list[str]) -> ServerHostPort:
    assert isinstance(value, str), "Server must be a string in hostname:port format"

    """Parse server argument in hostname:port format."""  # <-- not a docstring!
```

**Fix:** Move the docstring to immediately after the function signature:

```python
def parse_server(value: str | list[str]) -> ServerHostPort:
    """Parse server argument in hostname:port format."""
    assert isinstance(value, str), "Server must be a string in hostname:port format"
```

---

## 17. `typist.py` — Typing logic is commented out (non-functional)

**Location:** `packages/active-listener/src/eavesdrop/active_listener/typist.py`, lines 61-62, 78-79

**Problem:** The actual typing functionality is commented out, replaced with `pass`:

```python
def type_text(self, text: str) -> None:
    # ...
    try:
        # pydotool.type_string(text)
        pass  # <-- does nothing

def delete_characters(self, count: int) -> None:
    # ...
    for _ in range(count):
        # pydotool.key_combination([pydotool.KEY_BACKSPACE])
        pass  # <-- does nothing
```

The `YdoToolTypist` class initializes ydotool but never actually sends any keystrokes.

**Fix:** Uncomment the pydotool calls or implement the intended functionality.

---

## 18. `typist.py` — Mutates input parameter

**Location:** `packages/active-listener/src/eavesdrop/active_listener/typist.py`, lines 100, 104

**Problem:** The method mutates the `operation` parameter that was passed in:

```python
def execute_typing_operation(self, operation: TypingOperation) -> bool:
    # ...
    operation.completed = True   # line 100 - mutating input
    return True
except Exception:
    operation.completed = False  # line 104 - mutating input
    return False
```

This is a side-effect that can cause subtle bugs. The caller may not expect the object to be modified.

**Recommendation:** Either:
- Return a new `TypingOperation` with the updated status
- Document clearly that the method modifies the input object
- Use a return value to communicate completion status (which is already done via `bool`)

---

## 19. `typist.py` — Uses `text_manager.py` despite `workspace.py` having separate implementation

**Location:** `packages/active-listener/src/eavesdrop/active_listener/typist.py`, line 11

**Problem:** `typist.py` imports from `text_manager.py`:

```python
from eavesdrop.active_listener.text_manager import TypingOperation
```

But `workspace.py` doesn't use `text_manager.py` at all — it has its own segment tracking. And `app.py` doesn't use `typist.py`.

This suggests incomplete refactoring where:
- `typist.py` + `text_manager.py` are the old implementation
- `workspace.py` + `ui_channel.py` are the new implementation
- They haven't been fully integrated

**Recommendation:** Determine which implementation to use and consolidate.

---

## 20. `core.py` — Uses `Any` type (violates project rules)

**Location:** `packages/client/src/eavesdrop/client/core.py`, line 11

**Problem:** Per CLAUDE.md, "We never use the `any` type in Typescript, or 'Any' in Python. Never."

```python
from typing import Any
# ...
self._background_tasks: set[asyncio.Task[Any]] = set()  # line 68
```

**Fix:** Use a more specific type. For tasks that return nothing, use `asyncio.Task[None]`:

```python
self._background_tasks: set[asyncio.Task[None]] = set()
```

---

## 21. `core.py` — Recursive `__anext__` risks stack overflow

**Location:** `packages/client/src/eavesdrop/client/core.py`, line 315

**Problem:** The async iterator recursively calls itself on timeout:

```python
async def __anext__(self) -> TranscriptionMessage:
    try:
        message = await asyncio.wait_for(self._message_queue.get(), timeout=0.5)
        return message
    except asyncio.TimeoutError:
        if self._connected:
            return await self.__anext__()  # <-- recursive call
```

If the client stays connected but receives no messages, this will build up stack frames until hitting Python's recursion limit (typically 1000). At 0.5s per timeout, that's ~8 minutes to crash.

**Fix:** Use a loop instead of recursion:

```python
async def __anext__(self) -> TranscriptionMessage:
    while self._connected:
        try:
            return await asyncio.wait_for(self._message_queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
    raise StopAsyncIteration
```

---

## 22. `core.py` — Silent message drops when queue is full

**Location:** `packages/client/src/eavesdrop/client/core.py`, lines 341-343

**Problem:** Messages are silently dropped when the queue is full:

```python
def _on_transcription_message(self, message: TranscriptionMessage) -> None:
    try:
        self._message_queue.put_nowait(message)
    except asyncio.QueueFull:
        pass  # <-- silent data loss
```

There's no logging, no counter, no way to know transcription messages were lost.

**Fix:** At minimum, log when messages are dropped:

```python
except asyncio.QueueFull:
    logger.warning("Dropping transcription message - queue full")
```

Or use a bounded queue with a backpressure strategy.

---

## 23. `audio.py` — Silent audio frame drops

**Location:** `packages/client/src/eavesdrop/client/audio.py`, lines 38-39

**Problem:** Same issue as #22 — audio frames are silently dropped:

```python
try:
    self.audio_queue.put_nowait(audio_data.tobytes())
except asyncio.QueueFull:
    pass  # <-- silent data loss
```

Lost audio frames mean gaps in transcription with no indication to the user.

**Fix:** Log dropped frames or implement backpressure.

---

## 24. `core.py` — Uses Google-style docstrings instead of reStructuredText

**Location:** `packages/client/src/eavesdrop/client/core.py`, multiple methods

**Problem:** Per CLAUDE.md, the project uses reStructuredText format for docstrings. `core.py` uses Google-style:

```python
def transcriber(...):
    """
    Create a transcriber client for sending audio for transcription.

    Args:                              # <-- Google style
        host: Server hostname
        port: Server port

    Returns:                           # <-- Google style
        Configured EavesdropClient
    """
```

**Fix:** Convert to reStructuredText format:

```python
def transcriber(...):
    """Create a transcriber client for sending audio for transcription.

    :param host: Server hostname
    :type host: str
    :param port: Server port
    :type port: int
    :returns: Configured EavesdropClient in transcriber mode
    :rtype: EavesdropClient
    """
```

---

## 25. `CODE_STYLE.md` — Typos in newly added text

**Location:** `CODE_STYLE.md`, lines 3-4

**Problem:** The newly added text contains typos:

```markdown
And after you do, internalize this: We use types ALL the type. Because we do, we ALWAYS
trust the compiler, and never write ridiculous amateurish dynamic type testing code. Tat
kind of coding is STRICTLY forbidden.
```

- "all the type" should be "all the time"
- "Tat" should be "That"

**Fix:**
```markdown
And after you do, internalize this: We use types ALL the time. Because we do, we ALWAYS
trust the compiler, and never write ridiculous amateurish dynamic type testing code. That
kind of coding is STRICTLY forbidden.
```

---

## 26. `test_mocks.py` — Uses `Any` type (violates project rules)

**Location:** `packages/active-listener/tests/unit/test_mocks.py`, lines 5, 96, 97, 127, 131, 181, 185

**Problem:** Test code uses `Any` type throughout, violating CLAUDE.md rules:

```python
from typing import Any  # line 5

self.message_queue: list[Any] = []        # line 96
self.on_message_callback: Any = None       # line 97
def add_mock_message(self, message: Any)   # line 127
async def __aiter__(self) -> AsyncIterator[Any]  # line 131
```

**Fix:** Use proper types. For message queue, use `TranscriptionMessage`. For callbacks, use proper `Callable` types.

---

## 27. Multiple files — Generic `Exception` instead of specific types

**Locations:**
- `packages/active-listener/src/eavesdrop/active_listener/typist.py` (lines 49, 58, 65, 70, 83)
- `packages/active-listener/src/eavesdrop/active_listener/client.py` (lines 65, 85)
- `packages/active-listener/tests/unit/test_mocks.py` (lines 28, 34, 36, 41, 43, 101, 113, 162)

**Problem:** Code raises generic `Exception` instead of specific exception types:

```python
raise Exception("ydotool is not available")          # typist.py:58
raise Exception("Client not initialized")            # client.py:65
raise Exception(f"Failed to type text: {e}")         # typist.py:65
```

**Fix:** Use appropriate specific exception types:
- `RuntimeError` for runtime/state issues ("not initialized", "not available")
- `ValueError` for invalid input
- `ConnectionError` for connection issues
- Or define custom exception classes for the domain

---

## 28. Test suite tests dead code

**Location:** `packages/active-listener/tests/unit/test_text_manager.py`

**Problem:** The test suite thoroughly tests `text_manager.py`, but:
1. `text_manager.py` is likely dead code (see issue #12)
2. `workspace.py` (the active implementation) has no unit tests
3. The mocks in `test_mocks.py` mirror the old architecture (typist + text_manager)

This means the test suite validates orphaned code while the active code is untested.

**Fix:** Either:
- If `text_manager.py` is the canonical implementation, integrate it with `workspace.py`
- If `workspace.py` is the canonical implementation, write tests for it and remove `text_manager.py` tests

---

## 29. Multiple files — Bare `except Exception:` swallows errors

**Locations:** Many places in active-listener package (see grep results)

**Problem:** Many places catch all exceptions and swallow them or just log without proper handling:

```python
# ui_channel.py:158
except Exception:
    self.logger.debug("Error monitoring UI subprocess stderr")
    pass

# client.py:141
except Exception:
    self._connection_state.is_connected = False
    self.logger.exception("Reconnection failed")
    return False
```

While logging exceptions is good, catching `Exception` broadly can mask bugs and make debugging difficult.

**Recommendation:**
- Catch specific exceptions where possible
- Ensure all exception handlers at least log the error
- Consider whether the exception should propagate

---

## 30. `app.py` — `asyncio.create_task` from signal handler

**Location:** `packages/active-listener/src/eavesdrop/active_listener/app.py`, line 159

**Problem:** A signal handler creates an async task:

```python
def signal_handler(signum, _frame):
    self.logger.info("Received shutdown signal", signal=signum)
    self.shutdown()
    # Create a task to force client shutdown asynchronously
    asyncio.create_task(self._force_shutdown())  # <-- dangerous
```

Signal handlers run synchronously and may interrupt the event loop at any point. Creating a task from a signal handler is unreliable — the task may not be scheduled or awaited properly.

**Fix:** Use `loop.call_soon_threadsafe()` or set a flag and handle shutdown in the event loop:

```python
def signal_handler(signum, _frame):
    self.logger.info("Received shutdown signal", signal=signum)
    self.shutdown()
    loop = asyncio.get_running_loop()
    loop.call_soon_threadsafe(
        lambda: asyncio.create_task(self._force_shutdown())
    )
```

---

## 31. `ui_messages.py` — Line too long (ruff E501)

**Location:** `packages/active-listener/src/eavesdrop/active_listener/ui_messages.py`, line 22

**Problem:** Line exceeds 100 character limit:

```python
- COMMAND: Voice provides transformation instructions to an agent for processing the transcribed content
```

**Fix:** Wrap the line to stay under 100 characters.

---

## 32. `test_cli_interface.py` — Tests missing required `--ui-bin` argument

**Location:** `packages/active-listener/tests/contract/test_cli_interface.py`, multiple tests

**Problem:** CLI tests call `ActiveListener.parse([])` without the required `--ui-bin` argument:

```python
# Line 21
cmd = ActiveListener.parse([])  # Missing --ui-bin

# Line 86
cmd = ActiveListener.parse(["--server", "192.168.1.100:8080", "--audio-device", "hw:1,0"])  # Missing --ui-bin
```

Looking at `__main__.py:166`, `ui_bin` is a required argument with no default:

```python
ui_bin: Path = arg(parser=parse_ui_binary)  # No default!
```

These tests will fail when run because the required argument is missing.

**Fix:** Either:
- Add a mock/fixture for `--ui-bin` in all tests
- Make `ui_bin` optional with a default for testing

---

## 33. `test_ydotool_integration.py` — Tests patch code that's commented out

**Location:** `packages/active-listener/tests/contract/test_ydotool_integration.py`, lines 44-56

**Problem:** Tests patch `pydotool.type_string`, but the actual implementation has that call commented out:

```python
# In test (line 47):
with patch("pydotool.type_string") as mock_type:
    typist.type_text(text)
    mock_type.assert_called_once_with(text)  # Will FAIL

# In typist.py (lines 61-62):
# pydotool.type_string(text)  # <-- commented out
pass
```

The test expects `pydotool.type_string` to be called, but the actual code is `pass`. The test gives a false sense of security.

**Fix:** Either:
- Uncomment the actual pydotool calls in `typist.py`
- Update tests to reflect the current (non-functional) state

---

## 34. Inconsistent default port between packages

**Locations:**
- `packages/active-listener/src/eavesdrop/active_listener/__main__.py:164` — defaults to port **9090**
- `packages/client/src/eavesdrop/client/core.py:37,89,131` — defaults to port **8080**

**Problem:** The active-listener CLI defaults to port 9090, but the client library defaults to port 8080:

```python
# __main__.py
server: ServerHostPort = arg(default=ServerHostPort("localhost", 9090), ...)

# core.py
def __init__(self, ..., port: int = 8080, ...):
```

A user running the default active-listener against a server started with the client library's defaults would fail to connect.

**Fix:** Align on a single default port across all packages.

---

## 35. Contract tests test dead code architecture

**Location:** `packages/active-listener/tests/contract/`

**Problem:** All three contract test files import from `text_manager.py`:

```python
# test_text_processing.py:11
from eavesdrop.active_listener.text_manager import TextState, TextUpdate, ...

# test_ydotool_integration.py:13
from eavesdrop.active_listener.text_manager import TypingOperation

# test_cli_interface.py:11
from eavesdrop.active_listener.__main__ import ActiveListener  # Uses old architecture
```

These tests validate the old `text_manager.py` + `typist.py` architecture, not the new `workspace.py` + `ui_channel.py` architecture that's actually used by `app.py`.

This means:
1. Tests pass for code that isn't used
2. Active code has no contract tests
3. Refactoring broke the tests' relevance

**Fix:** Update contract tests to test the active architecture (`workspace.py`, `ui_channel.py`).

---

## 36. `__main__.py` — `isinstance` check violates CODE_STYLE.md

**Location:** `packages/active-listener/src/eavesdrop/active_listener/__main__.py`, lines 51, 79, 122

**Problem:** CODE_STYLE.md explicitly forbids dynamic type testing:

> "We use types ALL the time. Because we do, we ALWAYS trust the compiler, and never write ridiculous amateurish dynamic type testing code. That kind of coding is STRICTLY forbidden."

Yet the code has multiple `isinstance()` checks:

```python
# Line 51
if not isinstance(value, str):

# Line 79
if not isinstance(value, str):

# Line 122
default_input = sd.default.device[0] if isinstance(sd.default.device, tuple) else None
```

The first two are in argument parsers which receive `str | list[str]`, so the check may be justified for external input. But line 122 checks internal sounddevice types.

**Recommendation:** For external boundaries (arg parsers), type checks may be acceptable. For internal code (line 122), trust the library's types or use proper type narrowing.

---

## 37. `audio_flow.py` — Inverted condition logic in disconnect (Critical)

**Location:** `packages/server/src/eavesdrop/server/streaming/audio_flow.py`, lines 191-198

**Problem:** The `disconnect()` method has inverted condition logic:

```python
async def disconnect(self) -> None:
    """Send disconnect notification and clean up resources."""
    try:
      if self._closed or not self.websocket:
        await self.send_message(DisconnectMessage(stream=self.stream_name))
    finally:
      self._closed = True
```

This sends the disconnect message ONLY when `_closed is True` OR `websocket is None/falsy` — exactly when it CAN'T send. The logic should be inverted: send the message when the connection IS open.

**Fix:** Invert the condition:

```python
async def disconnect(self) -> None:
    """Send disconnect notification and clean up resources."""
    try:
      if not self._closed and self.websocket:
        await self.send_message(DisconnectMessage(stream=self.stream_name))
    finally:
      self._closed = True
```

---

## 38. `server.py` — Dead code in health check handler (High)

**Location:** `packages/server/src/eavesdrop/server/server.py`, line 210

**Problem:** The `_handle_health_check` method contains a method reference that does nothing:

```python
async def _handle_health_check(
    self, websocket: ServerConnection, message: HealthCheckRequest
) -> None:
    """Handle traditional transcriber client connections."""
    self.logger.info("Health check successful", websocket_id=websocket.id)
    self._send_error_and_close  # <-- Method reference, not a call!
    return None
```

This line is a no-op — it references the method but doesn't call it (missing `()` and arguments). The health check connection is never properly closed.

**Fix:** Either remove the dead line entirely, or complete the call:

```python
await self._send_and_close(websocket, HealthCheckResponse(...))
```

---

## 39. `websocket.py` — Returns `False` instead of `None` (Medium)

**Location:** `packages/server/src/eavesdrop/server/websocket.py`, lines 32-41

**Problem:** The `get_client()` method returns `False` when a client isn't found:

```python
def get_client(self, websocket):
    """
    Retrieves a client associated with the given websocket.

    :param websocket: The websocket associated with the client to retrieve.
    :returns: The client object if found, False otherwise.
    """
    if websocket in self.clients:
      return self.clients[websocket]
    return False  # <-- Should be None
```

Returning `False` instead of `None` is unconventional and can cause subtle bugs:
- `if client is None:` won't catch the "not found" case
- `if not client:` works but for the wrong reason
- Type annotation is missing, hiding the mixed return types

**Fix:** Return `None` for "not found" (Pythonic convention):

```python
def get_client(self, websocket: ServerConnection) -> WebSocketStreamingClient | None:
    """..."""
    return self.clients.get(websocket)
```

---

## 40. `server.py`, `rtsp/client.py` — Assert used for runtime validation (Medium)

**Locations:**
- `packages/server/src/eavesdrop/server/server.py`, line 275
- `packages/server/src/eavesdrop/server/rtsp/client.py`, lines 515, 531

**Problem:** Uses `assert` to validate runtime conditions that should always be checked:

```python
# server.py:275
case TranscriberConnection(client):
    assert client._completion_task is not None
    await client._completion_task

# rtsp/client.py:515
async def _streaming_processor_task(self) -> None:
    assert self.processor is not None, "Processor must be initialized"
    await self.processor.initialize()

# rtsp/client.py:531
if len(audio_array) > 0:
    assert self.processor is not None, "Processor must be initialized"
    self.processor.add_audio_frames(audio_array)
```

`assert` statements are removed when Python runs with `-O` (optimize) flag, causing these validations to silently disappear in production.

**Fix:** Replace with explicit validation:

```python
# server.py:275
if client._completion_task is None:
    raise RuntimeError("Client completion task not initialized")
await client._completion_task

# rtsp/client.py
if self.processor is None:
    raise RuntimeError("Processor must be initialized before use")
```

---

## 41. `__main__.py` (server) — Missing return type annotations (Low)

**Location:** `packages/server/src/eavesdrop/server/__main__.py`, lines 8 and 26

**Problem:** Two functions are missing return type annotations:

```python
# Line 8
def get_env_or_default(env_var, default, var_type: type = str):
    # Missing return type annotation

# Line 26
async def main():
    # Missing return type annotation
```

**Fix:** Add proper return types:

```python
def get_env_or_default(env_var: str, default: str | int | bool, var_type: type = str) -> str | int | bool:
    ...

async def main() -> None:
    ...
```

---

## 42. `buffer.py` — Race condition in `processed_duration` property (Medium)

**Location:** `packages/server/src/eavesdrop/server/streaming/buffer.py`, lines 239-242

**Problem:** The `processed_duration` property reads two instance variables without acquiring the lock:

```python
@property
def processed_duration(self) -> float:
    """Duration of audio that has been processed in seconds."""
    return self.processed_up_to_time - self.buffer_start_time
```

However, `add_frames()` (line 86) modifies `buffer_start_time` while holding the lock:

```python
with self.lock:
    # ...
    self.buffer_start_time += self.config.cleanup_duration
```

If `processed_duration` is called from a different thread while `add_frames()` is executing, it could read a stale `buffer_start_time` value, resulting in an incorrect (possibly negative) duration.

**Fix:** Acquire the lock in the property:

```python
@property
def processed_duration(self) -> float:
    """Duration of audio that has been processed in seconds."""
    with self.lock:
        return self.processed_up_to_time - self.buffer_start_time
```

---

## 43. `processor.py` — Misleading `np.frombuffer` on ndarray data (Medium)

**Location:** `packages/server/src/eavesdrop/server/streaming/processor.py`, lines 784-785

**Problem:** The code claims to "convert bytes back to numpy array" but operates on data that's already an ndarray:

```python
# Convert bytes back to numpy array for writing
audio_array = np.frombuffer(chunk.data, dtype=np.float32)
```

But `chunk.data` is typed as `np.ndarray` in the `AudioChunk` dataclass (line 32):

```python
@dataclass
class AudioChunk:
    data: np.ndarray  # <-- Already an ndarray!
```

Additionally, the variable `input_bytes` (which becomes `chunk.data`) comes from:

```python
input_bytes = self.frames_np[samples_take:].copy()  # Returns ndarray, not bytes
```

Calling `np.frombuffer()` on an ndarray works (ndarrays expose the buffer protocol), but it's semantically wrong and the comment is misleading.

**Fix:** Remove the unnecessary conversion:

```python
# Audio data is already a numpy array
audio_array = chunk.data
```

Or rename `input_bytes` to `input_array` throughout to avoid confusion.

---

## 44. `processor.py` — Bare `list` type without element annotation (Medium)

**Locations:** `packages/server/src/eavesdrop/server/streaming/processor.py`, lines 45, 452, 627, 722

**Problem:** Uses bare `list` type without specifying element type, violating project typing standards:

```python
# Line 45
speech_chunks: list | None = None

# Line 452
def _handle_transcription_output(
    self, result: list[Segment], duration: float, speech_chunks: list | None = None

# Line 627 (type annotation in parameter)
speech_chunks: list | None,

# Line 722 (same pattern)
speech_chunks: list | None,
```

The element type should be `SpeechChunk` (a TypedDict) from `eavesdrop.server.transcription.models`.

**Fix:** Add proper element type annotation:

```python
from eavesdrop.server.transcription.models import SpeechChunk

speech_chunks: list[SpeechChunk] | None = None
```

---

## 45. `manager.py` — `getattr` fallbacks suggest missing attributes (Medium)

**Location:** `packages/server/src/eavesdrop/server/rtsp/manager.py`, lines 278-279, 286

**Problem:** Uses `getattr` with default values to handle potentially missing attributes:

```python
status["streams"][stream_name] = {
    # ...
    "transcriptions_completed": getattr(client, "transcriptions_completed", 0),
    "transcription_errors": getattr(client, "transcription_errors", 0),
    # ...
    "segments_processed": getattr(client.processor, "segments_processed", 0),
}
```

This pattern indicates uncertainty about whether these attributes exist on the objects. Per CODE_STYLE.md, we "trust the compiler" and don't use dynamic type checking.

If these attributes should exist, they should be defined in the class. If they don't exist, accessing them shouldn't happen.

**Fix:** Either:
1. Add these attributes to `RTSPTranscriptionClient` and `StreamingTranscriptionProcessor` classes
2. Remove these fields from the status dict if they're not actually tracked
3. If processor can be None, handle that explicitly: `client.processor.segments_processed if client.processor else 0`

---

## 46. `manager.py` — `asyncio.Task` without type parameter (Low)

**Location:** `packages/server/src/eavesdrop/server/rtsp/manager.py`, line 73

**Problem:** Generic `asyncio.Task` type is used without specifying the return type:

```python
self.tasks: dict[str, asyncio.Task] = {}
```

Per project standards, all types should be fully parameterized.

**Fix:** Specify the task return type (likely `None` since `RTSPTranscriptionClient.run()` returns None):

```python
self.tasks: dict[str, asyncio.Task[None]] = {}
```

---

## 47. `rtsp/client.py` — Buffer not reset on reconnection (High)

**Location:** `packages/server/src/eavesdrop/server/rtsp/client.py`, lines 430 vs 456-462

**Problem:** When an RTSP stream disconnects and reconnects, the `stream_buffer` is NOT reset, but a new `processor` IS created:

```python
# In __init__ (line 430) - created ONCE:
self.stream_buffer = AudioStreamBuffer(transcription_config.buffer)

# In run() loop (lines 456-462) - created on EACH reconnection:
self.processor = StreamingTranscriptionProcessor(
    buffer=self.stream_buffer,  # <-- Reuses stale buffer!
    sink=self.transcription_sink,
    config=self.transcription_config,
    ...
)
```

After a reconnection, the buffer still contains:
- Old `buffer_start_time` and `processed_up_to_time` from the previous connection
- Potentially stale audio frames
- Incorrect timestamp offsets

This causes the new processor to work with completely wrong timing assumptions, leading to garbled transcription output.

**Fix:** Reset the buffer on each reconnection attempt:

```python
# In run() before creating processor:
self.stream_buffer.reset()  # Already has this method!
self.processor = StreamingTranscriptionProcessor(...)
```

---

## 48. `subscriber.py`, `manager.py` — Duplicate logger name (Medium)

**Locations:**
- `packages/server/src/eavesdrop/server/rtsp/subscriber.py`, line 46
- `packages/server/src/eavesdrop/server/rtsp/manager.py`, line 74

**Problem:** Both classes use the same logger name:

```python
# subscriber.py:46
self.logger = get_logger("rtsp/mgr")

# manager.py:74
self.logger = get_logger("rtsp/mgr")
```

When debugging, you can't tell whether a log message came from `RTSPSubscriberManager` or `RTSPClientManager`. This makes troubleshooting RTSP issues unnecessarily difficult.

**Fix:** Use distinct logger names:

```python
# subscriber.py
self.logger = get_logger("rtsp/sub")

# manager.py (keep as-is)
self.logger = get_logger("rtsp/mgr")
```

---

## 49. `subscriber.py` — `# type: ignore` suppressing type error (Medium)

**Location:** `packages/server/src/eavesdrop/server/rtsp/subscriber.py`, line 174

**Problem:** A type error is being suppressed with `# type: ignore`:

```python
await self.send_to_subscriber(
    stream_name,
    StreamStatusMessage(
        stream=stream_name,
        status=status,  # type: ignore  <-- Hiding a real problem
        message=message,
    ),
)
```

The `status` parameter is typed as `str`, but `StreamStatusMessage.status` likely expects a `Literal["online", "offline", "error"]` or similar enum type. The `type: ignore` hides this mismatch.

**Fix:** Either:
1. Use proper literal types: `status: Literal["online", "offline", "error"]`
2. Or cast the value: `status=cast(StreamStatus, status)`
3. Or fix the `StreamStatusMessage` definition if `str` is actually valid

---

## 50. `subscriber.py` — Bare `list` type without element annotation (Medium)

**Location:** `packages/server/src/eavesdrop/server/rtsp/subscriber.py`, line 180

**Problem:** The `segments` parameter uses bare `list` type:

```python
async def send_transcription(
    self, stream_name: str, segments: list, language: str | None = None
) -> None:
```

Per project typing standards, all types should be fully parameterized.

**Fix:** Add the element type (likely `Segment` from the wire package):

```python
async def send_transcription(
    self, stream_name: str, segments: list[Segment], language: str | None = None
) -> None:
```

---

## 51. `pipeline.py` — Dead code in `_TranscribeContext` (Low)

**Location:** `packages/server/src/eavesdrop/server/transcription/pipeline.py`, lines 540 and 585

**Problem:** The `_TranscribeContext` dataclass creates an `AnomalyDetector` that is never used:

```python
@dataclass
class _TranscribeContext:
    # ...
    anomaly_detector: AnomalyDetector = field(init=False)  # line 540
    # ...

    def __post_init__(self):
        # ...
        self.anomaly_detector = AnomalyDetector(_PUNCTUATION)  # line 585
```

But hallucination filtering in `_transcribe_segments` uses `self.hallucination_filter` from `WhisperModel` (line 145), not from the context:

```python
# In WhisperModel.__init__ (line 145):
self.hallucination_filter = HallucinationFilter(AnomalyDetector(_PUNCTUATION))

# In _transcribe_segments (line 412):
filtered_segments, new_seek = self.hallucination_filter.filter_segments(...)
```

The `ctx.anomaly_detector` is created but never accessed.

**Fix:** Remove the dead `anomaly_detector` field from `_TranscribeContext`.
