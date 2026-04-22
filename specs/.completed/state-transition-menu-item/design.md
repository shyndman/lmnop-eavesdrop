## Context

The GNOME indicator in `packages/active-listener-ui-gnome/src/extension.ts` already watches the active-listener session-bus name, reads the `State` property from `ca.lmnop.Eavesdrop.ActiveListener1`, and renders the icon plus overlay behavior from that state. Today the menu is operationally useful for service lifecycle (`Restart service`, `Stop service`) but it does not control dictation itself.

Inside active-listener, recording control semantics are currently keyboard-centric. `ActiveListenerService.handle_keyboard_action()` in `packages/active-listener/src/active_listener/app/service.py` drives the recording lifecycle, and the pure policy layer in `packages/active-listener/src/active_listener/app/state.py` models actions and decisions as keyboard-owned concepts. That shape is no longer truthful once the GNOME menu becomes a second control source.

There is one important timing constraint already present in the existing code. Finishing a recording transitions the foreground phase back to idle before background finalization emits the final transcript, and `GnomeShellExtensionTextEmitter.emit_text()` captures the focused window at emit time. That means a menu-triggered stop can still paste into the previously focused application after the menu closes, as long as the menu action shares the same finish semantics as the keyboard path.

## Current APIs and Implementation Touchpoints

### GNOME extension menu and state wiring

- File: `packages/active-listener-ui-gnome/src/extension.ts`
- Top-bar shell entrypoint: `ActiveListenerIndicatorExtension.enable()` creates `this.button = new PanelMenu.Button(0.5, this.metadata.name, false)`, adds `this.icon`, calls `this.addMenuItems()`, and registers the indicator with `Main.panel.addToStatusArea(this.uuid, this.button)`.
- Current menu construction happens directly in `addMenuItems()`. It creates four `PopupMenu.PopupMenuItem`s in this order: `Preferences`, `Show overlay`, `Restart service`, `Stop service`. They are added in that same order via `this.button.menu.addMenuItem(...)`.
- Existing menu state members are `restartServiceItem` and `stopServiceItem`; `updateMenuSensitivity()` keeps restart always enabled and disables stop only when `indicatorState === 'absent'`.
- D-Bus constants already used by the extension are:

```ts
const DBUS_BUS_NAME = 'ca.lmnop.Eavesdrop.ActiveListener';
const DBUS_OBJECT_PATH = '/ca/lmnop/Eavesdrop/ActiveListener';
const DBUS_INTERFACE_NAME = 'ca.lmnop.Eavesdrop.ActiveListener1';
const DBUS_STATE_PROPERTY = 'State';
```

- Service presence is tracked in `enable()` through `Gio.bus_watch_name(...)`. Bus appearance calls `attachProxy()`. Bus disappearance calls `detachProxy()` and `updateIndicator('absent')`.
- `attachProxy()` builds the active-listener proxy with `Gio.DBusProxy.new_for_bus_sync(...)`, subscribes `'g-properties-changed'` to `syncIndicatorState()`, subscribes `'g-signal'` to `handleProxySignal()`, then immediately calls `syncIndicatorState()`.
- `syncIndicatorState()` currently flattens everything except the literal property value `'recording'` into `'idle'`:

```ts
const value = this.proxy.get_cached_property(DBUS_STATE_PROPERTY)?.deepUnpack();
const nextState = value === 'recording' ? 'recording' : 'idle';
this.updateIndicator(nextState);
```

- The extension already has one visible notification pattern: `handleProxySignal()` responds to `PipelineFailed` by calling `Main.notifyError('Active Listener pipeline failed', detail)`. Existing menu-command failures only log with `console.error(...)`; they do not currently notify the user.

### Active-listener D-Bus boundary

- File: `packages/active-listener/src/active_listener/infra/dbus.py`
- Current exported constants are:

```python
DBUS_BUS_NAME = "ca.lmnop.Eavesdrop.ActiveListener"
DBUS_OBJECT_PATH = "/ca/lmnop/Eavesdrop/ActiveListener"
DBUS_INTERFACE_NAME = "ca.lmnop.Eavesdrop.ActiveListener1"
```

- The public interface is built in `_build_active_listener_dbus_interface()` and exported by `SdbusDbusService.connect()` via `request_default_bus_name_async(DBUS_BUS_NAME)` and `interface.export_to_dbus(DBUS_OBJECT_PATH, bus)`.
- Current active-listener D-Bus surface is one read-only property plus seven outbound signals:

```text
Property: State (type "s", read-only)
Signals:
- TranscriptionUpdated (a(ts)(ts))
- SpectrumUpdated (ay)
- RecordingAborted (s)
- PipelineFailed (ss)
- FatalError (s)
- Reconnecting ()
- Reconnected ()
```

- There are no current inbound `@dbus_method_async` methods on the active-listener interface.
- The `TYPE_CHECKING` stub in the same file mirrors the effective contract and will need to stay aligned with the runtime-built interface.
- `AppStateService` is currently an outbound publication protocol only: `set_state`, `transcription_updated`, `spectrum_updated`, `recording_aborted`, `pipeline_failed`, `fatal_error`, `reconnecting`, `reconnected`, and `close`. It does not currently model inbound control.
- Contract-locking tests live in `packages/active-listener/tests/test_dbus_service.py`:
  - `test_interface_contract_names_match_spec`
  - `test_property_is_read_only_over_dbus_and_empty_signals_emit`
  - `test_dbus_introspection_matches_locked_contract`

### App control model and runtime queue

- Files:
  - `packages/active-listener/src/active_listener/app/state.py`
  - `packages/active-listener/src/active_listener/app/signals.py`
  - `packages/active-listener/src/active_listener/app/service.py`
- Current control enums are keyboard-owned:

```python
class KeyboardAction(StrEnum):
    START_OR_FINISH = "start_or_finish"
    CANCEL = "cancel"

class KeyboardDecision(StrEnum):
    START_RECORDING = "start_recording"
    FINISH_RECORDING = "finish_recording"
    CANCEL_RECORDING = "cancel_recording"
    IGNORE = "ignore"
    SUPPRESS_RECONNECTING_START = "suppress_reconnecting_start"
```

- `decide_keyboard_action(phase, action)` currently maps:
  - `IDLE + START_OR_FINISH -> START_RECORDING`
  - `RECORDING + START_OR_FINISH -> FINISH_RECORDING`
  - `RECORDING + CANCEL -> CANCEL_RECORDING`
  - `RECONNECTING + START_OR_FINISH -> SUPPRESS_RECONNECTING_START`
  - everything else -> `IGNORE`
- Runtime queue types are currently:

```python
@dataclass(frozen=True)
class KeyboardSignal:
    action: KeyboardAction

@dataclass(frozen=True)
class ClientSignal:
    event: ConnectedEvent | DisconnectedEvent | ReconnectingEvent | ReconnectedEvent | TranscriptionEvent

RuntimeSignal = KeyboardSignal | ClientSignal
```

- `ActiveListenerService.run()` dispatches the queue with:

```python
if isinstance(signal, KeyboardSignal):
    await self.handle_keyboard_action(signal.action)
else:
    await self.handle_client_event(signal.event)
```

- `handle_keyboard_action()` is the current start/finish/cancel entrypoint. `handle_client_event()` owns connected/reconnecting/reconnected/disconnected transitions and can abort recording on disconnect.

### Expected end-state architecture after this feature

This section is intentionally more prescriptive than the earlier design notes. A junior engineer should be able to follow it without inventing extra architecture.

#### Keep one policy engine, two producers

The recording state machine should still have exactly one place where control intent is interpreted:

```text
ForegroundPhase + AppAction -> AppActionDecision
```

That logic lives in `packages/active-listener/src/active_listener/app/state.py`.

After this change there are two producers of `AppAction`:
- keyboard input
- the D-Bus method `StartOrFinishRecording`

Only the keyboard can produce `AppAction.CANCEL`.

The key rule is: the D-Bus method must reuse the same app-owned decision path as the keyboard. It must not duplicate start/finish logic in `infra/dbus.py`.

#### Recommended service method contract

Use this service-level shape:

```python
async def handle_action(self, action: AppAction) -> AppActionDecision:
    ...
```

Why this return type is recommended:
- keyboard callers can ignore the return value
- the D-Bus method can translate the returned `AppActionDecision` into the public D-Bus result string
- the decision enum remains the internal source of truth for what the service actually did

Recommended D-Bus result mapping:

```text
AppActionDecision.START_RECORDING -> "started"
AppActionDecision.FINISH_RECORDING -> "finished"
AppActionDecision.SUPPRESS_RECONNECTING_START -> "ignored"
AppActionDecision.IGNORE -> "ignored"
```

`AppActionDecision.CANCEL_RECORDING` should never be returned from the D-Bus path because the D-Bus interface does not expose cancel.

#### Avoid the circular dependency trap during startup

There is a non-obvious dependency cycle in the current bootstrap flow:
- `ActiveListenerService` already depends on `dbus_service`
- the new D-Bus method also needs a way to call back into `ActiveListenerService`

Do **not** solve this by moving recording policy into the D-Bus layer.

Instead, keep outbound state publication and inbound control delegation separate.

Recommended shape in `packages/active-listener/src/active_listener/infra/dbus.py`:

```python
class RecordingControl(Protocol):
    async def handle_action(self, action: AppAction) -> AppActionDecision: ...

class SdbusDbusService(AppStateService):
    def attach_recording_control(self, control: RecordingControl) -> None:
        ...
```

Then the exported D-Bus method simply delegates:

```python
decision = await self._recording_control.handle_action(AppAction.START_OR_FINISH)
```

Recommended startup sequence:

```text
1. construct SdbusDbusService
2. construct ActiveListenerService with that dbus_service
3. call dbus_service.attach_recording_control(service)
4. publish ForegroundPhase.IDLE over D-Bus
```

This ordering matters. The extension should not see an enabled `Start Recording` state before the D-Bus service is capable of delegating the method.

If the method is called before a control handler is attached, fail the method call. A built-in exception or `DbusFailedError` is sufficient; there is no need for a custom D-Bus error class for this feature.

#### Safe extension-state mapping, including startup

The current extension only distinguishes `absent`, `idle`, and `recording`, but the Python app also has a `starting` phase.

The user only locked four visible labels, so the safe implementation is:

```text
bus absent         -> No Service      disabled
state=reconnecting -> Reconnecting    disabled
state=recording    -> Stop Recording  enabled
state=idle         -> Start Recording enabled
state=starting     -> Start Recording disabled
unknown value      -> Start Recording disabled
```

This avoids exposing a fifth label while still keeping the menu safe during startup or unexpected state values.

Recommended derivation helper in `extension.ts`:

```python
def derive_menu_control(service_present: bool, phase: str) -> tuple[str, bool]:
    if not service_present:
        return ("No Service", False)
    if phase == "reconnecting":
        return ("Reconnecting", False)
    if phase == "recording":
        return ("Stop Recording", True)
    if phase == "idle":
        return ("Start Recording", True)
    return ("Start Recording", False)
```

The exact helper can be written in TypeScript, but the behavior above is the contract.

### Finish ordering and emitter timing

- Files:
  - `packages/active-listener/src/active_listener/app/service.py`
  - `packages/active-listener/src/active_listener/recording/session.py`
  - `packages/active-listener/src/active_listener/recording/finalizer.py`
  - `packages/active-listener/src/active_listener/infra/emitter.py`
- Current finish path while recording is ordered as follows:

```text
service.handle_keyboard_action(START_OR_FINISH)
-> self.phase = ForegroundPhase.IDLE
-> await self._recording_session.finish_recording()
   -> await self._exit_recording()
      -> release_recording_grab()
      -> await grab_stack.aclose()
      -> await self.client.stop_streaming()
-> await self._stop_spectrum_analysis()
-> await self.dbus_service.set_state(self.phase)
-> create_task(self._recording_finalizer.finalize_recording(...))
```

- `RecordingFinalizer.finalize_recording()` performs `client.flush(force_complete=True)`, text rendering/rewrite, then `self.emitter.emit_text(final_text)` on the background task.
- The default emitter is created eagerly in startup through `build_emitter()` in `packages/active-listener/src/active_listener/bootstrap.py`, which constructs `GnomeShellExtensionTextEmitter()` and immediately calls `initialize()`.
- `GnomeShellExtensionTextEmitter.initialize()` binds to GNOME Shell D-Bus interfaces:

```python
service_name="org.gnome.Shell"
object_path="/org/gnome/Shell/Extensions/Windows"
interface_name="org.gnome.Shell.Extensions.Windows"

service_name="org.gnome.Shell"
object_path="/org/gnome/Shell/Extensions/Clipboard"
interface_name="org.gnome.Shell.Extensions.Clipboard"
```

- `emit_text()` captures focus at emission time by calling `windows.get_focused_window_sync()` inside `emit_text()`, then uses `clipboard.set_content(...)` and `windows.send_shortcut(focused_window.id, 'v', modifiers)` for each chunk.

### Validation hooks already present in the repo

- `packages/active-listener/pyproject.toml` explicitly provides `basedpyright` and `pytest` in dev dependencies and configures `[tool.basedpyright]`, `[tool.ruff]`, and `[tool.pytest.ini_options]`.
- The same `pyproject.toml` does **not** currently declare a `ruff` executable in dependencies or dependency groups, even though Ruff config exists.
- `packages/active-listener-ui-gnome/package.json` exposes these scripts:

```json
{
  "build": "node esbuild.js",
  "typecheck": "tsc --noEmit",
  "wayland:test": "npm run install:extension && dbus-run-session gnome-shell --devkit --wayland"
}
```

### Third-party dependencies and external API surface

- No new third-party runtime dependency is required for this feature. The implementation should use the dependencies already present in the repo.
- Python D-Bus library in use:
  - Declared in `packages/active-listener/pyproject.toml` as `sdbus>=0.12.0`
  - Locked in root `uv.lock` as `sdbus 0.14.2`
  - Latest upstream found during spec review is also `0.14.2`, so the locked environment already includes the current documented API surface used here.
- GNOME extension TypeScript/GIR packages in use:
  - `@girs/gio-2.0` declared and locked as `2.88.0-4.0.0-rc.1`
  - `@girs/gjs` declared and locked as `4.0.0-rc.1`
  - `@girs/gnome-shell` declared and locked as `49.1.0`
  - Latest npm search during spec review confirms `@girs/gnome-shell` is still `49.1.0`; the repo should continue implementing against the locked versions above rather than inventing newer APIs.

The implementation should rely on these documented behaviors from the existing third-party APIs:

#### python-sdbus 0.14.2

- `DbusInterfaceCommonAsync` is the async interface base class. Derived classes must call `super().__init__()` in `__init__` if they define one.
- `dbus_method_async(...)` is the decorator for exported async D-Bus methods. Methods must be `async def`.
- `new_proxy(service_name, object_path, bus=...)` creates a proxy object for the interface class.
- `export_to_dbus(object_path, bus)` exports the object and returns a handle with `.stop()`.
- To expose the new method, the active-listener interface should use an async method declaration equivalent to:

```python
@dbus_method_async(
    input_signature="",
    result_signature="s",
    result_args_names=("result",),
    method_name="StartOrFinishRecording",
)
async def start_or_finish_recording(self) -> str:
    ...
```

- D-Bus errors in sdbus propagate from raised exceptions:
  - subclasses of `sdbus.exceptions.DbusFailedError` produce stable custom D-Bus error names via `dbus_error_name`
  - ordinary built-in Python exceptions are automatically mapped to `org.python.Error.<ExceptionName>`
- This feature does not require client-side branching on D-Bus error names. The extension only needs failure text for user notification, so a custom D-Bus error subclass is not required unless tests later choose to lock a stable error name.

#### Gio / GJS DBusProxy APIs already available in the extension toolchain

- `Gio.DBusProxy.get_cached_property(property_name)` reads the local property cache and does no blocking I/O.
- `Gio.DBusProxy::g-properties-changed` fires after the local cache has already been updated.
- `Gio.DBusProxy.call(method_name, parameters, flags, timeout_msec, cancellable, callback)` is the non-blocking way to invoke a D-Bus method from the extension.
  - `parameters` must be a `GLib.Variant` tuple or `null` when there are no input parameters.
  - `timeout_msec` may be `-1` to use the proxy default timeout.
  - The callback completes with `source.call_finish(result)`.
  - `call_finish(...)` returns a `GLib.Variant` tuple of return values.
- For this feature, the extension should use the async `call(...)`/`call_finish(...)` pattern instead of `call_sync(...)` so the GNOME Shell UI thread is not blocked by the menu command.
- The expected menu-command call shape is:

```ts
this.proxy.call(
  'StartOrFinishRecording',
  null,
  Gio.DBusCallFlags.NONE,
  -1,
  null,
  (source, result) => {
    const reply = source?.call_finish(result);
    const [commandResult] = reply?.deepUnpack() as [string];
    // commandResult is 'started' | 'finished' | 'ignored'
  },
);
```

- If the call fails, the callback should catch the thrown error from `call_finish(...)`, log it, and surface the existing user-visible notification style via `Main.notifyError(title, detail)`.

#### GNOME Shell notification API already in use

- The extension already imports `Main` from `resource:///org/gnome/shell/ui/main.js`.
- Current visible error-notification usage is `Main.notifyError('Active Listener pipeline failed', detail)`.
- The menu-command failure path should reuse that same two-string notification shape rather than introducing a custom `MessageTray` implementation.

## Goals / Non-Goals

**Goals:**
- Add a first GNOME menu item that acts as a stateful recording control.
- Make menu-issued start/finish requests use the exact same semantics as keyboard-issued start/finish requests.
- Keep cancel as a keyboard-only action.
- Preserve truthful UI rendering by deriving menu labels from D-Bus state and bus presence, not from optimistic local guesses.
- Refactor app control terminology so the domain model is app-action based instead of keyboard-centric.
- Extend the D-Bus contract in a typed, explicit way without turning the D-Bus layer into a second policy engine.

**Non-Goals:**
- Add a menu-visible cancel action.
- Add temporary UI states such as `Starting…` or `Stopping…`.
- Debounce or otherwise defend against duplicate `StartOrFinishRecording` delivery across a finish boundary; that behavior is explicitly accepted as out of scope.
- Introduce a generic D-Bus action dispatcher such as `HandleAction(action: str)`.
- Change finalization pipeline semantics, including background rewrite/final emission timing.

## Decisions

### 1. The first menu item is a stateful control and state indicator

The first GNOME menu item will always exist and will render one of four labels:

```text
No Service      disabled
Reconnecting    disabled
Start Recording enabled
Stop Recording  enabled
```

The extension will derive these states from two existing truths:
- session-bus name presence (`No Service` when the service is absent)
- the published foreground phase (`Reconnecting`, `Start Recording`, `Stop Recording`)

**Why:** this keeps the menu understandable at a glance and avoids adding a separate recording-status row.

**Alternatives considered:**
- Hiding the item when the service is absent: rejected because the user wants a visible state indicator.
- Showing `Start Recording` while reconnecting and letting the request no-op: rejected because the user wants reconnecting to be explicit and disabled.

### 2. The extension is a thin client and never performs optimistic UI updates

On activation, the menu item closes immediately and the extension sends a D-Bus request. The extension does not locally toggle labels, start animations, or infer the next phase from the click result.

```text
click -> menu closes -> D-Bus call -> service changes State -> extension re-renders from State
```

The extension shows a GNOME notification only when the D-Bus method call itself fails.

The menu command should use the existing non-blocking Gio proxy pattern, not a synchronous D-Bus round-trip. In practice, that means calling `this.proxy.call(...)` with `null` parameters for the zero-argument method, then handling the returned `GLib.Variant` tuple inside `call_finish(...)`.

**Why:** the service already owns authoritative state. Letting the extension speculate would create UI lies during reconnects, failures, or delayed transitions.

**Alternatives considered:**
- Optimistically swapping `Start Recording`/`Stop Recording` on click: rejected because it can drift from service truth.
- Silent failure handling: rejected because a failed command should be visible.
- `call_sync(...)` from the menu callback: rejected because it would block GNOME Shell on a D-Bus round-trip.

### 3. Add one explicit D-Bus control method: `StartOrFinishRecording`

The existing interface `ca.lmnop.Eavesdrop.ActiveListener1` will gain one inbound method:

```text
StartOrFinishRecording() -> "started" | "finished" | "ignored"
```

At the sdbus layer this is a zero-argument method with `result_signature="s"`, exported by the same `DbusInterfaceCommonAsync`-based interface that already owns `State` and the seven outbound signals.

Meaning:
- `started`: idle -> recording
- `finished`: recording -> idle, with background finalization started
- `ignored`: reconnecting suppressed the request

Real failures continue to use D-Bus errors/exceptions.

**Why:** the method shape stays explicit, matches the actual semantics discussed, and avoids inventing a generic action RPC.

**Alternatives considered:**
- `ToggleRecording()`: rejected because it hides the real semantics and makes accidental double-delivery harder to reason about.
- `HandleAction(action: str)`: rejected because it weakens the contract and needlessly exposes cancel on the bus.
- Returning nothing: rejected because the user wants an explicit result contract.
- Defining a new custom error type purely for this method: rejected as unnecessary for current client behavior because the GNOME extension only needs to notify on failure, not branch on a stable D-Bus error name.

### 4. Use typed app-level enums internally and plain strings on the D-Bus wire

The app layer will introduce app-owned control types, including:

```python
class AppAction(StrEnum):
    START_OR_FINISH = "start_or_finish"
    CANCEL = "cancel"

class AppActionDecision(StrEnum):
    ...

class StartOrFinishResult(StrEnum):
    STARTED = "started"
    FINISHED = "finished"
    IGNORED = "ignored"
```

The D-Bus method will still declare `result_signature='s'` and return `StartOrFinishResult.value` on the wire.

**Why:** this preserves strict typing in Python while keeping the D-Bus contract simple and explicit. The current repo already declares D-Bus signatures manually, and there is no verified project evidence that sdbus will transparently map `StrEnum` return types into a richer D-Bus contract.

**Alternatives considered:**
- Passing raw strings through the app layer: rejected because it weakens internal type safety.
- Relying on implicit enum mapping in sdbus: rejected because the verified docs only show explicit D-Bus signatures.

### 5. Move control semantics from keyboard-owned APIs to app-owned APIs

The service and pure policy layer will be renamed around app-level control intent:
- `KeyboardAction` -> `AppAction`
- `KeyboardDecision` -> `AppActionDecision`
- `KeyboardSignal` -> `AppActionSignal`
- `handle_keyboard_action(...)` -> `handle_action(...)`
- `decide_keyboard_action(...)` -> `decide_app_action(...)`

Keyboard input becomes one producer of `AppActionSignal`. The D-Bus method becomes another producer or caller of the same app-owned control path. Cancel remains an app action but only the keyboard produces it.

**Why:** once D-Bus can issue start/finish, the keyboard is no longer the domain owner of recording control.

**Alternatives considered:**
- Keep keyboard-centric names and let the D-Bus layer call `handle_keyboard_action(...)`: rejected because the architecture would describe the wrong ownership.

### 6. Preserve existing finalization and focus behavior

A menu-triggered stop uses the same finish path as the keyboard. It must not add a separate stop mode, soft stop, or menu-specific emission path.

This design relies on existing behavior:
- finishing transitions the app out of recording before final background emission
- the keyboard grab is released before the final flush/emission path completes
- the text emitter snapshots the focused window when emitting, after the menu has closed

**Why:** the user explicitly wants menu stop to mean the same thing as keyboard finish, and current focus timing already supports that model.

**Alternatives considered:**
- A menu-specific stop path: rejected because it would create semantic drift.

## Implementation Walkthrough by File

This section describes the intended edits in the order a junior engineer should make them.

### 1. `packages/active-listener/src/active_listener/app/state.py`

- Rename `KeyboardAction` to `AppAction`.
- Rename `KeyboardDecision` to `AppActionDecision`.
- Rename `decide_keyboard_action(...)` to `decide_app_action(...)`.
- Keep the existing string values for the action and decision members unless there is a test-mandated reason to change them. The rename is about ownership, not semantics.
- Add a new `StartOrFinishResult(StrEnum)` with values `started`, `finished`, `ignored`.
- Keep `ForegroundPhase` unchanged.

Expected pure-function behavior after the rename:

```text
IDLE + START_OR_FINISH -> START_RECORDING
RECORDING + START_OR_FINISH -> FINISH_RECORDING
RECORDING + CANCEL -> CANCEL_RECORDING
RECONNECTING + START_OR_FINISH -> SUPPRESS_RECONNECTING_START
all other combinations -> IGNORE
```

### 2. `packages/active-listener/src/active_listener/app/signals.py`

- Rename `KeyboardSignal` to `AppActionSignal`.
- Change its field type from `KeyboardAction` to `AppAction`.
- Keep `ClientSignal` unchanged.
- Update `RuntimeSignal` to `AppActionSignal | ClientSignal`.

This file is transport only. Do not move policy into it.

### 3. `packages/active-listener/src/active_listener/app/service.py`

- Rename `handle_keyboard_action(...)` to `handle_action(...)`.
- Update `run()` to dispatch `AppActionSignal` instead of `KeyboardSignal`.
- Keep `handle_client_event(...)` as the owner of reconnect/disconnect lifecycle.
- Have `handle_action(...)` return the `AppActionDecision` it executed.

Recommended branch behavior inside `handle_action(...)`:

```text
decision = decide_app_action(self.phase, action)

START_RECORDING:
  - enter recording
  - set phase RECORDING
  - publish State
  - return START_RECORDING

FINISH_RECORDING:
  - set phase IDLE
  - finish recording session
  - stop spectrum analysis
  - publish State
  - spawn background finalizer
  - return FINISH_RECORDING

CANCEL_RECORDING:
  - set phase IDLE
  - stop recording session
  - stop spectrum analysis
  - publish State
  - cancel utterance / emit recording-aborted path as today
  - return CANCEL_RECORDING

SUPPRESS_RECONNECTING_START:
  - keep current phase
  - return SUPPRESS_RECONNECTING_START

IGNORE:
  - no side effects
  - return IGNORE
```

Important: `FINISH_RECORDING` must preserve the current ordering already documented earlier in this spec. Do not “clean it up” into a different order.

### 4. `packages/active-listener/src/active_listener/infra/keyboard.py`

- Keep the keyboard boundary responsible for converting physical keyboard events into app actions.
- Update any imports and signal construction so keyboard input now produces `AppActionSignal(AppAction.START_OR_FINISH)` or `AppActionSignal(AppAction.CANCEL)`.
- Do not move keyboard event interpretation into the D-Bus layer.

### 5. `packages/active-listener/src/active_listener/infra/dbus.py`

- Keep the existing constants, property, and signals.
- Add a narrow inbound control delegation boundary (`RecordingControl` protocol plus an attachment method on the concrete DBus service).
- Extend the runtime-built interface with a zero-argument `StartOrFinishRecording` method returning type `s`.
- Inside that method:
  1. delegate to the attached control handler with `AppAction.START_OR_FINISH`
  2. translate the returned `AppActionDecision` into `StartOrFinishResult`
  3. return `StartOrFinishResult.value`

Recommended translation table:

```text
START_RECORDING -> StartOrFinishResult.STARTED
FINISH_RECORDING -> StartOrFinishResult.FINISHED
SUPPRESS_RECONNECTING_START -> StartOrFinishResult.IGNORED
IGNORE -> StartOrFinishResult.IGNORED
```

If no control handler is attached, fail the call instead of guessing.

Also update the `TYPE_CHECKING` interface stub so type-aware readers and tests see the new method.

### 6. `packages/active-listener/src/active_listener/bootstrap.py`

- Preserve the existing construction order as much as possible.
- After creating the `ActiveListenerService`, attach it to the concrete DBus service as the recording-control delegate, then publish `IDLE`.
- Do not publish `IDLE` before the D-Bus method can delegate successfully.

This is the main file where the circular dependency is resolved.

### 7. `packages/active-listener-ui-gnome/src/extension.ts`

- Add a new first menu item before `Preferences`.
- Store it as a field so label and sensitivity can be updated whenever the state changes.
- Stop flattening every non-`recording` state to `idle`.
- Preserve enough phase detail to distinguish at least `absent`, `starting/unknown`, `idle`, `recording`, and `reconnecting` for menu rendering.

Recommended menu update flow:

```text
bus appeared/disappeared
or
proxy properties changed
=> recompute service presence + phase
=> update icon state if needed
=> update first menu item label/sensitivity
=> update other menu sensitivity if needed
```

Recommended click flow for the new first item:

```text
activate menu item
=> if no proxy, return
=> call proxy.call('StartOrFinishRecording', ...)
=> in callback, call source.call_finish(result)
=> deepUnpack tuple to read the returned string
=> ignore the returned string for UI rendering
=> only notify on thrown error
```

The returned string is informational only. Do not use it to set the menu label.

### 8. `packages/active-listener/tests/test_state.py`

- Update or add tests for the renamed app-level decision function.
- The tests should still prove the exact decision matrix shown above.
- Add explicit coverage for the reconnecting suppression case returning `SUPPRESS_RECONNECTING_START`.

### 9. `packages/active-listener/tests/test_app.py`

- Update service tests for the renamed `handle_action(...)` entrypoint.
- Add coverage for:
  - idle start returns `START_RECORDING`
  - recording finish returns `FINISH_RECORDING`
  - recording cancel returns `CANCEL_RECORDING`
  - reconnecting start returns `SUPPRESS_RECONNECTING_START`
- The finish-path test should continue to prove that phase publication happens before background finalization runs.

### 10. `packages/active-listener/tests/test_dbus_service.py`

- Update the locked contract tests so introspection includes `StartOrFinishRecording`.
- Add a focused method-call test that asserts:
  - start path returns `started`
  - finish path returns `finished`
  - reconnecting suppression returns `ignored`
- Keep the existing property read-only assertion.

### 11. `packages/active-listener-ui-gnome/scripts/`

The GNOME package does not currently expose a dedicated unit-test runner in `package.json`, but it already uses script-style deterministic validation (`scripts/validate-transcript-controller.ts`).

Follow that existing pattern for any deterministic menu-state validation added by this feature. A small validation script is preferable to inventing a new test framework in this change.

## Explicit UI and Behavior Contracts

### Menu item ordering

After the change, the menu order should be:

```text
1. [stateful recording control]
2. Preferences
3. Show overlay
4. Restart service
5. Stop service
```

### Menu label/sensitivity contract

Use this table as the implementation contract:

| Service presence | Published phase | Visible label | Enabled |
|---|---|---|---|
| absent | n/a | `No Service` | no |
| present | `reconnecting` | `Reconnecting` | no |
| present | `recording` | `Stop Recording` | yes |
| present | `idle` | `Start Recording` | yes |
| present | `starting` | `Start Recording` | no |
| present | anything else | `Start Recording` | no |

### Command/result contract

Use this table when wiring the D-Bus method result:

| Internal decision | D-Bus return string | Meaning |
|---|---|---|
| `START_RECORDING` | `started` | recording began |
| `FINISH_RECORDING` | `finished` | recording stopped; finalization continues in background |
| `SUPPRESS_RECONNECTING_START` | `ignored` | reconnecting suppressed the request |
| `IGNORE` | `ignored` | no action was taken |

### Notification contract

- Show a notification only when the D-Bus method call itself fails.
- Reuse `Main.notifyError(title, detail)`.
- Do not show notifications for ordinary `ignored`, `No Service`, or `Reconnecting` UI states.

### What must stay unchanged

- Escape remains the only cancel path.
- The menu still closes immediately when the first item is activated.
- Final transcript emission still happens through the existing background finalization pipeline.
- The extension still renders from D-Bus truth; it never performs optimistic relabeling.

## Risks / Trade-offs

- [D-Bus contract expansion changes the locked interface] -> Update the DBus boundary tests to assert the new method and result contract alongside the existing property/signals.
- [Renaming keyboard-centric app types touches multiple files] -> Keep the refactor limited to the app state, signal, service, and boundary layers that truly participate in control flow.
- [Extension state rendering currently collapses non-recording phases to idle] -> Preserve the full published phase in the extension before wiring menu labels so reconnecting remains visible and disabled.
- [The extension could show stale UI if it uses the method return value as truth] -> Treat the method return as informational only; render labels exclusively from bus presence and `State` changes.
- [Duplicate `StartOrFinishRecording` delivery can still cross a phase boundary and start a new session] -> This is a deliberate non-goal for this feature and will not be addressed in this spec.

## Migration Plan

1. Extend the active-listener D-Bus interface and its tests to include `StartOrFinishRecording` and its string result contract.
2. Refactor active-listener app control types and service entry points from keyboard-owned naming to app-owned naming while preserving existing keyboard behavior.
3. Update the GNOME extension to render the first menu item from bus presence plus the real published phase, and invoke the new D-Bus method on activation.
4. Expand GNOME extension tests and active-listener tests to cover state rendering, command routing, and failure notification behavior.

Rollback is straightforward because the feature is additive at the D-Bus interface and local to the GNOME extension and active-listener packages. Reverting the method and menu item returns the system to the existing read-only indicator behavior.

## Open Questions

- `packages/active-listener/pyproject.toml` configures Ruff but does not currently declare a Ruff executable in package dependencies. Should this feature restore Ruff to the package dev dependencies and keep `uv run ruff check` as a required validation step, or should package-local validation omit Ruff until that tooling is restored?

All feature-shape decisions discussed with the user are locked: menu labels, disabled states, method naming, return values, notification behavior, and app-level naming.