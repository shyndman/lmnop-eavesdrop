## Why

`active-listener` currently has no machine-readable way to publish its lifecycle to the desktop. That makes it hard to add a simple always-on status surface, even though the service already has a clear foreground state machine and a handful of important one-shot lifecycle events.

This change is needed now because the service is stable enough to expose truthful app state, and the planned Ubuntu top-bar integration only needs a minimal passive communication mechanism rather than a full UI inside `active-listener`.

## What Changes

- Add a DBus-backed app-state publication surface to `active-listener` for passive desktop consumers.
- Publish a single durable state property with the values `starting`, `idle`, `recording`, and `reconnecting`.
- Publish only the explicitly requested one-shot signals: `RecordingAborted(reason)`, `Reconnecting()`, and `Reconnected()`.
- Initialize DBus at process startup alongside logging so consumers can observe `starting` and never miss early lifecycle publication.
- Make DBus required by default; fail fast on export/name-acquisition problems, including duplicate-instance startup.
- Add a `--no-dbus` CLI flag that swaps in `NoopDbusService` and allows the service to run without DBus when session-bus access is unavailable.
- Use `python-sdbus` on the user session bus with a stable exported interface for external consumers.
- Keep the desktop extension itself out of scope; this feature only adds the communication mechanism it will consume.

## Capabilities

### New Capabilities
- `broadcast-app-state`: Publish `active-listener` lifecycle state and selected one-shot lifecycle events over DBus for passive desktop consumers.

### Modified Capabilities
- `active-listener-hotkey-dictation`: The service startup and runtime lifecycle now publish a machine-readable app-state stream in addition to structured logs.
- `active-listener-cli-runtime`: Startup gains a DBus mode switch, with DBus enabled by default and `--no-dbus` as the explicit headless escape hatch.

## Impact

- **Active listener package**: DBus interface definition, startup/export lifecycle, state publication boundary, `NoopDbusService`, and integration at existing state-transition points.
- **CLI/runtime behavior**: default startup now requires a usable user session bus and exclusive ownership of the DBus name unless `--no-dbus` is supplied.
- **Dependencies**: add `python-sdbus` for async DBus serving on Linux desktop systems.
- **External consumers**: introduces a stable DBus contract intended for a future Ubuntu/GNOME top-bar extension, but does not include that extension in this feature.
- **Out of scope**: shell extension implementation, clickable DBus methods, richer event metadata, notifications, and any new in-app visual UI.
