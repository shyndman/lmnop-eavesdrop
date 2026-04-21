## Why

Active Listener already publishes foreground state to the GNOME top-bar indicator, but the menu is read-only for recording control. The user wants the first menu item to act as a stateful recording control so recording can be started and stopped from the menu without introducing a second set of semantics. This needs to work in both directions: a session started from the menu must remain stoppable from the keyboard, and a session started from the keyboard must remain stoppable from the menu. The change is needed now because the indicator already exposes recording state, and adding control at that same boundary is the natural next step.

## What Changes

- Add a first GNOME menu item that reflects service state as `No Service`, `Reconnecting`, `Start Recording`, or `Stop Recording`.
- Add an inbound D-Bus control method so the GNOME extension can request the exact same start/finish behavior currently triggered by the keyboard.
- Return an explicit human-readable result from the D-Bus control method: `started`, `finished`, or `ignored`.
- Keep cancel semantics keyboard-only; the menu does not add a cancel action.
- Refactor app control terminology so recording control is modeled as app actions instead of keyboard-owned actions.
- Show a GNOME notification when the menu-triggered D-Bus command fails.

## Scope

### New Capabilities
- `menu-recording-control`: Add a stateful GNOME menu item that can request recording start/finish through the active-listener D-Bus service and reflect disabled states for absent or reconnecting service.
- `dbus-recording-command`: Extend the active-listener D-Bus interface with an explicit `StartOrFinishRecording` command that reports whether the request started, finished, or was ignored.

### Modified Capabilities
- `app-action-state-machine`: Generalize foreground recording control from keyboard-centric actions and decisions to app-level actions shared by keyboard and D-Bus producers.
- `gnome-indicator-state-rendering`: Preserve and render the real foreground phase instead of collapsing every non-recording state into `idle`.
- `menu-command-failure-reporting`: Add visible GNOME error reporting for failed menu-issued recording commands.

## Impact

Affected code spans the GNOME extension menu and D-Bus proxy logic in `packages/active-listener-ui-gnome/src/extension.ts`, the active-listener D-Bus contract in `packages/active-listener/src/active_listener/infra/dbus.py`, and the app state/signal/service layers in `packages/active-listener/src/active_listener/app/`. The D-Bus interface gains a new method and method result contract, so DBus boundary tests and extension behavior tests will need to expand accordingly. No new external dependency is required; the feature builds on the existing session-bus integration between the extension and active-listener.