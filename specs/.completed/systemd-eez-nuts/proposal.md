## Why

`active-listener` already behaves like a long-running workstation service. It owns a session DBus name, enforces single-instance startup through that name, depends on per-user desktop resources such as `ydotoold`, keyboard devices, and the focused application, and is designed to stay connected across many recordings. The current manual CLI launch flow and repo-local config default are a mismatch between how the software really behaves and how it is installed and operated.

That mismatch matters operationally. A junior engineer could look at the current setup and reasonably assume `active-listener` is a developer-only command instead of a desktop-session component. They would then miss where config should live, how startup should be retried, where logs should be read, and how other desktop consumers should learn about fatal failure. This spec fixes that by making the installation model tell the same truth as the runtime model.

This change turns `active-listener` into a first-class user-session feature. It should start automatically with graphical login, read its primary config from the user's XDG config directory, keep the current logging path intact under `systemd --user`, and publish fatal startup/runtime failure as a one-shot DBus event when DBus is already available. The design deliberately avoids inventing a durable `failed` state because service absence already means the process is gone.

## What Changes

- Add a user-session systemd integration path for `active-listener` so it starts automatically under `systemd --user` as part of the graphical desktop session.
- Make `graphical-session.target` the owning lifecycle for the service so startup and shutdown follow the desktop session instead of a repo shell or lingering background user manager.
- Add Taskfile installation and uninstallation tasks so workstation setup and teardown have repo-standard entrypoints instead of ad hoc shell commands.
- Move the default runtime config path from `packages/active-listener/config.yaml` to `~/.config/eavesdrop/active-listener.yaml`.
- Move the rewrite prompt override path from `~/.config/active-listener/system.md` to `~/.config/eavesdrop/active-listener.system.md`.
- Keep explicit CLI config overrides working so manual runs and systemd runs use the same bootstrap path.
- Keep startup best-effort by using systemd ordering plus restart behavior instead of turning transient workstation prerequisites into a permanent failed session state.
- Order the service after the user-managed `ydotoold.service` unit because that is the actual user unit available on the target workstation.
- Extend the DBus contract with one fatal one-shot signal, emitted immediately before fatal exit when DBus has already been acquired.
- Preserve current stdout/stderr logging behavior and let the user service capture logs unchanged in journald.
- Require installation-path verification in actual `systemd --user` context so healthy startup proves `active-listener` reaches normal service state without emitting `FatalError`.
- Update GNOME-side prompt editing so the preferences UI writes to the new `eavesdrop` XDG location instead of the legacy `active-listener` path.

## Capabilities

### New Capabilities
- `active-listener-user-systemd-service`: Install and run `active-listener` as a user-session systemd service bound to the graphical desktop session, with best-effort startup, retry behavior, and an operator model based on `systemctl --user` and journald.
- `active-listener-taskfile-installation`: Install and uninstall `active-listener` user-service assets through `Taskfile.yaml` tasks so workstation setup follows one documented repo workflow.

### Modified Capabilities
- `active-listener-cli-runtime`: Default config discovery moves from repo-local package files to user XDG config, while manual CLI invocation remains supported through the same bootstrap path and explicit config overrides.
- `active-listener-llm-rewrite`: The prompt override location moves into the `eavesdrop` XDG namespace, but packaged prompt fallback remains the behavior when the override file is absent.
- `broadcast-app-state`: The DBus contract gains a fatal one-shot signal while keeping `State` as the durable source of current truth and keeping DBus-name absence as the signal that the service is not running.
- `gnome-settings-ui`: The prompt editor must target `~/.config/eavesdrop/active-listener.system.md` so the UI edits the same file the runtime reads.

## Impact

- **Active listener runtime**: `packages/active-listener/src/active_listener/config.py`, `cli.py`, `dbus_service.py`, and `rewrite.py` will change because default config discovery, bootstrap failure publication, and prompt override resolution all move.
- **Task runner workflow**: `Taskfile.yaml` will change to add install and uninstall entrypoints for the user service and to document how healthy startup is verified from those tasks.
- **GNOME integration**: `packages/active-listener-ui-gnome/src/prefs.ts` will change because it currently writes the prompt override to the old XDG path.
- **Operations**: operators now manage the service with `systemctl --user`, inspect logs with `journalctl --user -u active-listener.service`, and treat graphical login as the normal startup path.
- **Dependencies**: the user service must order against `graphical-session.target` and the existing user-managed `ydotoold.service` unit, but not against remote transcription server reachability.
- **DBus consumers**: passive consumers such as the GNOME indicator keep reading the durable `State` property but can also listen for a fatal one-shot event before the bus name disappears.
- **Migration**: developers and users who already have config or prompt files in legacy locations will need a clear move path because this spec intentionally treats the new `eavesdrop` XDG paths as the source of truth.
- **Docs and onboarding**: workstation setup instructions must shift from manual repo-local startup toward user-service installation, config placement, journald-based operations, and an explanation of what is still expected to fail fast versus retry.
