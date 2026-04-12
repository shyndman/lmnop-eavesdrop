# active-listener

`active-listener` is a user-session dictation service. The supported operator workflow is `systemd --user`, not a custom bootstrap shell script.

## Install or update the user service

Run from the repo root:

```bash
task install-active-listener-service
```

That task:

- installs `active-listener.service` into `~/.config/systemd/user/active-listener.service`
- enables it for `graphical-session.target`
- orders startup after `ydotoold.service`
- keeps restart behavior best-effort with `Restart=on-failure`
- seeds `~/.config/eavesdrop/active-listener.yaml` from `packages/active-listener/config.yaml` when the XDG config file does not exist yet

The canonical runtime paths are:

- config: `~/.config/eavesdrop/active-listener.yaml`
- prompt override: `~/.config/eavesdrop/active-listener.system.md`

If you still have a legacy prompt override at `~/.config/active-listener/system.md`, move it to `~/.config/eavesdrop/active-listener.system.md`. The install task copies it forward only when the new file does not exist yet.

## Uninstall the user service

Run from the repo root:

```bash
task uninstall-active-listener-service
```

This removes the installed user unit and reloads the user service manager. It does not delete `~/.config/eavesdrop/active-listener.yaml` or `~/.config/eavesdrop/active-listener.system.md` because those files are user data.

## Inspect and troubleshoot

Check service health:

```bash
systemctl --user status active-listener.service
```

Follow logs:

```bash
journalctl --user -u active-listener.service -f
```

The service is owned by `graphical-session.target`, so a fresh graphical login is the normal auto-start path. `ydotoold.service` is ordered ahead of `active-listener.service`, but ordering is not readiness: if workstation prerequisites are transiently unavailable, systemd retry behavior is expected to recover the service.

## Manual development run

Manual invocation still uses the same CLI entrypoint and config model as the service:

```bash
cd packages/active-listener
uv sync
uv run active-listener --config-path ~/.config/eavesdrop/active-listener.yaml
```