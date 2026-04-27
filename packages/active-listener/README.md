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
- keeps restart behavior best-effort with `Restart=on-failure`
- seeds `~/.config/eavesdrop/active-listener.yaml` from `packages/active-listener/config.yaml` when the XDG config file does not exist yet

## Vicinae GNOME extension dependency

`active-listener` now pastes committed text through Vicinae's GNOME Shell extension D-Bus services. `ydotoold` is no longer part of the runtime path.

The extension must be installed and exporting these session-bus objects under `org.gnome.Shell`:

- `/org/gnome/Shell/Extensions/Windows` (`org.gnome.Shell.Extensions.Windows`)
- `/org/gnome/Shell/Extensions/Clipboard` (`org.gnome.Shell.Extensions.Clipboard`)

Reference implementation: <https://github.com/dagimg-dot/vicinae-gnome-extension>

The canonical runtime paths are:

- config: `~/.config/eavesdrop/active-listener.yaml`
- prompt override: `~/.config/eavesdrop/active-listener.rewrite.system.md`

## Langfuse tracing

Langfuse tracing is env-driven.

- `task run-active-listener` already loads the repo-root `.env`.
- the installed `active-listener.service` now reads `@REPO_ROOT@/.env` too, if that file exists.

Set these variables to enable tracing:

- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_BASE_URL` when you are not using the default Langfuse cloud host

When tracing is enabled, each rewrite finalization is wrapped in a Langfuse root observation. Pydantic AI rewrite providers also emit their nested model spans via `Agent.instrument_all()` + `instrument=True`, so the rewrite trace shows both the active-listener request context and the underlying model call.

If you still have an older override at `~/.config/eavesdrop/active-listener.system.md` or `~/.config/active-listener/system.md`, move it to `~/.config/eavesdrop/active-listener.rewrite.system.md`.

## FFmpeg runtime dependency

Historical audio archival now depends on a usable `ffmpeg` binary at startup.

- If you leave `ffmpeg_path` unset, active-listener resolves `ffmpeg` from `PATH`.
- If you set `ffmpeg_path`, relative values resolve from the config file directory before startup validation runs.
- Startup fails fast when neither source provides an executable `ffmpeg` binary.

The sample config leaves `ffmpeg_path: null`, which means PATH lookup remains the default. Use an absolute override when your workstation keeps FFmpeg outside the normal shell PATH.

## Rewrite model and prompt files

Rewrite config now uses a provider block under `llm_rewrite.provider`. The seeded sample config uses:

- `prompt_path: "prompts/rewrite_prompt.md"`
- `provider.type: "litert"`
- `provider.model_path: "models/rewrite.litertlm"`

When the provider type is `litert`, both `llm_rewrite.provider.model_path` and `llm_rewrite.prompt_path` resolve relative to the config file directory, then normalize to absolute paths before the runtime starts.

That means a copied config at `~/.config/eavesdrop/active-listener.yaml` expects its fallback model bundle beside that config, and its fallback prompt under `prompts/rewrite_prompt.md`, unless you point them somewhere else.

Prompt files are markdown content only. No hidden routing metadata or template rendering remains in the active rewrite path.

GNOME prefs still edits `~/.config/eavesdrop/active-listener.rewrite.system.md`. Active Listener reloads that override on every rewrite request, so prompt edits take effect on the next recording without restarting the service.

## Uninstall the user service

Run from the repo root:

```bash
task uninstall-active-listener-service
```

This removes the installed user unit and reloads the user service manager. It does not delete `~/.config/eavesdrop/active-listener.yaml` or `~/.config/eavesdrop/active-listener.rewrite.system.md` because those files are user data.

## Inspect and troubleshoot

Check service health:

```bash
systemctl --user status active-listener.service
```

Follow logs:

```bash
journalctl --user -u active-listener.service -f
```

Check the Vicinae D-Bus objects:

```bash
gdbus introspect --session --dest org.gnome.Shell --object-path /org/gnome/Shell/Extensions/Windows
gdbus introspect --session --dest org.gnome.Shell --object-path /org/gnome/Shell/Extensions/Clipboard
```

The service is owned by `graphical-session.target`, so a fresh graphical login is the normal auto-start path. If GNOME Shell or the Vicinae extension is transiently unavailable during login, systemd retry behavior is expected to recover the service.

## Manual development run

Manual invocation still uses the same CLI entrypoint and config model as the service:

```bash
cd packages/active-listener
uv sync
uv run active-listener --config-path ~/.config/eavesdrop/active-listener.yaml
```