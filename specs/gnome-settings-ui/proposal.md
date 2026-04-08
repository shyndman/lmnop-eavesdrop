## Why

The new `active-listener-ui-gnome` package currently exposes only a top-bar indicator, which is enough for presence but not enough for testing or operating rewrite-related behavior from the workstation. We need a low-lift GNOME-native settings surface that can grow beyond the indicator and let the user edit the LLM system prompt without leaving the desktop.

## What Changes

- Add a GNOME extension preferences UI for `active-listener-ui-gnome`.
- Add rewrite-related settings controls to that preferences UI, including a large multiline text area for the system prompt.
- Store the editable prompt in a user override file at `~/.config/active-listener/system.md`.
- Seed the override editor from the repo default prompt when the override file does not exist yet.
- Update `active-listener` rewrite behavior so each LLM request reloads prompt contents from the user override when present, instead of relying only on the configured repo prompt path.
- Preserve direct file semantics for the prompt editor: raw file contents are editable and savable without validation gates.

## Capabilities

### New Capabilities
- `gnome-prefs-ui`: A GNOME extension preferences window that can host toggles, future settings controls, and a large multiline prompt editor.
- `rewrite-prompt-override`: A user-scoped rewrite prompt override stored at `~/.config/active-listener/system.md` and editable from the GNOME preferences UI.

### Modified Capabilities
- `llm-rewrite`: Rewrite prompt loading changes from a single configured prompt file to runtime resolution of a user override file plus fallback to the existing configured prompt path.
- `active-listener-ui-gnome-indicator`: The GNOME package grows from indicator-only UI to indicator plus preferences-based settings UI.

## Impact

- Affected packages: `packages/active-listener-ui-gnome`, `packages/active-listener`.
- Affected third-party dependencies: `packages/active-listener-ui-gnome` will need explicit GTK4 and libadwaita GIR type packages for prefs development (`@girs/gtk-4.0`, `@girs/adw-1`) rather than relying on transitive dependencies from `@girs/gnome-shell`.
- Affected user workflow: rewrite prompt edits can happen inside GNOME preferences instead of an external editor.
- Affected runtime behavior: prompt file contents may change between rewrite requests and must be re-read per request.
- Affected configuration surface: introduces a user config file at `~/.config/active-listener/system.md` in addition to the existing `llm_rewrite.prompt_path` setting.
- Likely affected areas: GNOME extension prefs entrypoints and packaging, rewrite prompt resolution/loading, logging around prompt source selection and load failures.