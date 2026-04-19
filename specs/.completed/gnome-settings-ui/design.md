## Context

`packages/active-listener-ui-gnome` is currently a GNOME Shell extension that exposes only a top-bar indicator backed by the `active-listener` DBus service state. It has no preferences UI yet. On the rewrite side, `packages/active-listener` currently loads the system prompt from the configured `llm_rewrite.prompt_path`, which points at the repo prompt file by default.

The desired change is intentionally pragmatic: add a low-lift GNOME settings surface that can grow over time and make prompt iteration easier during testing. The user wants to treat this as a settings window, not a special-purpose editor, but also wants the system prompt control to behave like a raw file: if the saved prompt is invalid, runtime prompt loading should fail the same way it would if edited externally.

This design touches two packages:
- `packages/active-listener-ui-gnome`: new preferences UI and prompt file editing workflow.
- `packages/active-listener`: new prompt source resolution behavior on each rewrite request.

Constraints:
- GNOME package remains a Shell extension, not a standalone app.
- The first UI should be low lift and easy to extend with toggles and future controls.
- Prompt editing must target `~/.config/active-listener/system.md`.
- Prompt loading must happen per LLM request so prompt edits take effect without restarting the service.
- Saving must not be blocked by front matter or markdown validation.

## Goals / Non-Goals

**Goals:**
- Add a GNOME extension preferences window as the package's settings surface.
- Include a multiline `System prompt` control in that window.
- Read and write a user override prompt file at `~/.config/active-listener/system.md`.
- Seed the editor with repo-default prompt contents when the override file is absent.
- Resolve prompt source on each rewrite request: override file first, configured prompt path second.
- Preserve room in the prefs UI for future toggles and controls.

**Non-Goals:**
- Rework the top-bar indicator beyond what is needed to expose preferences.
- Build a standalone GTK/libadwaita companion app.
- Add prompt validation, guarded save behavior, or schema-aware prompt editing.
- Replace `llm_rewrite.prompt_path` as the fallback/default prompt configuration.
- Introduce live preview, diffing, version history, or prompt-management workflows.

## Decisions

### 1. Use the GNOME extension preferences window as the new settings surface
Decision: implement the new UI in the extension prefs entrypoint rather than a panel popup, custom shell dialog, or standalone app.

Rationale:
- Lowest implementation lift for the current package.
- Fits the user's mental model of a settings window.
- Leaves room to add toggles and other conventional settings controls alongside the prompt text area.
- Avoids pushing a large editable text area into a Shell menu surface that is optimized for compact interactions.

Alternatives considered:
- Panel dropdown editor: possible, but awkward for multiline editing and a poor fit for future settings.
- Custom shell dialog: feasible, but more shell-specific complexity for little gain.
- Standalone GTK app: strongest long-term option for richer UI, but unnecessary for the current testing-driven need.

### 2. Model the prompt as a user override file, not a GSettings string
Decision: store editable prompt contents in `~/.config/active-listener/system.md`.

Rationale:
- The prompt is already a file-based artifact with front matter and markdown content.
- A file preserves the same semantics whether edited through GNOME prefs or an external editor.
- Using a config file avoids stuffing document-sized content into GSettings.
- The chosen path is explicit and user-scoped.

Alternatives considered:
- Store the prompt in GSettings: poor fit for large structured text and awkward for manual editing.
- Edit the repo prompt file in place: unsafe and surprising for a checked-in default.
- Introduce a new prompt storage backend: unnecessary abstraction for one override file.

### 3. Save raw prompt contents without validation gates
Decision: the prefs UI saves whatever text the user enters.

Rationale:
- Matches the behavior of editing the same file in a normal editor.
- Keeps the UI honest and low-friction for testing.
- Avoids duplicating or partially re-implementing `active-listener` prompt parsing rules in the extension.

Alternatives considered:
- Block save on parse failure: safer, but explicitly rejected by the user.
- Save with warnings only after validation: still adds validation coupling and extra behavior the user does not want.

### 4. Resolve the prompt source at request time inside `active-listener`
Decision: `active-listener` should look for the override file on each LLM rewrite request and fall back to `llm_rewrite.prompt_path` when no override exists.

Rationale:
- Makes prompt edits take effect without restarting the service.
- Keeps prompt source selection in the service that owns rewrite behavior.
- Preserves the existing configured prompt path as the canonical default.
- Produces a clean truth model: request-time prompt source is either user override or configured fallback.

Alternatives considered:
- Cache the prompt at startup: simpler, but fails the testing workflow.
- Have the extension notify the service to reload: adds coordination complexity with no benefit over direct request-time reads.
- Replace the configured prompt path entirely: removes a useful default/fallback mechanism.

### 5. Seed the editor from the fallback prompt when the override is missing
Decision: the prefs text area initially shows the repo/default prompt contents if `~/.config/active-listener/system.md` does not exist yet.

Rationale:
- Gives the user something immediately editable.
- Makes the first save produce a full override file rather than a partial patch concept.
- Keeps the override model easy to reason about: once saved, the user-owned file becomes the source of truth.

Alternatives considered:
- Start blank: confusing and forces the user to reconstruct required prompt structure.
- Require creating the override file externally first: defeats the convenience goal.

## Verified External APIs and Dependencies

### GNOME Shell extension prefs contract
- For GNOME 45+ and still valid on GNOME 49, preferences are exposed by shipping a `prefs.js` module beside `extension.js`; no extra `metadata.json` key is required just to enable prefs.
- The prefs entrypoint must export a default subclass of `ExtensionPreferences` imported from `resource:///org/gnome/Shell/Extensions/js/extensions/prefs.js` and implement `fillPreferencesWindow(window)`.
- `fillPreferencesWindow(window)` receives an `Adw.PreferencesWindow` that the extension populates; the implementation should add one or more `Adw.PreferencesPage` instances via `window.add(page)`.
- The main extension may open prefs with `this.openPreferences()` from the `Extension` base class.
- `prefs.js` runs in a separate process from GNOME Shell. It can use GTK4/libadwaita-compatible APIs, but it must not import or depend on Shell-only modules such as `St`, `Main`, `PanelMenu`, `Meta`, `Clutter`, or `Shell`.
- Import paths differ by process: shell code keeps using `resource:///org/gnome/shell/...`, while prefs code must use `resource:///org/gnome/Shell/Extensions/js/...`.

### Required package dependencies
- The current package manifest already includes `@girs/gio-2.0`, `@girs/gjs`, `@girs/gnome-shell`, and `@girs/st-17`, but it does not declare direct GTK4 or libadwaita type packages.
- Verified current latest packages are:
  - `@girs/gtk-4.0`: `4.23.0-4.0.0-rc.1`
  - `@girs/adw-1`: `1.10.0-4.0.0-rc.1`
- Recommended implementation choice: add both as direct `devDependencies` with normal caret ranges and keep them aligned with the repo's existing `4.0.0-rc.1` GIR family.
- Alternative: rely on transitive copies pulled in by `@girs/gnome-shell`. This is not recommended under `pnpm`, because direct imports in `prefs.ts` should be backed by direct dependencies and explicit ambient imports.
- `ambient.d.ts` must also add `@girs/gtk-4.0` and `@girs/adw-1` so TypeScript sees the prefs-side imports.

### Recommended prefs widget stack
- Use `Adw.PreferencesPage` to create the top-level page and `Adw.PreferencesGroup` to organize related controls.
- `Adw.PreferencesGroup` can contain normal `Adw.PreferencesRow` controls, but it can also host a non-row child below the list. This makes it suitable for embedding a larger custom editor widget without abandoning the native prefs layout.
- For the multiline prompt control, the recommended stack is:
  - `Gtk.ScrolledWindow` as the framed scroll container
  - `Gtk.TextView` as the editable multiline widget
  - `Gtk.TextBuffer` as the text model
- `Gtk.ScrolledWindow.set_child()` attaches the `Gtk.TextView`.
- `Gtk.TextView` is the GTK4 multiline editor widget; it is scrollable, supports clipboard/undo/redo actions, and exposes configuration such as `set_buffer()`, `set_monospace()`, and `set_wrap_mode()`.
- `Gtk.TextBuffer` holds the actual text and exposes `set_text()`, `get_bounds()`, `get_text()`, and the `changed` signal for save-state tracking.

### File and path APIs for the prompt override
- Do not build the override path by hand with a literal `~`. Resolve the config root with `GLib.get_user_config_dir()` and join path components with `GLib.build_filenamev()`.
- Use `Gio.File.new_for_path()` to create file handles for both the user override path and any fallback prompt path used in prefs.
- If the `active-listener` config directory does not exist, create it with `Gio.File.make_directory_with_parents()`. GIO does not provide an async variant for this helper.
- For reading prompt contents in prefs, use `Gio.File.load_contents_async()` and decode bytes with `TextDecoder('utf-8')`.
- For writing prompt contents in prefs, use `Gio.File.replace_contents_bytes_async()` with `GLib.Bytes`; the GJS file-operations guide explicitly recommends the bytes-based async replacement method to avoid corruption risks from async buffer lifetime mistakes.
- Because the prefs window is a UI process, prefer async file reads/writes even though sync methods exist.

### Packaging and manual verification details
- The current build only emits `dist/extension.js`; implementation must also emit `dist/prefs.js` so the ZIP contains both entrypoints.
- The extension ZIP anatomy for prefs-capable extensions is `metadata.json`, `extension.js`, and optional `prefs.js` at the root of the unpacked extension directory.
- Manual verification entrypoints are documented and should be used during implementation:
  - open prefs: `gnome-extensions prefs eavesdrop@shyndman.ca`
  - inspect prefs logs: `journalctl -f -o cat /usr/bin/gjs`

### Non-dependencies and deferred options
- No new Python package dependency is required on the `active-listener` side for prompt override resolution; existing `pathlib`, `frontmatter`, and `jinja2` usage remain sufficient.
- If future toggle-style settings are added to the prefs window, GNOME's documented pattern is `GSettings` plus a schema referenced by `metadata.json` via `settings-schema`. That is not required for the prompt override itself and remains out of scope for this feature.

## Risks / Trade-offs

- [Extension prefs text editing may feel more like a mini editor than a traditional settings form] -> Keep the layout simple and group it under rewrite settings so it still reads as part of a broader settings surface.
- [Invalid prompt saves can break runtime rewrite behavior] -> Accept this intentionally; surface the resulting runtime failure through existing prompt-load logging.
- [Per-request file reads add overhead] -> The prompt file is small, and request-time freshness is more important than micro-optimizing this path.
- [Two prompt locations can confuse maintainers] -> Make the resolution order explicit in code, logs, and user-facing labels: override file first, configured prompt path second.
- [Future prefs controls may need service integration beyond file editing] -> Keep this change scoped to the preferences window structure so additional controls can be added later without reworking the first surface.

## Migration Plan

1. Add the GNOME extension prefs entrypoint and package it with the extension.
2. Implement prompt file read/write behavior for `~/.config/active-listener/system.md`, including seeding from the fallback prompt when absent.
3. Update `active-listener` prompt resolution to check the override file per rewrite request before loading the configured prompt path.
4. Add targeted tests for prompt resolution and any pure file-resolution helpers.
5. Verify the manual flow: open prefs, edit prompt, save, trigger a rewrite, observe the new prompt taking effect.

Rollback strategy:
- Remove the prefs UI and revert `active-listener` prompt resolution to the configured `prompt_path` only.
- Existing repo-default behavior remains intact because the override file is additive.

## Open Questions

None currently. The key product decisions are locked:
- use the GNOME prefs window,
- store the override at `~/.config/active-listener/system.md`,
- reload on each LLM request,
- allow invalid saves.