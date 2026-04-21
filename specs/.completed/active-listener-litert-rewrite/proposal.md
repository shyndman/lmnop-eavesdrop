## Why

`active-listener` currently rewrites finalized dictation by calling an OpenAI-compatible chat endpoint and by treating the prompt file as both prose and hidden routing/config metadata. That no longer matches the desired product shape: the rewrite model should be a local LiteRT bundle kept resident in memory, while the GNOME settings UI remains the place where prompt text is edited. The current prompt loader also carries front matter and Jinja rendering that do not appear to serve any real runtime need.

This change makes the rewrite path tell the truth about the system. Model selection becomes explicit runtime configuration, prompt editing stays a markdown authoring workflow, and rewrite failures split cleanly into startup dependency failures versus per-recording prompt failures. The feature is needed now because the current remote-LLM path and prompt-file semantics are actively in the way of moving to the local SLM design.

## What Changes

- Replace the existing OpenAI-compatible rewrite client in `active-listener` with a LiteRT-LM Python integration backed by a resident compiled `.litertlm` model.
- **BREAKING**: replace `llm_rewrite.base_url` / `timeout_s` and prompt front-matter `model` routing with runtime model configuration centered on a local `model_path`.
- Keep prompt editing in GNOME prefs, but treat the prompt file as markdown content only instead of parsing YAML front matter or Jinja template variables.
- Preserve per-recording prompt reload behavior so prompt edits take effect on the next rewrite without restarting the service.
- Fail service startup when the configured LiteRT model cannot be opened, and surface prompt-load failures for individual recordings through the existing pipeline-failure DBus path.
- Resolve `llm_rewrite` file paths relative to the config file location, then normalize them to absolute paths during config load.

## Scope

### New Capabilities
- `active-listener-litert-rewrite-runtime`: Load a local LiteRT `.litertlm` bundle at service startup, keep a resident engine alive for the lifetime of the service, and create a fresh conversation for each rewrite request.

### Modified Capabilities
- `active-listener-llm-rewrite`: Rewrite execution moves from an OpenAI-compatible remote chat client to a local LiteRT Python engine, while keeping the same high-level role in the dictation finalization pipeline.
- `rewrite-prompt-override`: The override prompt remains a markdown file editable from GNOME prefs, but it no longer carries front matter, template variables, or model routing metadata.
- `active-listener-cli-runtime`: `llm_rewrite` path fields now resolve relative to the loaded config file and are normalized to absolute paths before the runtime starts.
- `broadcast-app-state`: Rewrite prompt failures continue to surface as per-recording pipeline failures, while LiteRT model startup failures must surface as startup fatal errors because the rewrite dependency is no longer optional once enabled.
- `gnome-settings-ui`: The prefs editor keeps the same markdown editing workflow, but its packaged fallback prompt becomes plain markdown content rather than a markdown-plus-front-matter template.

## Impact

- **Active listener runtime**: `packages/active-listener/src/active_listener/rewrite.py`, `settings.py`, `bootstrap.py`, `config.py`, `service.py`, `service_ports.py`, and `recording_finalizer.py` will change because rewrite model lifecycle, prompt loading, config resolution, and shutdown behavior all move.
- **Configuration surface**: `packages/active-listener/config.yaml`, `config.sample.yaml`, CLI/config tests, and operator docs will change because rewrite config now points at a local LiteRT model path instead of a network endpoint.
- **Dependencies**: `packages/active-listener` will gain a LiteRT Python dependency and likely drop the current OpenAI/prompt-parsing dependencies from the active rewrite path.
- **GNOME integration**: `packages/active-listener-ui-gnome/src/prefs.ts` and its packaged prompt asset will change so the fallback prompt remains markdown text without hidden metadata semantics.
- **DBus behavior**: prompt file problems remain observable through `PipelineFailed`, while missing or invalid LiteRT model bundles become startup failures that should publish `FatalError` when DBus is already available.
- **Operations**: the configured LiteRT bundle becomes a required local runtime asset for rewrite-enabled setups, and logs/errors should name the resolved absolute model and prompt paths actually in use.
