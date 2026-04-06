## Why

`active-listener` currently emits finalized dictation exactly as Whisper produced it. That leaves the last mile of punctuation, repetition cleanup, and voice shaping outside the product, even though the recording lifecycle already has a clear post-finalization boundary where this can happen safely.

This change is needed now because the app already has stable finished-recording assembly, and the user wants prompt-driven rewrite behavior they can tune quickly without code changes.

## What Changes

- Add an optional LLM rewrite step that runs only after a recording has been finalized and before text is emitted to the workstation.
- Introduce config-gated rewrite settings in `active-listener`, with the config file becoming the primary source of truth and CLI arguments acting as overrides.
- Add prompt-file-driven rewrite behavior using a Markdown file with YAML front matter and a Jinja2 template body, reloaded before every rewrite call.
- Reserve a `model` front-matter key for runtime model selection; all other parsed front-matter values become Jinja template context.
- Fall back to emitting the raw finalized transcript whenever prompt loading, template rendering, model invocation, or timeout handling fails.
- Add session-story logging around prompt loading, rewrite execution, fallback, and emission, with full raw and rewritten transcript content logged at info level.
- Add local-config support artifacts: a checked-in sample config and a gitignored real config initialized for the user's workstation values.

## Capabilities

### New Capabilities
- `active-listener-llm-rewrite`: Rewrite finalized dictation through a prompt-configured LLM before workstation emission, using an OpenAI-compatible endpoint and per-call prompt reload.
- `active-listener-file-config`: Load `active-listener` runtime settings from a primary config file with typed nested rewrite settings and sample/local config artifacts.

### Modified Capabilities
- `active-listener-hotkey-dictation`: Finished recordings no longer have to emit the raw transcript directly; when enabled, finalized text may be rewritten before emission, while preserving raw-transcript fallback on failures.
- `active-listener-cli-runtime`: CLI/environment-driven startup is narrowed so the config file becomes the main runtime configuration path and CLI arguments override loaded values.

## Impact

- Affected package: `packages/active-listener`
- Affected runtime flow: recording finalization, prompt loading, post-processing, emission, and startup configuration
- Dependency impact: existing `pydantic-ai` (1.74.x docs target) and `python-frontmatter` (1.1.x docs target) package dependencies gain first-party runtime use; `Jinja2` (3.1.x line) still needs to be added
- New config artifacts: checked-in sample config plus a gitignored local config initialized with workstation-specific defaults
- Logging impact: full transcript content logged at info level for rewrite and fallback paths
