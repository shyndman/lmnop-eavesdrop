## Context

`active-listener` already has a clean boundary for post-processing: `_finalize_recording()` flushes the remaining transcription, assembles a finalized raw transcript from committed segments, and emits a single string through the text emitter. The live transcription loop is timing-sensitive and should remain focused on recording control, reducer updates, reconnect handling, and final emission.

This feature adds a rewrite step only at that finalized boundary. The user wants prompt-driven behavior they can iterate on quickly, so the prompt file must be reloaded before every LLM call. The user also wants the rewrite endpoint to be OpenAI-compatible, defaulting to Ollama via `base_url`, and wants full transcript content logged at info level because session reconstruction matters more than log volume for this workflow.

Configuration also needs to shift. Today `active-listener` is primarily configured through CLI/env-derived values. For this feature, a config file becomes the primary source of truth, validated through Pydantic models, with CLI arguments acting as overrides. The prompt file owns the model choice and prompt variables; the runtime config owns whether rewrite is enabled and where requests are sent.

Current implementation touchpoints, verified from the repository:
- `packages/active-listener/src/active_listener/app.py`
  - `ActiveListenerConfig(BaseModel)` currently validates `keyboard_name`, `host`, `port`, `audio_device`, and `ydotool_socket` with `ConfigDict(strict=True)`.
  - `ActiveListenerService._finalize_recording(...)` currently executes this exact sequence: `await self.client.flush(force_complete=True)` → disconnect-generation check → `_ingest_transcription_message(...)` → `render_text(reducer_state.parts)` → `self.emitter.emit_text(text)`.
  - Emission is skipped when `disconnect_generation` changes, when the finalized text is empty, or when `emit_text(...)` raises.
  - `ActiveListenerLogger` is a protocol with `info(event: str, **kwargs)`, `warning(event: str, **kwargs)`, and `exception(event: str, **kwargs)`.
- `packages/active-listener/src/active_listener/cli.py`
  - `ActiveListenerCommand` currently sources defaults from environment variables: `ACTIVE_LISTENER_KEYBOARD_NAME`, `EAVESDROP_HOST`, `EAVESDROP_PORT`, `EAVESDROP_AUDIO_DEVICE`, and `YDOTOOL_SOCKET`.
  - Current precedence is env/defaults first, then CLI overrides, then `ActiveListenerConfig.model_validate(...)`.
  - There is no existing file-based config loader in this package.
- `packages/active-listener/pyproject.toml`
  - `pydantic-ai>=1.74.0` and `python-frontmatter>=1.1.0` are already declared dependencies.
  - `jinja2` is not currently declared and must be added if this design is implemented.
- `packages/active-listener/src/active_listener/rewrite_prompt.md`
  - The prompt file already exists at the path referenced by this spec and currently contains placeholder content.
- `packages/active-listener/tests/`
  - Existing tests already cover service flow (`test_app.py`), CLI startup (`test_cli.py`), emitter behavior (`test_emitter.py`), and reducer helpers (`test_reducer.py`).
  - `test_app.py` uses a `RecordingLogger` stand-in that captures structured `event` plus `fields`, so transcript-bearing info logs fit the current test style.
- Repository config convention
  - `packages/server/config.sample.yaml` is checked in and `packages/server/config.yaml` is gitignored via `packages/server/.gitignore`.
  - `packages/active-listener/` does not yet have corresponding config artifacts, so the spec should follow the server package naming convention rather than invent a new one.

Verified third-party baseline and implementation-facing APIs:
- `pydantic-ai`
  - Current PyPI release: `1.74.0`; implementation should target the 1.74.x docs rather than a remembered older API.
  - Relevant documented surface:
    - `from pydantic_ai import Agent`
    - `from pydantic_ai.models.openai import OpenAIChatModel`
    - `from pydantic_ai.providers.openai import OpenAIProvider`
    - `Agent(model, instructions=...)` or equivalent agent construction with no tools and default plain-text output.
    - `await agent.run(user_prompt)` returning a result whose `.output` is the final text.
    - `OpenAIChatModel('model_name', provider=OpenAIProvider(base_url=..., api_key=...))` for OpenAI-compatible endpoints.
  - The PydanticAI OpenAI docs show `base_url` and `api_key` on `OpenAIProvider`, and support OpenAI-compatible chat-completions backends.
- `python-frontmatter`
  - Current PyPI release: `1.1.0`; implementation should target the documented `frontmatter.load(...)` / `frontmatter.parse(...)` API.
  - Relevant documented surface:
    - `post = frontmatter.load(path_or_file, encoding='utf-8')`
    - `post.metadata` for parsed front matter
    - `post.content` for the Markdown body
    - `frontmatter.parse(text)` returning `(metadata, content)` if lower-level parsing is preferred
  - The docs explicitly note BOM-safe loading via `encoding='utf-8-sig'` when needed.
- `Jinja2`
  - Current PyPI release: `3.1.6`; add a non-patch-pinned dependency in the 3.1 line rather than freezing an exact patch number.
  - Relevant documented surface:
    - `from jinja2 import Environment, StrictUndefined`
    - `Environment(undefined=StrictUndefined, autoescape=False)`
    - `env.from_string(template_text)`
    - `template.render(**context)`
  - Jinja docs state that `StrictUndefined` causes undefined variables to raise rather than silently rendering empty output.
- Ollama OpenAI-compat API
  - Current docs describe `http://localhost:11434/v1/` as the OpenAI-compatible base URL shape.
  - Ollama examples require an API key field but state it is ignored, using `api_key='ollama'` in examples.
  - Supported endpoint for this feature is `/v1/chat/completions`; this spec does not need `/v1/responses`, tools, or structured outputs.

## Goals / Non-Goals

**Goals:**
- Add an optional rewrite path that runs only after recording finalization and before workstation text emission.
- Keep the live transcription and hotkey lifecycle unchanged while isolating LLM latency to background finalization.
- Represent rewrite settings as a nested Pydantic config model inside `ActiveListenerConfig`.
- Make a file-based config the main startup input, with CLI values overriding loaded config fields.
- Reload `rewrite_prompt.md` before each LLM call so prompt edits apply immediately.
- Parse prompt front matter with `python-frontmatter`, reserve `model` as a runtime-only key, and pass all remaining front-matter keys into Jinja2 as strict template context.
- Use PydanticAI with a plain-text output contract against an OpenAI-compatible backend configured by `base_url`.
- Fall back to raw transcript emission on prompt parsing, template rendering, missing model, timeout, or model-call failure.
- Add session-story logging, including full raw and rewritten transcript content at info level.
- Provide both a checked-in sample config and a gitignored local config initialized for the user workstation.
- Implement against the documented third-party APIs above so the implementation does not need ad hoc exploration.

**Non-Goals:**
- Performing any rewrite during live segment streaming.
- Defining or validating a schema for prompt front matter beyond the reserved `model` key.
- Returning structured model output.
- Adding provider-specific configuration beyond an OpenAI-compatible `base_url`.
- Retaining transcripts outside logs or introducing a persistence store.
- Designing prompt wording, voice policy, or rewrite instructions in code.

## Implementation Blueprint

This section is intentionally procedural. A junior engineer should be able to implement the feature by following it without having to rediscover the intended shape.

### Runtime data flow

1. Startup loads `packages/active-listener/config.yaml`.
2. Startup overlays any CLI-provided values on top of the file values.
3. The merged data is validated into `ActiveListenerConfig`, which now includes a nested rewrite config model.
4. During a normal recording, the existing foreground flow is unchanged.
5. When recording finishes, `_finalize_recording(...)` still flushes the server and assembles the raw finalized transcript.
6. If rewrite is disabled, emit the raw transcript exactly as today.
7. If rewrite is enabled:
   - read `rewrite_prompt.md`
   - parse front matter and body with `python-frontmatter`
   - remove the reserved `model` key from metadata
   - render the body as a Jinja2 template using the remaining metadata as context
   - build a PydanticAI agent for that `model`
   - submit the raw transcript as the user prompt
   - if a non-empty text response is returned within timeout, emit that rewritten text
   - otherwise log the failure and emit the raw transcript

### Concrete config shape

The intended validated config shape is:

```python
class LlmRewriteConfig(BaseModel):
    enabled: bool
    base_url: str
    timeout_s: int = 30
    prompt_path: str


class ActiveListenerConfig(BaseModel):
    keyboard_name: str
    host: str
    port: int
    audio_device: str
    ydotool_socket: str | None = None
    llm_rewrite: LlmRewriteConfig
```

Notes for the implementor:
- Keep strict validation enabled.
- `prompt_path` belongs in config even though the initial file lives at `packages/active-listener/src/active_listener/rewrite_prompt.md`; this keeps the path explicit and testable.
- `timeout_s` is part of config even though the current desired value is 30, because the user wants this tunable while prompt/model tuning is active.

### Concrete sample config shape

The sample and local config files should have this structure:

```yaml
keyboard_name: "AT Translated Set 2 keyboard"
host: "home-brainbox"
port: 9090
audio_device: "default"
ydotool_socket: "/run/user/1000/.ydotool_socket"

llm_rewrite:
  enabled: true
  base_url: "http://localhost:11434/v1"
  timeout_s: 30
  prompt_path: "packages/active-listener/src/active_listener/rewrite_prompt.md"
```

If implementation discovers a better local default for `audio_device`, it may change the value, but the field itself is required and must be present in both sample and local config.

### CLI override behavior

Do not make the CLI reconstruct config manually from scratch anymore.

Instead, the implementation should follow this sequence:

1. load the config file into a plain dict
2. collect CLI values
3. treat explicitly provided CLI values as overrides
4. merge them onto the loaded dict
5. validate the merged dict with `ActiveListenerConfig.model_validate(..., strict=True)`

Important junior-level detail: a parser default is not always the same thing as a user override. If the CLI library cannot tell you whether a value was explicitly passed, then you need a separate strategy for preserving file-config values rather than accidentally overwriting them with parser defaults.

### Prompt parsing contract

The prompt file contract is exact:

- file format: Markdown with optional YAML front matter
- parser: `frontmatter.load(...)`
- reserved key: `model`
- template source: `post.content`
- template context: every metadata key except `model`

Pseudo-code:

```python
post = frontmatter.load(prompt_path, encoding="utf-8")
metadata = dict(post.metadata)
model_name = metadata.pop("model", None)
template_text = post.content

if not model_name:
    raise RewritePromptError("rewrite prompt front matter must define 'model'")
```

Important junior-level detail: do not mutate `post.metadata` in place if that makes later logging or debugging confusing. Copy it first into a normal dict, then pop `model` from the copy.

### Jinja rendering contract

The rendering contract is also exact:

```python
env = Environment(undefined=StrictUndefined, autoescape=False)
template = env.from_string(template_text)
rendered_system_prompt = template.render(**metadata)
```

Important junior-level detail:
- `StrictUndefined` is required because missing variables must fail hard.
- `autoescape=False` is intentional because this is not HTML rendering.
- A render failure is not recoverable inside rewrite. Log it and fall back to the raw transcript.

### PydanticAI rewrite call contract

The implementation contract is:

```python
provider = OpenAIProvider(
    base_url=config.llm_rewrite.base_url,
    api_key="ollama",
)
model = OpenAIChatModel(model_name, provider=provider)
agent = Agent(model, instructions=rendered_system_prompt)

result = await agent.run(raw_transcript)
rewritten_text = result.output
```

Important junior-level detail:
- The raw transcript is the user prompt.
- The rendered prompt file becomes the system/developer instruction surface via `instructions=`.
- No tools are involved.
- No structured output type is involved.
- The result is expected to be plain text.

### Finalization integration contract

The exact place to integrate rewrite is after:

```python
text = render_text(reducer_state.parts)
```

and before:

```python
self.emitter.emit_text(text)
```

The intended control flow is:

```python
raw_text = render_text(reducer_state.parts)
if not raw_text:
    log and return

text_to_emit = raw_text

if config.llm_rewrite.enabled:
    try:
        text_to_emit = await rewrite(raw_text)
    except Exception:
        log fallback
        text_to_emit = raw_text

self.emitter.emit_text(text_to_emit)
```

Important junior-level detail: keep the disconnect-generation check before rewrite work starts. If the session has already become stale, do not spend time rendering prompts or calling the model.

### Logging contract

Expected lifecycle logs are:
- prompt file loaded
- prompt file missing or parse failed
- prompt model missing
- prompt rendered successfully
- rewrite started
- rewrite succeeded
- rewrite failed / timed out
- raw fallback selected
- final text emitted

Transcript content requirements:
- raw transcript content: full text at info level
- rewritten transcript content: full text at info level
- fallback path: log the raw transcript that was emitted

Important junior-level detail: log payload fields should stay structured, not concatenated into one giant human sentence. Match the existing logger style: event string plus keyword fields.

### Test plan mapping

The existing test layout should be extended rather than replaced:
- `tests/test_cli.py`
  - config loading and CLI override precedence
- `tests/test_app.py`
  - rewrite disabled / success / failure behavior in `_finalize_recording(...)`
  - logging assertions using the existing `RecordingLogger`
- new focused test module if helpful, for example `tests/test_rewrite.py`
  - prompt loading
  - Jinja rendering
  - PydanticAI client wrapper behavior

Important junior-level detail: keep pure parsing/rendering logic in small testable helpers. That will make it possible to test prompt failures without running the whole service loop.

## Decisions

### 1. Rewrite only after finalized transcript assembly
The rewrite step will live between `render_text(reducer_state.parts)` and `emitter.emit_text(...)` inside recording finalization.

Rationale:
- This preserves the existing truthful boundary between incremental transcription and finished dictation.
- It prevents model latency from affecting microphone control, reducer state, or reconnect policy.
- It matches the user requirement that rewrite happens when recording is finished and text is already in hand.

Alternatives considered:
- Rewriting incremental segments during recording: rejected because it would entangle model latency with the foreground recording loop and create unstable intermediate output.
- Rewriting after emitter output: rejected because it would require re-capturing or re-emitting text and breaks the existing emission boundary.

### 2. Use a dedicated nested Pydantic model for rewrite config
`ActiveListenerConfig` will contain a nested rewrite settings model, e.g. `llm_rewrite`, with typed fields such as `enabled`, `base_url`, `timeout_s`, and `prompt_path`.

Rationale:
- The rewrite path introduces a coherent subdomain with its own runtime policy.
- Nesting keeps configuration truthful and discoverable rather than scattering rewrite flags among unrelated fields.
- It gives startup validation a single typed boundary for deciding whether rewrite can run.

Alternatives considered:
- Flat top-level config fields: rejected because it muddies the meaning of `ActiveListenerConfig` and scales poorly as rewrite-specific settings grow.
- Ad hoc dict parsing: rejected because it weakens type safety and startup validation.

### 3. Make the config file the primary runtime source, with CLI overrides
Startup will load `packages/active-listener/config.yaml` first, validate it through Pydantic, then apply CLI overrides on top. A checked-in `packages/active-listener/config.sample.yaml` will document the expected shape, matching the existing `packages/server/` convention.

Rationale:
- The user wants a real editable local config waiting in the repo, not only env-driven values.
- File-based config is a better home for nested rewrite settings and prompt paths.
- CLI overrides preserve one-off experimentation without forcing edits to the local file.

Alternatives considered:
- Keep env vars as primary and add more flags: rejected because the feature now has enough configuration to warrant a file-first workflow.
- Config-file-only without overrides: rejected because CLI overrides are explicitly desired.

### 4. Prompt file is Markdown with YAML front matter parsed by `python-frontmatter`
The prompt document at `packages/active-listener/src/active_listener/rewrite_prompt.md` will be loaded from disk on every rewrite call using `python-frontmatter`. Implementation should use `frontmatter.load(...)` against the file path, then read `post.metadata` and `post.content`. The Markdown body is the Jinja2 template source. Front matter is parsed as-is with no schema validation. If BOM handling becomes necessary, the docs support loading with `encoding='utf-8-sig'`.

Rationale:
- The user wants an off-the-shelf front-matter parser rather than hand-rolled parsing.
- This keeps prompt iteration fast and local to a single editable file.
- Front matter naturally supports lists such as `related_words` and future arbitrary metadata.

Alternatives considered:
- Hand-rolled `---` splitting: rejected because an off-the-shelf parser is simpler and less error-prone.
- Separate YAML + prompt files: rejected because it slows iteration and fragments the prompt surface.

### 5. Reserve `model` as the only runtime-owned front-matter key
Prompt front matter has exactly one key with built-in meaning to code: `model`. Runtime removes it from the parsed metadata before Jinja rendering. All remaining keys become template context untouched.

Rationale:
- The user wants model choice to travel with prompt tuning.
- Reserving only `model` keeps the runtime’s front-matter contract minimal and avoids embedding prompt semantics in code.
- Not passing `model` into Jinja prevents the template surface from depending on a runtime control field.

Alternatives considered:
- Put model in config: rejected because prompt tuning and model selection should change together.
- Validate known prompt keys such as `related_words`: rejected because the user explicitly does not want code to understand the front-matter shape beyond `model`.

### 6. Render the prompt body with Jinja2 using strict undefined behavior
The Markdown body will be rendered through Jinja2 using a local `Environment(undefined=StrictUndefined, autoescape=False)` and `env.from_string(template_text).render(**context)`. Missing variables are fatal to the rewrite attempt.

Rationale:
- The prompt is user-authored and expected to change often.
- Silent empty substitutions would hide prompt bugs and produce misleading model behavior.
- Failing hard keeps prompt mistakes observable and makes raw-transcript fallback deterministic.

Alternatives considered:
- Lenient undefined behavior: rejected because it conceals prompt errors during active tuning.
- `str.format(...)`: rejected in favor of Jinja2 because the user wants a richer template surface and decided against Python t-strings.

### 7. Use PydanticAI as a thin plain-text rewrite client
The rewrite subsystem will use a small, single-purpose PydanticAI agent with plain-text output only, no tools, and a 30-second timeout. The documented construction to target is:

- `provider = OpenAIProvider(base_url=config.llm_rewrite.base_url, api_key='ollama')`
- `model = OpenAIChatModel(model_name, provider=provider)`
- `agent = Agent(model, instructions=rendered_system_prompt)`
- `result = await agent.run(raw_transcript)`
- `rewritten_text = result.output`

`pydantic-ai` is already present in `packages/active-listener/pyproject.toml`, so implementation should wire the existing dependency rather than add a parallel client library.

The hard-coded `api_key='ollama'` is based on Ollama's OpenAI-compat documentation, which says the API key field is required by client shape but ignored by Ollama. This keeps the public config surface at `base_url` only, as requested.

Rationale:
- The user wants PydanticAI even for the simple case because it keeps the path open for future complexity without making the initial use case awkward.
- Plain-text output is more reliable for smaller local models than structured return schemas.
- An OpenAI-compatible provider cleanly supports the user’s default Ollama setup.

Alternatives considered:
- Direct OpenAI client calls without PydanticAI: rejected because the user explicitly wants PydanticAI.
- Structured output schema: rejected because smaller rewrite models may not obey it reliably and the feature only needs rewritten text.

### 8. Failure mode is raw-transcript emission
If any rewrite-stage step fails — prompt load, front-matter parse, missing `model`, Jinja rendering, model timeout, or model response failure — the service emits the original raw transcript instead.

For avoidance of doubt, "fails" here includes these concrete cases:
- `prompt_path` does not exist
- front matter is malformed YAML
- the prompt file has no `model`
- Jinja raises because a template variable is missing
- the rewrite client raises any exception
- the rewrite client exceeds `timeout_s`
- the rewrite client returns an unusable value such as empty text, if implementation chooses to treat empty text as failure

Rationale:
- The user explicitly chose raw-transcript fallback.
- Dictation completion is more important than rewrite success.
- This preserves the pre-feature behavior as the failure mode.

Alternatives considered:
- Emit nothing on failure: rejected because it would make a finished recording silently disappear.
- Block until success with no timeout: rejected because the rewrite path must not wedge finalization indefinitely.

### 9. Log full transcript content at info level
Raw transcript content and rewritten transcript content will be logged in full at info level, alongside lifecycle logs for prompt loading, rewrite start, rewrite success/failure, fallback, and emission.

Rationale:
- The user explicitly wants no shortening and does not want missing transcript detail when debugging a low-volume dictation session.
- Full-content logs plus lifecycle events tell the complete session story.

Alternatives considered:
- Metadata-only logs: rejected because they are insufficient for debugging rewrite quality and fallback behavior.
- Full transcript content at debug only: rejected because the user considers that too lossy operationally.

### 10. Provide both sample and local config artifacts
The repo will gain `packages/active-listener/config.sample.yaml` and `packages/active-listener/config.yaml`, matching the existing `packages/server/` convention. The local config will be initialized with user-specific defaults, including keyboard name, host, port, ydotool socket, and an explicit `audio_device` value.

Rationale:
- The user wants a real local config waiting in the repo and also wants a sample artifact checked in.
- This keeps onboarding and local iteration simple without leaking workstation-specific config into version control.

Alternatives considered:
- Checked-in real config: rejected because the user explicitly does not want that.
- No sample config: rejected because the config file is now the primary source of truth and needs a discoverable shape in version control.

## Risks / Trade-offs

- [Prompt template errors break rewriting] → Use strict Jinja rendering, log the failure cause, and fall back to the raw transcript.
- [Prompt front matter omits `model`] → Treat missing `model` as a hard rewrite failure and emit the raw transcript.
- [Model latency delays post-recording emission] → Keep the timeout at 30 seconds for tuning, run rewrite in background finalization, and preserve raw fallback on timeout/failure.
- [Base-url-only config is Ollama-specific in practice] → Use `api_key='ollama'` internally per Ollama docs now; if authenticated OpenAI-compatible providers are required later, add explicit auth config in a follow-up change.
- [Full transcript logging increases exposure in logs] → Accept the exposure explicitly because the user values complete session reconstruction over low-volume log minimization.
- [Config source transition may temporarily confuse startup behavior] → Make the precedence explicit in design and implementation: config file first, CLI overrides second.
- [Per-call prompt reload adds repeated file I/O] → Accept the small overhead because rewrite volume is low and immediate prompt iteration is a core requirement.
- [Future prompt metadata needs may grow beyond simple scalar/list values] → Preserve a generic passthrough front-matter model so new metadata usually requires no code change unless runtime semantics change.

## Migration Plan

1. Introduce the nested file-based config model and define precedence between config file values and CLI overrides.
2. Add sample/local config artifacts and document the local default path expected at startup.
3. Introduce the rewrite prompt-loading path using `python-frontmatter` plus strict Jinja rendering.
4. Add the PydanticAI rewrite client and wire it into recording finalization before emission.
5. Add info-level transcript and lifecycle logging around success and fallback paths.
6. Verify that disabling rewrite preserves current raw-emission behavior and that rewrite failures emit raw text.

Suggested implementation order for a junior engineer:

1. Add the new config models and load/merge logic first.
2. Add config artifacts (`config.sample.yaml`, `config.yaml`) and ignore rules.
3. Add a small prompt-loader helper and test it thoroughly.
4. Add a small rewrite-client helper and test it separately from the service.
5. Integrate the helper into `_finalize_recording(...)`.
6. Add logging assertions.
7. Run the targeted test suite.

Rollback strategy:
- Disable `llm_rewrite.enabled` in local config to restore raw-emission behavior without removing the feature.
- If needed, revert the feature code while leaving config artifacts in place; the preexisting finalization flow remains the behavioral fallback.

## Open Questions

- The current design intentionally keeps rewrite config at `base_url` only and uses Ollama's documented placeholder `api_key='ollama'` internally. If you want this feature to work against authenticated OpenAI-compatible providers during the first implementation, the config needs an explicit auth field (for example `api_key`) or a documented environment-variable contract.
