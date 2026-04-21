## Context

`packages/active-listener-ui-gnome/src/extension.ts` already receives live transcription updates and renders them through a single `St.Label` inside the overlay.

Today, the extension still swaps the whole visible transcript at once. The current path is:

```text
renderOverlay()
  -> compute completedTranscript / transcript
  -> setOverlayText(completedText, incompleteText)
  -> getOverlayClutterText()
  -> set_markup(buildOverlayTextMarkup(...))
```

That means the overlay already keeps one flowing label and dims the incomplete tail inline, but every transcript update still replaces the whole markup at once.

The result is abrupt tail edits and abrupt multiline height changes.

Earlier design work confirmed two constraints:
- the visible transcript must stay a single flowing text layout
- per-range styling is available through `St.Label.get_clutter_text()` plus `ClutterText.set_attributes(Pango.AttrList)`

Two GNOME details shape the design:
- Pango range attributes are applied by UTF-8 byte range, not JavaScript string index.
- Wrapped text height is width-dependent and recomputed during relayout, so swapping to the full target text immediately produces the final wrapped layout even if part of the tail is visually hidden.

The feature is therefore an animation/state-management change inside the GNOME extension.

It does not require:
- D-Bus contract changes
- server-side changes
- a second transcript representation

Current baseline, spelled out for implementation:
- Transcript updates arrive through `handleTranscriptionUpdated(parameters)`.
- That handler appends normalized completed segments into `completedTranscriptParts`, stores the latest incomplete tail in `incompleteTranscriptText`, then calls `renderOverlay()`.
- `renderOverlay()` computes two strings:
  - `completedTranscript = completedTranscriptParts.join(' ')`
  - `transcript = [completedTranscript, incompleteTranscriptText].filter(...).join(' ')`
- `renderOverlay()` currently calls `setOverlayText(completedTranscript, incompleteTranscriptText)` and then `showOverlay(false)`.
- `setOverlayText()` reaches the label’s borrowed `ClutterText` and calls `set_markup(buildOverlayTextMarkup(...))`.

That current pipeline matters because the new animation controller is replacing the whole-markup swap, not inventing a second data source.

## Goals / Non-Goals

**Goals:**
- Animate transcript updates inside one wrapped `ClutterText` flow.
- Fade outgoing tail graphemes out and incoming tail graphemes in using Pango range attributes.
- Start overlay height tweening at the text swap so multiline transitions do not visibly jump.
- Keep transcription truth anchored to the latest full transcript string coming from existing extension state.
- Keep the implementation local to the GNOME extension.

**Non-Goals:**
- Changing the D-Bus transcription payloads or reducer semantics.
- Splitting completed and incomplete transcript into separate visible label actors.
- Recomputing audio spectrum or changing spectrum transport/rendering.
- Designing a physically perfect continuous reflow animation.
- Adding shader effects, alpha masks, or whole-label animation as the primary transcript effect.

## Decisions

### 1. Replace whole-markup replacement with a transcript animation controller

The extension will stop treating transcript rendering as one whole-markup replacement.

Today the relevant path is:

```ts
setOverlayText(completedText, incompleteText)
overlayClutterText.set_markup(buildOverlayTextMarkup(completedText, incompleteText))
```

Instead, it will introduce a small controller that owns:
- the latest canonical transcript string
- the currently installed label text buffer
- the active animation phase (`idle`, `erasing-old-tail`, `revealing-new-tail`)
- timing metadata for grapheme-level alpha envelopes

At the extension boundary, transcription updates still produce one canonical transcript string:

```ts
const canonicalText = [completedTranscriptParts.join(' '), incompleteTranscriptText]
  .filter((part) => part.length > 0)
  .join(' ');
```

The controller consumes that full string and decides whether to animate, but the visible surface remains the existing single `St.Label`.

Recommended controller shape for a junior engineer:

```ts
type TranscriptPhase = 'idle' | 'erasing-old-tail' | 'revealing-new-tail';

type GraphemeSpan = {
  text: string;
  graphemeIndex: number;
  startByte: number;
  endByte: number;
};

type TransitionPlan = {
  sourceText: string;
  targetText: string;
  sourceSpans: GraphemeSpan[];
  targetSpans: GraphemeSpan[];
  commonPrefixCount: number;
};

type TranscriptAnimationState = {
  canonicalText: string;
  installedText: string;
  phase: TranscriptPhase;
  plan: TransitionPlan | null;
  frameSourceId: number | null;
  phaseStartedAtMs: number;
};
```

The spec does not require these exact type names, but it does require this level of separation:
- `canonicalText`: newest transcript the UI should converge to
- `installedText`: actual text currently installed in `ClutterText`
- `phase`: whether the controller is idle, erasing, or revealing
- `plan`: precomputed source/target grapheme info for the current transition
- `frameSourceId`: the active GLib timer source, if any

Without this separation, interruption handling becomes guesswork.

Why this over alternatives:
- Keeps all animation policy in one place instead of spreading it across D-Bus handlers and overlay rendering.
- Preserves the existing transcript truth model: one canonical transcript string at a time.
- Makes interruption handling explicit instead of implicit.

Alternatives considered:
- Keep `setOverlayText()` as whole-markup replacement and bolt animation onto CSS/label properties: rejected because the animation target is the text content itself, not the whole actor.
- Create separate actors for stable text and animated tail: rejected because it breaks the agreed single-flow layout model.

### 2. Animate grapheme clusters, not JavaScript code units

Per-character animation will be defined over grapheme clusters. The controller will segment both source and target transcript strings with `Intl.Segmenter` at `{ granularity: 'grapheme' }`, then derive UTF-8 byte ranges for each grapheme using `TextEncoder`. Those byte ranges become the `start_index` / `end_index` values for Pango attributes.

Implementation consequence:
- range math is done in grapheme order for animation sequencing
- range application is done in UTF-8 bytes for Pango

Why this over alternatives:
- Matches the visual idea of “individual characters” better than code-unit indexing.
- Prevents broken animation for emoji, combining marks, and non-ASCII text.
- GJS in this environment exposes both `Intl.Segmenter` and `TextEncoder`, so this is available without new dependencies.

Implementation-facing API details locked for this decision:
- Runtime support is verified in the current environment with `gjs 1.86.0`; both `Intl.Segmenter` and `TextEncoder` are present.
- Package-level typing is not ready today: `packages/active-listener-ui-gnome/package.json` does not declare `@girs/pango`, and `ambient.d.ts` does not import it. Implementation must add `@girs/pango` to `devDependencies`, add `import '@girs/pango';` to `ambient.d.ts`, and add `import Pango from 'gi://Pango';` in `src/extension.ts` before using `Pango.AttrList` APIs.
- Pango attribute ranges are UTF-8 byte offsets, not JS string indices. Each animated grapheme span must therefore be converted to a half-open byte range `[start, end)` before assigning `start_index` / `end_index`.

Recommended helper output for a junior engineer:

```ts
function buildGraphemeSpans(text: string): GraphemeSpan[];
function computeCommonPrefixCount(source: GraphemeSpan[], target: GraphemeSpan[]): number;
function buildTransitionPlan(sourceText: string, targetText: string): TransitionPlan;
```

Expected behavior of `buildGraphemeSpans(text)`:
- Use `new Intl.Segmenter(undefined, { granularity: 'grapheme' })`
- Iterate in grapheme order
- Track cumulative UTF-8 byte length with one `TextEncoder`
- Return one span per visible grapheme with both grapheme index and `[startByte, endByte)`

Expected behavior of `computeCommonPrefixCount(...)`:
- Compare grapheme text in order
- Stop at the first mismatch
- Return the number of leading graphemes that are identical

That common-prefix count is the boundary between the stable prefix and the animated tail.

Alternatives considered:
- Use `string[index]` / UTF-16 code units: rejected because it mis-segments visible characters.
- Use code points via `Array.from()`: rejected because it still splits some grapheme clusters incorrectly.

### 3. Use a two-phase tail transition with one text swap

Each transcript change will be animated in three logical steps:
1. Compute the longest common grapheme prefix between the current label text and the next canonical transcript.
2. Keep the old full text buffer installed and fade only the outgoing tail graphemes toward alpha 0.
3. When the erase phase reaches the common prefix, swap the installed label text to the full target string once, immediately hide the new tail graphemes with attributes, and then fade that new tail in.

This mirrors the approved algorithm shape from the canvas prototype while adapting it to Pango attributes:
- text changes only at the swap boundary
- every other animation frame updates attributes only

Reference flow:

```text
old text buffer
  -> fade old tail out by grapheme range
  -> set_text(full target) once
  -> set_attributes(hide new tail)
  -> fade new tail in by grapheme range
```

The transition state machine should be implemented literally, not implicitly:

```text
idle
  -> if new canonical text differs from installed text: build plan, enter erasing-old-tail

erasing-old-tail
  -> keep installedText = sourceText
  -> update only attributes each frame
  -> when erase phase completes: install targetText once, clear/reset attributes, enter revealing-new-tail

revealing-new-tail
  -> keep installedText = targetText
  -> update only attributes each frame
  -> when reveal phase completes: clear transition state, enter idle
```

Reference pseudocode:

```python
def on_transcript_changed(next_text: str) -> None:
    if next_text == state.canonical_text and state.phase == 'idle':
        return

    state.canonical_text = next_text
    restart_transition(from_text=state.installed_text, to_text=next_text)


def restart_transition(from_text: str, to_text: str) -> None:
    stop_frame_loop()
    clutter_text.set_attributes(None)

    plan = build_transition_plan(from_text, to_text)
    if plan.source_text == plan.target_text:
        state.installed_text = to_text
        state.phase = 'idle'
        return

    state.plan = plan
    state.installed_text = plan.source_text
    state.phase = 'erasing-old-tail'
    state.phase_started_at_ms = now_ms()
    start_frame_loop()
```

The crucial point for a junior engineer: `installedText` does not change during erase, and it does not change during reveal. It changes exactly once, at the swap.

Why this over alternatives:
- Preserves the approved “set text once at swap; animate attributes otherwise” model.
- Gives the reveal phase access to final wrapped layout immediately after the swap.
- Avoids per-frame `set_text()` churn.

Implementation-facing API details locked for this decision:
- `St.Label.get_clutter_text()` returns the internal `ClutterText` with borrowed ownership. The returned actor is owned by the `St.Label` and must not be unreffed or destroyed.
- `ClutterText.set_attributes(attrs)` is the styling entry point for the animation. Passing `null` clears attributes; `get_attributes()` returns `null` when none were set.
- The controller must clear attributes explicitly at transition boundaries and interruptions with `clutterText.set_attributes(null)` before applying a new list.
- `Pango.AttrList` is the container passed into `set_attributes()`. The implementation should build a fresh list for each animation frame or phase application and set it atomically on the current `ClutterText`.
- `Pango.AttrList.insert(attr)` takes ownership of the inserted `PangoAttribute`; do not reuse a `PangoAttribute` after inserting it into the list.

Alternatives considered:
- Append/remove characters directly in the label text buffer during the whole animation: rejected because it contradicts the approved swap model and destabilizes wrapping.
- Replace the whole string immediately and animate the entire changed span uniformly: rejected because it loses the per-grapheme feel of the agreed algorithm.

### 4. Height tweening starts at the swap and is handled by explicit allocation + clipping

Because `set_text(full target)` causes the label to prefer its final wrapped height, the overlay must decouple preferred height from visible height during the transition. The overlay actor tree will therefore gain an explicit clipping wrapper around the transcript label.

Conceptual structure:

```text
overlay shell
  -> content box
       -> spectrum section
       -> transcript clip
            -> transcript label
```

At the swap boundary the controller will:
- measure the old transcript preferred width/height through `get_preferred_width(-1)` and `get_preferred_height(fixedTranscriptWidth)`
- freeze the transcript clip height and outer shell height at their old values
- call `set_text(full target)`
- measure the new transcript preferred height for the same width
- start tweens from old height to new height on the transcript clip and shell
- run the reveal phase while those tweens are in flight

For a junior engineer, the actor tree change should be understood as a before/after refactor:

Current tree:

```text
overlay
  -> overlayContent
       -> spectrumFrame
       -> overlayLabel
```

Target tree:

```text
overlay
  -> overlayContent
       -> spectrumFrame
       -> transcriptClip
            -> overlayLabel
```

Responsibilities of the new `transcriptClip` actor:
- owns the explicit animated height during swap/reveal
- clips any overflow from the already-reflowed final text layout
- does not own text styling logic

Responsibilities that stay on `overlayLabel` / `ClutterText`:
- installed transcript text buffer
- line wrapping
- current `Pango.AttrList`

Reference swap pseudocode:

```python
old_text = state.installed_text
old_text_h = measure_text_height_for_current_width()
old_shell_h = measure_shell_height_for_current_width()

transcript_clip.set_height(old_text_h)
overlay.set_height(old_shell_h)
transcript_clip.set_clip_to_allocation(True)

clutter_text.set_text(plan.target_text)
state.installed_text = plan.target_text

new_text_h = measure_text_height_for_current_width()
new_shell_h = measure_shell_height_for_current_width()

apply_hidden_new_tail_attributes(plan)
animate_height(transcript_clip, old_text_h, new_text_h)
animate_height(overlay, old_shell_h, new_shell_h)
```

Important implementation note: measure shell height after the label text is swapped, because shell height includes padding and any content layout changes above the transcript.

The text reflow still happens internally at swap time; the clip and shell tween smooth what is visible.

Why this over alternatives:
- Matches the approved “height tween at the swap” decision.
- Works with the chosen attribute-only reveal phase.
- Keeps line wrapping authoritative to Pango instead of trying to fake reflow.

Implementation-facing API details locked for this decision:
- Height/width measurement must stay in `get_preferred_*` calls, not allocation-box reads. `get_preferred_width(-1)` is the unconstrained baseline; `get_preferred_height(forWidth)` computes wrapped height for that width.
- Preferred sizes are layout values only; scale and translation do not affect them.
- `set_height(height)` is the API for freezing/tweening wrapper and shell height. `set_height(-1)` restores preferred-height behavior after the transition if the implementation wants to relinquish the override.
- The transcript clip wrapper must enable `set_clip_to_allocation(true)` and must not also use `set_clip(...)`, because an explicit clip rect overrides `clip-to-allocation`.
- Height tweening must target the animatable `height` property and should cancel any in-flight transitions first via the existing `remove_all_transitions()` pattern already used in `extension.ts`.

Alternatives considered:
- Let the label allocate immediately to its new height and accept the snap: rejected because the user explicitly wants height tweening.
- Keep the old layout until reveal finishes, then swap: rejected because it delays final wrapping too long and breaks the agreed algorithm.
- Animate height by progressively appending real text instead of swapping: rejected because it reintroduces repeated `set_text()` during reveal.

### 5. Drive per-grapheme alpha with a timer-backed frame loop

Pango range styling does not expose native tweening for span alpha, so the controller will own a timer-backed frame loop that recalculates a fresh `Pango.AttrList` for the current phase and elapsed time. Each frame will:
- identify the graphemes that should currently be visible, fading, or hidden
- coalesce adjacent graphemes with equal alpha where practical
- build a new `Pango.AttrList`
- apply it to the label’s `ClutterText`

The timing model stays algorithmic rather than paint-driven:
- erase phase: outgoing tail graphemes fade in sequence from the end of the old tail back toward the common prefix
- reveal phase: incoming tail graphemes fade in sequence from the common prefix outward toward the new end

The exact constants stay local to the extension as tuning values, but the structure is fixed: staggered grapheme starts plus an overlap fade duration.

Recommended timing model for a junior engineer:
- Choose constants near the existing overlay constants block in `extension.ts`
- Keep them as named values, not literals inside the controller
- Use one frame timer (for example `GLib.timeout_add(..., 16, ...)`) to drive attribute recomputation

Recommended per-phase math:

```python
def erase_alpha_for_tail_index(tail_index: int, elapsed_ms: float) -> int:
    start_ms = tail_index * ERASE_STAGGER_MS
    progress = clamp((elapsed_ms - start_ms) / FADE_DURATION_MS, 0.0, 1.0)
    return alpha16(1.0 - progress)


def reveal_alpha_for_tail_index(tail_index: int, elapsed_ms: float) -> int:
    start_ms = tail_index * REVEAL_STAGGER_MS
    progress = clamp((elapsed_ms - start_ms) / FADE_DURATION_MS, 0.0, 1.0)
    return alpha16(progress)
```

Where:
- `tail_index` is the grapheme offset inside the animated tail, not inside the full string
- `alpha16(x)` converts `0.0..1.0` to the Pango alpha domain
- graphemes in the stable prefix do not receive animation attributes

Recommended attr-list construction rules:
- Start each frame with `const attrs = new Pango.AttrList()`
- Insert only the attributes required for the current phase
- Prefer one attribute per contiguous run of equal alpha instead of one attribute per grapheme when multiple adjacent graphemes have the same alpha
- Call `clutterText.set_attributes(attrs)` once per frame

Implementation-facing API details locked for this decision:
- The fade effect should use `Pango.attr_foreground_alpha_new(alpha)` on each animated range.
- `alpha` must be mapped into Pango’s 16-bit domain, not CSS or byte alpha. Treat the accepted range as `0..65535` in code, with doc/source guidance showing the constructor in the `guint16` / `1..65536` domain; clamp explicitly before creating the attribute.
- `PangoAttribute.start_index` and `end_index` are byte offsets, and `end_index` is exclusive.
- Because `Pango.AttrList` is linear, the implementation should coalesce adjacent graphemes with the same computed alpha before inserting attributes where practical.

Why this over alternatives:
- Matches the approved algorithm while staying inside native text/layout primitives.
- Keeps the text actor truthful: one label, one text buffer, one attr list at any point in time.

Alternatives considered:
- Rebuild Pango markup strings every frame: rejected in favor of direct attributes.
- Use `ClutterText` property animation for inline spans: rejected because inline span alpha is not exposed as a first-class tween target.

### 6. New transcript updates interrupt cleanly and retarget from the currently installed text buffer

Live transcription updates can arrive while an animation is in flight. The controller will support one active transition at a time. When a new canonical transcript arrives:
- if the current phase is `idle` and the text is unchanged, do nothing
- otherwise stop the frame loop, clear transient attributes, and use the label’s currently installed text buffer as the source string for the next transition
- start a fresh transition toward the new canonical transcript

This means interruption behavior differs by phase:
- during erase, the source string is still the old full text buffer
- during reveal, the source string is the already-swapped target buffer

Why this over alternatives:
- Guarantees convergence to the latest transcript without stacking animations.
- Keeps state space small and debuggable.
- Preserves transcript truth without inventing a second partially materialized string model.

Alternatives considered:
- Queue every intermediate transcript update: rejected because live incomplete-tail updates can arrive faster than the animation should run.
- Blend multiple overlapping transitions at once: rejected as unnecessary complexity for v1.

Canonical interruption sequence:
1. stop the active timer/loop
2. call `clutterText.set_attributes(null)`
3. read the controller’s currently installed text buffer as the next source string
4. compute the new target transition plan
5. start the next erase/reveal cycle

Two examples, to remove ambiguity:

```text
Case A: interruption during erase
- source text on screen is still the old transcript
- new transition starts from that old transcript

Case B: interruption during reveal
- installed text was already swapped to the previous target
- new transition starts from that already-swapped text, not from the original old text
```

This is why `installedText` must be stored explicitly.

### 7. Validation and package setup are explicit parts of the feature

This package currently has build, typecheck, install, and manual GNOME-session scripts, but it does not have a package-local automated test runner.

The available scripts today are:

```text
build
typecheck
install:extension
wayland:test
```

The implementation must therefore treat package setup and scripted validation as part of the feature rather than assuming those facilities already exist.

Implementation-facing details locked for this decision:
- Type/build validation is available today through `npm run typecheck` and `npm run build` in `packages/active-listener-ui-gnome`.
- Manual GNOME Shell validation is available today through `npm run wayland:test`.
- Helper/controller validation for grapheme planning and transition sequencing must use a package-local scripted harness unless the same change introduces a dedicated automated test runner. The current package layout does not provide one.
- The spec does not require a full test framework adoption; a focused scripted harness is sufficient as long as it covers ASCII, emoji, multiline input, and interruption sequencing.

For a junior engineer, the minimum scripted-harness cases are:
- identical strings: no transition should start
- pure append: common prefix is full source length
- pure delete: common prefix is full target length
- middle divergence with shared prefix: common prefix stops at first changed grapheme
- emoji / combining-mark examples: grapheme boundaries remain intact
- multiline-width-sensitive strings: byte range generation must remain correct even when the final layout wraps differently
- interruption during erase and interruption during reveal: source text for the restarted transition is correct

The harness does not need to boot GNOME Shell. It only needs to exercise the pure planning/controller logic that can run outside the UI.

## Implementation Walkthrough

Recommended coding order for a junior engineer:
1. Add Pango typing/runtime imports and make `typecheck` pass.
2. Extract pure grapheme-planning helpers with a small scripted harness.
3. Add a `transcriptClip` wrapper without changing transcript behavior yet.
4. Replace `setOverlayText(...)` with a controller that can still do an immediate no-animation install.
5. Add erase/reveal attr-list generation.
6. Add swap-time height tweening.
7. Add interruption handling.
8. Run package validation and manual GNOME verification.

This order is important because it isolates failures:
- if grapheme planning is wrong, debug it outside GNOME Shell first
- if height tweening is wrong, debug it after the text controller already works
- if interruption is wrong, debug it after the normal one-shot transition works

## Failure Modes To Watch For

- Text appears but never clears styling after animation completes
  - likely cause: missing `set_attributes(null)` on phase completion
- Height tween runs but text is visibly cut off forever
  - likely cause: explicit `set_height(...)` never reset or clip wrapper height not updated to final value
- Range styling applies to the wrong characters
  - likely cause: JS indices were used instead of UTF-8 byte offsets
- Emoji or accented characters split into separate fades
  - likely cause: code-point iteration instead of grapheme segmentation
- New updates cause flicker or jump back to stale text
  - likely cause: restart logic is using canonical text instead of installed text as the next source
- `typecheck` fails after adding Pango usage
  - likely cause: missing `@girs/pango` dependency or missing ambient import

## Risks / Trade-offs

- [Frequent live updates restart the animation often] → Keep interruption semantics simple and always retarget to the freshest transcript so the overlay stays truthful even if some intermediate animation beats are skipped.
- [Incorrect byte-range math breaks styling on non-ASCII text] → Base sequencing on grapheme segmentation and derive Pango indices from UTF-8 bytes, not JS indices.
- [Attribute rebuilds become expensive on long transcripts] → Keep the overlay transcript window bounded by the existing overlay usage pattern and coalesce adjacent equal-alpha runs when building the attr list.
- [Height tween exposes clipped or awkward intermediate states] → Freeze old allocation before swap, clip transcript overflow, and tween shell and transcript clip in sync.
- [Layout metrics drift if styling changes affect glyph measurement] → Restrict the animated attributes to alpha/color-only changes so wrapping and preferred height remain tied to the installed text buffer rather than the current frame attributes.

## Migration Plan

1. Refactor overlay transcript rendering behind a dedicated controller/helper inside `packages/active-listener-ui-gnome/src/extension.ts`.
2. Introduce the transcript clip wrapper and explicit height tween path without changing D-Bus handling.
3. Replace the current whole-markup `setOverlayText(...)` path with the two-phase grapheme animation controller.
4. Verify interruption behavior with repeated incomplete-tail updates and multiline transcript transitions.
5. Rollback strategy: revert the extension to the current `setOverlayText(...)` / `buildOverlayTextMarkup(...)` path and remove the controller/wrapper code. No data migration or protocol rollback is required.

## Open Questions

None. The spec review resolved the codebase assumptions by tightening the package setup and API contract in this document; no unresolved design questions remain.