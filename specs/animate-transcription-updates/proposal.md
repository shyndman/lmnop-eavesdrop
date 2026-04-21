## Why

The GNOME overlay already receives live transcription updates, but it renders each update as an immediate text replacement.

That works functionally, but it makes the overlay feel abrupt, especially when the tail changes rapidly or a wrapped transcript jumps to a different height.

The goal of this feature is to make transcription updates feel intentional and readable while preserving the existing single flowing transcript model.

This matters now because the overlay is moving from proof-of-concept rendering toward a UI Scott will actually live with while recording.

## What Changes

- Animate transcription updates inside the existing single text flow instead of replacing the whole visible transcript abruptly.
- Fade the outgoing tail of the previous transcript state out, then fade the incoming tail of the next transcript state in, at per-character range granularity.
- Start overlay height tweening at the transcript swap point so multiline reflow changes are visually smoothed instead of snapping.
- Keep the transcript as one wrapped text layout; do not split completed and incomplete text into separate visible blocks.
- Preserve the current D-Bus transcription contract and existing overlay responsibilities outside transcript animation.

## Scope

### New Capabilities
- `animated-transcription-transitions`: Animate GNOME overlay transcript updates by driving per-range text attributes over one flowing `ClutterText`, including outgoing-tail fade, incoming-tail fade, and synchronized container height tweening at the swap boundary.

### Modified Capabilities
- `gnome-overlay-transcript-rendering`: Change transcript rendering from immediate whole-string replacement to phased animated transitions while keeping one wrapped label as the visible transcript surface.
- `gnome-overlay-layout`: Change overlay sizing behavior so transcript height changes are tweened when wrapped layout changes at the transcript swap point.

## Impact

Affected systems are concentrated in `packages/active-listener-ui-gnome`, especially `src/extension.ts` and any supporting tests or design assets for the overlay.

The feature relies on existing GNOME Shell text/layout primitives:

```text
St.Label
ClutterText
Pango attributes
actor sizing / clipping
```

It does not require new D-Bus APIs or server-side changes.

Existing transcription delivery, spectrum updates, and recording-state wiring remain in place. This feature changes presentation, state management, and layout behavior in the GNOME extension.