## Why

`active-listener` currently treats every finished recording as one flat transcript string. That is enough for raw dictation, but it cannot express “this portion of the recording was spoken as an editing instruction” even though the user’s workflow depends on that distinction. The user wants to hold `Caps Lock` during an active recording to mark command text, see that command text colored differently in the live overlay, and hand the final rewrite model one inline-marked string that preserves surrounding context.

This change is needed now because the timing contract has already moved in the right direction: word timestamps are recording-relative, which makes command-text classification a local `active-listener` concern instead of a server reconstruction problem. The remaining gap is to preserve that timing truth long enough to classify words, group them into meaningful runs, and serialize them for both the overlay and the rewrite step.

## What Changes

- Add command-text gesture handling during active recordings: a second `Caps Lock` press inside an active recording becomes either a finish tap or a command-text hold depending on a fixed 250 ms threshold.
- Record command-text timing on the recording timeline using closed spans plus one optional open-start marker while a committed hold is still in progress.
- Classify transcript words as normal text vs command text by checking whether each word midpoint falls inside a command-text span.
- Replace plain transcript display strings on the D-Bus path with normalized `TextRun` values carrying `text`, `is_command`, and `is_complete`.
- Render command text in the overlay by color only, while preserving the existing incomplete-tail alpha treatment so incomplete command text appears more transparent than committed command text.
- Serialize the same `TextRun` sequence into one flat rewrite input string, wrapping command-text runs in literal `<instruction>...</instruction>` markers.
- Keep the existing no-emission rewrite-failure behavior and make that failure visible in GNOME via the existing D-Bus error path.

## Scope

### New Capabilities
- `active-listener-command-text`: Mark spoken command text during an active recording by holding `Caps Lock`, then preserve that distinction through overlay rendering and final rewrite serialization.

### Modified Capabilities
- `active-listener-hotkey-dictation`: a `Caps Lock` press during recording is no longer always finish; it becomes a pending finish-or-command-text gesture resolved by a fixed hold threshold.
- `active-listener-transcript-reducer`: live transcription reduction must preserve enough timing and completeness truth to produce normalized `TextRun` output instead of plain strings.
- `active-listener-dbus-overlay`: the D-Bus display contract shifts from transcript segments to preclassified `TextRun` values so the GNOME side paints color/alpha rather than reconstructing transcript semantics.
- `active-listener-llm-rewrite`: rewrite input becomes one inline-marked string where command text is rendered as `<instruction>...</instruction>` inside the dictated context.

## Impact

Affected systems include `packages/active-listener` input handling, recording reduction, finalization, and D-Bus broadcasting; `packages/active-listener-ui-gnome` transcript display rendering; and the existing rewrite path that already consumes one finalized transcript string. The feature intentionally does not define any new idle-mode `Caps Lock` behavior yet: command-text holds only mean something during an active recording, and the first `Caps Lock` press still starts recording under the existing policy.
