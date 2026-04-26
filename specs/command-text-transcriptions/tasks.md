## 1. Recording-time Caps Lock gesture handling
Files likely touched: `packages/active-listener/src/active_listener/infra/keyboard.py`, `packages/active-listener/src/active_listener/app/state.py`, `packages/active-listener/src/active_listener/app/service.py`, existing active-listener input/state tests.

- [ ] 1.1 Replace the current key-down-only `AppAction` keyboard pump with a low-level Caps/Escape event stream that preserves Caps down/up timing, while keeping `handle_action(AppAction.START_OR_FINISH)` available for non-keyboard callers such as D-Bus menu control.
- [ ] 1.1a Verify gesture resolution with focused active-listener tests covering finish tap before threshold, hold commit at threshold, hold span start at key-down, and release closing the command-text span.
- [ ] 1.2 Add recording-session-owned pending gesture state, threshold-task cleanup, and recording-relative command-span timing so the second `Caps Lock` press during an active recording becomes either a finish tap or a committed command-text hold using a fixed 250 ms threshold.
- [ ] 1.2a Verify idle-path behavior with focused tests showing recording still starts under the existing policy and that no new idle hold contract is introduced.

## 2. Command-text timeline and word classification
Files likely touched: `packages/active-listener/src/active_listener/recording/`, active-listener recording tests, and any shared transcript models used on the D-Bus path.

- [ ] 2.1 Replace the current `parts`-based reducer snapshot with a word-aware recording state that preserves completed words, incomplete words, closed command spans, one optional open command start, and duration bounds derived from words rather than stripped segment text.
- [ ] 2.1a Verify timeline invariants with focused tests covering no overlapping open spans, closed-span accumulation, and the invariant that finalization never sees an open command-text span.
- [ ] 2.2 Add word-level command-text classification using recording-relative word timestamps and midpoint-in-span matching for both completed words and the unstable tail, and fail explicitly when a command-text recording lacks required `Segment.words` payloads.
- [ ] 2.2a Verify classification with focused tests covering normal words, command-text words, midpoint boundary behavior, and words matched by an open command-text span.

## 3. TextRun reduction and D-Bus contract
Files likely touched: `packages/active-listener/src/active_listener/recording/reducer.py`, `packages/active-listener/src/active_listener/app/`, D-Bus broadcaster/client types, `packages/active-listener-ui-gnome/src/active-listener-service-client.ts`, `packages/active-listener-ui-gnome/src/transcript-overlay.ts`, `packages/active-listener-ui-gnome/src/transcript-attributes.ts`, and new or existing GNOME verification tests.

- [ ] 3.1 Replace plain transcript display text on the active-listener side with normalized `TextRun` values carrying `text`, `is_command`, and `is_complete`.
- [ ] 3.1a Verify `TextRun` normalization with focused tests proving empty runs are impossible and adjacent runs with identical flags are merged immediately.
- [ ] 3.2 Preserve the current incremental reducer semantics by appending newly completed material once and rebuilding the incomplete tail wholesale on each update, now in terms of `TextRun`s.
- [ ] 3.2a Verify reducer behavior with focused tests covering completed-prefix accumulation, unstable-tail replacement, and tail revisions that change command-text coloring without duplicating committed text.
- [ ] 3.3 Change the D-Bus transcription display contract from `a(ts)(ts)` to one ordered run payload (`a(sbb)`), and update both the Python signal emitter and the TypeScript client decode path in the same change.
- [ ] 3.3a Verify the D-Bus/display contract with focused active-listener and GNOME tests showing command text color selection and incomplete-tail transparency driven only by `TextRun` flags.

## 4. Rewrite serialization and failure behavior
Files likely touched: `packages/active-listener/src/active_listener/recording/finalizer.py`, rewrite-related service modules/tests, `packages/active-listener-ui-gnome/src/active-listener-service-client.ts`, `packages/active-listener-ui-gnome/src/panel-indicator.ts`, and the chosen GNOME-visible error surface.

- [ ] 4.1 Serialize normalized `TextRun`s into one flat rewrite input string, wrapping command-text runs in literal `<instruction>...</instruction>` markers while leaving normal runs unwrapped.
- [ ] 4.1a Verify rewrite serialization with focused tests covering mixed normal/command runs, punctuation-attached words, and adjacent command-text grouping.
- [ ] 4.2 Preserve the existing no-workstation-emission pipeline-failure behavior for command-text recordings and make rewrite failure visibly consumable in GNOME through the existing `PipelineFailed(step, reason)` -> `onError(title, detail)` path.
- [ ] 4.2a Verify finalization failure behavior with focused tests covering successful rewrite emission, rewrite failure with no workstation emission, D-Bus `PipelineFailed` signaling, and whichever GNOME-visible failure presentation is chosen.

## 5. Targeted validation
This task is intentionally last. Do not run it until gesture handling, classification, D-Bus presentation, and rewrite behavior are complete.

- [ ] 5.1 Run targeted validation for affected packages (`packages/active-listener`, `packages/active-listener-ui-gnome`, and any touched shared package) using the exact package-local test/typecheck commands documented in `AGENTS.md` and package configuration.
- [ ] 5.1a Verify validation artifacts show the command-text path works end-to-end at the package level, including gesture resolution, overlay presentation, rewrite serialization, and rewrite-failure signaling.

```mermaid
graph TD
  "1.1" --> "1.1a"
  "1.2" --> "1.2a"
  "1.1" --> "2.1"
  "2.1" --> "2.1a"
  "2.1" --> "2.2"
  "2.2" --> "2.2a"
  "2.2" --> "3.1"
  "3.1" --> "3.1a"
  "3.1" --> "3.2"
  "3.2" --> "3.2a"
  "3.2" --> "3.3"
  "3.3" --> "3.3a"
  "3.2" --> "4.1"
  "4.1" --> "4.1a"
  "4.1" --> "4.2"
  "4.2" --> "4.2a"
  "1.1a" --> "5.1"
  "2.2a" --> "5.1"
  "3.3a" --> "5.1"
  "4.2a" --> "5.1"
  "5.1" --> "5.1a"
```
