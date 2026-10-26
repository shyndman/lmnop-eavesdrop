## Why

Clients need a model-native estimate of whether speech has ended after a speech-to-silence transition. Client-side timing heuristics add state-machine complexity and cannot reconstruct missing server decisions. Make turn-ending detection a first-class optional part of live transcription output.

## What Changes

- Add optional Smart Turn metadata to `TranscriptionMessage` so consumers receive transcript content and turn-ending probability in one message.
- Run Smart Turn after any speech-to-silence transition using buffered turn audio, then attach its result to the transcription update for that evaluated boundary.
- Omit Smart Turn data from ordinary in-progress transcription updates where no speech-to-silence evaluation occurred or evaluation could not be produced.
- Keep Smart Turn as a raw signal only: expose probability plus evaluated audio boundary, not a server-side complete/incomplete decision.
- Extend live server transcription flows to emit Smart Turn-enriched transcription updates.

## Scope

### New Capabilities
- `turn-ending-signal`: Emit optional Smart Turn probability metadata alongside transcription updates when a speech-to-silence transition is evaluated.

### Modified Capabilities
- `transcription-message-contract`: `TranscriptionMessage` may now carry optional turn-ending metadata while continuing to omit unavailable optional fields on the wire.
- `streaming-transcription-output`: Server transcription pipelines must coordinate ASR results with Smart Turn results for pause-evaluated updates, while leaving ordinary incremental updates unchanged.

## Impact

Affected systems include `packages/wire` message models and codec behavior, `packages/server` live transcription emission, and consumers of `TranscriptionMessage` in `packages/client` and `packages/active-listener`. The feature introduces Smart Turn model integration plus its runtime dependencies, while limiting client-facing API expansion to one optional field on existing transcription messages.
