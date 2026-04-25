## Why

Consumers need turn-ending signal in same payload as transcript updates, not as separate sideband messages they have to correlate by timestamp. In this project, transcript text and turn-ending probability are usually consumed together, and making clients wait on two message types would add state-machine complexity while still leaving gaps when no turn-ending result exists for a given update. Smart Turn gives Eavesdrop a model-native estimate of whether speech has ended after a speech-to-silence transition, which is especially useful for RTSP consumers where post-processing decisions matter more than dictation latency. This change is needed now to make turn-ending detection a first-class part of transcription output without forcing clients to reconstruct server timing decisions.

## What Changes

- Add optional Smart Turn metadata to `TranscriptionMessage` so consumers receive transcript content and turn-ending probability in one message.
- Run Smart Turn after any speech-to-silence transition using buffered turn audio, then attach its result to the transcription update for that evaluated boundary.
- Omit Smart Turn data from ordinary in-progress transcription updates where no speech-to-silence evaluation occurred or evaluation could not be produced.
- Keep Smart Turn as a raw signal only: expose probability plus evaluated audio boundary, not a server-side complete/incomplete decision.
- Extend server-side transcription flows so live and RTSP sessions can emit Smart Turn-enriched transcription updates under the same wire contract.

## Scope

### New Capabilities
- `turn-ending-signal`: Emit optional Smart Turn probability metadata alongside transcription updates when a speech-to-silence transition is evaluated.

### Modified Capabilities
- `transcription-message-contract`: `TranscriptionMessage` may now carry optional turn-ending metadata while continuing to omit unavailable optional fields on the wire.
- `streaming-transcription-output`: Server transcription pipelines must coordinate ASR results with Smart Turn results for pause-evaluated updates, while leaving ordinary incremental updates unchanged.
- `rtsp-transcription-consumption`: RTSP consumers receive same integrated turn-ending signal as live clients instead of needing a separate message type or source-specific contract.

## Impact

Affected systems include `packages/wire` message models and codec behavior, `packages/server` streaming and RTSP transcription emission, and any client consumer of `TranscriptionMessage` in `packages/client` and `packages/active-listener`. The feature introduces Smart Turn model integration plus any runtime dependency needed to execute it, but keeps client-facing API expansion limited to one optional field on existing transcription messages.
