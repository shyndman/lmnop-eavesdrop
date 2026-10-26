# Retired design: finite-file ingestion

## Status

Historical context only. Issue 20 supersedes this design with live WebSocket transcription as the only supported ingestion workflow. None of the former implementation instructions remain active.

## Historical rationale

Finite inputs motivated a distinction between ingestion completion and processing completion: ending input should not silently discard accepted audio. The design explored normalization, bounded canonical-audio buffering, a drain/finalization lifecycle, deterministic result reduction, and cleanup on cancellation or timeout.

The file decoder, dedicated queue, source routing, upload lifecycle, single-shot client operation, and upload metrics are retired. Their former dependency and deployment assumptions are not requirements for the live-only server.

## Retained live concerns

Accepted work must still survive normal recording transitions. A completed recording's force-completing flush and final transcript are delivered before the next recording processes audio. Explicit cancellation instead discards its targeted utterance and suppresses outstanding results.

Flush remains independent of socket lifetime and completes even without pending audio. Native segment and individual word timestamps are mandatory and remain recording-relative across silence. With VAD disabled, every pending segment remains provisional until explicit force completion; with VAD enabled, silence-based completion remains supported.

These retained concerns belong to the active live contracts, not to a dormant finite-file implementation. Reintroducing alternative ingestion is outside this cutover.
