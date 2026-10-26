# Retired capability: file-sourced transcription

## Status

This historical capability is withdrawn by issue 20. It defines no active added requirements or supported scenarios.

## Historical intent

The former capability sought complete finite-audio results using server-side decoding, bounded canonical-audio buffering, processing drain before terminal signaling, a single-shot client operation, deterministic reduction, and upload metrics.

## Superseding contract

Only live transcriber audio over WebSocket is supported. File uploads and all file-exclusive lifecycle, decoding, buffering, client, and protocol surfaces are removed. No source selector, deprecated API, compatibility shim, or alternate file-upload route replaces them.

Normal live recording completion preserves accepted work through its force-completing flush. Cancellation discards its targeted utterance. Native segment and individual word timestamps are always present. VAD-disabled output stays provisional until explicit force completion; VAD-enabled silence completion remains available. Active live specifications govern these retained behaviors.
