# Retired proposal: file-sourced transcription

## Status

Retired by issue 20's live-only cutover. This historical change is not a supported capability, implementation plan, or migration guide.

## Historical intent

The proposal explored deterministic finite-audio transcription with server-side normalization, bounded buffering, explicit drain/finalization, and a single-shot client result. Those goals explain the former implementation but do not authorize retaining or restoring it.

## Current boundary

The sole supported ingestion path is live audio from a transcriber over WebSocket. File uploads, encoded-file decoding, file queues, file-specific observability, source selection, and one-shot file client APIs are removed without compatibility shims. Do not route files through a renamed live-upload API.

Live recording boundaries, flush, cancellation, mandatory native segment and word timing, and connection cleanup remain supported. See the active live transcription specifications for their contracts.
