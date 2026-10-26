# Retired baseline engineer note

## Status

This note described the implementation before the historical file-sourced transcription change. It is not a description of the current server or client.

## Historical observations

The earlier live pipeline paired ingestion and processing tasks, recognized an end-of-input sentinel, and had incomplete handling of terminal disconnect messages. The file proposal used those observations to motivate finite-input draining and explicit terminal cleanup.

These observations do not prescribe current APIs, task orchestration, or input signaling. The finite-file capability is retired by issue 20, and its upload instructions and implementation references have been removed from this record.

## Current scope

The supported server ingests live transcriber audio over WebSocket. Recording boundaries, flush, cancellation, mandatory native segment and word timing, and connection-owned cleanup remain governed by the active live specifications. Ending a normal recording must preserve accepted work; cancelling an utterance explicitly discards it.
