# Baseline Engineer Note (pre-implementation)

## Live mode setup path
1. Client transcriber connects via `WebSocketConnection.connect(...)` and sends a `TranscriptionSetupMessage` with `UserTranscriptionOptions`.
2. Server `WebSocketConnectionHandler.handle_connection(...)` reads the first message and routes `(ClientType.TRANSCRIBER, TranscriptionSetupMessage)` to `_handle_transcriber_connection(...)`.
3. `TranscriptionServer.initialize_client(...)` merges user options into server transcription config and builds `WebSocketStreamingClient` with `get_audio_from_websocket` as the source callback.
4. `WebSocketStreamingClient.start()` initializes processor, starts processing + ingestion tasks, and returns completion task.

## END_OF_AUDIO handling
1. Client signals EOF by sending `b"END_OF_AUDIO"` through `WebSocketConnection.send_end_of_audio()`.
2. Server `TranscriptionServer.get_audio_from_websocket(...)` maps `b"END_OF_AUDIO"` to `False`.
3. `WebSocketAudioSource.read_audio()` converts `False` into `None` (end-of-stream).
4. `WebSocketStreamingClient._audio_ingestion_loop()` breaks when audio source returns `None`.

## Disconnect handling (current)
1. Server sink (`WebSocketTranscriptionSink.disconnect`) emits a `DisconnectMessage` and marks sink closed.
2. Client `WebSocketConnection._process_message(...)` currently handles `ServerReadyMessage`, `TranscriptionMessage`, `ErrorMessage` only.
3. `DisconnectMessage` currently falls through to default/unknown path and is reported as unexpected.

## Completion orchestration (current)
1. `WebSocketStreamingClient._wait_for_completion()` uses `asyncio.wait(..., return_when=FIRST_COMPLETED)` over audio and processing tasks.
2. As soon as either task completes, it cancels the other task.
3. This is safe for open-ended live mode but does not provide explicit finite-source drain/finalization guarantees for file uploads.
