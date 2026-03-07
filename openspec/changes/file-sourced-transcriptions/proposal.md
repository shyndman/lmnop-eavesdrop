## Why

Verified current behavior is live-stream centric and lacks a first-party finite-file path. The server currently ingests transcriber audio as raw float32 websocket frames with `b"END_OF_AUDIO"` as an ingest sentinel, while client APIs are microphone-stream oriented and do not expose one-shot file transcription.

## Verified Baseline (Current Code)

- **Wire setup contract has no source mode today**: `TranscriptionSetupMessage.options` uses `UserTranscriptionOptions`, which currently contains `send_last_n_segments`, `initial_prompt`, `hotwords`, `word_timestamps`, and `beam_size` only (`packages/wire/src/eavesdrop/wire/messages.py`, `packages/wire/src/eavesdrop/wire/transcription.py`).
- **Client API is stream-oriented**: `EavesdropClient` exposes transcriber/subscriber factories and streaming lifecycle methods (`connect/start_streaming/stop_streaming`), with no `transcribe_file(...)` entrypoint (`packages/client/src/eavesdrop/client/core.py`).
- **Server transcriber ingest expects pre-decoded float32 frames**: `get_audio_from_websocket` converts websocket payloads via `np.frombuffer(..., dtype=np.float32)` and treats `b"END_OF_AUDIO"` specially (`packages/server/src/eavesdrop/server/server.py`).
- **Current completion orchestration is FIRST_COMPLETED**: `WebSocketStreamingClient._wait_for_completion` waits for either ingest or processing task completion, then cancels the other (`packages/server/src/eavesdrop/server/streaming/client.py`).
- **Output shape is windowed completed history + incomplete tail**: `StreamingTranscriptionProcessor` sends `most_recent_completed_segments(send_last_n_segments)` and ensures an incomplete tail segment (synthetic if needed) (`packages/server/src/eavesdrop/server/streaming/processor.py`, `packages/server/src/eavesdrop/server/config.py`).
- **Buffer behavior is trim-based, not bounded enqueue backpressure**: `AudioStreamBuffer` appends frames and trims old audio when `max_buffer_duration` is exceeded (`packages/server/src/eavesdrop/server/streaming/buffer.py`).
- **Disconnect exists on wire but client does not handle it explicitly**: server sink emits `DisconnectMessage`, while client message handling currently matches only ready/transcription/error and treats others as unexpected (`packages/server/src/eavesdrop/server/streaming/audio_flow.py`, `packages/client/src/eavesdrop/client/connection.py`).
- **Verified dependency constraint**: server package already depends on `soundfile`, but official `soundfile`/libsndfile documentation only guarantees libsndfile-supported formats; MP3 is supported and AAC is not listed. Therefore `soundfile` alone cannot satisfy the proposed WAV/MP3/AAC requirement (`packages/server/pyproject.toml`; https://python-soundfile.readthedocs.io/en/0.13.1/; https://libsndfile.github.io/libsndfile/formats.html).
- **FFmpeg runtime is already a first-party pattern in server deployment**: server Docker base images install `ffmpeg`, and RTSP ingestion already shells out to `ffmpeg` to normalize audio to mono 16k PCM before downstream processing (`packages/server/docker/Dockerfile.cuda-base`, `packages/server/docker/Dockerfile.rocm-base`, `packages/server/src/eavesdrop/server/rtsp/client.py`).

## What Changes

- Extend transcriber setup options with explicit file-source intent (rather than adding a new websocket client type).
- Add an explicit FFmpeg-backed server-side file decode/normalization path to canonical mono 16kHz float32 before queue residency.
- Add bounded in-memory file ingestion buffering (900 seconds canonical audio) with blocking backpressure.
- Add finite-source lifecycle semantics that separate EOF ingest completion from processing drain/finalization.
- Add single-shot `EavesdropClient.transcribe_file(..., timeout_s=...)` with non-reentrant operation guard and deterministic reducer semantics.
- Add periodic observability for file-session queue/throughput health.

## Capabilities

### New Capabilities
- `file-sourced-transcription`: Deterministic end-to-end transcription of finite audio files, including ingestion, buffering/backpressure, lifecycle completion, and single-shot client API behavior.

### Modified Capabilities
- None.

## Impact

- **Server (`packages/server`)**: add finite-file ingest/decode path, bounded queueing, drain/finalization state handling, and file-session observability; implementation will rely on the existing `ffmpeg` runtime already present in server container images unless a different FFmpeg-backed decoder is explicitly introduced.
- **Client (`packages/client`)**: add one-shot file API, reducer logic for windowed segment outputs, and explicit terminal handling for disconnect/close completion.
- **Wire (`packages/wire`)**: extend setup options to declare file-source mode while preserving existing transcriber compatibility.
- **Runtime/ops**: predictable bounded memory behavior (900s canonical audio queue) and periodic queue/throughput telemetry.
- **Validation**: new contract-focused tests across server/client/wire for setup compatibility, EOF drain correctness, reducer behavior, and backpressure semantics.
