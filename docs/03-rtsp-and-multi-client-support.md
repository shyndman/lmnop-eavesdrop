# RTSP and Multi-Client Support Design

This document outlines a plan to refactor the eavesdrop server to support multiple concurrent clients (both WebSocket and RTSP streams) and to introduce the ability to transcribe audio from RTSP streams.

## 1. High-Level Architecture

The current server is single-threaded and can only handle one client at a time. To support multiple clients and I/O-bound tasks like RTSP streaming, we will move to an asynchronous architecture using Python's `asyncio` library.

- **Main `asyncio` Event Loop:** The main thread will run the `asyncio` event loop, which will manage all concurrent tasks.
- **WebSocket Server Task:** The existing WebSocket server will be adapted to run as an `asyncio` task. Each new WebSocket connection will also be handled as a separate `asyncio` task.
- **RTSP Stream Tasks:** Each configured RTSP stream will be managed by a dedicated `asyncio` task, created via `asyncio.create_task()`. This provides inherent context isolation.
- **Transcription Tasks:** The CPU-intensive work of audio transcription will be run in a separate thread for each task using `asyncio.to_thread`. A shared thread pool will not be used at this time. The `faster-whisper` model is thread-safe, so no locking mechanism is required for the transcription calls.
- **Shared Resources:** Resources like the transcription model will be loaded once and shared safely across all tasks.

This asynchronous approach is well-suited for I/O-bound operations like network communication and will allow the server to handle many clients efficiently.

## 2. Configuration

RTSP streams will be configured via a YAML file. The path to this file will be provided as a new command-line argument to the server.

**Example `config.yaml`:**

```yaml
streams:
  office: rtsp://brainbox:8554/office?audio=all
  kitchen: rtsp://brainbox:8554/kitchen?audio=all
```

The server will parse this file at startup and create a new `RTSPClient` task for each stream.

## 3. Implementation Details

### 3.1. `RTSPClient` Error Handling

- **Reconnection Strategy:** When an `RTSPClient` fails to connect to a stream or the `ffmpeg` process terminates unexpectedly, it will attempt to reconnect every **30 seconds**. This will be a fixed delay.
- **Logging:** Errors will be logged with detailed information, including the stream name and the error message from `ffmpeg`.

### 3.2. Logging

- **Structured Logging:** We will use `structlog` for structured, context-aware logging.
- **Context Binding:** For each RTSP client task, we will use the `structlog.contextvars.bound_contextvars` context manager to bind the stream's name to the context.
- **State Transitions:** The `RTSPClient` will log its state transitions (e.g., "Connecting", "Transcribing", "Reconnecting").

### 3.3. Configuration Validation

- **Strict Validation:** The server will validate the `config.yaml` file at startup. If the file is not found, is not valid YAML, or does not contain a top-level `streams` key with a dictionary of streams, the server will log a critical error and **fail to start**.

### 3.4. `ffmpeg` Command

- **Command:** The following `ffmpeg` command will be used to capture and decode the RTSP streams:
  ```bash
  ffmpeg -fflags nobuffer -flags low_delay -rtsp_transport tcp -i {rtsp_url} -vn -acodec pcm_s16le -ar 16000 -ac 1 -f s16le -
  ```

### 3.5. Audio Queue

- **Size:** The `asyncio.Queue` used to buffer audio between `ffmpeg` and the transcription task will have a `maxsize` of 100, based on the example project. This will not be configurable for the initial implementation.

### 3.6. Graceful Shutdown

- **Shutdown Sequence:** When the server receives a `SIGINT` or `SIGTERM`, it will initiate a graceful shutdown:
    1. Stop accepting new WebSocket connections.
    2. Signal all active `RTSPClient` and WebSocket client tasks to terminate.
    3. Each `RTSPClient` task will terminate its `ffmpeg` subprocess.
    4. The main event loop will wait for all running tasks to complete.

### 3.7. Testing

- **Framework:** `pytest` will be used as the testing framework.
- **Unit Tests:** We will create unit tests for the `RTSPClient` class, mocking `asyncio.create_subprocess_exec`.
- **Integration Tests:** We will set up integration tests that run against a local RTSP server.

## 4. Implementation Steps

### Milestone 1: Asynchronous WebSocket Server (Completed)

The existing `TranscriptionServer` has been refactored to be fully asynchronous using `asyncio`. The WebSocket handling logic was separated into a new `websocket.py` module, and the entire codebase was updated to use modern, non-deprecated `websockets` library APIs with correct type hints. The CPU-bound transcription work is now correctly dispatched to a background thread using `asyncio.to_thread`, preventing blocking of the main event loop.

### Milestone 2: RTSP Client Implementation (Completed)

This milestone focused on creating a self-contained client for handling RTSP streams. A new `RTSPClient` class was implemented that manages an `ffmpeg` subprocess to ingest RTSP streams and puts the resulting audio data onto an `asyncio.Queue` for downstream processing.

**Completed Implementation:**
1.  **Created `src/eavesdrop/rtsp.py`:** Houses the complete `RTSPClient` class with full async/await architecture.
2.  **RTSPClient Class Features:**
    *   Constructor accepts stream name, RTSP URL, and `asyncio.Queue[bytes]` reference
    *   Main `async def run()` method handles the complete client lifecycle
    *   Structured logging with stream context binding using `structlog.contextvars.bound_contextvars`
3.  **FFmpeg Subprocess Management:**
    *   Uses `asyncio.create_subprocess_exec` with the exact command from the design document
    *   Proper stdout/stderr pipe management for audio capture and error monitoring
    *   Process cleanup with graceful termination and force-kill fallback
4.  **Audio Stream Processing:**
    *   `_read_audio_stream()` method reads 4096-byte chunks from FFmpeg stdout
    *   Continuous audio data production to the shared asyncio.Queue
    *   Statistics tracking with periodic logging to avoid spam
5.  **Comprehensive Error Handling:**
    *   `_monitor_process_errors()` categorizes FFmpeg stderr messages by severity
    *   Distinguishes connection errors, authentication issues, and stream format problems
    *   Infinite reconnection loop with 30-second delays between attempts
6.  **Graceful Shutdown:**
    *   `stop()` method for clean termination with process cleanup
    *   Async context manager support (`__aenter__`/`__aexit__`)
    *   Proper resource management and exception handling

**Key Design Features Implemented:**
- Thread-safe async operations throughout
- Clean separation of concerns (audio ingestion only, no transcription)
- Robust error categorization and logging
- Fixed 30-second reconnection delays as specified
- Integration with existing structured logging patterns
- Complete adherence to the codebase's async/await architecture

### Milestone 3: Transcription and Integration

This milestone integrates the `RTSPClient` with the transcription backend and the main server. It introduces a second producer-consumer relationship where a transcription worker pulls audio from the queue, transcribes it, and (for now) logs the output.

**Tasks:**
1.  **Add `PyYAML` Dependency:** Add `PyYAML` to the project's dependencies in `pyproject.toml` to handle the new configuration file.
2.  **Transcription Worker:**
    *   In the `RTSPClient` class, create a new `async` method (e.g., `_transcribe_worker`).
    *   This method will run in a `while True` loop, pulling audio chunks from the `asyncio.Queue` using `await queue.get()`.
    *   It will aggregate these chunks into a buffer suitable for the transcription model (e.g., 1-second of audio).
    *   Use `await asyncio.to_thread(transcriber.transcribe, audio_buffer)` to perform the transcription.
    *   For now, simply log the transcribed segments to the console.
3.  **Server Integration:**
    *   In `server.py`, modify the `TranscriptionServer.run` method.
    *   Add logic to read the YAML file specified by the `--config` argument.
    *   Use `PyYAML` to parse the file. Implement the strict validation described in the design document (file exists, is valid YAML, has `streams` key).
    *   Loop through the configured streams. For each stream, create an `asyncio.Queue` and an `RTSPClient` instance.
    *   Create two `asyncio.Task`s for each client: one for `rtsp_client.run()` (the ffmpeg producer) and one for `rtsp_client._transcribe_worker()`.
4.  **Shared Model:**
    *   Ensure the transcription model is loaded once in the `TranscriptionServer` and passed as a reference to each `RTSPClient` instance.

## 5. Libraries

-   **`PyYAML`:** For parsing the RTSP configuration file.
-   **`pytest`:** For testing.
-   **`asyncio`:** For the core asynchronous architecture. (Part of the standard library)
-   **`ffmpeg`:** The `ffmpeg` command-line tool will need to be installed on the server.
