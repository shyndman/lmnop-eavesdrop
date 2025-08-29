# RTSP and Multi-Client Support Design

This document outlines a plan to refactor the eavesdrop server to support multiple concurrent clients (both WebSocket and RTSP streams) and to introduce the ability to transcribe audio from RTSP streams.

## 1. High-Level Architecture

The current server is single-threaded and can only handle one client at a time. To support multiple clients and I/O-bound tasks like RTSP streaming, we will move to an asynchronous architecture using Python's `asyncio` library.

- **Main `asyncio` Event Loop:** The main thread will run the `asyncio` event loop, which will manage all concurrent tasks.
- **WebSocket Server Task:** The existing WebSocket server will be adapted to run as an `asyncio` task. Each new WebSocket connection will also be handled as a separate `asyncio` task.
- **RTSP Stream Tasks:** Each configured RTSP stream will be managed by a dedicated `asyncio` task. This task will handle the `ffmpeg` subprocess and the audio data pipeline.
- **Transcription Tasks:** The CPU-intensive work of audio transcription will be run in a separate thread pool using `asyncio.to_thread` to avoid blocking the main event loop.
- **Shared Resources:** Resources like the transcription model will be loaded once and shared safely across all tasks.

This asynchronous approach is well-suited for I/O-bound operations like network communication and will allow the server to handle many clients efficiently.

## 2. Configuration

RTSP streams will be configured via a YAML file, as requested. The path to this file will be provided as a new command-line argument to the server.

**Example `config.yaml`:**

```yaml
streams:
  office: rtsp://brainbox:8554/office?audio=all
  kitchen: rtsp://brainbox:8554/kitchen?audio=all
```

The server will parse this file at startup and create a new `RTSPClient` task for each stream.

## 3. Implementation Steps

1.  **Adopt `asyncio`:**
    *   Refactor the main entry point of the server (in `src/eavesdrop/__main__.py` and `src/eavesdrop/server.py`) to use `asyncio.run()`.
    *   The existing `websockets` library has good `asyncio` support, so the `TranscriptionServer` can be adapted to work in this new paradigm.

2.  **Create an `RTSPClient`:**
    *   Create a new class, `RTSPClient`, that is responsible for handling a single RTSP stream.
    *   **Producer (within `RTSPClient`):** This part of the client will run `ffmpeg` as a subprocess using `asyncio.create_subprocess_exec`. It will read the decoded, raw PCM audio from the subprocess's stdout.
    *   **Queue:** A `asyncio.Queue` will be used to buffer the audio chunks received from `ffmpeg`.
    *   **Consumer (within `RTSPClient`):** A separate `async` method in the `RTSPClient` will consume audio chunks from the queue.
    *   **Transcription:** The consumer will then call the transcription model to process the audio. To avoid blocking the event loop, the transcription itself should be run in a thread pool using `loop.run_in_executor()` or `asyncio.to_thread()`.
    *   The `RTSPClient` should be designed to be resilient, with logic to automatically restart the `ffmpeg` subprocess if the connection is lost.

3.  **Integrate `RTSPClient` into the Server:**
    *   Add a new command-line argument to accept the path to the RTSP configuration file.
    *   At startup, the server will read the config file and create an `asyncio.Task` for each configured stream, running the main loop of an `RTSPClient` instance.

4.  **Refactor WebSocket Handling:**
    *   Modify the `TranscriptionServer` to handle each WebSocket client as a separate `asyncio` task.
    *   This will allow multiple WebSocket clients to be connected and transcribing simultaneously.

5.  **Shared Transcription Model:**
    *   The transcription model should be loaded once when the server starts.
    *   A reference to the model will be passed to each `RTSPClient` and WebSocket client task. Care must be taken to ensure that access to the model is thread-safe, especially if the model itself is not stateless.

## 4. Libraries

-   **`PyYAML`:** For parsing the RTSP configuration file.
-   **`asyncio`:** For the core asynchronous architecture. (Part of the standard library)
-   **`ffmpeg`:** The `ffmpeg` command-line tool will need to be installed on the server. The Python code will interact with it as a subprocess.

## 5. Future Considerations

-   **GPU Management:** The initial implementation will rely on the GPU's own scheduler. If performance issues arise due to GPU contention, a more sophisticated GPU management strategy could be implemented. This might involve creating a central, size-limited queue for transcription tasks that all clients feed into.
-   **Dynamic Configuration:** In the future, it might be desirable to add and remove RTSP streams without restarting the server. This could be achieved by adding an API endpoint or another mechanism to update the configuration dynamically.
-   **Transcription Output:** For now, RTSP transcriptions will be logged. Later, the output could be sent to other destinations like a message queue or a database.