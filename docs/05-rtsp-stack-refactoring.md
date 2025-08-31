# Refactoring the RTSP Transcription Stack

**Document ID:** 05-rtsp-stack-refactoring.md

## 1. Introduction

This document outlines a plan to refactor the RTSP (Real-Time Streaming Protocol) transcription stack to align it with the modern, more maintainable architecture used by the WebSocket stack. The primary goal is to eliminate the legacy `ServeClientFasterWhisper` and `RTSPModelManager` components, replacing them with the `StreamingTranscriptionProcessor`.

This refactoring will result in a cleaner, more consistent, and more maintainable codebase by removing redundant code and unifying the transcription process for both WebSocket and RTSP streams.

## 2. Analysis of the Current State

The current implementation of the RTSP stack has several drawbacks:

*   **Divergent Architectures:** The RTSP and WebSocket stacks use different architectures for transcription, leading to code duplication and increased maintenance overhead.
*   **Legacy Components:** The RTSP stack relies on the `RTSPModelManager` and `ServeClientFasterWhisper` classes, which have been superseded by the more flexible `StreamingTranscriptionProcessor` used in the WebSocket stack.
*   **Tight Coupling:** The `RTSPClientManager` is tightly coupled to the `RTSPModelManager`, making it difficult to modify or extend the transcription process.
*   **Dead Code:** As the WebSocket stack has already been refactored, the `ServeClientFasterWhisper` class is only used by the RTSP stack. Once the RTSP stack is updated, this class and the entire `src/eavesdrop/backend.py` file will become dead code.

## 3. Proposed Architecture

The proposed architecture will unify the RTSP and WebSocket stacks by using the `StreamingTranscriptionProcessor` for both. This will be achieved by making the following changes:

*   **`RTSPTranscriptionClient`:** This class will be made self-contained, creating and managing its own `StreamingTranscriptionProcessor` instance.
*   **`RTSPClientManager`:** This class will be updated to create and manage `RTSPTranscriptionClient` instances directly, without the need for an `RTSPModelManager`.
*   **Configuration:** All transcription-related configuration will be handled by the `TranscriptionConfig` dataclass, providing a single, consistent way to configure the transcription process.
*   **Code Removal:** The legacy `RTSPModelManager` and `ServeClientFasterWhisper` classes, along with their containing files (`rtsp_models.py` and `backend.py`), will be deleted.

## 4. Detailed Implementation Steps

This section provides a file-by-file guide to implementing the proposed refactoring.

### Step 1: Make `RTSPTranscriptionClient` Self-Contained

*   **File:** `src/eavesdrop/rtsp.py`
*   **Class:** `RTSPTranscriptionClient`

The `RTSPTranscriptionClient` will be modified to create its own `StreamingTranscriptionProcessor` instead of receiving a `transcriber` object.

**Current `__init__` method:**

```python
class RTSPTranscriptionClient(RTSPClient):
  def __init__(self, stream_name: str, rtsp_url: str, transcriber):
    # ...
    self.transcriber = transcriber
    # ...
    self.processor = StreamingTranscriptionProcessor(
      transcriber=transcriber,
      # ...
    )
```

**New `__init__` method:**

```python
class RTSPTranscriptionClient(RTSPClient):
  def __init__(self, stream_name: str, rtsp_url: str):
    # Initialize parent with internal queue
    super().__init__(stream_name, rtsp_url, asyncio.Queue(maxsize=100))

    # Configuration from environment variables with defaults
    buffer_config = BufferConfig(
      sample_rate=16000,
      max_buffer_duration=get_env_float("EAVESDROP_RTSP_BUFFER_DURATION", 45.0),
      cleanup_duration=get_env_float("EAVESDROP_RTSP_CLEANUP_DURATION", 30.0),
      min_chunk_duration=get_env_float("EAVESDROP_RTSP_MIN_CHUNK_DURATION", 1.0),
      clip_audio=get_env_bool("EAVESDROP_RTSP_CLIP_AUDIO", False),
      max_stall_duration=get_env_float("EAVESDROP_RTSP_MAX_STALL_DURATION", 25.0),
    )

    transcription_config = TranscriptionConfig(
      send_last_n_segments=10,  # Not used for RTSP
      no_speech_thresh=get_env_float("EAVESDROP_RTSP_NO_SPEECH_THRESH", 0.45),
      same_output_threshold=get_env_int("EAVESDROP_RTSP_SAME_OUTPUT_THRESH", 10),
      use_vad=get_env_bool("EAVESDROP_RTSP_USE_VAD", True),
      clip_audio=get_env_bool("EAVESDROP_RTSP_CLIP_AUDIO", False),
      # These will need to be passed in or configured globally
      model="distil-small.en",
      task="transcribe",
      language=None,
      initial_prompt=None,
      vad_parameters=None,
      single_model=True,
      cache_path="~/.cache/eavesdrop/",
      device_index=0,
    )

    # Create abstracted components
    self.audio_source = RTSPAudioSource(self.audio_queue)
    self.stream_buffer = AudioStreamBuffer(buffer_config)
    self.transcription_sink = RTSPTranscriptionSink(stream_name)
    self.processor = StreamingTranscriptionProcessor(
      buffer=self.stream_buffer,
      sink=self.transcription_sink,
      config=transcription_config,
      client_uid=stream_name,
      logger_name=f"rtsp_processor_{stream_name}",
    )

    # Statistics (preserved from current implementation)
    self.transcriptions_completed = 0
    self.transcription_errors = 0
```

### Step 2: Update `RTSPClientManager`

*   **File:** `src/eavesdrop/rtsp_manager.py`
*   **Class:** `RTSPClientManager`

The `RTSPClientManager` will be updated to work without the `RTSPModelManager`.

**Current `__init__` method:**

```python
class RTSPClientManager:
  def __init__(self, model_manager: RTSPModelManager):
    self.model_manager = model_manager
    # ...
```

**New `__init__` method:**

```python
class RTSPClientManager:
  def __init__(self):
    self.clients: dict[str, RTSPTranscriptionClient] = {}
    self.tasks: dict[str, asyncio.Task] = {}
    self.logger = get_logger("rtsp_client_manager")

    # Statistics
    self.total_streams_created = 0
    self.active_streams = 0
    self.failed_streams = 0
```

**Current `add_stream` method:**

```python
  async def add_stream(self, stream_name: str, rtsp_url: str) -> bool:
    # ...
    transcriber = await self.model_manager.get_transcriber()
    client = RTSPTranscriptionClient(stream_name, rtsp_url, transcriber)
    # ...
```

**New `add_stream` method:**

```python
  async def add_stream(self, stream_name: str, rtsp_url: str) -> bool:
    if stream_name in self.clients:
      self.logger.warning("Stream already exists", stream=stream_name)
      return False

    try:
      self.logger.info("Adding RTSP stream", stream=stream_name, url=rtsp_url)

      # Create RTSP client
      client = RTSPTranscriptionClient(stream_name, rtsp_url)
      self.clients[stream_name] = client

      # Create and start task for this client
      task = asyncio.create_task(client.run())
      task.set_name(f"rtsp_stream_{stream_name}")
      self.tasks[stream_name] = task

      self.total_streams_created += 1
      self.active_streams += 1

      self.logger.info(
        "RTSP stream added successfully", stream=stream_name, active_streams=self.active_streams
      )

      return True

    except Exception:
      self.failed_streams += 1
      self.logger.exception("Failed to add RTSP stream", stream=stream_name)

      # Clean up partial state
      self.clients.pop(stream_name, None)
      task = self.tasks.pop(stream_name, None)
      if task and not task.done():
        task.cancel()

      return False
```

**Update `stop_all_streams`:**

Remove the call to `self.model_manager.cleanup()`.

**Update `get_stream_status`:**

Remove the `model_info` from the status dictionary.

### Step 3: Update `TranscriptionServer`

*   **File:** `src/eavesdrop/server.py`
*   **Class:** `TranscriptionServer`

The `TranscriptionServer` needs to be updated to instantiate the `RTSPClientManager` without the `RTSPModelManager`.

**Current `_initialize_rtsp_streams` method:**

```python
  async def _initialize_rtsp_streams(
    self,
    rtsp_streams: dict[str, str],
    faster_whisper_custom_model_path: str | None,
    single_model: bool,
    cache_path: str,
  ) -> None:
    # ...
    self.rtsp_client_manager = RTSPClientManager(RTSPModelManager(backend_params))
    # ...
```

**New `_initialize_rtsp_streams` method:**

```python
  async def _initialize_rtsp_streams(
    self,
    rtsp_streams: dict[str, str],
    # These parameters are no longer needed here
    # faster_whisper_custom_model_path: str | None,
    # single_model: bool,
    # cache_path: str,
  ) -> None:
    try:
      self.logger.info("Initializing RTSP transcription system")

      # Create RTSP client manager
      self.rtsp_client_manager = RTSPClientManager()

      # Start all configured streams
      await self.rtsp_client_manager.start_all_streams(rtsp_streams)

      self.logger.info(
        "RTSP transcription system initialized",
        active_streams=self.rtsp_client_manager.get_stream_count(),
      )

    except Exception:
      self.logger.exception("Failed to initialize RTSP system")
      # Clean up on failure
      self.rtsp_client_manager = None
```

The parameters `faster_whisper_custom_model_path`, `single_model`, and `cache_path` will now be handled by the `TranscriptionConfig` within the `RTSPTranscriptionClient`, likely configured via environment variables.

### Step 4: Delete `rtsp_models.py`

*   **File:** `src/eavesdrop/rtsp_models.py`

This file contains the `RTSPModelManager`, which is no longer needed. The entire file can be deleted.

### Step 5: Delete `backend.py`

*   **File:** `src/eavesdrop/backend.py`

This file contains the `ServeClientFasterWhisper` class, which is no longer used by either the RTSP or WebSocket stacks. The entire file can be deleted.

## 5. Verification and Testing

After implementing these changes, the following steps should be taken to verify the refactoring:

1.  **Unit Tests:** Any existing unit tests for the RTSP stack will need to be updated to reflect the new architecture. New unit tests should be added for the `RTSPTranscriptionClient`.
2.  **Integration Tests:** An integration test should be performed to ensure that the RTSP streams are correctly transcribed. This can be done by running the server with an RTSP configuration and monitoring the logs for transcription output.
3.  **Manual Testing:** Manually start the server with an RTSP stream and verify that the transcriptions are being logged correctly.

## 6. Conclusion

This refactoring will significantly improve the maintainability and consistency of the codebase. By unifying the transcription architecture, we will reduce code duplication, eliminate dead code, and make it easier to develop and maintain the application in the future.
