import asyncio

import numpy as np
import structlog

from .logs import get_logger


class RTSPClient:
  """
  A client for ingesting RTSP audio streams using FFmpeg subprocesses.

  This class manages the lifecycle of an FFmpeg subprocess that connects to an RTSP stream,
  extracts audio data, and feeds it into an asyncio.Queue for downstream processing.

  The client handles automatic reconnection with a 30-second delay between attempts,
  comprehensive error logging, and graceful shutdown capabilities.
  """

  def __init__(self, stream_name: str, rtsp_url: str, audio_queue: asyncio.Queue[bytes]):
    """
    Initialize the RTSP client.

    Args:
        stream_name: A human-readable name for this stream (e.g., "office", "kitchen")
        rtsp_url: The RTSP URL to connect to
        audio_queue: An asyncio.Queue to receive audio chunks
    """
    self.stream_name = stream_name
    self.rtsp_url = rtsp_url
    self.audio_queue = audio_queue

    # Process tracking
    self.process: asyncio.subprocess.Process | None = None
    self.stopped = False

    # Logging with stream context
    self.logger = get_logger("rtsp_client").bind(stream=stream_name)

    # Statistics tracking
    self.chunks_read = 0
    self.total_bytes = 0
    self.reconnect_count = 0

  async def _create_ffmpeg_process(self) -> asyncio.subprocess.Process:
    """
    Create and start the FFmpeg subprocess for RTSP stream ingestion.

    Uses the exact command specified in the design document to capture audio
    at 16kHz sample rate in PCM format suitable for Whisper transcription.

    Returns:
        The created subprocess.Process object

    Raises:
        Exception: If the process fails to start
    """
    cmd = [
      "ffmpeg",
      "-fflags",
      "nobuffer",
      "-flags",
      "low_delay",
      "-rtsp_transport",
      "tcp",
      "-i",
      self.rtsp_url,
      "-vn",  # No video
      "-acodec",
      "pcm_s16le",  # 16-bit PCM audio
      "-ar",
      "16000",  # 16kHz sample rate
      "-ac",
      "1",  # Mono audio
      "-f",
      "s16le",  # Raw 16-bit little-endian format
      "-",  # Output to stdout
    ]

    self.logger.debug("Starting FFmpeg process", command=" ".join(cmd))

    process = await asyncio.create_subprocess_exec(
      *cmd,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
    )

    self.process = process
    self.logger.info("FFmpeg process started", pid=process.pid)
    return process

  async def _read_audio_stream(self) -> None:
    """
    Read audio data from the FFmpeg process stdout and feed it to the audio queue.

    Continuously reads 4096-byte chunks from the process and places them on the
    audio queue. Logs statistics periodically to avoid spam while providing
    visibility into stream health.

    Returns when the process terminates or when stopped.
    """
    if not self.process or not self.process.stdout:
      self.logger.error("No process or stdout available for reading")
      return

    chunk_size = 4096
    log_every_n_chunks = 100

    self.logger.info("Starting audio stream reading")

    try:
      while not self.stopped:
        chunk = await self.process.stdout.read(chunk_size)

        if not chunk:  # EOF - process has terminated
          self.logger.info("Audio stream ended (EOF)")
          break

        # Add chunk to queue for downstream processing
        await self.audio_queue.put(chunk)

        # Update statistics
        self.chunks_read += 1
        self.total_bytes += len(chunk)

        # Log statistics periodically to avoid spam
        if self.chunks_read % log_every_n_chunks == 0:
          self.logger.debug(
            "Audio streaming progress",
            chunks_read=self.chunks_read,
            total_mb=f"{self.total_bytes / (1024 * 1024):.1f}",
            chunk_size=len(chunk),
          )

    except asyncio.CancelledError:
      self.logger.debug("Audio reading cancelled")
      raise
    except Exception:
      self.logger.exception("Error reading audio stream")
      raise
    finally:
      self.logger.info(
        "Audio reading finished",
        chunks_read=self.chunks_read,
        total_mb=f"{self.total_bytes / (1024 * 1024):.1f}",
      )

  async def _monitor_process_errors(self) -> None:
    """
    Monitor FFmpeg stderr for error messages and log them appropriately.

    Reads stderr output from the FFmpeg process and categorizes messages
    by severity. Distinguishes between expected warnings and critical errors
    that might indicate connection problems.

    Returns when the process terminates or when stopped.
    """
    if not self.process or not self.process.stderr:
      self.logger.error("No process or stderr available for monitoring")
      return

    self.logger.debug("Starting FFmpeg error monitoring")

    try:
      while not self.stopped:
        line = await self.process.stderr.readline()

        if not line:  # EOF - process has terminated
          self.logger.debug("FFmpeg stderr ended (EOF)")
          break

        error_msg = line.decode("utf-8", errors="replace").strip()
        if not error_msg:
          continue

        # Categorize FFmpeg messages by content
        error_lower = error_msg.lower()

        if any(
          keyword in error_lower
          for keyword in [
            "connection refused",
            "connection failed",
            "network unreachable",
            "timeout",
            "host not found",
            "no route to host",
          ]
        ):
          self.logger.error("FFmpeg connection error", message=error_msg)
        elif any(
          keyword in error_lower
          for keyword in ["unauthorized", "forbidden", "authentication failed", "401", "403"]
        ):
          self.logger.error("FFmpeg authentication error", message=error_msg)
        elif any(
          keyword in error_lower
          for keyword in [
            "invalid data",
            "corrupt",
            "decode error",
            "unsupported",
            "invalid format",
          ]
        ):
          self.logger.warning("FFmpeg stream format issue", message=error_msg)
        elif any(keyword in error_lower for keyword in ["deprecated", "option", "using"]):
          # Common FFmpeg informational messages - log at debug level
          self.logger.debug("FFmpeg info", message=error_msg)
        elif "error" in error_lower or "fatal" in error_lower:
          self.logger.error("FFmpeg error", message=error_msg)
        else:
          # Generic FFmpeg output - likely informational
          self.logger.debug("FFmpeg output", message=error_msg)

    except asyncio.CancelledError:
      self.logger.debug("Error monitoring cancelled")
      raise
    except Exception:
      self.logger.exception("Error monitoring FFmpeg stderr")
      raise

  async def run(self) -> None:
    """
    Main entry point for the RTSP client lifecycle.

    Implements an infinite retry loop that attempts to connect to the RTSP stream,
    manages concurrent audio reading and error monitoring tasks, and handles
    automatic reconnection with a 30-second delay between attempts.

    This method will run until stop() is called or an unrecoverable error occurs.
    """
    # Use structured logging context to bind stream name to all log messages
    with structlog.contextvars.bound_contextvars(stream=self.stream_name):
      self.logger.info("Starting RTSP client", url=self.rtsp_url)

      while not self.stopped:
        try:
          # Reset statistics for this connection attempt
          self.chunks_read = 0
          self.total_bytes = 0

          if self.reconnect_count > 0:
            self.logger.info("Reconnecting to RTSP stream", attempt=self.reconnect_count + 1)
          else:
            self.logger.info("Connecting to RTSP stream")

          # Create and start the FFmpeg process
          process = await self._create_ffmpeg_process()

          # Start concurrent tasks for audio reading and error monitoring
          audio_task = asyncio.create_task(self._read_audio_stream())
          error_task = asyncio.create_task(self._monitor_process_errors())

          self.logger.info("RTSP stream connected, processing audio")

          # Wait for either task to complete (indicates process death or error)
          done, pending = await asyncio.wait(
            [audio_task, error_task], return_when=asyncio.FIRST_COMPLETED
          )

          # Cancel any remaining tasks
          for task in pending:
            task.cancel()
            try:
              await task
            except asyncio.CancelledError:
              pass

          # Check if any task raised an exception
          for task in done:
            if task.exception():
              self.logger.error("Task failed", task=task.get_name(), error=str(task.exception()))

          # Wait for process to finish and check exit code
          exit_code = await process.wait()
          if exit_code != 0:
            self.logger.warning("FFmpeg process exited with error", exit_code=exit_code)
          else:
            self.logger.info("FFmpeg process exited normally")

          # Clean up process reference
          self.process = None
        except Exception:
          self.logger.exception("RTSP connection attempt failed")
          self.process = None

          # Only reconnect if we haven't been stopped
          if not self.stopped:
            self.reconnect_count += 1
            self.logger.info("Reconnecting in 30 seconds", attempt=self.reconnect_count + 1)

            try:
              await asyncio.sleep(30)
            except asyncio.CancelledError:
              self.logger.info("Reconnection wait interrupted")
              break

    self.logger.info("RTSP client stopped")

  async def stop(self) -> None:
    """
    Gracefully stop the RTSP client.

    Sets the stopped flag, terminates any running FFmpeg process, and ensures
    clean shutdown of all resources. This method can be called from external
    signal handlers or shutdown sequences.

    Safe to call multiple times.
    """
    if self.stopped:
      self.logger.debug("Stop called but client already stopped")
      return

    self.logger.info("Stopping RTSP client")
    self.stopped = True

    # Clean up any running process
    await self._cleanup_process()

    self.logger.info("RTSP client shutdown complete")

  async def _cleanup_process(self) -> None:
    """
    Clean up the FFmpeg process gracefully.

    Attempts graceful termination first, then force kills if necessary.
    Handles cases where the process is already dead or was never started.
    """
    if not self.process:
      return

    self.logger.debug("Cleaning up FFmpeg process", pid=self.process.pid)

    try:
      # Check if process is still running
      if self.process.returncode is None:
        # First try graceful termination
        self.logger.debug("Terminating FFmpeg process gracefully")
        self.process.terminate()

        try:
          # Wait up to 5 seconds for graceful shutdown
          await asyncio.wait_for(self.process.wait(), timeout=5.0)
          self.logger.debug("FFmpeg process terminated gracefully")
        except asyncio.TimeoutError:
          # Force kill if graceful termination fails
          self.logger.warning("FFmpeg process did not terminate gracefully, force killing")
          self.process.kill()
          await self.process.wait()
          self.logger.debug("FFmpeg process force killed")
      else:
        self.logger.debug("FFmpeg process already terminated", exit_code=self.process.returncode)

    except ProcessLookupError:
      # Process was already dead
      self.logger.debug("FFmpeg process was already dead during cleanup")
    except Exception:
      self.logger.exception("Error during process cleanup")
    finally:
      self.process = None

  async def __aenter__(self):
    """Async context manager entry."""
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    """Async context manager exit - ensures cleanup."""
    await self.stop()


class AudioBuffer:
  """
  Manages audio chunks from RTSP streams and prepares them for transcription.

  Accumulates raw audio bytes from FFmpeg until enough data is available for
  transcription (target duration), then converts to numpy arrays suitable for
  Whisper processing.
  """

  def __init__(self, sample_rate: int = 16000, target_duration: float = 1.0):
    """
    Initialize the audio buffer.

    Args:
        sample_rate: Audio sample rate in Hz (16kHz for Whisper)
        target_duration: Target buffer duration in seconds before transcription
    """
    self.sample_rate = sample_rate
    self.target_duration = target_duration
    self.target_samples = int(sample_rate * target_duration)
    self.bytes_per_sample = 2  # 16-bit PCM = 2 bytes per sample
    self.target_bytes = self.target_samples * self.bytes_per_sample

    # Internal buffer state
    self.buffer = b""
    self.total_chunks_added = 0
    self.total_arrays_produced = 0

    self.logger = get_logger("audio_buffer")

  def add_chunk(self, chunk: bytes) -> np.ndarray | None:
    """
    Add an audio chunk to the buffer.

    Args:
        chunk: Raw audio bytes from FFmpeg (16-bit PCM, little-endian)

    Returns:
        numpy.ndarray if buffer has enough data for transcription, None otherwise
    """
    if not chunk:
      return None

    self.buffer += chunk
    self.total_chunks_added += 1

    # Check if we have enough data for transcription
    if len(self.buffer) >= self.target_bytes:
      # Extract target amount of data
      audio_bytes = self.buffer[: self.target_bytes]
      self.buffer = self.buffer[self.target_bytes :]  # Keep overflow

      # Convert bytes to numpy array
      # FFmpeg outputs 16-bit signed PCM little-endian (s16le)
      audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

      # Convert to float32 normalized to [-1.0, 1.0] for Whisper
      audio_array = audio_array.astype(np.float32) / 32768.0

      self.total_arrays_produced += 1

      # Log statistics periodically
      if self.total_arrays_produced % 10 == 0:
        self.logger.debug(
          "Audio buffer statistics",
          chunks_added=self.total_chunks_added,
          arrays_produced=self.total_arrays_produced,
          buffer_bytes=len(self.buffer),
          array_samples=len(audio_array),
        )

      return audio_array

    return None

  def get_remaining_audio(self) -> np.ndarray | None:
    """
    Get any remaining audio in the buffer as a numpy array.

    Used during shutdown to process any partial buffer data.

    Returns:
        numpy.ndarray if buffer has any data, None if empty
    """
    if not self.buffer:
      return None

    # Convert remaining bytes to numpy array
    audio_array = np.frombuffer(self.buffer, dtype=np.int16)
    audio_array = audio_array.astype(np.float32) / 32768.0

    # Clear buffer
    self.buffer = b""

    self.logger.debug(
      "Extracted remaining audio",
      samples=len(audio_array),
      duration_seconds=len(audio_array) / self.sample_rate,
    )

    return audio_array

  def reset(self):
    """Reset the buffer state."""
    self.buffer = b""
    self.total_chunks_added = 0
    self.total_arrays_produced = 0
    self.logger.debug("Audio buffer reset")


class RTSPTranscriptionClient(RTSPClient):
  """
  RTSP client with integrated transcription capabilities.

  Extends RTSPClient to add audio transcription functionality. Manages both
  the FFmpeg subprocess for audio ingestion and a transcription worker that
  processes audio chunks using a shared Whisper model.
  """

  def __init__(self, stream_name: str, rtsp_url: str, transcriber):
    """
    Initialize the RTSP transcription client.

    Args:
        stream_name: Human-readable name for this stream
        rtsp_url: RTSP URL to connect to
        transcriber: ServeClientFasterWhisper instance for transcription
    """
    # Create internal queue for audio data
    self.audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)

    # Initialize parent class with our internal queue
    super().__init__(stream_name, rtsp_url, self.audio_queue)

    # Transcription components
    self.transcriber = transcriber
    self.audio_buffer = AudioBuffer()

    # Task management
    self.transcription_task: asyncio.Task | None = None

    # Statistics
    self.transcriptions_completed = 0
    self.transcription_errors = 0

  async def run(self) -> None:
    """
    Main entry point - runs both FFmpeg ingestion and transcription concurrently.

    Overrides parent run() to start both the FFmpeg process and transcription
    worker as concurrent tasks.
    """
    with structlog.contextvars.bound_contextvars(stream=self.stream_name):
      self.logger.info("Starting RTSP transcription client", url=self.rtsp_url)

      while not self.stopped:
        try:
          # Reset statistics for this connection attempt
          self.chunks_read = 0
          self.total_bytes = 0
          self.transcriptions_completed = 0
          self.transcription_errors = 0
          self.audio_buffer.reset()

          if self.reconnect_count > 0:
            self.logger.info("Reconnecting to RTSP stream", attempt=self.reconnect_count + 1)
          else:
            self.logger.info("Connecting to RTSP stream")

          # Create and start the FFmpeg process
          process = await self._create_ffmpeg_process()

          # Start concurrent tasks for audio ingestion, error monitoring, and transcription
          audio_task = asyncio.create_task(self._read_audio_stream())
          error_task = asyncio.create_task(self._monitor_process_errors())
          transcription_task = asyncio.create_task(self._transcription_worker())

          # Store transcription task for cleanup
          self.transcription_task = transcription_task

          self.logger.info("RTSP stream connected, processing audio and transcribing")

          # Wait for any task to complete (indicates process death or error)
          done, pending = await asyncio.wait(
            [audio_task, error_task, transcription_task], return_when=asyncio.FIRST_COMPLETED
          )

          # Cancel any remaining tasks
          for task in pending:
            task.cancel()
            try:
              await task
            except asyncio.CancelledError:
              pass

          # Check if any task raised an exception
          for task in done:
            if task.exception():
              self.logger.error("Task failed", task=task.get_name(), error=str(task.exception()))

          # Wait for process to finish and check exit code
          exit_code = await process.wait()
          if exit_code != 0:
            self.logger.warning("FFmpeg process exited with error", exit_code=exit_code)
          else:
            self.logger.info("FFmpeg process exited normally")

          # Clean up process reference
          self.process = None
          self.transcription_task = None

        except Exception:
          self.logger.exception("RTSP connection attempt failed")
          self.process = None
          self.transcription_task = None

        # Only reconnect if we haven't been stopped
        if not self.stopped:
          self.reconnect_count += 1
          self.logger.info("Reconnecting in 30 seconds", attempt=self.reconnect_count + 1)

          try:
            await asyncio.sleep(30)
          except asyncio.CancelledError:
            self.logger.info("Reconnection wait interrupted")
            break

      self.logger.info("RTSP transcription client stopped")

  async def _transcription_worker(self) -> None:
    """
    Worker task that processes audio from the queue and performs transcription.

    Continuously pulls audio chunks from the queue, accumulates them using
    AudioBuffer, and performs transcription when enough audio is available.
    """
    self.logger.info("Starting transcription worker")

    try:
      while not self.stopped:
        try:
          # Get audio chunk from queue with timeout
          chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=1.0)

          # Add to buffer and check if ready for transcription
          audio_array = self.audio_buffer.add_chunk(chunk)

          if audio_array is not None:
            # We have enough audio data - transcribe it
            await self._transcribe_audio_chunk(audio_array)

        except asyncio.TimeoutError:
          # No audio data available - continue loop
          continue
        except asyncio.CancelledError:
          self.logger.debug("Transcription worker cancelled")
          break
        except Exception:
          self.transcription_errors += 1
          self.logger.exception(
            "Error in transcription worker", error_count=self.transcription_errors
          )
          # Brief pause to avoid tight error loops
          await asyncio.sleep(0.1)

      # Process any remaining audio on shutdown
      remaining_audio = self.audio_buffer.get_remaining_audio()
      if remaining_audio is not None:
        await self._transcribe_audio_chunk(remaining_audio)

    except Exception:
      self.logger.exception("Fatal error in transcription worker")
    finally:
      self.logger.info(
        "Transcription worker stopped",
        transcriptions_completed=self.transcriptions_completed,
        transcription_errors=self.transcription_errors,
      )

  async def _transcribe_audio_chunk(self, audio_array: np.ndarray) -> None:
    """
    Transcribe a single audio chunk using the shared model.

    Args:
        audio_array: Audio data as numpy array, normalized to [-1.0, 1.0]
    """
    try:
      duration = len(audio_array) / 16000  # 16kHz sample rate
      self.logger.debug("Transcribing audio chunk", duration_seconds=f"{duration:.2f}")

      # Perform transcription in background thread to avoid blocking
      result, info = await asyncio.to_thread(self.transcriber.transcribe_audio, audio_array)

      # Handle transcription results
      if result:
        segments = list(result)  # Convert generator to list
        if segments:
          # Log transcription results (for now)
          for segment in segments:
            segment_text = getattr(segment, "text", "").strip()
            if segment_text:
              segment_start = getattr(segment, "start", 0)
              segment_end = getattr(segment, "end", duration)

              self.logger.info(
                "Transcription result",
                stream=self.stream_name,
                text=segment_text,
                start=f"{segment_start:.2f}s",
                end=f"{segment_end:.2f}s",
                duration=f"{duration:.2f}s",
              )
        else:
          self.logger.debug("No transcription segments produced")
      else:
        self.logger.debug("Transcription returned no result")

      self.transcriptions_completed += 1

      # Log progress periodically
      if self.transcriptions_completed % 10 == 0:
        self.logger.debug(
          "Transcription progress",
          completed=self.transcriptions_completed,
          errors=self.transcription_errors,
        )

    except Exception:
      self.transcription_errors += 1
      self.logger.exception("Failed to transcribe audio chunk")

  async def stop(self) -> None:
    """
    Enhanced stop method that also cancels transcription task.

    Extends parent stop() to properly clean up transcription worker.
    """
    if self.stopped:
      self.logger.debug("Stop called but client already stopped")
      return

    self.logger.info("Stopping RTSP transcription client")
    self.stopped = True

    # Cancel transcription task if running
    if self.transcription_task and not self.transcription_task.done():
      self.logger.debug("Cancelling transcription task")
      self.transcription_task.cancel()
      try:
        await self.transcription_task
      except asyncio.CancelledError:
        pass

    # Clean up FFmpeg process
    await self._cleanup_process()

    self.logger.info("RTSP transcription client shutdown complete")
