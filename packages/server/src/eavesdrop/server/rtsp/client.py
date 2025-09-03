import asyncio
from typing import TYPE_CHECKING

import numpy as np
import structlog

from eavesdrop.server.constants import SAMPLE_RATE
from eavesdrop.server.logs import get_logger

if TYPE_CHECKING:
  from eavesdrop.server.rtsp.cache import RTSPTranscriptionCache
  from eavesdrop.server.rtsp.subscriber import RTSPSubscriberManager
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.interfaces import (
  AudioSource,
  TranscriptionResult,
  TranscriptionSink,
)
from eavesdrop.server.streaming.processor import (
  StreamingTranscriptionProcessor,
  TranscriptionConfig,
)


class RTSPAudioSource(AudioSource):
  """
  RTSP implementation of AudioSource protocol.

  Converts raw FFmpeg audio bytes to numpy arrays suitable for Whisper transcription.
  Handles the complex audio format conversion chain from RTSP streams through FFmpeg
  to the streaming transcription processor.

  Audio Processing Chain:
    RTSP stream → FFmpeg (s16le PCM) → RTSPClient queue → RTSPAudioSource → StreamingProcessor

  Data Conversion:
    1. FFmpeg outputs 16-bit signed PCM little-endian (s16le) as bytes
    2. np.frombuffer() converts bytes → int16 numpy array
    3. Normalize int16 → float32 [-1.0, 1.0] for Whisper compatibility
    4. Division by 32768.0 handles int16 → float32 normalization

  Format Specifications:
    - Input: Raw bytes from FFmpeg (16-bit PCM, little-endian, 16kHz)
    - Output: numpy.ndarray (float32, normalized [-1.0, 1.0], 16kHz)
    - Sample Rate: 16000 Hz (required by Whisper)
    - Bit Depth: 16-bit signed → float32 conversion

  Timeout Behavior:
    - 1-second timeout on queue reads (prevents blocking on stream issues)
    - Returns empty array on timeout (keeps processor alive vs ending stream)
    - Different from WebSocket which returns None to end stream

  Error Handling:
    - Empty chunks or closed state → None (end stream)
    - Queue timeout → empty array (maintain connection)
    - No exception propagation (graceful degradation)

  Threading:
    - Safe for single-threaded async use within StreamingTranscriptionProcessor
    - Queue operations are thread-safe but source instance is not
  """

  def __init__(self, audio_queue: asyncio.Queue[bytes]) -> None:
    """
    Initialize RTSP audio source with FFmpeg byte queue.

    Args:
        audio_queue: asyncio.Queue containing raw audio bytes from FFmpeg.
                    Queue should contain 16-bit PCM little-endian data at 16kHz.
    """
    self.audio_queue: asyncio.Queue[bytes] = audio_queue
    self.sample_rate: int = SAMPLE_RATE
    self.bytes_per_sample: int = 2  # 16-bit PCM
    self.closed: bool = False

  async def read_audio(self) -> np.ndarray | None:
    """
    Read from FFmpeg queue and convert to numpy array.
    Returns None when stream ends.
    """
    try:
      # Get audio chunk from FFmpeg (via RTSPClient)
      chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=1.0)

      if not chunk or self.closed:
        return None

      # Convert FFmpeg bytes to numpy array (like WebSocket does)
      # FFmpeg outputs 16-bit signed PCM little-endian (s16le)
      audio_array = np.frombuffer(chunk, dtype=np.int16)

      # Convert to float32 normalized to [-1.0, 1.0] for Whisper
      audio_array = audio_array.astype(np.float32) / 32768.0

      return audio_array

    except asyncio.TimeoutError:
      # No audio available - return empty array to keep processor alive
      return np.array([], dtype=np.float32)

  def close(self) -> None:
    self.closed = True


class RTSPTranscriptionSink(TranscriptionSink):
  """
  RTSP implementation of TranscriptionSink protocol.

  Logs transcription results using structured logging instead of sending them
  over a network connection. Designed for RTSP streaming scenarios where
  transcription output is consumed via log aggregation systems.

  Message Flow:
    StreamingProcessor → RTSPTranscriptionSink → Structured logs

  Logging Output:
    - Transcription results: Structured log entries with text, timestamps, completion
    - Error messages: Transcription failures logged at ERROR level
    - Language detection: Detected language and confidence scores
    - Server status: No-op (RTSP doesn't have connection handshakes)

  Log Structure:
    - All logs include stream name for multi-stream identification
    - Transcription count provides sequence numbering
    - Structured fields: text, start, end, completed, transcription_number
    - Uses logger.exception() for proper error stack traces

  Differences from WebSocket:
    - Logs instead of sending JSON messages
    - Server ready/disconnect are no-ops (no connection state)
    - Error level logging instead of error message sending

  Threading:
    - Safe for single-threaded async use within StreamingTranscriptionProcessor
    - Logging operations are thread-safe
  """

  def __init__(
    self,
    stream_name: str,
    subscriber_manager: "RTSPSubscriberManager",
    transcription_cache: "RTSPTranscriptionCache",
    logger_name: str = "rtsp/sink",
  ) -> None:
    """
    Initialize RTSP transcription sink with structured logging and caching.

    Args:
        stream_name: Unique identifier for the RTSP stream (used in log context).
        subscriber_manager: Manager for WebSocket subscribers that will receive
                           transcription results.
        transcription_cache: Cache manager for storing transcription history.
        logger_name: Logger name for log routing and filtering.
    """
    self.stream_name: str = stream_name
    self.logger = get_logger(logger_name).bind(stream=stream_name)
    self.transcription_count: int = 0
    self.subscriber_manager = subscriber_manager
    self.transcription_cache = transcription_cache

  async def send_result(self, result: TranscriptionResult) -> None:
    """Log transcription results and send to WebSocket subscribers if available."""
    self.transcription_count += 1

    # Separate completed and incomplete segments
    completed_segments = []
    incomplete_segments = []

    for segment in result.segments:
      if segment.text.strip():
        if segment.completed:
          completed_segments.append(segment)
        else:
          incomplete_segments.append(segment)

    # Log concatenated completed segments as single entry
    if completed_segments:
      concatenated_text = " ".join(seg.text.strip() for seg in completed_segments)
      start_time = min(seg.start for seg in completed_segments)
      end_time = max(seg.end for seg in completed_segments)

      self.logger.info(
        "Transcription result",
        text=concatenated_text,
        start=start_time,
        end=end_time,
        completed=True,
        transcription_number=self.transcription_count,
      )

    # Log each incomplete segment separately
    for segment in incomplete_segments:
      self.logger.info(
        "Transcription result",
        text=segment.text.strip(),
        start=segment.start,
        end=segment.end,
        completed=False,
        transcription_number=self.transcription_count,
      )

    # Store transcription in cache for later retrieval by new subscribers
    try:
      await self.transcription_cache.add_transcription(
        self.stream_name, result.segments, result.language
      )

      self.logger.debug(
        "Stored transcription in cache",
        transcription_number=self.transcription_count,
        segments=len(result.segments),
      )

    except Exception as e:
      self.logger.error(
        "Failed to store transcription in cache",
        error=str(e),
        transcription_number=self.transcription_count,
      )

    # Send to WebSocket subscribers
    try:
      await self.subscriber_manager.send_transcription(
        self.stream_name, result.segments, result.language
      )

      self.logger.debug(
        "Sent transcription result to subscribers",
        transcription_number=self.transcription_count,
        segments=len(result.segments),
      )

    except Exception as e:
      self.logger.error(
        "Failed to send transcription result to subscribers",
        error=str(e),
        transcription_number=self.transcription_count,
      )

  async def send_error(self, error: str) -> None:
    """Log transcription errors and notify subscribers."""
    self.logger.error("Transcription error", error=error)

    # Notify subscribers of stream error
    try:
      await self.subscriber_manager.send_stream_status(
        self.stream_name, "error", f"Transcription error: {error}"
      )
    except Exception as e:
      self.logger.error(
        "Failed to notify subscribers of transcription error",
        original_error=error,
        notification_error=str(e),
      )

  async def send_language_detection(self, language: str, probability: float) -> None:
    """Log language detection results (no subscriber notification needed)."""
    self.logger.info("Language detected", language=language, probability=probability)

  async def send_server_ready(self, backend: str) -> None:
    """Log server ready status and notify subscribers stream is online."""
    self.logger.info("Server ready", backend=backend)

    # Notify subscribers that stream is online
    try:
      await self.subscriber_manager.send_stream_status(
        self.stream_name, "online", f"Stream started with {backend} backend"
      )
    except Exception as e:
      self.logger.error("Failed to notify subscribers of stream online status", error=str(e))

  async def disconnect(self) -> None:
    """Log disconnection and notify subscribers stream is offline."""
    self.logger.info("Stream disconnected", total_transcriptions=self.transcription_count)

    # Notify subscribers that stream is offline
    try:
      await self.subscriber_manager.send_stream_status(
        self.stream_name, "offline", "Stream disconnected"
      )
    except Exception as e:
      self.logger.error("Failed to notify subscribers of stream offline status", error=str(e))


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
    self.logger = get_logger("rtsp/client").bind(stream=stream_name)

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
      *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=1024 * 1024
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

    Handles oversized lines gracefully by reading in chunks and reconstructing
    complete messages when possible.

    Returns when the process terminates or when stopped.
    """
    if not self.process or not self.process.stderr:
      self.logger.error("No process or stderr available for monitoring")
      return

    self.logger.debug("Starting FFmpeg error monitoring")

    try:
      while not self.stopped:
        try:
          # Use a reasonable limit to prevent memory issues with very long lines
          # If line exceeds limit, we'll get the truncated portion
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
            # Generic FFmpeg output - log at warn level to ensure we see diagnostics
            self.logger.warning("FFmpeg output", message=error_msg)

        except asyncio.LimitOverrunError as e:
          # Handle oversized lines by reading the available partial data
          self.logger.warning(
            "FFmpeg output line exceeded buffer limit, truncating",
            limit=e.consumed,
            partial_message="<truncated - line too long>",
          )
          # Consume the partial data to prevent stream corruption and show what we can
          try:
            partial = await self.process.stderr.read(e.consumed)
            if partial:
              truncated_msg = partial.decode("utf-8", errors="replace").strip()[
                :1000
              ]  # Show more context
              self.logger.warning(
                "FFmpeg oversized output (partial)", message=f"{truncated_msg}..."
              )
          except Exception as read_error:
            self.logger.warning("Could not read oversized FFmpeg output", error=str(read_error))

        except UnicodeDecodeError:
          self.logger.warning("FFmpeg output contained invalid UTF-8, skipping line")

    except asyncio.CancelledError:
      self.logger.debug("Error monitoring cancelled")
      raise
    except (asyncio.LimitOverrunError, ValueError) as e:
      self.logger.warning(
        "FFmpeg stderr buffer overflow, continuing without stderr monitoring", error=str(e)
      )
      # Don't re-raise - continue without stderr monitoring to allow reconnection
    except Exception:
      self.logger.exception("Error monitoring FFmpeg stderr")
      # Don't re-raise - we want the connection to continue and retry

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


class RTSPTranscriptionClient(RTSPClient):
  def __init__(
    self,
    stream_name: str,
    rtsp_url: str,
    transcription_config: TranscriptionConfig,
    subscriber_manager: "RTSPSubscriberManager",
    transcription_cache: "RTSPTranscriptionCache",
  ):
    # Initialize parent with internal queue
    super().__init__(stream_name, rtsp_url, asyncio.Queue(maxsize=100))

    # Create abstracted components
    self.audio_source = RTSPAudioSource(self.audio_queue)
    self.stream_buffer = AudioStreamBuffer(transcription_config.buffer)
    self.transcription_sink = RTSPTranscriptionSink(
      stream_name, subscriber_manager, transcription_cache
    )
    self.processor = StreamingTranscriptionProcessor(
      buffer=self.stream_buffer,
      sink=self.transcription_sink,
      config=transcription_config,
      stream_name=stream_name,
      logger_name=f"rtsp/proc.{stream_name}",
    )

    # Statistics (preserved from current implementation)
    self.transcriptions_completed = 0
    self.transcription_errors = 0

  async def run(self) -> None:
    """Enhanced run method with new processing architecture"""
    with structlog.contextvars.bound_contextvars(stream=self.stream_name):
      self.logger.info("Starting RTSP transcription client", url=self.rtsp_url)

      while not self.stopped:
        try:
          if self.reconnect_count > 0:
            self.logger.info("Reconnecting to RTSP stream", attempt=self.reconnect_count + 1)
          else:
            self.logger.info("Connecting to RTSP stream")

          process = await self._create_ffmpeg_process()

          audio_task = asyncio.create_task(self._read_audio_stream())
          error_task = asyncio.create_task(self._monitor_process_errors())
          streaming_task = asyncio.create_task(self._streaming_processor_task())
          audio_feeding_task = asyncio.create_task(self._feed_audio_to_buffer())

          self.logger.info("RTSP stream connected, processing audio and transcribing")

          done, pending = await asyncio.wait(
            [audio_task, error_task, streaming_task, audio_feeding_task],
            return_when=asyncio.FIRST_COMPLETED,
          )

          for task in pending:
            task.cancel()
            try:
              await task
            except asyncio.CancelledError:
              pass

          for task in done:
            if task.exception():
              self.logger.error("Task failed", task=task.get_name(), error=str(task.exception()))

          exit_code = await process.wait()
          if exit_code != 0:
            self.logger.warning("FFmpeg process exited with error", exit_code=exit_code)
          else:
            self.logger.info("FFmpeg process exited normally")

          self.process = None

        except Exception:
          self.logger.exception("RTSP connection attempt failed")
          self.process = None

          if not self.stopped:
            self.reconnect_count += 1
            self.logger.info("Reconnecting in 30 seconds", attempt=self.reconnect_count + 1)

            try:
              await asyncio.sleep(30)
            except asyncio.CancelledError:
              self.logger.info("Reconnection wait interrupted")
              break
      self.logger.info("RTSP transcription client stopped")

  async def _streaming_processor_task(self) -> None:
    """New task to run the streaming transcription processor"""
    try:
      await self.processor.initialize()
      await self.processor.start_processing()
    except Exception:
      self.logger.exception("Streaming processor failed")

  async def _feed_audio_to_buffer(self) -> None:
    """New task to feed audio from source to buffer"""
    try:
      while not self.stopped:
        audio_array = await self.audio_source.read_audio()

        if audio_array is None:
          break

        if len(audio_array) > 0:
          self.processor.add_audio_frames(audio_array)

    except Exception:
      self.logger.exception("Audio feeding task failed")

  async def stop(self) -> None:
    """Enhanced stop with new component cleanup"""
    if self.stopped:
      return

    self.logger.info("Stopping RTSP transcription client")
    self.stopped = True

    await self.processor.stop_processing()
    self.audio_source.close()

    await self._cleanup_process()
    self.logger.info("RTSP transcription client shutdown complete")
