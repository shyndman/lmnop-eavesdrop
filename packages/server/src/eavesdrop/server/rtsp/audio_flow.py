import asyncio
from typing import TYPE_CHECKING

import numpy as np

from eavesdrop.common import get_logger
from eavesdrop.server.constants import SAMPLE_RATE
from eavesdrop.server.streaming.interfaces import (
  AudioSource,
  TranscriptionResult,
  TranscriptionSink,
)
from eavesdrop.wire import Segment

if TYPE_CHECKING:
  from eavesdrop.server.rtsp.cache import RTSPTranscriptionCache
  from eavesdrop.server.rtsp.subscriber import RTSPSubscriberManager


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

    :param audio_queue: asyncio.Queue containing raw audio bytes from FFmpeg.
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

    :param stream_name: Unique identifier for the RTSP stream (used in log context).
    :param subscriber_manager: Manager for WebSocket subscribers that will receive
                           transcription results.
    :param transcription_cache: Cache manager for storing transcription history.
    :param logger_name: Logger name for log routing and filtering.
    """
    self.stream_name: str = stream_name
    self.logger = get_logger(logger_name, stream=stream_name)
    self.transcription_count: int = 0
    self.subscriber_manager = subscriber_manager
    self.transcription_cache = transcription_cache

  async def send_result(self, result: TranscriptionResult) -> None:
    """Log transcription results and send to WebSocket subscribers if available."""
    self.transcription_count += 1

    # Separate completed and incomplete segments
    completed_segments: list[Segment] = []
    incomplete_segments: list[Segment] = []

    for segment in filter(lambda s: s.text.strip(), result.segments):
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
        f"Transcription: {concatenated_text}",
        start=start_time,
        end=end_time,
        completed=True,
        transcription_number=self.transcription_count,
      )

    # Log each incomplete segment separately
    for segment in incomplete_segments:
      self.logger.info(
        f"Transcription: {segment.text.strip()}",
        start=segment.start,
        end=segment.end,
        completed=False,
        temp=segment.temperature,
        compression=segment.compression_ratio,
        avg_logprob=segment.avg_logprob,
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
