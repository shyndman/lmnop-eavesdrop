"""Contract tests for file-mode ingest decode, queueing, and EOF tail handling."""

import asyncio
import math
import subprocess
import wave
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from io import BytesIO
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

from eavesdrop.server.config import BufferConfig, TranscriptionConfig
from eavesdrop.server.streaming.audio_flow import WebSocketTranscriptionSink
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.file_decoder import decode_file_bytes_to_canonical_audio
from eavesdrop.server.streaming.file_queue import FileAudioQueue
from eavesdrop.server.streaming.interfaces import TranscriptionResult
from eavesdrop.server.streaming.processor import AudioChunk, StreamingTranscriptionProcessor
from eavesdrop.server.transcription.session import create_session
from eavesdrop.wire import TranscriptionSourceMode

Float32Audio = NDArray[np.float32]
EncodedPayloadBuilder = Callable[[bytes], bytes]
_decode_file_bytes_to_canonical_audio = cast(
  Callable[[bytes], Awaitable[Float32Audio]],
  decode_file_bytes_to_canonical_audio,
)


class _LifecycleProcessorDouble:
  """Processor double that records finite-source lifecycle sequencing."""

  _events: list[str]

  def __init__(self, events: list[str]) -> None:
    self._events = events

  async def start_processing(self) -> None:
    self._events.append("processing_completed")

  def mark_source_exhausted(self) -> None:
    self._events.append("source_exhausted")

  async def stop_processing(self) -> None:
    self._events.append("stop_processing")


@dataclass
class NoopSink:
  """Minimal sink used to instantiate processor for buffer-level contract checks."""

  async def send_result(self, result: TranscriptionResult) -> None:
    del result
    return

  async def send_error(self, error: str) -> None:
    del error
    return

  async def send_language_detection(self, language: str, probability: float) -> None:
    del language, probability
    return

  async def send_server_ready(self, backend: str) -> None:
    del backend
    return

  async def disconnect(self) -> None:
    return


@dataclass
class _SinkDouble:
  disconnect: AsyncMock
  send_error: AsyncMock


def _build_noncanonical_wav_bytes(duration_s: float = 0.35) -> bytes:
  """Create stereo 22.05kHz WAV bytes to validate canonical decode normalization."""
  sample_rate = 22_050
  sample_count = int(duration_s * sample_rate)

  left: list[float] = []
  right: list[float] = []
  for index in range(sample_count):
    t = index / sample_rate
    left.append(math.sin(2 * math.pi * 440.0 * t))
    right.append(math.sin(2 * math.pi * 660.0 * t))

  stereo: NDArray[np.float64] = np.column_stack((left, right))
  pcm16: NDArray[np.int16] = (np.clip(stereo, -1.0, 1.0) * 32767).astype(np.int16)

  with BytesIO() as wav_buffer:
    with wave.open(wav_buffer, "wb") as wav_writer:
      wav_writer.setnchannels(2)
      wav_writer.setsampwidth(2)
      wav_writer.setframerate(sample_rate)
      wav_writer.writeframes(pcm16.tobytes())
    return wav_buffer.getvalue()


def _encode_with_ffmpeg(wav_bytes: bytes, output_format: str, codec: str | None = None) -> bytes:
  """Encode WAV bytes to another format using local ffmpeg for decode contract tests."""
  command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "wav", "-i", "pipe:0"]
  if codec:
    command.extend(["-c:a", codec])
  command.extend(["-f", output_format, "pipe:1"])

  result = subprocess.run(command, input=wav_bytes, capture_output=True, check=True)
  return result.stdout


def _as_wav(wav_bytes: bytes) -> bytes:
  return wav_bytes


def _as_mp3(wav_bytes: bytes) -> bytes:
  return _encode_with_ffmpeg(wav_bytes, "mp3")


def _as_aac(wav_bytes: bytes) -> bytes:
  return _encode_with_ffmpeg(wav_bytes, "adts", codec="aac")


def _enqueue(queue: FileAudioQueue, chunk: Float32Audio) -> Coroutine[object, object, float]:
  return cast(Callable[[Float32Audio], Coroutine[object, object, float]], queue.enqueue)(chunk)


def _dequeue(queue: FileAudioQueue) -> Coroutine[object, object, Float32Audio | None]:
  return cast(Callable[[], Coroutine[object, object, Float32Audio | None]], queue.dequeue)()


def _add_audio_frames(processor: StreamingTranscriptionProcessor, frames: Float32Audio) -> None:
  cast(Callable[[Float32Audio], None], processor.add_audio_frames)(frames)


def _get_next_audio_chunk(
  processor: StreamingTranscriptionProcessor,
) -> Coroutine[object, object, AudioChunk | None]:
  return cast(
    Callable[[], Coroutine[object, object, AudioChunk | None]],
    getattr(processor, "_get_next_audio_chunk"),
  )()


def _set_attr(target: object, name: str, value: object) -> None:
  setattr(target, name, value)


@pytest.mark.asyncio
@pytest.mark.parametrize("encoded_payload_builder", [_as_wav, _as_mp3, _as_aac])
async def test_decoder_normalizes_wav_mp3_aac_to_mono_16khz_float32(
  encoded_payload_builder: EncodedPayloadBuilder,
) -> None:
  """Decoder must output canonical mono 16kHz float32 audio for all supported file formats."""
  wav_bytes = _build_noncanonical_wav_bytes(duration_s=0.35)
  encoded_payload = encoded_payload_builder(wav_bytes)

  decoded = await _decode_file_bytes_to_canonical_audio(encoded_payload)

  assert decoded.dtype == np.float32
  assert decoded.ndim == 1
  assert decoded.shape[0] >= int(0.30 * 16_000)
  assert decoded.shape[0] <= int(0.45 * 16_000)


@pytest.mark.asyncio
async def test_file_queue_blocks_enqueue_when_capacity_would_be_exceeded() -> None:
  """Queue must block producers instead of dropping canonical audio when full."""
  queue = FileAudioQueue(capacity_seconds=0.01)  # 160 samples at 16kHz
  first_chunk: Float32Audio = np.ones(120, dtype=np.float32)
  second_chunk: Float32Audio = np.ones(80, dtype=np.float32)

  first_block = await _enqueue(queue, first_chunk)
  assert first_block < 0.01

  blocked_enqueue = asyncio.create_task(_enqueue(queue, second_chunk))
  await asyncio.sleep(0.05)
  assert blocked_enqueue.done() is False

  dequeued = await _dequeue(queue)
  assert dequeued is not None
  assert dequeued.shape[0] == 120

  second_block = await asyncio.wait_for(blocked_enqueue, timeout=0.2)
  assert second_block > 0.0


@pytest.mark.asyncio
async def test_source_exhausted_short_tail_is_returned_for_processing() -> None:
  """Final short tail must still be eligible for processing once EOF/source exhaustion is known."""
  buffer_config = BufferConfig(
    sample_rate=16_000,
    max_buffer_duration=10.0,
    cleanup_duration=5.0,
    min_chunk_duration=0.30,
    transcription_interval=0.05,
  )
  processor = StreamingTranscriptionProcessor(
    buffer=AudioStreamBuffer(buffer_config),
    sink=NoopSink(),
    config=TranscriptionConfig(),
    session=create_session("file-tail"),
    stream_name="file-tail",
  )

  _add_audio_frames(processor, np.ones(int(0.20 * buffer_config.sample_rate), dtype=np.float32))

  before_eof = await _get_next_audio_chunk(processor)
  assert before_eof is None

  processor.mark_source_exhausted()
  after_eof = await _get_next_audio_chunk(processor)

  assert after_eof is not None
  assert after_eof.duration == pytest.approx(0.20, abs=0.01)
  assert after_eof.data.dtype == np.float32


@pytest.mark.asyncio
async def test_file_lifecycle_disconnects_after_drain_and_processing() -> None:
  """File lifecycle must drain ingest/feed before terminal disconnect/finalization."""
  from eavesdrop.server.streaming.client import FileLifecycleState, WebSocketStreamingClient

  events: list[str] = []
  ingest_completed = asyncio.Event()
  client = WebSocketStreamingClient.__new__(WebSocketStreamingClient)
  client.logger = MagicMock()
  client.stream_name = "file-order"
  client.source_mode = TranscriptionSourceMode.FILE
  _set_attr(client, "_file_state", FileLifecycleState.TERMINAL)
  _set_attr(client, "_exit", False)
  _set_attr(client, "_stopped", False)
  _set_attr(client, "_file_ingest_task", None)
  _set_attr(client, "_file_feed_task", None)
  _set_attr(client, "_file_observability_task", None)
  _set_attr(client, "_audio_task", None)
  _set_attr(client, "_processing_task", None)
  _set_attr(client, "_file_queue", None)
  _set_attr(client, "_file_ingested_seconds", 0.0)
  client.audio_source = MagicMock()

  async def _ingest() -> None:
    events.append("ingest_done")
    ingest_completed.set()

  async def _feed() -> None:
    _ = await ingest_completed.wait()
    events.append("feed_done")

  async def _observe() -> None:
    file_state = cast(FileLifecycleState, getattr(client, "_file_state"))
    while file_state in (FileLifecycleState.INGESTING, FileLifecycleState.DRAINING):
      await asyncio.sleep(0.01)
      file_state = cast(FileLifecycleState, getattr(client, "_file_state"))

  _set_attr(client, "_ingest_file_upload", _ingest)
  _set_attr(client, "_feed_file_queue_to_processor", _feed)
  _set_attr(client, "_file_observability_loop", _observe)
  client.processor = cast(
    StreamingTranscriptionProcessor,
    cast(object, _LifecycleProcessorDouble(events)),
  )

  async def _disconnect() -> None:
    events.append("disconnect")

  client.transcription_sink = cast(
    WebSocketTranscriptionSink,
    cast(
      object,
      _SinkDouble(
        disconnect=AsyncMock(side_effect=_disconnect),
        send_error=AsyncMock(),
      ),
    ),
  )

  async def _stop() -> None:
    events.append("stop")

  client.stop = AsyncMock(side_effect=_stop)

  await cast(Callable[[], Awaitable[None]], getattr(client, "_run_file_mode_lifecycle"))()

  assert events[0] == "processing_completed"
  assert events[1:] == ["ingest_done", "feed_done", "source_exhausted", "disconnect", "stop"]
  assert cast(FileLifecycleState, getattr(client, "_file_state")) == FileLifecycleState.TERMINAL
