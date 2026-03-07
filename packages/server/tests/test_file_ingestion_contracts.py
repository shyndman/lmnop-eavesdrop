"""Contract tests for file-mode ingest decode, queueing, and EOF tail handling."""

import asyncio
import math
import subprocess
import wave
from dataclasses import dataclass
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from eavesdrop.server.config import BufferConfig, TranscriptionConfig
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.file_decoder import decode_file_bytes_to_canonical_audio
from eavesdrop.server.streaming.file_queue import FileAudioQueue
from eavesdrop.server.streaming.interfaces import TranscriptionResult
from eavesdrop.server.streaming.processor import StreamingTranscriptionProcessor
from eavesdrop.server.transcription.session import create_session


class _LifecycleProcessorDouble:
  """Processor double that records finite-source lifecycle sequencing."""

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
    return

  async def send_error(self, error: str) -> None:
    return

  async def send_language_detection(self, language: str, probability: float) -> None:
    return

  async def send_server_ready(self, backend: str) -> None:
    return

  async def disconnect(self) -> None:
    return


def _build_noncanonical_wav_bytes(duration_s: float = 0.35) -> bytes:
  """Create stereo 22.05kHz WAV bytes to validate canonical decode normalization."""
  sample_rate = 22_050
  sample_count = int(duration_s * sample_rate)

  left = []
  right = []
  for index in range(sample_count):
    t = index / sample_rate
    left.append(math.sin(2 * math.pi * 440.0 * t))
    right.append(math.sin(2 * math.pi * 660.0 * t))

  stereo = np.column_stack((left, right))
  pcm16 = np.int16(np.clip(stereo, -1.0, 1.0) * 32767)

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


@pytest.mark.asyncio
@pytest.mark.parametrize(
  "encoded_payload_builder",
  [
    lambda wav_bytes: wav_bytes,
    lambda wav_bytes: _encode_with_ffmpeg(wav_bytes, "mp3"),
    lambda wav_bytes: _encode_with_ffmpeg(wav_bytes, "adts", codec="aac"),
  ],
)
async def test_decoder_normalizes_wav_mp3_aac_to_mono_16khz_float32(
  encoded_payload_builder,
) -> None:
  """Decoder must output canonical mono 16kHz float32 audio for all supported file formats."""
  wav_bytes = _build_noncanonical_wav_bytes(duration_s=0.35)
  encoded_payload = encoded_payload_builder(wav_bytes)

  decoded = await decode_file_bytes_to_canonical_audio(encoded_payload)

  assert decoded.dtype == np.float32
  assert decoded.ndim == 1
  assert decoded.shape[0] >= int(0.30 * 16_000)
  assert decoded.shape[0] <= int(0.45 * 16_000)


@pytest.mark.asyncio
async def test_file_queue_blocks_enqueue_when_capacity_would_be_exceeded() -> None:
  """Queue must block producers instead of dropping canonical audio when full."""
  queue = FileAudioQueue(capacity_seconds=0.01)  # 160 samples at 16kHz
  first_chunk = np.ones(120, dtype=np.float32)
  second_chunk = np.ones(80, dtype=np.float32)

  first_block = await queue.enqueue(first_chunk)
  assert first_block < 0.01

  blocked_enqueue = asyncio.create_task(queue.enqueue(second_chunk))
  await asyncio.sleep(0.05)
  assert blocked_enqueue.done() is False

  dequeued = await queue.dequeue()
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

  processor.add_audio_frames(np.ones(int(0.20 * buffer_config.sample_rate), dtype=np.float32))

  before_eof = await processor._get_next_audio_chunk()
  assert before_eof is None

  processor.mark_source_exhausted()
  after_eof = await processor._get_next_audio_chunk()

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
  client.source_mode = "file"
  client._file_state = FileLifecycleState.TERMINAL
  client._exit = False
  client._stopped = False
  client._file_ingest_task = None
  client._file_feed_task = None
  client._file_observability_task = None
  client._audio_task = None
  client._processing_task = None
  client._file_queue = None
  client._file_ingested_seconds = 0.0
  client.audio_source = MagicMock()

  async def _ingest() -> None:
    events.append("ingest_done")
    ingest_completed.set()

  async def _feed() -> None:
    await ingest_completed.wait()
    events.append("feed_done")

  async def _observe() -> None:
    while client._file_state in (FileLifecycleState.INGESTING, FileLifecycleState.DRAINING):
      await asyncio.sleep(0.01)

  client._ingest_file_upload = _ingest
  client._feed_file_queue_to_processor = _feed
  client._file_observability_loop = _observe
  client.processor = _LifecycleProcessorDouble(events)

  async def _disconnect() -> None:
    events.append("disconnect")

  client.transcription_sink = MagicMock()
  client.transcription_sink.disconnect = AsyncMock(side_effect=_disconnect)
  client.transcription_sink.send_error = AsyncMock()

  async def _stop() -> None:
    events.append("stop")

  client.stop = AsyncMock(side_effect=_stop)

  await client._run_file_mode_lifecycle()

  assert events[0] == "processing_completed"
  assert events[1:] == ["ingest_done", "feed_done", "source_exhausted", "disconnect", "stop"]
  assert client._file_state == FileLifecycleState.TERMINAL
