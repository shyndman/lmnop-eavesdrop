"""
Streaming transcription processor with integrated Faster Whisper transcriber.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from structlog.stdlib import BoundLogger

from eavesdrop.common import Pretty, Range, Seconds, get_logger
from eavesdrop.server.config import TranscriptionConfig
from eavesdrop.server.constants import CACHE_PATH, SINGLE_MODEL
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.debug_capture import AudioDebugCapture
from eavesdrop.server.streaming.flush_state import LiveSessionFlushState, PendingFlush
from eavesdrop.server.streaming.interfaces import TranscriptionResult, TranscriptionSink
from eavesdrop.server.transcription.models import SpeechChunk, TranscriptionInfo, VadParameters
from eavesdrop.server.transcription.session import TranscriptionSession
from eavesdrop.server.transcription.utils import summarize_array
from eavesdrop.server.transcription.vendor_types import (
  ContainsModel,
  TransformersConverterFactory,
  load_contains_model,
  load_torch,
  load_transformers_converter,
)
from eavesdrop.wire import Segment

if TYPE_CHECKING:
  from eavesdrop.server.transcription.pipeline import WhisperModel
  from eavesdrop.server.transcription.session import TranscriptionSessionProtocol

tracing_logger = get_logger("tracing")

Float32Audio = NDArray[np.float32]
SnapshotDownload = Callable[..., str]


def _load_snapshot_download() -> SnapshotDownload:
  return cast(SnapshotDownload, getattr(import_module("huggingface_hub"), "snapshot_download"))


class ClientSegmentDict(TypedDict):
  id: str
  start: str
  end: str
  text: str
  completed: bool


class FormattedSegmentDict(TypedDict):
  start: str
  end: str
  text: str
  completed: bool


class TracingClientSegmentDict(TypedDict):
  id: int
  start: str
  end: str
  text: str
  completed: bool
  synthetic: bool


@dataclass
class AudioChunk:
  """Audio data with associated metadata for processing."""

  data: Float32Audio
  duration: float
  start_time: float


@dataclass
class ChunkTranscriptionResult:
  """Result of transcribing an audio chunk."""

  status: "TranscriptionPassStatus"
  chunk_start_sample: int
  chunk_sample_count: int
  segments: list[Segment] | None
  info: TranscriptionInfo | None
  processing_time: float
  audio_duration: float
  speech_chunks: list[SpeechChunk] | None = None
  utterance_generation: int = 0
  recording_id: str | None = None


class TranscriptionPassStatus(str, Enum):
  """Outcome for one transcription pass over a buffered audio chunk."""

  TRANSCRIBED = "transcribed"
  INTERRUPTED_BEFORE_COMMIT = "interrupted_before_commit"


class StreamingTranscriptionProcessor:
  """
  Orchestrates streaming transcription processing with integrated Faster Whisper.

  This class combines audio buffering, transcription processing, and result
  delivery by combining an AudioStreamBuffer with integrated Faster Whisper
  model management.
  """

  # Class variable for single model mode
  SINGLE_MODEL: WhisperModel | None = None

  def __init__(
    self,
    buffer: AudioStreamBuffer,
    sink: TranscriptionSink,
    config: TranscriptionConfig,
    session: TranscriptionSession,
    stream_name: str,
    flush_state: LiveSessionFlushState | None = None,
  ) -> None:
    self.buffer: AudioStreamBuffer = buffer
    self.sink: TranscriptionSink = sink
    self.config: TranscriptionConfig = config
    self.stream_name: str = stream_name
    self.session: TranscriptionSession = session
    self.flush_state: LiveSessionFlushState | None = flush_state
    self.logger: BoundLogger = get_logger("proc", stream=self.stream_name)

    # Transcription state
    self.exit: bool = False
    self._source_exhausted: bool = False
    self._minimum_chunk_wait_logged: bool = False
    self.language: str | None = config.language
    self.text: list[str] = []
    self.recording_id: str | None = None

    # Segment processing state - repetition logic removed

    # Model attributes
    self.transcriber: WhisperModel | None = None
    self.compute_type: str = ""
    self.model_sizes: list[str] = [
      "tiny",
      "tiny.en",
      "base",
      "base.en",
      "small",
      "small.en",
      "medium",
      "medium.en",
      "large-v2",
      "large-v3",
      "distil-small.en",
      "distil-medium.en",
      "distil-large-v2",
      "distil-large-v3",
      "large-v3-turbo",
      "turbo",
    ]
    self.vad_parameters: VadParameters = config.vad_parameters

    # Initialize debug capture if configured
    self._debug_capture: AudioDebugCapture | None = None
    if config.debug_audio and config.debug_audio.post_buffer:
      self._debug_capture = AudioDebugCapture(
        output_path=Path(str(config.debug_audio.post_buffer)),
        stream_name=stream_name,
        sample_rate=buffer.config.sample_rate,
      )

  async def initialize(self) -> None:
    """Initialize the transcriber model."""
    torch = load_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    self.logger.debug("Selected device", device=device)

    if device == "cuda":
      major_version, _ = torch.cuda.get_device_capability(device)
      # self.compute_type = "float16"
      self.compute_type = "int8"

      self.logger.debug(
        "CUDA device capability", major=major_version, compute_type=self.compute_type
      )
    else:
      self.compute_type = "int8"
      self.logger.debug("Using CPU with compute_type", compute_type=self.compute_type)

    self.logger.info("Using Device with precision", device=device, precision=self.compute_type)

    try:
      await asyncio.to_thread(self._create_model, device)
    except Exception:
      self.logger.exception("Failed to load model")
      # TODO: Add try
      await self.sink.send_error(f"Failed to load model: {self.config.model}")
      raise

    # Send server ready message
    await self.sink.send_server_ready("faster_whisper")

  def _create_model(self, device: str) -> None:
    """Create and initialize the Whisper model."""
    self.logger.debug("Creating model", model_reference=self.config.model)

    if SINGLE_MODEL:
      self.logger.debug("Using single model mode", stream=self.stream_name)
      if StreamingTranscriptionProcessor.SINGLE_MODEL is None:
        self.logger.debug("Creating new single model instance")
        self._create_model_instance(device, self.config.model_path)
        StreamingTranscriptionProcessor.SINGLE_MODEL = self.transcriber
      else:
        self.logger.debug("Reusing existing single model instance")
        self.transcriber = StreamingTranscriptionProcessor.SINGLE_MODEL
    else:
      self.logger.debug("Creating dedicated model", stream=self.stream_name)
      self._create_model_instance(device, self.config.model_path)

  def _create_model_instance(self, device: str, model_ref: str) -> None:
    """Create a single model instance."""
    from eavesdrop.server.transcription.pipeline import WhisperModel

    model_to_load = self._resolve_model_path(model_ref)

    self.logger.info("Loading model", model=model_to_load)
    self.transcriber = WhisperModel(
      model_to_load,
      device=device,
      device_index=self.config.device_index,
      compute_type=self.compute_type,
      num_workers=self.config.num_workers,
      download_root=CACHE_PATH,
      local_files_only=False,
    )
    self.logger.debug("Model loaded successfully")

  def _resolve_model_path(self, model_ref: str) -> str:
    """Resolve model reference to a loadable path."""
    if model_ref in self.model_sizes:
      self.logger.debug("Model found in standard model sizes", model=model_ref)
      return model_ref

    if self._is_local_ct2_model(model_ref):
      self.logger.debug("Found local CTranslate2 model", path=model_ref)
      return model_ref

    return self._download_and_convert_model(model_ref)

  def _is_local_ct2_model(self, model_ref: str) -> bool:
    """Check if model reference points to a local CTranslate2 model."""
    contains_model: ContainsModel = load_contains_model()
    return os.path.isdir(model_ref) and contains_model(model_ref)

  def _download_and_convert_model(self, model_ref: str) -> str:
    """Download model from HuggingFace and convert to CTranslate2 if needed."""
    contains_model: ContainsModel = load_contains_model()
    snapshot_download = _load_snapshot_download()
    self.logger.debug("Downloading model from HuggingFace", model=model_ref)
    local_snapshot = snapshot_download(repo_id=model_ref, repo_type="model")
    self.logger.debug("Downloaded model", path=local_snapshot)

    if contains_model(local_snapshot):
      self.logger.debug("Downloaded model is already in CTranslate2 format")
      return local_snapshot

    return self._convert_to_ct2(model_ref, local_snapshot)

  def _convert_to_ct2(self, model_ref: str, local_snapshot: str) -> str:
    """Convert downloaded model to CTranslate2 format."""
    contains_model: ContainsModel = load_contains_model()
    transformers_converter: TransformersConverterFactory = load_transformers_converter()
    cache_root = os.path.expanduser(os.path.join(CACHE_PATH, "whisper-ct2-models/"))
    os.makedirs(cache_root, exist_ok=True)
    safe_name = model_ref.replace("/", "--")
    ct2_dir = os.path.join(cache_root, safe_name)
    self.logger.debug("CTranslate2 cache directory", path=ct2_dir)

    if contains_model(ct2_dir):
      self.logger.debug("CTranslate2 model already exists in cache")
      return ct2_dir

    self.logger.info("Converting to CTranslate2", model=model_ref, output_dir=ct2_dir)
    ct2_converter = transformers_converter(
      local_snapshot,
      copy_files=["tokenizer.json", "preprocessor_config.json"],
    )
    _ = ct2_converter.convert(
      output_dir=ct2_dir,
      quantization=self.compute_type,
      force=False,
    )
    self.logger.debug("Model conversion completed")
    return ct2_dir

  async def start_processing(self) -> None:
    """Start the transcription processing loop."""
    self.logger.info("Starting transcription processing", stream=self.stream_name)
    await self._transcription_loop()

  async def stop_processing(self) -> None:
    """Stop the transcription processing loop."""
    self.logger.info("Stopping transcription processing", stream=self.stream_name)
    self.exit = True
    await self.sink.disconnect()

  def mark_source_exhausted(self) -> None:
    """Mark that no additional audio will arrive for this session."""
    self._source_exhausted = True

  def add_audio_frames(self, frames: Float32Audio) -> None:
    """Add audio frames to the buffer for processing."""
    self.buffer.add_frames(frames)

  def reset_live_recording(
    self,
    *,
    recording_id: str | None,
    generation: int,
    preserve_language: bool = True,
  ) -> None:
    """Reset processor-local recording state after a new epoch or cancellation."""
    self.recording_id = recording_id
    self.text = []
    if not preserve_language:
      self.language = self.config.language

    self.logger.info(
      "processor live recording reset",
      recording_id=recording_id,
      generation=generation,
      preserve_language=preserve_language,
    )

  async def _transcription_loop(self) -> None:
    """Main processing loop: get audio → transcribe → send results → wait."""
    while not self.exit:
      loop_start = time.monotonic()
      try:
        chunk_fetch_start = time.monotonic()
        audio_chunk = await self._get_next_audio_chunk()
        chunk_fetch_duration = time.monotonic() - chunk_fetch_start
        if audio_chunk is None:
          if self._source_exhausted and self.buffer.available_duration <= 0:
            break
          continue

        result_processing_start = time.monotonic()
        transcription_result = await self._transcribe_chunk(audio_chunk)
        await self._process_transcription_result(transcription_result)
        result_processing_duration = time.monotonic() - result_processing_start

        if transcription_result.status is TranscriptionPassStatus.INTERRUPTED_BEFORE_COMMIT:
          continue

        interval_wait_start = time.monotonic()
        await self._wait_for_next_interval(transcription_result.processing_time)
        interval_wait_duration = time.monotonic() - interval_wait_start

        loop_duration = time.monotonic() - loop_start
        self.logger.info(
          "Transcription loop timing",
          chunk_duration_s=f"{audio_chunk.duration:.3f}",
          chunk_fetch_s=f"{chunk_fetch_duration:.3f}",
          inference_and_result_s=f"{result_processing_duration:.3f}",
          inference_s=f"{transcription_result.processing_time:.3f}",
          interval_wait_s=f"{interval_wait_duration:.3f}",
          total_loop_s=f"{loop_duration:.3f}",
        )
      except Exception:
        await self._handle_transcription_error()

    self.logger.info("Exiting speech to text thread")

  async def _get_next_audio_chunk(self) -> AudioChunk | None:
    """Get the next audio chunk, handling buffer management and minimum duration."""
    self.buffer.clip_if_stalled()
    input_bytes, duration, start_time = self.buffer.get_chunk_for_processing()

    # Debug logging to track buffer catchup patterns
    available_duration = self.buffer.available_duration
    total_duration = self.buffer.total_duration
    processed_duration = self.buffer.processed_duration

    self.logger.debug(
      "Buffer status before transcription",
      available_for_processing=f"{available_duration:.2f}s",
      total_buffered=f"{total_duration:.2f}s",
      already_processed=f"{processed_duration:.2f}s",
      is_caught_up=available_duration < 0.5,
    )

    if duration < self.buffer.config.min_chunk_duration:
      if self._source_exhausted and duration > 0:
        return AudioChunk(data=input_bytes, duration=duration, start_time=start_time)
      remaining_wait = self.buffer.config.min_chunk_duration - duration
      if not self._minimum_chunk_wait_logged:
        self.logger.info(
          "Transcription loop minimum chunk wait",
          buffered_duration_s=f"{duration:.3f}",
          required_duration_s=f"{self.buffer.config.min_chunk_duration:.3f}",
          wait_s=f"{remaining_wait:.3f}",
        )
        self._minimum_chunk_wait_logged = True
      observed_generation = self._current_utterance_generation()
      wait_completed = await self._wait_for_live_control_wakeup(
        remaining_wait,
        observed_generation=observed_generation,
      )
      pending_flush = self._pending_flush()
      if not wait_completed and pending_flush is not None and duration > 0:
        self.logger.info(
          "Pending flush interrupts minimum chunk wait",
          buffered_duration_s=f"{duration:.3f}s",
          boundary_sample=pending_flush.boundary_sample,
        )
        return AudioChunk(data=input_bytes, duration=duration, start_time=start_time)
      if not wait_completed and self._current_utterance_generation() != observed_generation:
        self.logger.info(
          "Cancelled utterance interrupts minimum chunk wait",
          buffered_duration_s=f"{duration:.3f}s",
          utterance_generation=observed_generation,
        )
      return None

    # TRACING: Audio chunk range being processed
    tracing_logger.info("\n\n")
    tracing_logger.info(
      "Processing audio chunk",
      chunk_start=f"{start_time:.2f}s",
      chunk_end=f"{start_time + duration:.2f}s",
      chunk_duration=f"{duration:.2f}s",
      buffer_available=f"{available_duration:.2f}s",
      buffer_total=f"{total_duration:.2f}s",
      buffer_processed=f"{processed_duration:.2f}s",
    )

    return AudioChunk(data=input_bytes, duration=duration, start_time=start_time)

  async def _transcribe_chunk(self, chunk: AudioChunk) -> ChunkTranscriptionResult:
    """Transcribe an audio chunk and return results with timing information."""
    # Debug audio capture (post-buffer)
    if self._debug_capture:
      self._debug_capture.capture(chunk)

    transcription_start = time.monotonic()
    pass_result = await asyncio.to_thread(self._run_transcription_pass, chunk)
    processing_time = time.monotonic() - transcription_start

    if pass_result.status is TranscriptionPassStatus.INTERRUPTED_BEFORE_COMMIT:
      return pass_result

    self.logger.debug(
      "Transcription performance",
      audio_duration=f"{chunk.duration:.2f}s",
      transcription_time=f"{processing_time:.2f}s",
      speed_vs_realtime=f"{chunk.duration / processing_time:.1f}x faster",
    )

    pass_result.processing_time = processing_time
    return pass_result

  async def _process_transcription_result(self, result: ChunkTranscriptionResult) -> None:
    """Process transcription results, update language, and handle segments."""
    if result.status is TranscriptionPassStatus.INTERRUPTED_BEFORE_COMMIT:
      self.logger.info("Skipping interrupted transcription pass before commit")
      return

    active_generation = self._current_utterance_generation()
    if result.utterance_generation != active_generation:
      self.logger.info(
        "Dropping stale transcription result after utterance cancel",
        result_generation=result.utterance_generation,
        active_generation=active_generation,
      )
      return

    if result.recording_id != self.recording_id:
      self.logger.info(
        "Dropping stale transcription result after recording boundary",
        result_recording_id=result.recording_id,
        active_recording_id=self.recording_id,
      )
      return

    pending_flush = self._pending_flush()
    flush_boundary_reached = pending_flush is not None and self._covers_flush_boundary(
      result, pending_flush
    )

    if self.language is None and result.info is not None:
      await self._set_language(result.info)

    await self._handle_transcription_output(
      result.segments or [],
      result.audio_duration,
      result.speech_chunks,
      emit_result=pending_flush is None or flush_boundary_reached,
      flush_complete=flush_boundary_reached,
      force_complete=bool(
        flush_boundary_reached and pending_flush and pending_flush.force_complete
      ),
    )

    if flush_boundary_reached:
      self._discard_flushed_audio(pending_flush)
      if self.flush_state is not None:
        _ = self.flush_state.complete()

  def _discard_flushed_audio(self, pending_flush: PendingFlush | None) -> None:
    """Drop buffered audio through the accepted flush boundary after emitting the response."""
    if pending_flush is None:
      raise RuntimeError("Cannot discard flushed audio without a pending flush boundary")

    discarded_duration = self.buffer.discard_through_sample(pending_flush.boundary_sample)
    self.logger.info(
      "Discarded flushed audio",
      discarded_duration_s=f"{discarded_duration:.3f}",
      boundary_sample=pending_flush.boundary_sample,
    )

  async def _wait_for_next_interval(self, processing_time: float) -> None:
    """Wait to maintain consistent transcription intervals."""
    remaining_wait = self.buffer.config.transcription_interval - processing_time
    if remaining_wait > 0:
      self.logger.debug(f"Waiting {remaining_wait:.2f}s to maintain transcription interval")
      observed_generation = self._current_utterance_generation()
      wait_completed = await self._wait_for_live_control_wakeup(
        remaining_wait,
        observed_generation=observed_generation,
      )
      if not wait_completed:
        pending_flush = self._pending_flush()
        if pending_flush is not None:
          self.logger.info("Pending flush interrupts interval wait")
        elif self._current_utterance_generation() != observed_generation:
          self.logger.info(
            "Cancelled utterance interrupts interval wait",
            utterance_generation=observed_generation,
          )
    else:
      self.logger.warning("Transcription took longer than interval, proceeding immediately")

  async def _handle_transcription_error(self) -> None:
    """Handle transcription errors with logging and recovery."""
    self.logger.exception("Failed to transcribe audio chunk")
    await asyncio.sleep(self.buffer.config.transcription_interval)

  def _pending_flush(self) -> PendingFlush | None:
    """Return the currently pending flush request, if any."""
    if self.flush_state is None:
      return None
    return self.flush_state.pending()

  def _chunk_start_sample(self, chunk: AudioChunk) -> int:
    """Convert an audio chunk's absolute start time to an absolute sample index."""
    return int(round(chunk.start_time * self.buffer.config.sample_rate))

  def _consume_precommit_interrupt(self) -> bool:
    """Consume one live-session control interrupt before expensive work begins."""
    if self.flush_state is None or not self.flush_state.interrupt.is_set():
      return False
    _ = self.flush_state.clear_interrupt()
    self.logger.info("Live-session control interrupts worker pass before commit")
    return True

  def _current_utterance_generation(self) -> int:
    """Return the active utterance generation for the live session."""
    if self.flush_state is None:
      return 0
    return self.flush_state.current_generation()

  def _run_transcription_pass(self, chunk: AudioChunk) -> ChunkTranscriptionResult:
    """Run one worker-thread transcription pass or report interruption before commit."""
    utterance_generation = self._current_utterance_generation()
    if self._consume_precommit_interrupt():
      return ChunkTranscriptionResult(
        status=TranscriptionPassStatus.INTERRUPTED_BEFORE_COMMIT,
        chunk_start_sample=self._chunk_start_sample(chunk),
        chunk_sample_count=int(chunk.data.shape[0]),
        segments=None,
        info=None,
        processing_time=0.0,
        audio_duration=chunk.duration,
        speech_chunks=None,
        utterance_generation=utterance_generation,
        recording_id=self.recording_id,
      )

    result, info, speech_chunks = self._transcribe_audio(chunk)
    return ChunkTranscriptionResult(
      status=TranscriptionPassStatus.TRANSCRIBED,
      chunk_start_sample=self._chunk_start_sample(chunk),
      chunk_sample_count=int(chunk.data.shape[0]),
      segments=result,
      info=info,
      processing_time=0.0,
      audio_duration=chunk.duration,
      speech_chunks=speech_chunks,
      utterance_generation=utterance_generation,
      recording_id=self.recording_id,
    )

  def _covers_flush_boundary(
    self, result: ChunkTranscriptionResult, pending_flush: PendingFlush
  ) -> bool:
    """Return whether the transcribed chunk covers the accepted flush boundary."""
    chunk_end_sample = result.chunk_start_sample + result.chunk_sample_count
    boundary_reached = chunk_end_sample >= pending_flush.boundary_sample
    self.logger.debug(
      "Evaluated flush boundary coverage",
      chunk_start_sample=result.chunk_start_sample,
      chunk_end_sample=chunk_end_sample,
      boundary_sample=pending_flush.boundary_sample,
      boundary_reached=boundary_reached,
    )
    return boundary_reached

  async def _wait_for_live_control_wakeup(
    self,
    delay: float,
    *,
    observed_generation: int,
  ) -> bool:
    """Sleep unless live-session control needs the loop to resume immediately."""
    if delay <= 0:
      return True
    if self.flush_state is None:
      await asyncio.sleep(delay)
      return True
    if not self.flush_state.begin_wait(observed_generation=observed_generation):
      return False

    sleep_task = asyncio.create_task(asyncio.sleep(delay))
    wake_task = asyncio.create_task(self.flush_state.wakeup.wait())
    try:
      done, pending = await asyncio.wait(
        {sleep_task, wake_task},
        return_when=asyncio.FIRST_COMPLETED,
      )
      for task in pending:
        _ = task.cancel()
      for task in pending:
        try:
          await task
        except asyncio.CancelledError:
          pass
      return sleep_task in done
    finally:
      if not sleep_task.done():
        _ = sleep_task.cancel()
      if not wake_task.done():
        _ = wake_task.cancel()

  def _transcribe_audio(
    self, chunk: AudioChunk
  ) -> tuple[list[Segment] | None, TranscriptionInfo, list[SpeechChunk] | None]:
    """Transcribe audio sample using the Faster Whisper model."""
    if not self.transcriber:
      raise RuntimeError("Transcriber not initialized")

    self.logger.debug(
      "Entering transcriber.transcribe",
      start_time=chunk.start_time,
      duration=chunk.duration,
      processed_up_to_time=self.buffer.processed_up_to_time,
      language=self.language,
      use_vad=self.config.use_vad,
      beam_size=self.config.beam_size,
      word_timestamps=self.config.word_timestamps,
      has_hotwords=bool(self.config.hotwords),
      **summarize_array("audio", chunk.data),
    )

    result, info = self.transcriber.transcribe(
      chunk.data,
      initial_prompt=self.config.initial_prompt,
      language=self.language,
      vad_filter=self.config.use_vad,
      vad_parameters=self.vad_parameters,
      absolute_stream_start=chunk.start_time,
      hotwords=" ".join(self.config.hotwords) if self.config.hotwords else None,
      session=cast("TranscriptionSessionProtocol", cast(object, self.session)),
      start_offset=self.buffer.processed_up_to_time,
      beam_size=self.config.beam_size,
      word_timestamps=self.config.word_timestamps,
    )
    self.logger.debug(
      "Returned from transcriber.transcribe",
      info_present=True,
      speech_chunk_count=len(info.speech_chunks) if info.speech_chunks else 0,
    )
    result_list = list(result) if result else None
    self.logger.debug(
      "Materialized transcriber result",
      segment_count=len(result_list) if result_list else 0,
    )

    # Store VAD speech chunks for silence analysis
    speech_chunks = info.speech_chunks

    # TRACING: Audio characteristics analysis
    sample_rate = 16000  # Standard Whisper sample rate
    rms = np.sqrt(np.mean(chunk.data**2)) if len(chunk.data) > 0 else 0.0
    peak = np.max(np.abs(chunk.data)) if len(chunk.data) > 0 else 0.0
    zero_crossings = np.sum(np.diff(np.signbit(chunk.data))) if len(chunk.data) > 1 else 0
    zcr = zero_crossings / len(chunk.data) * sample_rate if len(chunk.data) > 0 else 0

    # TRACING: VAD speech detection results
    if speech_chunks:
      speech_ranges = [
        f"{chunk['start'] / sample_rate:.2f}-{chunk['end'] / sample_rate:.2f}s"
        for chunk in speech_chunks
      ]
      total_speech_duration = sum(
        (chunk["end"] - chunk["start"]) / sample_rate for chunk in speech_chunks
      )

      # Build timeline showing speech vs silence (S=speech, ~=silence)
      timeline = ["~"] * int(chunk.duration * 10)  # 100ms resolution
      for speech_chunk in speech_chunks:
        start_idx = int((speech_chunk["start"] / sample_rate) * 10)
        end_idx = int((speech_chunk["end"] / sample_rate) * 10)
        for i in range(start_idx, min(end_idx, len(timeline))):
          timeline[i] = "S"
      timeline_str = "".join(timeline)

      tracing_logger.info(
        "VAD analysis",
        speech_ranges=speech_ranges,
        speech_duration=f"{total_speech_duration:.2f}s",
        silence_duration=f"{chunk.duration - total_speech_duration:.2f}s",
        audio_duration=f"{chunk.duration:.2f}s",
        speech_ratio=f"{total_speech_duration / chunk.duration:.1%}",
        timeline=timeline_str,
        audio_rms=f"{rms:.6f}",
        audio_peak=f"{peak:.6f}",
        audio_zcr=f"{zcr:.1f}",
      )
    else:
      tracing_logger.info(
        "VAD analysis",
        speech_ranges=[],
        speech_duration="0.00s",
        silence_duration=f"{chunk.duration:.2f}s",
        audio_duration=f"{chunk.duration:.2f}s",
        speech_ratio="0.0%",
        timeline="~" * int(chunk.duration * 10),
        audio_rms=f"{rms:.6f}",
        audio_peak=f"{peak:.6f}",
        audio_zcr=f"{zcr:.1f}",
      )

    return result_list, info, speech_chunks

  async def _set_language(self, info: TranscriptionInfo) -> None:
    """Update language based on detection info."""
    self.logger.debug(
      "Language detection info",
      language=info.language,
      probability=info.language_probability,
    )

    if info.language_probability > 0.5:
      self.language = info.language
      self.logger.info(
        "Detected language",
        language=self.language,
        probability=info.language_probability,
      )
      await self.sink.send_language_detection(info.language, info.language_probability)

  async def _handle_transcription_output(
    self,
    result: list[Segment],
    duration: float,
    speech_chunks: list[SpeechChunk] | None = None,
    *,
    emit_result: bool = True,
    flush_complete: bool = False,
    force_complete: bool = False,
  ) -> None:
    """Handle transcription output and send to client."""
    result_count = len(result) if result else 0
    self.logger.debug(
      "Handling transcription output",
      duration=f"{duration:.2f}s",
      segments=result_count,
    )

    if not result and not flush_complete:
      self.logger.debug("No segments to send to client")
      self.buffer.advance_processed_boundary(duration)
      return

    if result:
      # TRACING: Raw Whisper segments generated
      raw_segments = [
        {
          "id": seg.id,
          "timespan": Range(Seconds(seg.start), Seconds(seg.end)),
          "text": seg.text[:50] + ("..." if len(seg.text) > 50 else ""),
          "logprob": f"{seg.avg_logprob:.3f}",
          "completed": seg.completed,
        }
        for seg in result
      ]
      tracing_logger.info(
        "Whisper generated segments",
        segment_count=len(result),
        segments=Pretty(raw_segments),
        audio_duration=Seconds(duration),
      )

    self.logger.debug("Processing transcription segments")
    # Update internal transcript (for translation queue, etc.)
    self._update_segments(result, duration)

    # Apply explicit completion rule: all but last segment are completed
    # Plus enhanced rule: last segment can also complete based on silence
    current_incomplete_segment: Segment | None = None
    silence_threshold = self.config.silence_completion_threshold

    for i, segment in enumerate(result):
      is_last_segment = i == len(result) - 1

      # Explicit rule: all segments except last are marked complete
      should_complete = not is_last_segment

      # Enhanced rule: last segment can also complete if silence analysis says so
      if is_last_segment:
        should_complete = self._should_complete_segment_by_silence(
          segment, speech_chunks, silence_threshold, duration
        )

      if should_complete and not segment.completed:
        # Mark as completed and assign chain-based ID
        preceding_segment = self.session.get_last_completed_segment()
        segment.mark_completed(preceding_segment)

        # Add to session's completed segments chain
        self.session.add_completed_segment(segment)

        # TRACING: Completion decision
        completion_reason = "positional" if not is_last_segment else "silence_analysis"
        tracing_logger.info(
          "Segment marked complete",
          segment_id=segment.id,
          reason=completion_reason,
          text=segment.text[:50] + ("..." if len(segment.text) > 50 else ""),
          time_range=f"{segment.start:.2f}-{segment.end:.2f}s",
        )
      else:
        # This is the incomplete (current) segment
        current_incomplete_segment = segment

        # TRACING: Incomplete segment
        if is_last_segment:
          tracing_logger.info(
            "Last segment remains incomplete",
            segment_id=segment.id,
            reason="insufficient_silence",
            text=segment.text[:50] + ("..." if len(segment.text) > 50 else ""),
            time_range=f"{segment.start:.2f}-{segment.end:.2f}s",
          )

    if force_complete and current_incomplete_segment is not None:
      preceding_segment = self.session.get_last_completed_segment()
      current_incomplete_segment.mark_completed(preceding_segment)
      self.session.add_completed_segment(current_incomplete_segment)
      current_incomplete_segment = None

      tracing_logger.info(
        "Forced completion of tentative tail for flush",
        flush_complete=flush_complete,
      )

    if not result:
      self.buffer.advance_processed_boundary(duration)

    # Prepare segments for client: last N completed + current incomplete
    segments_for_client: list[Segment] = []

    # Add the last N completed segments from session
    recent_completed_segments: list[Segment] = list(
      self.session.most_recent_completed_segments(self.config.send_last_n_segments)
    )
    segments_for_client.extend(recent_completed_segments)

    # Ensure client invariant: always have an incomplete segment at the tail
    if current_incomplete_segment:
      segments_for_client.append(current_incomplete_segment)
    else:
      # Create synthetic incomplete segment to maintain client state machine invariant
      synthetic_segment = self._create_synthetic_incomplete_segment(
        speech_chunks, silence_threshold, duration
      )
      segments_for_client.append(synthetic_segment)

    # Advance buffer pointer based on completed segments
    self._advance_buffer_by_completed_segments(
      speech_chunks, silence_threshold, duration, current_incomplete_segment
    )

    if not emit_result:
      self.logger.debug("Suppressing ordinary emission while flush remains pending")
      return

    # TRACING: Final client output
    client_segments: list[TracingClientSegmentDict] = [
      {
        "id": seg.id,
        "start": f"{seg.start:.2f}s",
        "end": f"{seg.end:.2f}s",
        "text": seg.text[:30] + ("..." if len(seg.text) > 30 else ""),
        "completed": seg.completed,
        "synthetic": seg.text == "" and seg.start == seg.end,
      }
      for seg in segments_for_client
    ]
    completed_count = sum(1 for seg in segments_for_client if seg.completed)
    incomplete_count = len(segments_for_client) - completed_count

    tracing_logger.info(
      "Sending segments to client",
      total_segments=len(segments_for_client),
      completed_segments=completed_count,
      incomplete_segments=incomplete_count,
      segments=client_segments,
    )

    # Send rich Segment objects to client
    transcription_result = TranscriptionResult(
      segments=segments_for_client,
      language=self.language,
      recording_id=self.recording_id,
      flush_complete=flush_complete or None,
    )
    self.logger.debug("Sending segments to client", segments=segments_for_client)
    await self.sink.send_result(transcription_result)

  def _update_segments(self, segments: list[Segment], _duration: float) -> None:
    """Process segments and update transcript."""
    # Update internal transcript for translation queue
    for segment in segments:
      if segment.completed and segment.text.strip():
        self.text.append(segment.text)

  def _prepare_segments(
    self, last_segment: ClientSegmentDict | None = None
  ) -> list[ClientSegmentDict]:
    """Prepare segments for client using session completed segments."""
    segments: list[ClientSegmentDict] = []

    if self.session and self.session.completed_segments:
      # Convert Segment objects to dict format for client
      completed_dicts: list[ClientSegmentDict] = []
      for seg in self.session.completed_segments:
        completed_dicts.append(
          {
            "id": str(seg.id),
            "start": "{:.3f}".format(seg.absolute_start_time),
            "end": "{:.3f}".format(seg.absolute_end_time),
            "text": seg.text,
            "completed": seg.completed,
          }
        )

      # Apply send_last_n_segments limit
      if len(completed_dicts) >= self.config.send_last_n_segments:
        segments = completed_dicts[-self.config.send_last_n_segments :].copy()
      else:
        segments = completed_dicts.copy()

    if last_segment is not None:
      segments = segments + [last_segment]

    return segments

  def _should_complete_segment_by_silence(
    self,
    segment: Segment,
    speech_chunks: list[SpeechChunk] | None,
    silence_threshold: float,
    audio_duration: float,
  ) -> bool:
    """Determine if segment should be completed based on silence after speech."""
    if not speech_chunks:
      # No speech in current chunk - check if we have a segment from previous speech
      # If the segment exists and we have sufficient silence duration, complete it
      if segment.text.strip() and audio_duration >= silence_threshold:
        tracing_logger.info(
          "Completing segment due to silence chunk",
          segment_id=segment.id,
          silence_duration=f"{audio_duration:.2f}s",
          threshold=f"{silence_threshold:.2f}s",
        )
        return True
      return False

    # VAD speaks chunk-relative samples. Segment timestamps are recording-relative;
    # convert before comparing because recording time is the canonical timeline.
    chunk_start_recording_time = self.buffer.processed_up_to_time
    last_speech_chunk_time = self._last_speech_chunk_time(speech_chunks)
    last_speech_recording_time = chunk_start_recording_time + last_speech_chunk_time

    # Time since last speech in this audio chunk.
    chunk_end_recording_time = chunk_start_recording_time + audio_duration
    silence_duration = chunk_end_recording_time - last_speech_recording_time

    # If we have enough silence after the segment ends, mark it complete
    if (
      segment.absolute_end_time <= last_speech_recording_time + silence_threshold
      and silence_duration >= silence_threshold
    ):
      tracing_logger.info(
        "Completing segment due to silence after speech",
        segment_id=segment.id,
        silence_duration=f"{silence_duration:.2f}s",
        last_speech_recording_time=f"{last_speech_recording_time:.2f}s",
        threshold=f"{silence_threshold:.2f}s",
      )
      return True

    return False

  def _last_speech_chunk_time(self, speech_chunks: list[SpeechChunk]) -> float:
    """Return the last VAD speech end as chunk-relative seconds."""
    sample_rate = self.buffer.config.sample_rate
    return max(chunk["end"] for chunk in speech_chunks) / sample_rate

  def _advance_buffer_by_completed_segments(
    self,
    speech_chunks: list[SpeechChunk] | None,
    silence_threshold: float,
    audio_duration: float,
    current_incomplete_segment: Segment | None,
  ) -> None:
    """Advance buffer pointer based on speech boundaries and completed segments."""
    # Get the last completed segment to determine how far we can advance
    last_completed = self.session.get_last_completed_segment()

    if not last_completed:
      return

    # Calculate how far we can safely advance
    completed_end_time = last_completed.absolute_end_time
    current_processed_time = self.buffer.processed_up_to_time

    # Only advance if we're moving forward
    if completed_end_time > current_processed_time:
      advance_amount = completed_end_time - current_processed_time
      self.buffer.advance_processed_boundary(advance_amount)

      # TRACING: Buffer pointer advancement
      tracing_logger.info(
        "Buffer advanced by completed segment",
        advance_amount=f"{advance_amount:.2f}s",
        old_processed_time=f"{current_processed_time:.2f}s",
        new_processed_time=f"{completed_end_time:.2f}s",
        last_completed_id=last_completed.id,
      )

    # Additionally, if we have pure silence after completion, advance through it
    if speech_chunks and current_incomplete_segment is None:
      last_speech_time = self._last_speech_chunk_time(speech_chunks)
      chunk_end_time = audio_duration
      silence_after_speech = chunk_end_time - last_speech_time

      # This advance amount is chunk-relative by nature. Only apply it after the
      # recording-relative tail is closed, or the client can lose an open segment.
      if silence_after_speech > silence_threshold * 2:
        additional_advance = silence_after_speech - silence_threshold
        self.buffer.advance_processed_boundary(additional_advance)

        # TRACING: Additional silence advancement
        tracing_logger.info(
          "Buffer advanced through silence",
          silence_duration=f"{silence_after_speech:.2f}s",
          additional_advance=f"{additional_advance:.2f}s",
          silence_threshold=f"{silence_threshold:.2f}s",
        )

  def _create_synthetic_incomplete_segment(
    self,
    speech_chunks: list[SpeechChunk] | None,
    silence_threshold: float,
    duration: float,
  ) -> Segment:
    """Create a synthetic incomplete segment to maintain client state machine invariant."""
    from eavesdrop.wire.transcription import compute_segment_chain_id

    # Determine where the synthetic segment should start
    if speech_chunks:
      # Start after the last detected speech + threshold
      last_speech_time = self._last_speech_chunk_time(speech_chunks)
      synthetic_start = last_speech_time + silence_threshold
    else:
      # No speech detected, start at current buffer position
      synthetic_start = duration

    # Always create to maintain client invariant - place at end if necessary
    actual_start = min(synthetic_start, duration)
    recording_start = self.buffer.processed_up_to_time + actual_start

    return Segment(
      id=compute_segment_chain_id(0, ""),  # Baseline ID for incomplete
      seek=0,
      start=recording_start,
      end=recording_start,  # Zero-duration synthetic segment
      text="",  # Empty text is fine for client state machine
      tokens=[],
      avg_logprob=0.0,
      compression_ratio=0.0,
      words=None,
      temperature=0.0,
      completed=False,
      time_offset=0.0,
    )

  def _format_segment(
    self, start: float, end: float, text: str, completed: bool = False
  ) -> FormattedSegmentDict:
    """Format a transcription segment."""
    return {
      "start": "{:.3f}".format(start),
      "end": "{:.3f}".format(end),
      "text": text,
      "completed": completed,
    }
