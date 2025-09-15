"""
Streaming transcription processor with integrated Faster Whisper transcriber.
"""

import asyncio
import os
import queue
import threading
import time
from dataclasses import dataclass

import ctranslate2
import numpy as np
import torch
from faster_whisper.vad import VadOptions
from huggingface_hub import snapshot_download

from eavesdrop.server.config import TranscriptionConfig
from eavesdrop.server.constants import CACHE_PATH, SINGLE_MODEL
from eavesdrop.server.logs import get_logger
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.interfaces import TranscriptionResult, TranscriptionSink
from eavesdrop.server.transcription.models import TranscriptionInfo
from eavesdrop.server.transcription.pipeline import WhisperModel
from eavesdrop.server.transcription.session import TranscriptionSession
from eavesdrop.wire import Segment


@dataclass
class AudioChunk:
  """Audio data with associated metadata for processing."""

  data: np.ndarray
  duration: float
  start_time: float


@dataclass
class ChunkTranscriptionResult:
  """Result of transcribing an audio chunk."""

  segments: list[Segment] | None
  info: TranscriptionInfo | None
  processing_time: float
  audio_duration: float


class StreamingTranscriptionProcessor:
  """
  Orchestrates streaming transcription processing with integrated Faster Whisper.

  This class combines audio buffering, transcription processing, and result
  delivery by combining an AudioStreamBuffer with integrated Faster Whisper
  model management.
  """

  # Class variables for single model mode
  SINGLE_MODEL: WhisperModel | None = None
  SINGLE_MODEL_LOCK = threading.Lock()

  def __init__(
    self,
    buffer: AudioStreamBuffer,
    sink: TranscriptionSink,
    config: TranscriptionConfig,
    stream_name: str,
    translation_queue: queue.Queue[dict] | None = None,
    logger_name: str = "proc",
    session: TranscriptionSession | None = None,
  ) -> None:
    self.buffer = buffer
    self.sink = sink
    self.config = config
    self.stream_name = stream_name
    self.translation_queue = translation_queue
    self.session = session
    self.logger = get_logger("proc", stream=self.stream_name)

    # Transcription state
    self.exit: bool = False
    self.language: str | None = config.language
    self.transcript: list[dict] = []
    self.text: list[str] = []

    # Segment processing state
    self.current_out: str = ""
    self.prev_out: str = ""
    self.same_output_count: int = 0
    self.end_time_for_same_output: float | None = None

    # Model attributes
    self.transcriber: WhisperModel | None = None
    self.compute_type: str = ""
    self.model_sizes = [
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

    # Prepare VAD parameters
    if config.vad_parameters is None:
      self.vad_parameters = VadOptions()
    elif isinstance(config.vad_parameters, dict):
      self.vad_parameters = VadOptions(**config.vad_parameters)
    else:
      self.vad_parameters = config.vad_parameters

  async def initialize(self) -> None:
    """Initialize the transcriber model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    self.logger.debug("Selected device", device=device)

    if device == "cuda":
      major_version, _ = torch.cuda.get_device_capability(device)
      self.compute_type = "float16"
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
      with StreamingTranscriptionProcessor.SINGLE_MODEL_LOCK:
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
    return os.path.isdir(model_ref) and ctranslate2.contains_model(model_ref)

  def _download_and_convert_model(self, model_ref: str) -> str:
    """Download model from HuggingFace and convert to CTranslate2 if needed."""
    self.logger.debug("Downloading model from HuggingFace", model=model_ref)
    local_snapshot = snapshot_download(repo_id=model_ref, repo_type="model")
    self.logger.debug("Downloaded model", path=local_snapshot)

    if ctranslate2.contains_model(local_snapshot):
      self.logger.debug("Downloaded model is already in CTranslate2 format")
      return local_snapshot

    return self._convert_to_ct2(model_ref, local_snapshot)

  def _convert_to_ct2(self, model_ref: str, local_snapshot: str) -> str:
    """Convert downloaded model to CTranslate2 format."""
    cache_root = os.path.expanduser(os.path.join(CACHE_PATH, "whisper-ct2-models/"))
    os.makedirs(cache_root, exist_ok=True)
    safe_name = model_ref.replace("/", "--")
    ct2_dir = os.path.join(cache_root, safe_name)
    self.logger.debug("CTranslate2 cache directory", path=ct2_dir)

    if ctranslate2.contains_model(ct2_dir):
      self.logger.debug("CTranslate2 model already exists in cache")
      return ct2_dir

    self.logger.info("Converting to CTranslate2", model=model_ref, output_dir=ct2_dir)
    ct2_converter = ctranslate2.converters.TransformersConverter(
      local_snapshot,
      copy_files=["tokenizer.json", "preprocessor_config.json"],
    )
    ct2_converter.convert(
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

  def add_audio_frames(self, frames: np.ndarray) -> None:
    """Add audio frames to the buffer for processing."""
    self.buffer.add_frames(frames)

  async def _transcription_loop(self) -> None:
    """Main processing loop: get audio → transcribe → send results → wait."""
    while not self.exit:
      try:
        audio_chunk = await self._get_next_audio_chunk()
        if audio_chunk is None:
          continue

        transcription_result = await self._transcribe_chunk(audio_chunk)
        await self._process_transcription_result(transcription_result)
        await self._wait_for_next_interval(transcription_result.processing_time)
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
      await asyncio.sleep(self.buffer.config.min_chunk_duration - duration)
      return None

    return AudioChunk(data=input_bytes, duration=duration, start_time=start_time)

  async def _transcribe_chunk(self, chunk: AudioChunk) -> ChunkTranscriptionResult:
    """Transcribe an audio chunk and return results with timing information."""
    import time

    transcription_start = time.time()
    result, info = await asyncio.to_thread(self._transcribe_audio, chunk)
    processing_time = time.time() - transcription_start

    self.logger.debug(
      "Transcription performance",
      audio_duration=f"{chunk.duration:.2f}s",
      transcription_time=f"{processing_time:.2f}s",
      speed_vs_realtime=f"{chunk.duration / processing_time:.1f}x faster",
    )

    return ChunkTranscriptionResult(
      segments=result, info=info, processing_time=processing_time, audio_duration=chunk.duration
    )

  async def _process_transcription_result(self, result: ChunkTranscriptionResult) -> None:
    """Process transcription results, update language, and handle segments."""
    if self.language is None and result.info is not None:
      await self._set_language(result.info)

    if result.segments is None or len(result.segments) == 0:
      self.buffer.advance_processed_boundary(result.audio_duration)
    else:
      await self._handle_transcription_output(result.segments, result.audio_duration)

  async def _wait_for_next_interval(self, processing_time: float) -> None:
    """Wait to maintain consistent transcription intervals."""
    remaining_wait = self.buffer.config.transcription_interval - processing_time
    if remaining_wait > 0:
      self.logger.debug(f"Waiting {remaining_wait:.2f}s to maintain transcription interval")
      await asyncio.sleep(remaining_wait)
    else:
      self.logger.warning("Transcription took longer than interval, proceeding immediately")

  async def _handle_transcription_error(self) -> None:
    """Handle transcription errors with logging and recovery."""
    self.logger.exception("Failed to transcribe audio chunk")
    await asyncio.sleep(self.buffer.config.transcription_interval)

  def _transcribe_audio(
    self, chunk: AudioChunk
  ) -> tuple[list[Segment] | None, TranscriptionInfo | None]:
    """Transcribe audio sample using the Faster Whisper model."""
    if not self.transcriber:
      raise RuntimeError("Transcriber not initialized")

    shape = chunk.data.shape
    self.logger.debug("Transcribing audio sample", shape=shape, start_time=chunk.start_time)

    # if SINGLE_MODEL:
    #   self.logger.debug("Acquiring single model lock")
    #   StreamingTranscriptionProcessor.SINGLE_MODEL_LOCK.acquire()

    try:
      result, info = self.transcriber.transcribe(
        chunk.data,
        initial_prompt=self.config.initial_prompt,
        language=self.language,
        vad_filter=self.config.use_vad,
        vad_parameters=self.vad_parameters,
        absolute_stream_start=chunk.start_time,
        hotwords=" ".join(self.config.hotwords) if self.config.hotwords else None,
        session=self.session,
        start_offset=self.buffer.processed_up_to_time,
      )
      result_list = list(result) if result else None
      return result_list, info

    finally:
      pass
      # if SINGLE_MODEL:
      #   self.logger.debug("Releasing single model lock")
      #   StreamingTranscriptionProcessor.SINGLE_MODEL_LOCK.release()

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

  async def _handle_transcription_output(self, result: list[Segment], duration: float) -> None:
    """Handle transcription output and send to client."""
    result_count = len(result) if result else 0
    self.logger.debug(
      "Handling transcription output",
      duration=f"{duration:.2f}s",
      segments=result_count,
    )

    if len(result):
      self.logger.debug("Processing transcription segments")
      # Update internal transcript (for translation queue, etc.)
      self._update_segments(result, duration)

      # Mark segments as completed/incomplete based on position
      # Typically all segments except the last are completed
      segments_for_client = []
      for i, segment in enumerate(result):
        # Create a copy and set the completed field
        segment_copy = Segment(
          id=segment.id,
          seek=segment.seek,
          start=segment.start,
          end=segment.end,
          text=segment.text,
          tokens=segment.tokens,
          avg_logprob=segment.avg_logprob,
          compression_ratio=segment.compression_ratio,
          words=segment.words,
          temperature=segment.temperature,
          completed=i < len(result) - 1,  # All but last segment are completed
        )
        segments_for_client.append(segment_copy)

      # Send rich Segment objects to client
      transcription_result = TranscriptionResult(
        segments=segments_for_client,
        language=self.language,
      )
      self.logger.debug("Sending segments to client", segment_count=len(segments_for_client))
      await self.sink.send_result(transcription_result)
    else:
      self.logger.debug("No segments to send to client")

  def _update_segments(self, segments: list[Segment], duration: float) -> None:
    """Process segments and update transcript."""
    offset: float | None = None
    self.current_out = ""

    # Process complete segments
    if len(segments) > 1:
      for s in segments[:-1]:
        text_: str = s.text
        self.text.append(text_)
        start = self.buffer.processed_up_to_time + s.start
        end = self.buffer.processed_up_to_time + min(duration, s.end)

        if start >= end:
          continue

        completed_segment = self._format_segment(start, end, text_, completed=True)
        self.transcript.append(completed_segment)

        if self.translation_queue:
          try:
            self.translation_queue.put(completed_segment.copy(), timeout=0.1)
          except queue.Full:
            self.logger.warning("Translation queue is full, skipping segment")

        offset = min(duration, s.end)

    # Process last segment
    self.current_out += segments[-1].text

    # Handle repeated output
    if self.current_out.strip() == self.prev_out.strip() and self.current_out != "":
      self.same_output_count += 1
      if self.end_time_for_same_output is None:
        self.end_time_for_same_output = segments[-1].end
      time.sleep(0.1)
    else:
      self.same_output_count = 0
      self.end_time_for_same_output = None

    # Complete repeated segments
    if self.same_output_count > self.config.same_output_threshold:
      if not self.text or self.text[-1].strip().lower() != self.current_out.strip().lower():
        self.text.append(self.current_out)
        completed_segment = self._format_segment(
          self.buffer.processed_up_to_time,
          self.buffer.processed_up_to_time + min(duration, self.end_time_for_same_output),  # type: ignore
          self.current_out,
          completed=True,
        )
        self.transcript.append(completed_segment)

        if self.translation_queue:
          try:
            self.translation_queue.put(completed_segment.copy(), timeout=0.1)
          except queue.Full:
            self.logger.warning("Translation queue is full, skipping segment")

      self.current_out = ""
      offset = min(duration, self.end_time_for_same_output)  # type: ignore
      self.same_output_count = 0
      self.end_time_for_same_output = None
    else:
      self.prev_out = self.current_out

    if offset is not None:
      self.buffer.advance_processed_boundary(offset)

  def _prepare_segments(self, last_segment: dict | None = None) -> list[dict]:
    """Prepare segments for client."""
    segments: list[dict] = []
    if len(self.transcript) >= self.config.send_last_n_segments:
      segments = self.transcript[-self.config.send_last_n_segments :].copy()
    else:
      segments = self.transcript.copy()

    if last_segment is not None:
      segments = segments + [last_segment]

    return segments

  def _format_segment(self, start: float, end: float, text: str, completed: bool = False) -> dict:
    """Format a transcription segment."""
    return {
      "start": "{:.3f}".format(start),
      "end": "{:.3f}".format(end),
      "text": text,
      "completed": completed,
    }
