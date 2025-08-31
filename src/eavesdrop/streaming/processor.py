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

from ..logs import get_logger
from ..transcription.models import Segment, TranscriptionInfo
from ..transcription.whisper_model import WhisperModel
from .buffer import AudioStreamBuffer
from .interfaces import TranscriptionResult, TranscriptionSink


@dataclass
class TranscriptionConfig:
  """Configuration for transcription processing behavior."""

  # Transcription behavior
  send_last_n_segments: int = 10
  """Number of most recent segments to send to the client."""

  no_speech_thresh: float = 0.45
  """Segments with no speech probability above this threshold will be discarded."""

  same_output_threshold: int = 10
  """Number of repeated outputs before considering it as a valid segment."""

  use_vad: bool = True
  """Whether to use Voice Activity Detection."""

  clip_audio: bool = False
  """Whether to clip audio with no valid segments."""

  # Model configuration
  model: str = "distil-small.en"
  """Whisper model size or path."""

  task: str = "transcribe"
  """The task type, e.g., "transcribe"."""

  language: str | None = None
  """Language for transcription."""

  initial_prompt: str | None = None
  """Initial prompt for whisper inference."""

  vad_parameters: VadOptions | dict | None = None
  """Voice Activity Detection parameters."""

  single_model: bool = False
  """Whether to use single shared model instance."""

  cache_path: str = "~/.cache/eavesdrop/"
  """Path for model caching."""

  device_index: int = 0
  """GPU device index to use."""


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
    client_uid: str,
    translation_queue: queue.Queue[dict] | None = None,
    logger_name: str = "transcription_processor",
  ) -> None:
    self.buffer = buffer
    self.sink = sink
    self.config = config
    self.client_uid = client_uid
    self.translation_queue = translation_queue
    self.logger = get_logger(logger_name)

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
      major, _ = torch.cuda.get_device_capability(device)
      self.compute_type = "float16" if major >= 7 else "float32"
      self.logger.debug("CUDA device capability", major=major, compute_type=self.compute_type)
    else:
      self.compute_type = "int8"
      self.logger.debug("Using CPU with compute_type", compute_type=self.compute_type)

    self.logger.info("Using Device with precision", device=device, precision=self.compute_type)

    try:
      await asyncio.to_thread(self._create_model, device)
    except Exception:
      self.logger.exception("Failed to load model")
      await self.sink.send_error(f"Failed to load model: {self.config.model}")
      raise

    # Send server ready message
    await self.sink.send_server_ready("faster_whisper")

  def _create_model(self, device: str) -> None:
    """Create and initialize the Whisper model."""
    self.logger.debug("Creating model", model_reference=self.config.model)

    if self.config.single_model:
      self.logger.debug("Using single model mode", client_uid=self.client_uid)
      with StreamingTranscriptionProcessor.SINGLE_MODEL_LOCK:
        if StreamingTranscriptionProcessor.SINGLE_MODEL is None:
          self.logger.debug("Creating new single model instance")
          self._create_model_instance(device, self.config.model)
          StreamingTranscriptionProcessor.SINGLE_MODEL = self.transcriber
        else:
          self.logger.debug("Reusing existing single model instance")
          self.transcriber = StreamingTranscriptionProcessor.SINGLE_MODEL
    else:
      self.logger.debug("Creating dedicated model", client_uid=self.client_uid)
      self._create_model_instance(device, self.config.model)

  def _create_model_instance(self, device: str, model_ref: str) -> None:
    """Create a single model instance."""
    if model_ref in self.model_sizes:
      self.logger.debug("Model found in standard model sizes", model=model_ref)
      model_to_load = model_ref
    else:
      self.logger.debug("Model not in standard model sizes, checking if custom", model=model_ref)
      if os.path.isdir(model_ref) and ctranslate2.contains_model(model_ref):
        self.logger.debug("Found local CTranslate2 model", path=model_ref)
        model_to_load = model_ref
      else:
        self.logger.debug("Downloading model from HuggingFace", model=model_ref)
        local_snapshot = snapshot_download(repo_id=model_ref, repo_type="model")
        self.logger.debug("Downloaded model", path=local_snapshot)

        if ctranslate2.contains_model(local_snapshot):
          self.logger.debug("Downloaded model is already in CTranslate2 format")
          model_to_load = local_snapshot
        else:
          cache_root = os.path.expanduser(
            os.path.join(self.config.cache_path, "whisper-ct2-models/")
          )
          os.makedirs(cache_root, exist_ok=True)
          safe_name = model_ref.replace("/", "--")
          ct2_dir = os.path.join(cache_root, safe_name)
          self.logger.debug("CTranslate2 cache directory", path=ct2_dir)

          if not ctranslate2.contains_model(ct2_dir):
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
          else:
            self.logger.debug("CTranslate2 model already exists in cache")
          model_to_load = ct2_dir

    self.logger.info("Loading model", model=model_to_load)
    self.transcriber = WhisperModel(
      model_to_load,
      device=device,
      device_index=self.config.device_index,
      compute_type=self.compute_type,
      download_root=self.config.cache_path,
      local_files_only=False,
    )
    self.logger.debug("Model loaded successfully")

  async def start_processing(self) -> None:
    """Start the transcription processing loop."""
    self.logger.info("Starting transcription processing", client_uid=self.client_uid)
    await self._transcription_loop()

  async def stop_processing(self) -> None:
    """Stop the transcription processing loop."""
    self.logger.info("Stopping transcription processing", client_uid=self.client_uid)
    self.exit = True
    await self.sink.disconnect()

  def add_audio_frames(self, frames: np.ndarray) -> None:
    """Add audio frames to the buffer for processing."""
    self.buffer.add_frames(frames)

  async def _transcription_loop(self) -> None:
    """
    Process an audio stream in an infinite loop, continuously transcribing the speech.

    This method continuously processes audio frames, performs real-time transcription,
    and sends transcribed segments to the client via the sink.
    """
    while True:
      if self.exit:
        self.logger.info("Exiting speech to text thread")
        break

      if self.buffer.available_duration == 0:
        await asyncio.sleep(0.1)
        continue

      if self.config.clip_audio:
        self.buffer.clip_if_stalled()

      input_bytes, duration = self.buffer.get_chunk_for_processing()
      if duration < self.buffer.config.min_chunk_duration:
        await asyncio.sleep(0.1)
        continue

      try:
        input_sample = input_bytes.copy()
        result, info = await asyncio.to_thread(self._transcribe_audio, input_sample)

        if self.language is None and info is not None:
          await self._set_language(info)

        if result is None:
          self.buffer.advance_processed_boundary(duration)
          await asyncio.sleep(0.25)
          continue

        await self._handle_transcription_output(result, duration)

      except Exception:
        self.logger.exception("Failed to transcribe audio chunk")
        await asyncio.sleep(0.01)

  def _transcribe_audio(
    self, input_sample: np.ndarray
  ) -> tuple[list[Segment] | None, TranscriptionInfo | None]:
    """Transcribe audio sample using the Faster Whisper model."""
    if not self.transcriber:
      raise RuntimeError("Transcriber not initialized")

    shape = input_sample.shape if hasattr(input_sample, "shape") else "unknown"
    self.logger.debug("Transcribing audio sample", shape=shape)

    if self.config.single_model:
      self.logger.debug("Acquiring single model lock")
      StreamingTranscriptionProcessor.SINGLE_MODEL_LOCK.acquire()

    try:
      self.logger.debug(
        "Starting transcription",
        language=self.language,
        task=self.config.task,
        vad=self.config.use_vad,
      )

      result, info = self.transcriber.transcribe(
        input_sample,
        initial_prompt=self.config.initial_prompt,
        language=self.language,
        task=self.config.task,
        vad_filter=self.config.use_vad,
        vad_parameters=self.vad_parameters,
      )

      result_list = list(result) if result else None
      result_count = len(result_list) if result_list else 0
      self.logger.debug("Transcription completed", segments=result_count)
      return result_list, info

    finally:
      if self.config.single_model:
        self.logger.debug("Releasing single model lock")
        StreamingTranscriptionProcessor.SINGLE_MODEL_LOCK.release()

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

    segments: list[dict] = []
    if len(result):
      self.logger.debug("Processing transcription segments")
      last_segment = self._update_segments(result, duration)
      segments = self._prepare_segments(last_segment)
      self.logger.debug("Prepared segments for client", segment_count=len(segments))

    if len(segments):
      self.logger.debug("Sending segments to client", segment_count=len(segments))
      transcription_result = TranscriptionResult(
        segments=segments,
        language=self.language,
      )
      await self.sink.send_result(transcription_result)
    else:
      self.logger.debug("No segments to send to client")

  def _update_segments(self, segments: list[Segment], duration: float) -> dict | None:
    """Process segments and update transcript."""
    offset: float | None = None
    self.current_out = ""
    last_segment: dict | None = None

    # Process complete segments
    if len(segments) > 1 and segments[-1].no_speech_prob <= self.config.no_speech_thresh:
      for s in segments[:-1]:
        text_: str = s.text
        self.text.append(text_)
        start = self.buffer.timestamp_offset + s.start
        end = self.buffer.timestamp_offset + min(duration, s.end)

        if start >= end or s.no_speech_prob > self.config.no_speech_thresh:
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
    if segments[-1].no_speech_prob <= self.config.no_speech_thresh:
      self.current_out += segments[-1].text
      last_segment = self._format_segment(
        self.buffer.timestamp_offset + segments[-1].start,
        self.buffer.timestamp_offset + min(duration, segments[-1].end),
        self.current_out,
        completed=False,
      )

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
          self.buffer.timestamp_offset,
          self.buffer.timestamp_offset + min(duration, self.end_time_for_same_output),  # type: ignore
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
      last_segment = None
      self.end_time_for_same_output = None
    else:
      self.prev_out = self.current_out

    if offset is not None:
      self.buffer.advance_processed_boundary(offset)

    return last_segment

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
