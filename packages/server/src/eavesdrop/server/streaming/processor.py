"""
Streaming transcription processor with integrated Faster Whisper transcriber.
"""

import asyncio
import os
import threading
from dataclasses import dataclass

import ctranslate2
import numpy as np
import torch
from huggingface_hub import snapshot_download

from eavesdrop.common import Pretty, Range, Seconds, get_logger
from eavesdrop.server.config import TranscriptionConfig
from eavesdrop.server.constants import CACHE_PATH, SINGLE_MODEL
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.interfaces import TranscriptionResult, TranscriptionSink
from eavesdrop.server.transcription.models import TranscriptionInfo
from eavesdrop.server.transcription.pipeline import WhisperModel
from eavesdrop.server.transcription.session import TranscriptionSession
from eavesdrop.wire import Segment

tracing_logger = get_logger("tracing")


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
  speech_chunks: list | None = None


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
    session: TranscriptionSession,
    stream_name: str,
  ) -> None:
    self.buffer = buffer
    self.sink = sink
    self.config = config
    self.stream_name = stream_name
    self.session: TranscriptionSession = session
    self.logger = get_logger("proc", stream=self.stream_name)

    # Transcription state
    self.exit: bool = False
    self.language: str | None = config.language
    self.text: list[str] = []

    # Segment processing state - repetition logic removed

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
    self.vad_parameters = config.vad_parameters

  async def initialize(self) -> None:
    """Initialize the transcriber model."""
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
    import time

    transcription_start = time.time()
    result, info, speech_chunks = await asyncio.to_thread(self._transcribe_audio, chunk)
    processing_time = time.time() - transcription_start

    self.logger.debug(
      "Transcription performance",
      audio_duration=f"{chunk.duration:.2f}s",
      transcription_time=f"{processing_time:.2f}s",
      speed_vs_realtime=f"{chunk.duration / processing_time:.1f}x faster",
    )

    # Store speech chunks for silence-based completion
    return ChunkTranscriptionResult(
      segments=result,
      info=info,
      processing_time=processing_time,
      audio_duration=chunk.duration,
      speech_chunks=speech_chunks,
    )

  async def _process_transcription_result(self, result: ChunkTranscriptionResult) -> None:
    """Process transcription results, update language, and handle segments."""
    if self.language is None and result.info is not None:
      await self._set_language(result.info)

    if result.segments is None or len(result.segments) == 0:
      self.buffer.advance_processed_boundary(result.audio_duration)
    else:
      await self._handle_transcription_output(
        result.segments, result.audio_duration, result.speech_chunks
      )

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
  ) -> tuple[list[Segment] | None, TranscriptionInfo | None, list | None]:
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

      # Store VAD speech chunks for silence analysis
      speech_chunks = getattr(info, "speech_chunks", None) if info else None

      # TRACING: VAD speech detection results
      sample_rate = 16000  # Standard Whisper sample rate

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
        )

      return result_list, info, speech_chunks

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

  async def _handle_transcription_output(
    self, result: list[Segment], duration: float, speech_chunks: list | None = None
  ) -> None:
    """Handle transcription output and send to client."""
    result_count = len(result) if result else 0
    self.logger.debug(
      "Handling transcription output",
      duration=f"{duration:.2f}s",
      segments=result_count,
    )

    if not result:
      self.logger.debug("No segments to send to client")
      return

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
    current_incomplete_segment = None
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

    # Prepare segments for client: last N completed + current incomplete
    segments_for_client = []

    # Add the last N completed segments from session
    segments_for_client.extend(
      self.session.most_recent_completed_segments(self.config.send_last_n_segments)
    )

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
    self._advance_buffer_by_completed_segments(speech_chunks, silence_threshold, duration)

    # TRACING: Final client output
    client_segments = [
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
    )
    self.logger.debug("Sending segments to client", segments=segments_for_client)
    await self.sink.send_result(transcription_result)

  def _update_segments(self, segments: list[Segment], duration: float) -> None:
    """Process segments and update transcript."""
    # Update internal transcript for translation queue
    for segment in segments:
      if segment.completed and segment.text.strip():
        self.text.append(segment.text)

  def _prepare_segments(self, last_segment: dict | None = None) -> list[dict]:
    """Prepare segments for client using session completed segments."""
    segments: list[dict] = []

    if self.session and self.session.completed_segments:
      # Convert Segment objects to dict format for client
      completed_dicts = []
      for seg in self.session.completed_segments:
        completed_dicts.append(
          {
            "id": str(seg.id),
            "start": "{:.3f}".format(seg.absolute_start_time()),
            "end": "{:.3f}".format(seg.absolute_end_time()),
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
    speech_chunks: list | None,
    silence_threshold: float,
    audio_duration: float,
  ) -> bool:
    """Determine if segment should be completed based on silence after speech."""
    sample_rate = self.buffer.config.sample_rate

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

    # Convert speech chunks from samples to seconds
    last_speech_time = max(chunk["end"] for chunk in speech_chunks) / sample_rate

    # Time since last speech in this audio chunk
    chunk_end_time = audio_duration
    silence_duration = chunk_end_time - last_speech_time

    # If we have enough silence after the segment ends, mark it complete
    if (
      segment.end <= last_speech_time + silence_threshold and silence_duration >= silence_threshold
    ):
      tracing_logger.info(
        "Completing segment due to silence after speech",
        segment_id=segment.id,
        silence_duration=f"{silence_duration:.2f}s",
        last_speech_time=f"{last_speech_time:.2f}s",
        threshold=f"{silence_threshold:.2f}s",
      )
      return True

    return False

  def _advance_buffer_by_completed_segments(
    self,
    speech_chunks: list | None,
    silence_threshold: float,
    audio_duration: float,
  ) -> None:
    """Advance buffer pointer based on speech boundaries and completed segments."""
    # Get the last completed segment to determine how far we can advance
    last_completed = self.session.get_last_completed_segment()

    if not last_completed:
      return

    # Calculate how far we can safely advance
    completed_end_time = last_completed.absolute_end_time()
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
    if speech_chunks:
      sample_rate = self.buffer.config.sample_rate
      last_speech_time = max(chunk["end"] for chunk in speech_chunks) / sample_rate
      chunk_end_time = audio_duration
      silence_after_speech = chunk_end_time - last_speech_time

      # If we have significant silence, advance through most of it
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
    speech_chunks: list | None,
    silence_threshold: float,
    duration: float,
  ) -> Segment:
    """Create a synthetic incomplete segment to maintain client state machine invariant."""
    from eavesdrop.wire.transcription import compute_segment_chain_id

    # Determine where the synthetic segment should start
    if speech_chunks:
      # Start after the last detected speech + threshold
      sample_rate = self.buffer.config.sample_rate
      last_speech_time = max(chunk["end"] for chunk in speech_chunks) / sample_rate
      synthetic_start = last_speech_time + silence_threshold
    else:
      # No speech detected, start at current buffer position
      synthetic_start = duration

    # Always create to maintain client invariant - place at end if necessary
    actual_start = min(synthetic_start, duration)

    return Segment(
      id=compute_segment_chain_id(0, ""),  # Baseline ID for incomplete
      seek=0,
      start=actual_start,
      end=actual_start,  # Zero-duration synthetic segment
      text="",  # Empty text is fine for client state machine
      tokens=[],
      avg_logprob=0.0,
      compression_ratio=0.0,
      words=None,
      temperature=0.0,
      completed=False,
      time_offset=self.buffer.processed_up_to_time,
    )

  def _format_segment(self, start: float, end: float, text: str, completed: bool = False) -> dict:
    """Format a transcription segment."""
    return {
      "start": "{:.3f}".format(start),
      "end": "{:.3f}".format(end),
      "text": text,
      "completed": completed,
    }
