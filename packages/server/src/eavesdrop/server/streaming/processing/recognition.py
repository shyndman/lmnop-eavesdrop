"""Model caching and cancellation-safe ownership of native recognition workers."""

from __future__ import annotations

import asyncio
import time
from threading import Lock

import numpy as np
from structlog.stdlib import BoundLogger

from eavesdrop.common import get_logger
from eavesdrop.server.config import TranscriptionConfig
from eavesdrop.server.constants import SINGLE_MODEL
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.flush_state import LiveSessionFlushState
from eavesdrop.server.streaming.processing.models import (
  AudioChunk,
  ChunkTranscriptionResult,
  TranscriptionPassStatus,
)
from eavesdrop.server.transcription.models import SpeechChunk, TranscriptionInfo
from eavesdrop.server.transcription.orukeet import OrukeetTranscriber
from eavesdrop.server.transcription.session import TranscriptionSession
from eavesdrop.wire import Segment

tracing_logger = get_logger("tracing")


class RecognitionWorker:
  """Retain model lifetime and await native work before releasing its session."""

  # Class variable for single model mode
  _models: dict[tuple[str, str, bool], OrukeetTranscriber] = {}
  _model_lock: Lock = Lock()

  def __init__(
    self,
    buffer: AudioStreamBuffer,
    config: TranscriptionConfig,
    logger: BoundLogger,
    flush_state: LiveSessionFlushState | None,
  ) -> None:
    self.buffer: AudioStreamBuffer = buffer
    self.config: TranscriptionConfig = config
    self.logger: BoundLogger = logger
    self.flush_state: LiveSessionFlushState | None = flush_state
    self.transcriber: OrukeetTranscriber | None = None

  async def initialize(self) -> None:
    """Load the recognition model off the event loop."""
    self.transcriber = await asyncio.to_thread(self._load_model)

  def _load_model(self) -> OrukeetTranscriber:
    if not SINGLE_MODEL:
      return OrukeetTranscriber.load(self.config)
    key = (self.config.model, self.config.device, self.config.use_vad)
    with self._model_lock:
      if key not in self._models:
        self._models[key] = OrukeetTranscriber.load(self.config)
      return self._models[key]

  async def transcribe(
    self,
    chunk: AudioChunk,
    generation: int,
    recording_id: str | None,
    session: TranscriptionSession,
  ) -> ChunkTranscriptionResult:
    """Transcribe an audio chunk and return results with timing information."""
    transcription_start = time.monotonic()
    worker = asyncio.create_task(
      asyncio.to_thread(
        self._run_transcription_pass,
        chunk,
        generation,
        recording_id,
        session,
      )
    )
    try:
      pass_result = await asyncio.shield(worker)
    except asyncio.CancelledError:
      # Cancelling to_thread does not stop native inference. Await it before
      # releasing connection/session ownership, but never commit its output.
      await worker
      raise
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

  def _chunk_start_sample(self, chunk: AudioChunk) -> int:
    """Convert a chunk's buffer start offset to an absolute buffer sample index.

    Live WebSocket recording boundaries reset the buffer, so this sample index
    is recording-relative in live mode.
    """
    return int(round(chunk.start_time * self.buffer.config.sample_rate))

  def _consume_precommit_interrupt(self) -> bool:
    """Consume one live-session control interrupt before expensive work begins."""
    if self.flush_state is None or not self.flush_state.interrupt.is_set():
      return False
    _ = self.flush_state.clear_interrupt()
    self.logger.info("Live-session control interrupts worker pass before commit")
    return True

  def _run_transcription_pass(
    self,
    chunk: AudioChunk,
    utterance_generation: int,
    recording_id: str | None,
    session: TranscriptionSession,
  ) -> ChunkTranscriptionResult:
    """Run one worker-thread pass with identity captured before scheduling."""
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
        recording_id=recording_id,
      )

    result, info, speech_chunks = self._transcribe_audio(chunk, session, recording_id)
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
      recording_id=recording_id,
    )

  def _transcribe_audio(
    self, chunk: AudioChunk, session: TranscriptionSession, recording_id: str | None
  ) -> tuple[list[Segment] | None, TranscriptionInfo, list[SpeechChunk] | None]:
    """Transcribe audio sample using the speech recognition model."""
    if not self.transcriber:
      raise RuntimeError("Transcriber not initialized")

    session.update_recording_context(recording_id)

    result, info = self.transcriber.transcribe(
      chunk.data, config=self.config, absolute_stream_start=chunk.start_time, session=session
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
    sample_rate = 16000  # Required sample rate
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
