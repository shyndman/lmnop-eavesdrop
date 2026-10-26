"""Chunk selection and interruptible live-control scheduling."""

from __future__ import annotations

import asyncio

from structlog.stdlib import BoundLogger

from eavesdrop.common import get_logger
from eavesdrop.server.config import TranscriptionConfig
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.flush_state import LiveSessionFlushState, PendingFlush
from eavesdrop.server.streaming.processing.models import AudioChunk

tracing_logger = get_logger("tracing")


class ProcessingSchedule:
  """Select buffer views and own the tasks used for interruptible waits."""

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
    self._minimum_chunk_wait_logged: bool = False

  async def next_chunk(self) -> AudioChunk | None:
    """Get the next audio chunk, handling buffer management and minimum duration."""
    pending_flush = self.pending_flush()
    if self.config.use_vad and pending_flush is None:
      self.buffer.clip_if_stalled()
    input_bytes, duration, start_time = self.buffer.get_chunk_for_processing()
    if pending_flush is not None:
      boundary_offset = pending_flush.boundary_sample - int(
        round(start_time * self.buffer.config.sample_rate)
      )
      input_bytes = input_bytes[: max(0, boundary_offset)]
      duration = input_bytes.shape[0] / self.buffer.config.sample_rate
    if pending_flush is not None:
      return AudioChunk(data=input_bytes, duration=duration, start_time=start_time)

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
      remaining_wait = self.buffer.config.min_chunk_duration - duration
      if not self._minimum_chunk_wait_logged:
        self.logger.info(
          "Transcription loop minimum chunk wait",
          buffered_duration_s=f"{duration:.3f}",
          required_duration_s=f"{self.buffer.config.min_chunk_duration:.3f}",
          wait_s=f"{remaining_wait:.3f}",
        )
        self._minimum_chunk_wait_logged = True
      observed_generation = self.current_generation()
      wait_completed = await self._wait_for_live_control_wakeup(
        remaining_wait,
        observed_generation=observed_generation,
      )
      pending_flush = self.pending_flush()
      if not wait_completed and pending_flush is not None and duration > 0:
        self.logger.info(
          "Pending flush interrupts minimum chunk wait",
          buffered_duration_s=f"{duration:.3f}s",
          boundary_sample=pending_flush.boundary_sample,
        )
        return AudioChunk(data=input_bytes, duration=duration, start_time=start_time)
      if not wait_completed and self.current_generation() != observed_generation:
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
      recording_relative_chunk_start=f"{start_time:.2f}s",
      recording_relative_chunk_end=f"{start_time + duration:.2f}s",
      chunk_duration=f"{duration:.2f}s",
      buffer_available=f"{available_duration:.2f}s",
      buffer_total=f"{total_duration:.2f}s",
      buffer_processed=f"{processed_duration:.2f}s",
    )

    return AudioChunk(data=input_bytes, duration=duration, start_time=start_time)

  async def wait_for_interval(self, processing_time: float) -> None:
    """Wait to maintain consistent transcription intervals."""
    remaining_wait = self.buffer.config.transcription_interval - processing_time
    if remaining_wait > 0:
      self.logger.debug(f"Waiting {remaining_wait:.2f}s to maintain transcription interval")
      observed_generation = self.current_generation()
      wait_completed = await self._wait_for_live_control_wakeup(
        remaining_wait,
        observed_generation=observed_generation,
      )
      if not wait_completed:
        pending_flush = self.pending_flush()
        if pending_flush is not None:
          self.logger.info("Pending flush interrupts interval wait")
        elif self.current_generation() != observed_generation:
          self.logger.info(
            "Cancelled utterance interrupts interval wait",
            utterance_generation=observed_generation,
          )
    else:
      self.logger.warning("Transcription took longer than interval, proceeding immediately")

  def pending_flush(self) -> PendingFlush | None:
    """Return the currently pending flush request, if any."""
    if self.flush_state is None:
      return None
    return self.flush_state.pending()

  def current_generation(self) -> int:
    """Return the active utterance generation for the live session."""
    if self.flush_state is None:
      return 0
    return self.flush_state.current_generation()

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
      _ = await asyncio.gather(sleep_task, wake_task, return_exceptions=True)
