"""Recording-lifetime state and resource ownership for active-listener."""

from __future__ import annotations

import asyncio
import time

from structlog.stdlib import BoundLogger

from active_listener.app.ports import (
  ActiveListenerClient,
  ActiveListenerRuntimeError,
  CapturedRecordingAudio,
  CorrectionLoadTask,
  FinishedRecording,
)
from active_listener.infra.corrections import CorrectionMap
from active_listener.recording.reducer import (
  RecordingReducerState,
  SegmentReduction,
  TextRun,
  TimeSpan,
  TranscriptionUpdate,
  apply_segment_reduction,
  build_completed_text_runs,
  build_incomplete_text_runs,
  reduce_new_segments,
)
from active_listener.recording.spectrum import SAMPLE_RATE_HZ
from active_listener.recording.text_shaping import shape_runs
from eavesdrop.wire import TranscriptionMessage

RECORDING_CHANNEL_COUNT = 1


class RecordingAudioBuffer:
  """Retain captured frames only for the currently active recording."""

  def __init__(self) -> None:
    self._chunks: list[bytes] = []
    self._active: bool = False

  def start(self) -> None:
    self._chunks.clear()
    self._active = True

  def append(self, chunk: bytes) -> None:
    if not self._active:
      return

    self._chunks.append(chunk)

  def finish(self) -> bytes:
    combined = b"".join(self._chunks)
    self._chunks.clear()
    self._active = False
    return combined

  def discard(self) -> None:
    self._chunks.clear()
    self._active = False


class RecordingTranscript:
  """Own one recording's captured audio and revisable transcript, without hardware.

  :param client: Capture/streaming dependency.
  :param logger: Recording diagnostics.
  :param audio_buffer: Buffer receiving microphone frames during capture.
  """

  def __init__(
    self, *, client: ActiveListenerClient, logger: BoundLogger, audio_buffer: RecordingAudioBuffer
  ) -> None:
    self.client: ActiveListenerClient = client
    self.logger: BoundLogger = logger
    self.audio_buffer: RecordingAudioBuffer = audio_buffer
    self._connection_last_id: int | None = None
    self._recording_id: str | None = None
    self._recording_reducer_state: RecordingReducerState | None = None
    self._recording_started_monotonic_s: float | None = None
    self._correction_load_task: CorrectionLoadTask | None = None
    self._last_transcription_runs: list[TextRun] = []
    self._loaded_corrections: CorrectionMap = {}
    self._corrections_harvested: bool = False
    self._shaped_completed_runs: list[TextRun] = []
    self._shaped_completed_source: list[TextRun] = []

  @property
  def is_recording(self) -> bool:
    return self._recording_reducer_state is not None

  @property
  def active_recording_id(self) -> str | None:
    return self._recording_id

  def reset_connection_cursor(self) -> None:
    self.logger.debug(
      "transcription connection cursor reset",
      previous_last_id=self._connection_last_id,
    )
    self._connection_last_id = None

  async def start_recording(
    self,
    recording_id: str,
    *,
    correction_load_task: CorrectionLoadTask | None = None,
  ) -> None:
    if self.is_recording:
      self.logger.error("recording transcript started twice")
      raise ActiveListenerRuntimeError("recording transcript started twice")
    self._recording_id = recording_id
    self._recording_started_monotonic_s = time.monotonic()
    self._correction_load_task = correction_load_task
    self._recording_reducer_state = RecordingReducerState()
    self.audio_buffer.start()
    self._last_transcription_runs = []
    self._loaded_corrections = {}
    self._corrections_harvested = False
    self._shaped_completed_runs = []
    self._shaped_completed_source = []
    self._connection_last_id = None
    self.logger.debug(
      "recording transcript started",
      recording_id=recording_id,
    )

  async def stop_recording(self) -> None:
    await self._exit_recording(close_open_command_span=False)

  async def finish_recording(self) -> FinishedRecording:
    recording_id = self._require_recording_id()
    reducer_state = self._require_recording_reducer_state()
    correction_load_task = self._correction_load_task
    captured_audio = CapturedRecordingAudio(
      pcm_f32le=self.audio_buffer.finish(),
      sample_rate_hz=SAMPLE_RATE_HZ,
      channels=RECORDING_CHANNEL_COUNT,
    )
    await self._exit_recording(close_open_command_span=True, cancel_correction_load=False)
    self.logger.debug(
      "recording transcript finished",
      last_id=reducer_state.last_id,
      committed_word_count=len(reducer_state.completed_words),
      closed_command_span_count=len(reducer_state.closed_command_spans),
      captured_audio_bytes=len(captured_audio.pcm_f32le),
    )
    return FinishedRecording(
      recording_id=recording_id,
      reducer_state=reducer_state,
      captured_audio=captured_audio,
      correction_load_task=correction_load_task,
    )

  def open_command_span(self, start_s: float) -> None:
    """Mark a recording-relative command interval's start.

    :param start_s: Start time on the captured recording timeline.
    """
    self._require_recording_reducer_state().open_command_start_s = start_s

  def close_command_span(self, end_s: float) -> None:
    """Close the current command interval, if any.

    :param end_s: End time on the captured recording timeline.
    """
    state = self._require_recording_reducer_state()
    if state.open_command_start_s is not None:
      state.closed_command_spans.append(TimeSpan(state.open_command_start_s, end_s))
      state.open_command_start_s = None

  def ingest_live_transcription_message(
    self,
    message: TranscriptionMessage,
  ) -> TranscriptionUpdate | None:
    recording_id = self._require_recording_id()
    if message.recording_id != recording_id:
      self.logger.warning(
        "transcription message ignored for inactive recording",
        session_id=message.stream,
        recording_id=message.recording_id,
        active_recording_id=recording_id,
      )
      return None

    reducer_state = self._require_recording_reducer_state()
    return self.ingest_transcription_message(reducer_state, message)

  def ingest_transcription_message(
    self,
    state: RecordingReducerState,
    message: TranscriptionMessage,
  ) -> TranscriptionUpdate | None:
    self.logger.debug(
      "reducing transcription window",
      session_id=message.stream,
      prior_last_id=state.last_id,
      segment_count=len(message.segments),
      flush_complete=message.flush_complete is True,
    )
    reduction = reduce_new_segments(message.segments, state.last_id)
    if reduction.missing_last_id:
      self.logger.warning(
        "transcription reducer sentinel missing",
        session_id=message.stream,
        last_id=state.last_id,
      )
    apply_segment_reduction(state, reduction)
    word_timings = [
      f"{word.text}   {word.start_s:.3f}-{word.end_s:.3f} ({word.end_s - word.start_s:.3f})"
      for word in [*state.completed_words, *state.incomplete_words]
    ]
    if word_timings:
      self.logger.info(
        "transcription word timings",
        words=word_timings,
      )
    state.last_id = reduction.last_id
    self._connection_last_id = reduction.last_id
    # Shape stable completed runs only; revisions keep the raw incomplete tail.
    self._harvest_corrections()
    completed_source = build_completed_text_runs(state)
    if completed_source != self._shaped_completed_source:
      self._shaped_completed_source = completed_source
      self._shaped_completed_runs = shape_runs(completed_source, self._loaded_corrections)
    overlay_runs = [*self._shaped_completed_runs, *build_incomplete_text_runs(state)]
    transcription_update = TranscriptionUpdate(runs=overlay_runs) if overlay_runs else None
    if (
      transcription_update is not None
      and transcription_update.runs == self._last_transcription_runs
    ):
      transcription_update = None
    elif transcription_update is None:
      self._last_transcription_runs = []
    else:
      self._last_transcription_runs = list(transcription_update.runs)
    self._log_reduction(state, message, reduction, transcription_update is not None)
    return transcription_update

  def _log_reduction(
    self,
    state: RecordingReducerState,
    message: TranscriptionMessage,
    reduction: SegmentReduction,
    overlay_update: bool,
  ) -> None:
    incomplete = reduction.incomplete_segment
    self.logger.debug(
      "transcription window reduced",
      session_id=message.stream,
      new_last_id=reduction.last_id,
      new_segment_ids=[segment.id for segment in reduction.segments],
      incomplete_segment_id=incomplete.id if incomplete is not None else None,
      incomplete_segment_text=incomplete.text.strip() if incomplete is not None else None,
      committed_segment_count=len(reduction.segments),
      committed_word_count=len(state.completed_words),
      incomplete_word_count=len(state.incomplete_words),
      overlay_update=overlay_update,
    )

  def _harvest_corrections(self) -> None:
    # Non-blocking live path: adopt stored corrections as soon as the load task settles,
    # so live windows shape with corrections without ever stalling on the load.
    task = self._correction_load_task
    if task is None or self._corrections_harvested or not task.done():
      return
    self._corrections_harvested = True
    try:
      self._loaded_corrections = task.result()
    except Exception:
      self.logger.exception("stored correction load failed")
      self._loaded_corrections = {}
    # Corrections changed: invalidate the shaped cache so the next window reshapes committed
    # runs with them. The reshape happens later in this same ingest, so no window goes stale.
    self._shaped_completed_source = []

  async def _exit_recording(
    self,
    *,
    close_open_command_span: bool,
    cancel_correction_load: bool = True,
  ) -> None:
    reducer_state = self._recording_reducer_state

    recording_end_s: float | None = None
    if close_open_command_span and reducer_state is not None:
      recording_end_s = self.recording_elapsed_s(time.monotonic())

    if reducer_state is not None:
      if close_open_command_span and reducer_state.open_command_start_s is not None:
        assert recording_end_s is not None
        reducer_state.closed_command_spans.append(
          TimeSpan(start_s=reducer_state.open_command_start_s, end_s=recording_end_s)
        )
      reducer_state.open_command_start_s = None
    self.audio_buffer.discard()

    self._recording_id = None
    self._recording_reducer_state = None
    self._recording_started_monotonic_s = None
    correction_load_task = self._correction_load_task
    self._correction_load_task = None
    self._last_transcription_runs = []
    self._loaded_corrections = {}
    self._corrections_harvested = False
    self._shaped_completed_runs = []
    self._shaped_completed_source = []

    if cancel_correction_load and correction_load_task is not None:
      if not correction_load_task.done():
        _ = correction_load_task.cancel()
      _ = await asyncio.gather(correction_load_task, return_exceptions=True)

    await self.client.stop_streaming()

  def _require_recording_reducer_state(self) -> RecordingReducerState:
    reducer_state = self._recording_reducer_state
    if reducer_state is None:
      raise ActiveListenerRuntimeError("recording finish requested without reducer state")
    return reducer_state

  def _require_recording_id(self) -> str:
    recording_id = self._recording_id
    if recording_id is None:
      raise ActiveListenerRuntimeError("recording id requested without active recording")
    return recording_id

  def recording_elapsed_s(self, now_monotonic_s: float) -> float:
    recording_started_monotonic_s = self._recording_started_monotonic_s
    if recording_started_monotonic_s is None:
      raise ActiveListenerRuntimeError("recording timeline requested without start anchor")
    elapsed_samples = max(
      0,
      int(round((now_monotonic_s - recording_started_monotonic_s) * SAMPLE_RATE_HZ)),
    )
    return elapsed_samples / SAMPLE_RATE_HZ
