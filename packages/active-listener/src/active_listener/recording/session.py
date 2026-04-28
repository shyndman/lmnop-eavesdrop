"""Recording-lifetime state and resource ownership for active-listener."""

from __future__ import annotations

import asyncio
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from structlog.stdlib import BoundLogger

from active_listener.app.ports import (
  ActiveListenerClient,
  ActiveListenerRuntimeError,
  CapturedRecordingAudio,
  FinishedRecording,
)
from active_listener.infra.keyboard import KeyboardInput, RecordingGrabRelease
from active_listener.recording.reducer import (
  RecordingReducerState,
  TextRun,
  TimedWord,
  TimeSpan,
  TranscriptionUpdate,
  apply_segment_reduction,
  build_transcription_update,
  reduce_new_segments,
)
from active_listener.recording.spectrum import SAMPLE_RATE_HZ
from eavesdrop.wire import TranscriptionMessage

HOLD_THRESHOLD_S = 0.250
RECORDING_CHANNEL_COUNT = 1


@dataclass
class RecordingAudioBuffer:
  _chunks: list[bytes] = field(default_factory=list)
  _active: bool = False

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


@dataclass
class PendingCapsGesture:
  start_s: float
  threshold_task: asyncio.Task[None] | None


@dataclass
class RecordingSession:
  """Own recording state, reducer state, and keyboard-grab resources."""

  keyboard: KeyboardInput
  client: ActiveListenerClient
  logger: BoundLogger
  audio_buffer: RecordingAudioBuffer
  _connection_last_id: int | None = None
  _recording_id: str | None = None
  _recording_reducer_state: RecordingReducerState | None = None
  _recording_grab_stack: AsyncExitStack | None = None
  _release_recording_grab: RecordingGrabRelease | None = None
  _recording_started_monotonic_s: float | None = None
  _pending_caps_gesture: PendingCapsGesture | None = None
  _last_transcription_runs: list[TextRun] = field(default_factory=list)

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

  async def start_recording(self, recording_id: str) -> None:
    if self._recording_grab_stack is not None or self._release_recording_grab is not None:
      raise ActiveListenerRuntimeError("recording grab entered twice")

    grab_stack = AsyncExitStack()
    try:
      release_recording_grab = await grab_stack.enter_async_context(self.keyboard.recording_grab())
    except Exception:
      await grab_stack.aclose()
      raise

    self._recording_grab_stack = grab_stack
    self._release_recording_grab = release_recording_grab
    self._recording_id = recording_id
    self._recording_started_monotonic_s = time.monotonic()
    self._recording_reducer_state = RecordingReducerState()
    self.audio_buffer.start()
    self._last_transcription_runs = []
    self._connection_last_id = None
    self.logger.debug(
      "recording session started",
      recording_id=recording_id,
    )

  async def stop_recording(self) -> None:
    await self._exit_recording(close_open_command_span=False)

  async def finish_recording(self) -> FinishedRecording:
    recording_id = self._require_recording_id()
    reducer_state = self._require_recording_reducer_state()
    captured_audio = CapturedRecordingAudio(
      pcm_f32le=self.audio_buffer.finish(),
      sample_rate_hz=SAMPLE_RATE_HZ,
      channels=RECORDING_CHANNEL_COUNT,
    )
    await self._exit_recording(close_open_command_span=True)
    self.logger.debug(
      "recording session finished",
      last_id=reducer_state.last_id,
      committed_word_count=len(reducer_state.completed_words),
      closed_command_span_count=len(reducer_state.closed_command_spans),
      captured_audio_bytes=len(captured_audio.pcm_f32le),
    )
    return FinishedRecording(
      recording_id=recording_id,
      reducer_state=reducer_state,
      captured_audio=captured_audio,
    )

  async def handle_capslock_down(self, received_monotonic_s: float) -> None:
    reducer_state = self._require_recording_reducer_state()
    if self._pending_caps_gesture is not None or reducer_state.open_command_start_s is not None:
      return

    start_s = self.recording_elapsed_s(received_monotonic_s)
    threshold_task = asyncio.create_task(self._commit_pending_caps_hold(start_s))
    self._pending_caps_gesture = PendingCapsGesture(
      start_s=start_s,
      threshold_task=threshold_task,
    )
    self.logger.debug("caps gesture pending", start_s=start_s, threshold_s=HOLD_THRESHOLD_S)

  async def handle_capslock_up(self, received_monotonic_s: float) -> bool:
    reducer_state = self._require_recording_reducer_state()
    pending_gesture = self._pending_caps_gesture
    if pending_gesture is not None:
      end_s = self.recording_elapsed_s(received_monotonic_s)
      hold_duration_s = end_s - pending_gesture.start_s
      await self._clear_pending_caps_gesture()
      if hold_duration_s >= HOLD_THRESHOLD_S:
        reducer_state.closed_command_spans.append(
          TimeSpan(start_s=pending_gesture.start_s, end_s=end_s)
        )
        self.logger.debug(
          "caps hold resolved on release",
          start_s=pending_gesture.start_s,
          end_s=end_s,
          hold_duration_s=hold_duration_s,
        )
        return False

      self.logger.debug("caps finish tap resolved", start_s=pending_gesture.start_s)
      return True

    open_command_start_s = reducer_state.open_command_start_s
    if open_command_start_s is None:
      return False

    end_s = self.recording_elapsed_s(received_monotonic_s)
    reducer_state.closed_command_spans.append(TimeSpan(start_s=open_command_start_s, end_s=end_s))
    reducer_state.open_command_start_s = None
    self.logger.debug(
      "caps hold closed",
      start_s=open_command_start_s,
      end_s=end_s,
    )
    return False

  def ingest_live_transcription_message(
    self,
    message: TranscriptionMessage,
  ) -> TranscriptionUpdate | None:
    recording_id = self._require_recording_id()
    if message.recording_id != recording_id:
      self.logger.warning(
        "transcription message ignored for inactive recording",
        stream=message.stream,
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
      stream=message.stream,
      prior_last_id=state.last_id,
      segment_count=len(message.segments),
      flush_complete=message.flush_complete is True,
    )
    reduction = reduce_new_segments(message.segments, state.last_id)
    if reduction.missing_last_id:
      self.logger.warning(
        "transcription reducer sentinel missing",
        stream=message.stream,
        last_id=state.last_id,
      )
    apply_segment_reduction(state, reduction)
    word_timings = [
      _format_word_timing(index, word)
      for index, word in enumerate([*state.completed_words, *state.incomplete_words])
    ]
    if word_timings:
      self.logger.info(
        "transcription word timings",
        words=word_timings,
      )
    state.last_id = reduction.last_id
    self._connection_last_id = reduction.last_id
    transcription_update = build_transcription_update(state)
    if (
      transcription_update is not None
      and transcription_update.runs == self._last_transcription_runs
    ):
      transcription_update = None
    elif transcription_update is None:
      self._last_transcription_runs = []
    else:
      self._last_transcription_runs = list(transcription_update.runs)
    incomplete_segment = reduction.incomplete_segment
    self.logger.debug(
      "transcription window reduced",
      stream=message.stream,
      new_last_id=reduction.last_id,
      new_segment_ids=[segment.id for segment in reduction.segments],
      incomplete_segment_id=None if incomplete_segment is None else incomplete_segment.id,
      incomplete_segment_text=(
        None if incomplete_segment is None else incomplete_segment.text.strip()
      ),
      committed_segment_count=len(reduction.segments),
      committed_word_count=len(state.completed_words),
      incomplete_word_count=len(state.incomplete_words),
      overlay_update=transcription_update is not None,
    )
    return transcription_update

  async def _exit_recording(self, *, close_open_command_span: bool) -> None:
    release_recording_grab = self._release_recording_grab
    grab_stack = self._recording_grab_stack
    reducer_state = self._recording_reducer_state

    if (release_recording_grab is None) != (grab_stack is None):
      raise ActiveListenerRuntimeError("recording grab ownership is inconsistent")

    recording_end_s: float | None = None
    if close_open_command_span and reducer_state is not None:
      recording_end_s = self.recording_elapsed_s(time.monotonic())

    await self._clear_pending_caps_gesture()
    if reducer_state is not None:
      if close_open_command_span and reducer_state.open_command_start_s is not None:
        assert recording_end_s is not None
        reducer_state.closed_command_spans.append(
          TimeSpan(start_s=reducer_state.open_command_start_s, end_s=recording_end_s)
        )
      reducer_state.open_command_start_s = None
    self.audio_buffer.discard()

    self._release_recording_grab = None
    self._recording_grab_stack = None
    self._recording_id = None
    self._recording_reducer_state = None
    self._recording_started_monotonic_s = None
    self._last_transcription_runs = []

    try:
      if release_recording_grab is not None:
        _ = release_recording_grab()
    finally:
      if grab_stack is not None:
        await grab_stack.aclose()

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

  async def _commit_pending_caps_hold(self, start_s: float) -> None:
    try:
      await asyncio.sleep(HOLD_THRESHOLD_S)
    except asyncio.CancelledError:
      return

    pending_gesture = self._pending_caps_gesture
    reducer_state = self._recording_reducer_state
    if pending_gesture is None or reducer_state is None or pending_gesture.start_s != start_s:
      return

    reducer_state.open_command_start_s = start_s
    self._pending_caps_gesture = None
    self.logger.debug("caps hold committed", start_s=start_s)

  async def _clear_pending_caps_gesture(self) -> None:
    pending_gesture = self._pending_caps_gesture
    self._pending_caps_gesture = None
    if pending_gesture is None or pending_gesture.threshold_task is None:
      return

    threshold_task = pending_gesture.threshold_task
    if threshold_task.done():
      return

    _ = threshold_task.cancel()
    _ = await asyncio.gather(threshold_task, return_exceptions=True)


def _format_word_timing(_index: int, word: TimedWord) -> str:
  duration_s = word.end_s - word.start_s
  return f"{word.text}   {word.start_s:.3f}-{word.end_s:.3f} ({duration_s:.3f})"
