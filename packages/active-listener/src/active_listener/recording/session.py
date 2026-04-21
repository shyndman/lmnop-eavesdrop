"""Recording-lifetime state and resource ownership for active-listener."""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass

from structlog.stdlib import BoundLogger

from active_listener.app.ports import (
  ActiveListenerClient,
  ActiveListenerRuntimeError,
)
from active_listener.infra.keyboard import KeyboardInput, RecordingGrabRelease
from active_listener.recording.reducer import (
  RecordingReducerState,
  TranscriptionUpdate,
  append_segment_text,
  build_transcription_update,
  reduce_new_segments,
)
from eavesdrop.wire import TranscriptionMessage


@dataclass
class RecordingSession:
  """Own recording state, reducer state, and keyboard-grab resources."""

  keyboard: KeyboardInput
  client: ActiveListenerClient
  logger: BoundLogger
  _connection_last_id: int | None = None
  _recording_reducer_state: RecordingReducerState | None = None
  _recording_grab_stack: AsyncExitStack | None = None
  _release_recording_grab: RecordingGrabRelease | None = None

  @property
  def is_recording(self) -> bool:
    return self._recording_reducer_state is not None

  def reset_connection_cursor(self) -> None:
    self.logger.debug(
      "transcription connection cursor reset",
      previous_last_id=self._connection_last_id,
    )
    self._connection_last_id = None

  async def start_recording(self) -> None:
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
    self._recording_reducer_state = RecordingReducerState(last_id=self._connection_last_id)
    self.logger.debug(
      "recording session started",
      seeded_last_id=self._connection_last_id,
    )

  async def stop_recording(self) -> None:
    await self._exit_recording()

  async def finish_recording(self) -> RecordingReducerState:
    reducer_state = self._require_recording_reducer_state()
    await self._exit_recording()
    self.logger.debug(
      "recording session finished",
      last_id=reducer_state.last_id,
      committed_part_count=len(reducer_state.parts),
    )
    return reducer_state

  def ingest_live_transcription_message(
    self,
    message: TranscriptionMessage,
  ) -> TranscriptionUpdate | None:
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
    append_segment_text(state.parts, reduction.segments)
    state.last_id = reduction.last_id
    self._connection_last_id = reduction.last_id
    transcription_update = build_transcription_update(reduction)
    incomplete_segment = reduction.incomplete_segment
    self.logger.debug(
      "transcription window reduced",
      stream=message.stream,
      new_last_id=reduction.last_id,
      new_segment_ids=[segment.id for segment in reduction.segments],
      incomplete_segment_id=None if incomplete_segment is None else incomplete_segment.id,
      incomplete_segment_text=None
      if incomplete_segment is None
      else incomplete_segment.text.strip(),
      committed_segment_count=len(reduction.segments),
      committed_part_count=len(state.parts),
      overlay_update=transcription_update is not None,
    )
    return transcription_update

  async def _exit_recording(self) -> None:
    release_recording_grab = self._release_recording_grab
    grab_stack = self._recording_grab_stack

    if (release_recording_grab is None) != (grab_stack is None):
      raise ActiveListenerRuntimeError("recording grab ownership is inconsistent")

    self._release_recording_grab = None
    self._recording_grab_stack = None
    self._recording_reducer_state = None

    try:
      if release_recording_grab is not None:
        release_recording_grab()
    finally:
      if grab_stack is not None:
        await grab_stack.aclose()

    await self.client.stop_streaming()

  def _require_recording_reducer_state(self) -> RecordingReducerState:
    reducer_state = self._recording_reducer_state
    if reducer_state is None:
      raise ActiveListenerRuntimeError("recording finish requested without reducer state")
    return reducer_state
