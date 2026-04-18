"""Recording-lifetime state and resource ownership for active-listener."""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass

from active_listener.input import KeyboardInput, RecordingGrabRelease
from active_listener.reducer import (
  RecordingReducerState,
  append_segment_text,
  reduce_new_segments,
)
from active_listener.service_ports import (
  ActiveListenerClient,
  ActiveListenerLogger,
  ActiveListenerRuntimeError,
)
from eavesdrop.wire import TranscriptionMessage


@dataclass
class RecordingSession:
  """Own recording state, reducer state, and keyboard-grab resources."""

  keyboard: KeyboardInput
  client: ActiveListenerClient
  logger: ActiveListenerLogger
  _connection_last_id: int | None = None
  _recording_reducer_state: RecordingReducerState | None = None
  _recording_grab_stack: AsyncExitStack | None = None
  _release_recording_grab: RecordingGrabRelease | None = None

  @property
  def is_recording(self) -> bool:
    return self._recording_reducer_state is not None

  def reset_connection_cursor(self) -> None:
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

  async def stop_recording(self) -> None:
    await self._exit_recording()

  async def finish_recording(self) -> RecordingReducerState:
    reducer_state = self._require_recording_reducer_state()
    await self._exit_recording()
    return reducer_state

  def ingest_live_transcription_message(self, message: TranscriptionMessage) -> None:
    reducer_state = self._require_recording_reducer_state()
    self.ingest_transcription_message(reducer_state, message)

  def ingest_transcription_message(
    self,
    state: RecordingReducerState,
    message: TranscriptionMessage,
  ) -> None:
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
