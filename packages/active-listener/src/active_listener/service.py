from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Protocol

import active_listener.rewrite as rewrite_module
from active_listener.dbus_service import AppStateService
from active_listener.emitter import TextEmitter
from active_listener.input import KeyboardInput, RecordingGrabRelease
from active_listener.reducer import (
  RecordingReducerState,
  append_segment_text,
  reduce_new_segments,
  render_text,
)
from active_listener.rewrite import (
  RewriteClientError,
  RewriteClientTimeoutError,
  RewritePromptError,
)
from active_listener.settings import ActiveListenerConfig
from active_listener.state import (
  ConnectionDecision,
  ForegroundPhase,
  KeyboardAction,
  KeyboardDecision,
  decide_client_event,
  decide_keyboard_action,
)
from eavesdrop.client import (
  ConnectedEvent,
  DisconnectedEvent,
  ReconnectedEvent,
  ReconnectingEvent,
  TranscriptionEvent,
)
from eavesdrop.wire import TranscriptionMessage

UNKNOWN_DISCONNECT_REASON = "unknown disconnect reason"


class ActiveListenerClient(Protocol):
  """Protocol for the live transcription client dependency."""

  async def connect(self) -> None:
    """Establish the live connection."""
    ...

  async def disconnect(self) -> None:
    """Close the live connection."""
    ...

  async def start_streaming(self) -> None:
    """Begin microphone capture and upstream streaming."""
    ...

  async def stop_streaming(self) -> None:
    """Stop microphone capture for the active recording."""
    ...

  async def cancel_utterance(self) -> None:
    """Discard the current live utterance without closing the session."""
    ...

  async def flush(self, *, force_complete: bool = True) -> TranscriptionMessage:
    """Request a committed transcription flush from the server."""
    ...

  def __aiter__(
    self,
  ) -> AsyncIterator[
    ConnectedEvent | DisconnectedEvent | ReconnectingEvent | ReconnectedEvent | TranscriptionEvent
  ]:
    """Iterate ordered client lifecycle events."""
    ...


class ActiveListenerLogger(Protocol):
  """Minimal structured logger API used by active-listener."""

  def info(self, event: str, **kwargs: object) -> None:
    """Emit an informational event."""
    ...

  def warning(self, event: str, **kwargs: object) -> None:
    """Emit a warning event."""
    ...

  def exception(self, event: str, **kwargs: object) -> None:
    """Emit an exception event with stack trace."""


class ActiveListenerRewriteClient(Protocol):
  async def rewrite_text(
    self,
    *,
    model_name: str,
    instructions: str,
    transcript: str,
  ) -> str: ...


@dataclass(frozen=True)
class KeyboardSignal:
  """Signal carrying a workstation keyboard action into app policy."""

  action: KeyboardAction


@dataclass(frozen=True)
class ClientSignal:
  """Signal carrying a live client event into app policy."""

  event: (
    ConnectedEvent | DisconnectedEvent | ReconnectingEvent | ReconnectedEvent | TranscriptionEvent
  )


RuntimeSignal = KeyboardSignal | ClientSignal


class ActiveListenerRuntimeError(RuntimeError):
  """Raised when the service cannot satisfy runtime prerequisites."""


@dataclass
class ActiveListenerService:
  """Long-running active-listener service instance."""

  config: ActiveListenerConfig
  keyboard: KeyboardInput
  client: ActiveListenerClient
  emitter: TextEmitter
  logger: ActiveListenerLogger
  rewrite_client: ActiveListenerRewriteClient
  dbus_service: AppStateService
  phase: ForegroundPhase = ForegroundPhase.IDLE
  disconnect_generation: int = 0
  _recording_reducer_state: RecordingReducerState | None = None
  _recording_grab_stack: AsyncExitStack | None = None
  _release_recording_grab: RecordingGrabRelease | None = None
  _finalization_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
  _background_tasks: set[asyncio.Task[None]] = field(default_factory=set)

  async def run(self) -> None:
    self.logger.info(
      "active-listener ready",
      keyboard_name=self.config.keyboard_name,
      host=self.config.host,
      port=self.config.port,
      audio_device=self.config.audio_device,
    )

    signal_queue: asyncio.Queue[RuntimeSignal] = asyncio.Queue()
    producer_tasks = {
      asyncio.create_task(self._pump_keyboard_actions(signal_queue)),
      asyncio.create_task(self._pump_client_events(signal_queue)),
    }

    try:
      while True:
        signal = await signal_queue.get()
        if isinstance(signal, KeyboardSignal):
          await self.handle_keyboard_action(signal.action)
        else:
          await self.handle_client_event(signal.event)
    finally:
      for task in producer_tasks:
        _ = task.cancel()
      if producer_tasks:
        _ = await asyncio.gather(*producer_tasks, return_exceptions=True)
      await self.close()

  async def close(self) -> None:
    for task in list(self._background_tasks):
      _ = task.cancel()
    if self._background_tasks:
      _ = await asyncio.gather(*self._background_tasks, return_exceptions=True)
      self._background_tasks.clear()

    cleanup_errors: list[Exception] = []

    try:
      if self.phase is ForegroundPhase.RECORDING:
        try:
          await self._exit_recording(next_phase=ForegroundPhase.IDLE)
        except Exception as exc:
          cleanup_errors.append(exc)

      try:
        await self.client.disconnect()
      except Exception as exc:
        cleanup_errors.append(exc)
    finally:
      self.keyboard.close()

    if len(cleanup_errors) == 1:
      raise cleanup_errors[0]
    if cleanup_errors:
      raise ExceptionGroup("active-listener close failed", cleanup_errors)

  async def wait_for_background_tasks(self) -> None:
    if self._background_tasks:
      _ = await asyncio.gather(*list(self._background_tasks), return_exceptions=True)

  async def handle_keyboard_action(self, action: KeyboardAction) -> None:
    decision = decide_keyboard_action(self.phase, action)

    if decision is KeyboardDecision.IGNORE:
      if action is KeyboardAction.CANCEL:
        self.logger.info("cancel ignored while idle")
      return

    if decision is KeyboardDecision.SUPPRESS_RECONNECTING_START:
      self.logger.info("recording start suppressed while reconnecting")
      return

    if decision is KeyboardDecision.START_RECORDING:
      await self.client.start_streaming()
      await self._enter_recording()
      await self.dbus_service.set_state(self.phase)
      self.logger.info("recording started")
      return

    if decision is KeyboardDecision.CANCEL_RECORDING:
      await self._exit_recording(next_phase=ForegroundPhase.IDLE)
      await self.dbus_service.set_state(self.phase)
      await self.client.cancel_utterance()
      self.logger.info("recording cancelled")
      return

    reducer_state = self._require_recording_reducer_state()

    await self._exit_recording(next_phase=ForegroundPhase.IDLE)
    await self.dbus_service.set_state(self.phase)
    finalization_task = asyncio.create_task(
      self._finalize_recording(
        disconnect_generation=self.disconnect_generation,
        reducer_state=reducer_state,
      )
    )
    self._background_tasks.add(finalization_task)
    finalization_task.add_done_callback(self._background_tasks.discard)
    self.logger.info("recording finished", background_finalization=True)

  async def handle_client_event(
    self,
    event: (
      ConnectedEvent | DisconnectedEvent | ReconnectingEvent | ReconnectedEvent | TranscriptionEvent
    ),
  ) -> None:
    if isinstance(event, TranscriptionEvent):
      reducer_state = self._recording_reducer_state
      if self.phase is ForegroundPhase.RECORDING and reducer_state is not None:
        self._ingest_transcription_message(state=reducer_state, message=event.message)
      return

    decision = decide_client_event(self.phase, event)

    if decision is ConnectionDecision.IGNORE:
      return

    if decision is ConnectionDecision.CONNECTED:
      self.phase = ForegroundPhase.IDLE
      await self.dbus_service.set_state(self.phase)
      self.logger.info("client connected", stream=event.stream)
      return

    if decision is ConnectionDecision.RECONNECTING:
      self.phase = ForegroundPhase.RECONNECTING
      await self.dbus_service.set_state(self.phase)
      await self.dbus_service.reconnecting()
      reconnect_event = event
      if isinstance(reconnect_event, ReconnectingEvent):
        self.logger.warning(
          "client reconnecting",
          stream=reconnect_event.stream,
          attempt=reconnect_event.attempt,
          retry_delay_s=reconnect_event.retry_delay_s,
        )
      return

    if decision is ConnectionDecision.RECONNECTED:
      self.phase = ForegroundPhase.IDLE
      await self.dbus_service.set_state(self.phase)
      await self.dbus_service.reconnected()
      reconnected_event = event
      assert isinstance(reconnected_event, ReconnectedEvent)
      self.logger.info("client reconnected", stream=reconnected_event.stream)
      return

    self.disconnect_generation += 1
    self.phase = ForegroundPhase.RECONNECTING
    await self.dbus_service.set_state(self.phase)
    disconnected_event = event
    assert isinstance(disconnected_event, DisconnectedEvent)
    disconnect_reason = disconnected_event.reason or UNKNOWN_DISCONNECT_REASON

    if decision is ConnectionDecision.ABORT_RECORDING:
      await self._exit_recording(next_phase=ForegroundPhase.RECONNECTING)
      await self.dbus_service.set_state(self.phase)
      await self.dbus_service.recording_aborted(disconnect_reason)
      await self.dbus_service.reconnecting()
      self.logger.warning(
        "recording aborted by disconnect",
        stream=disconnected_event.stream,
        reason=disconnect_reason,
      )
      return

    self.logger.warning(
      "client disconnected",
      stream=disconnected_event.stream,
      reason=disconnect_reason,
    )

  async def _enter_recording(self) -> None:
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
    self._recording_reducer_state = RecordingReducerState()
    self.phase = ForegroundPhase.RECORDING

  async def _exit_recording(self, *, next_phase: ForegroundPhase) -> None:
    release_recording_grab = self._release_recording_grab
    grab_stack = self._recording_grab_stack

    if (release_recording_grab is None) != (grab_stack is None):
      raise ActiveListenerRuntimeError("recording grab ownership is inconsistent")

    self._release_recording_grab = None
    self._recording_grab_stack = None
    self._recording_reducer_state = None
    self.phase = next_phase

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

  async def _pump_keyboard_actions(self, signal_queue: asyncio.Queue[RuntimeSignal]) -> None:
    async for action in self.keyboard.actions():
      await signal_queue.put(KeyboardSignal(action=action))

  async def _pump_client_events(self, signal_queue: asyncio.Queue[RuntimeSignal]) -> None:
    async for event in self.client:
      await signal_queue.put(ClientSignal(event=event))

  def _ingest_transcription_message(
    self,
    *,
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

  async def _finalize_recording(
    self,
    *,
    disconnect_generation: int,
    reducer_state: RecordingReducerState,
  ) -> None:
    async with self._finalization_lock:
      try:
        message = await self.client.flush(force_complete=True)
      except Exception:
        self.logger.exception("recording finalization failed")
        return

      if self.disconnect_generation != disconnect_generation:
        self.logger.warning("skipping emission after disconnect", stream=message.stream)
        return

      self._ingest_transcription_message(state=reducer_state, message=message)
      raw_text = render_text(reducer_state.parts)
      if not raw_text:
        self.logger.info("recording finalized without committed text", stream=message.stream)
        return

      self.logger.info("finalized raw transcript", stream=message.stream, raw_text=raw_text)

      text_to_emit = raw_text
      emitted_text_source = "raw"

      if self.config.llm_rewrite.enabled:
        prompt_path: str | None = None
        try:
          loaded_prompt = rewrite_module.load_active_listener_rewrite_prompt(
            self.config.llm_rewrite.prompt_path
          )
          prompt = loaded_prompt.prompt
          prompt_path = str(loaded_prompt.prompt_path)
          self.logger.info(
            "rewrite prompt loaded",
            stream=message.stream,
            prompt_path=prompt_path,
            model_name=prompt.model_name,
            prompt_metadata=prompt.metadata,
          )
          self.logger.info(
            "rewrite prompt rendered",
            stream=message.stream,
            prompt_path=prompt_path,
            model_name=prompt.model_name,
            instructions=prompt.instructions,
          )
          self.logger.info(
            "rewrite started",
            stream=message.stream,
            base_url=self.config.llm_rewrite.base_url,
            prompt_path=prompt_path,
            model_name=prompt.model_name,
            raw_text=raw_text,
          )
          text_to_emit = await self.rewrite_client.rewrite_text(
            model_name=prompt.model_name,
            instructions=prompt.instructions,
            transcript=raw_text,
          )
          emitted_text_source = "rewritten"
          self.logger.info(
            "rewrite succeeded",
            stream=message.stream,
            prompt_path=prompt_path,
            model_name=prompt.model_name,
            raw_text=raw_text,
            rewritten_text=text_to_emit,
          )
        except RewritePromptError as exc:
          self.logger.exception(
            "rewrite prompt load failed",
            stream=message.stream,
            prompt_path=str(exc.prompt_path) if exc.prompt_path is not None else None,
          )
          self.logger.warning(
            "rewrite raw fallback selected",
            stream=message.stream,
            raw_text=raw_text,
          )
        except RewriteClientTimeoutError:
          self.logger.exception(
            "rewrite timed out",
            stream=message.stream,
            prompt_path=prompt_path,
            timeout_s=self.config.llm_rewrite.timeout_s,
          )
          self.logger.warning(
            "rewrite raw fallback selected",
            stream=message.stream,
            raw_text=raw_text,
          )
        except RewriteClientError:
          self.logger.exception(
            "rewrite model failed",
            stream=message.stream,
            prompt_path=prompt_path,
          )
          self.logger.warning(
            "rewrite raw fallback selected",
            stream=message.stream,
            raw_text=raw_text,
          )

      try:
        self.emitter.emit_text(text_to_emit)
      except Exception:
        self.logger.exception("text emission failed", stream=message.stream)
        return

      self.logger.info(
        "text emitted",
        stream=message.stream,
        emitted_text=text_to_emit,
        text_length=len(text_to_emit),
        source=emitted_text_source,
      )
