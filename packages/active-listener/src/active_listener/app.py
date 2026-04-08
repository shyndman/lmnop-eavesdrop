"""Application assembly and runtime policy for active-listener."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import ClassVar, Protocol

from pydantic import BaseModel, ConfigDict, Field

from active_listener.dbus_service import AppStateService, NoopDbusService
from active_listener.emitter import PydotoolTextEmitter, TextEmitter
from active_listener.input import KeyboardInput, RecordingGrabRelease, resolve_keyboard
from active_listener.reducer import (
  RecordingReducerState,
  append_segment_text,
  reduce_new_segments,
  render_text,
)
from active_listener.rewrite import (
  LlmRewriteClient,
  RewriteClientError,
  RewriteClientTimeoutError,
  RewritePromptError,
  load_rewrite_prompt,
)
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
  EavesdropClient,
  ReconnectedEvent,
  ReconnectingEvent,
  TranscriptionEvent,
)
from eavesdrop.common import get_logger
from eavesdrop.wire import TranscriptionMessage

UNKNOWN_DISCONNECT_REASON = "unknown disconnect reason"


class LlmRewriteConfig(BaseModel):
  model_config: ClassVar[ConfigDict] = ConfigDict(strict=True)

  enabled: bool
  base_url: str = Field(min_length=1)
  timeout_s: int = Field(default=30, ge=1)
  prompt_path: str = Field(min_length=1)


class ActiveListenerConfig(BaseModel):
  """Validated runtime configuration for the active-listener service.

  :param keyboard_name: Exact evdev device name to capture during dictation.
  :type keyboard_name: str
  :param host: Eavesdrop server hostname.
  :type host: str
  :param port: Eavesdrop server port.
  :type port: int
  :param audio_device: PortAudio capture device name passed to the client.
  :type audio_device: str
  :param ydotool_socket: Optional custom ydotool daemon socket path.
  :type ydotool_socket: str | None
  :param llm_rewrite: Nested rewrite configuration.
  :type llm_rewrite: LlmRewriteConfig
  """

  model_config: ClassVar[ConfigDict] = ConfigDict(strict=True)

  keyboard_name: str = Field(min_length=1)
  host: str = Field(min_length=1)
  port: int = Field(ge=1, le=65535)
  audio_device: str = Field(min_length=1)
  ydotool_socket: str | None = None
  llm_rewrite: LlmRewriteConfig


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
  ) -> str:
    ...
    ...


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
  """Long-running active-listener service instance.

  :param config: Validated runtime configuration.
  :type config: ActiveListenerConfig
  :param keyboard: Workstation keyboard boundary used for hotkey input and grabs.
  :type keyboard: KeyboardInput
  :param client: Live transcriber client boundary.
  :type client: ActiveListenerClient
  :param emitter: Text emission boundary.
  :type emitter: TextEmitter
  :param logger: Structured logger for service lifecycle events.
  :type logger: ActiveListenerLogger
  """

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
    """Run the steady-state hotkey and client-event loop.

    :returns: This coroutine only returns on cancellation or unrecoverable producer exit.
    :rtype: None
    """

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
    """Release service-owned resources.

    :returns: None
    :rtype: None
    """

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
    """Await all currently scheduled background finalization tasks.

    :returns: None
    :rtype: None
    """

    if self._background_tasks:
      _ = await asyncio.gather(*list(self._background_tasks), return_exceptions=True)

  async def handle_keyboard_action(self, action: KeyboardAction) -> None:
    """Apply a hotkey action to the foreground policy.

    :param action: Normalized hotkey action from the input boundary.
    :type action: KeyboardAction
    :returns: None
    :rtype: None
    """

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
    """Apply a live-client lifecycle event to the foreground policy.

    :param event: Live client event.
    :type event: ConnectedEvent | DisconnectedEvent | ReconnectingEvent |
                 ReconnectedEvent | TranscriptionEvent
    :returns: None
    :rtype: None
    """

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
    """Acquire recording-owned state and keyboard grab.

    :returns: None
    :rtype: None
    :raises ActiveListenerRuntimeError: If recording grab ownership already exists.
    """

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
    """End the local recording scope before awaited downstream cleanup.

    This helper exists because a leaked evdev grab is operationally dangerous.
    If the service leaves the workstation keyboard grabbed while it awaits
    network shutdown or finalization work, the desktop can lose normal input
    exactly when the process is already on a failure path. ``run()`` still has
    outer cleanup, but that is a last resort. Recording exit must release the
    keyboard locally, in one place, before any slow or exception-prone await.

    The method tells the truth about local state first: it clears reducer and
    grab ownership, updates the foreground phase, then releases the keyboard in
    ``finally`` by invoking the early-release callback and closing the stored
    async context manager. Only after that does it await ``stop_streaming()``.
    That ordering keeps cancel, finish, disconnect-abort, and shutdown on the
    same release-first invariant even if later work raises.

    :param next_phase: Foreground phase visible after local recording cleanup.
    :type next_phase: ForegroundPhase
    :returns: None
    :rtype: None
    :raises ActiveListenerRuntimeError: If recording grab ownership is inconsistent.
    """

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
    """Return the reducer state required to finalize a recording.

    :returns: Active recording reducer state.
    :rtype: RecordingReducerState
    :raises ActiveListenerRuntimeError: If finish is requested without reducer state.
    """

    reducer_state = self._recording_reducer_state
    if reducer_state is None:
      raise ActiveListenerRuntimeError("recording finish requested without reducer state")
    return reducer_state

  async def _pump_keyboard_actions(self, signal_queue: asyncio.Queue[RuntimeSignal]) -> None:
    """Forward normalized workstation input into the internal signal queue.

    :param signal_queue: Service-owned queue of normalized runtime signals.
    :type signal_queue: asyncio.Queue[RuntimeSignal]
    :returns: None
    :rtype: None
    """

    async for action in self.keyboard.actions():
      await signal_queue.put(KeyboardSignal(action=action))

  async def _pump_client_events(self, signal_queue: asyncio.Queue[RuntimeSignal]) -> None:
    """Forward live client lifecycle events into the internal signal queue.

    :param signal_queue: Service-owned queue of normalized runtime signals.
    :type signal_queue: asyncio.Queue[RuntimeSignal]
    :returns: None
    :rtype: None
    """

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
    """Flush and emit a finished recording without re-grabbing the keyboard.

    :param disconnect_generation: Disconnect epoch observed when finish was requested.
    :type disconnect_generation: int
    :param reducer_state: Recording-owned transcription reducer state handed off from foreground.
    :type reducer_state: RecordingReducerState
    :returns: None
    :rtype: None
    """

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
        try:
          prompt = load_rewrite_prompt(self.config.llm_rewrite.prompt_path)
          self.logger.info(
            "rewrite prompt loaded",
            stream=message.stream,
            prompt_path=self.config.llm_rewrite.prompt_path,
            model_name=prompt.model_name,
            prompt_metadata=prompt.metadata,
          )
          self.logger.info(
            "rewrite prompt rendered",
            stream=message.stream,
            prompt_path=self.config.llm_rewrite.prompt_path,
            model_name=prompt.model_name,
            instructions=prompt.instructions,
          )
          self.logger.info(
            "rewrite started",
            stream=message.stream,
            base_url=self.config.llm_rewrite.base_url,
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
            model_name=prompt.model_name,
            raw_text=raw_text,
            rewritten_text=text_to_emit,
          )
        except RewritePromptError:
          self.logger.exception(
            "rewrite prompt load failed",
            stream=message.stream,
            prompt_path=self.config.llm_rewrite.prompt_path,
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
            timeout_s=self.config.llm_rewrite.timeout_s,
          )
          self.logger.warning(
            "rewrite raw fallback selected",
            stream=message.stream,
            raw_text=raw_text,
          )
        except RewriteClientError:
          self.logger.exception("rewrite model failed", stream=message.stream)
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


async def create_service(
  config: ActiveListenerConfig,
  *,
  dbus_service: AppStateService | None = None,
  keyboard_resolver: Callable[[str], KeyboardInput] = resolve_keyboard,
  client_factory: Callable[[ActiveListenerConfig], ActiveListenerClient] | None = None,
  emitter_factory: Callable[[str | None], TextEmitter] | None = None,
  rewrite_client_factory: Callable[[LlmRewriteConfig], ActiveListenerRewriteClient] | None = None,
) -> ActiveListenerService:
  """Construct a fully initialized service instance.

  :param config: Validated runtime configuration.
  :type config: ActiveListenerConfig
  :param keyboard_resolver: Resolver for the exact-name keyboard dependency.
  :type keyboard_resolver: Callable[[str], KeyboardInput]
  :param client_factory: Factory for the live transcriber dependency.
  :type client_factory: Callable[[ActiveListenerConfig], ActiveListenerClient] | None
  :param emitter_factory: Factory for the text emitter dependency.
  :type emitter_factory: Callable[[str | None], TextEmitter] | None
  :returns: Ready-to-run service instance.
  :rtype: ActiveListenerService
  :raises ActiveListenerRuntimeError: If startup prerequisites cannot be satisfied.
  """

  logger = get_logger("al/app")
  resolved_dbus_service = dbus_service or NoopDbusService()
  resolved_client_factory = client_factory or build_client
  resolved_emitter_factory = emitter_factory or build_emitter
  resolved_rewrite_client_factory = rewrite_client_factory or build_rewrite_client

  try:
    keyboard = keyboard_resolver(config.keyboard_name)
  except Exception as exc:
    logger.exception("keyboard resolution failed", keyboard_name=config.keyboard_name)
    raise ActiveListenerRuntimeError(str(exc)) from exc

  try:
    emitter = resolved_emitter_factory(config.ydotool_socket)
    client = resolved_client_factory(config)
    rewrite_client = resolved_rewrite_client_factory(config.llm_rewrite)
    await client.connect()
  except Exception as exc:
    keyboard.close()
    logger.exception(
      "startup prerequisite failed",
      keyboard_name=config.keyboard_name,
      host=config.host,
      port=config.port,
    )
    raise ActiveListenerRuntimeError(str(exc)) from exc

  logger.info(
    "startup prerequisites satisfied",
    keyboard_name=config.keyboard_name,
    host=config.host,
    port=config.port,
  )
  await resolved_dbus_service.set_state(ForegroundPhase.IDLE)
  return ActiveListenerService(
    config=config,
    keyboard=keyboard,
    client=client,
    emitter=emitter,
    logger=logger,
    rewrite_client=rewrite_client,
    dbus_service=resolved_dbus_service,
  )


async def run_service(
  config: ActiveListenerConfig,
  *,
  dbus_service: AppStateService | None = None,
  keyboard_resolver: Callable[[str], KeyboardInput] = resolve_keyboard,
  client_factory: Callable[[ActiveListenerConfig], ActiveListenerClient] | None = None,
  emitter_factory: Callable[[str | None], TextEmitter] | None = None,
  rewrite_client_factory: Callable[[LlmRewriteConfig], ActiveListenerRewriteClient] | None = None,
) -> None:
  """Create and run the long-lived active-listener service.

  :param config: Validated runtime configuration.
  :type config: ActiveListenerConfig
  :param keyboard_resolver: Resolver for the exact-name keyboard dependency.
  :type keyboard_resolver: Callable[[str], KeyboardInput]
  :param client_factory: Factory for the live transcriber dependency.
  :type client_factory: Callable[[ActiveListenerConfig], ActiveListenerClient] | None
  :param emitter_factory: Factory for the text emitter dependency.
  :type emitter_factory: Callable[[str | None], TextEmitter] | None
  :returns: None
  :rtype: None
  """

  resolved_dbus_service = dbus_service or NoopDbusService()
  try:
    service = await create_service(
      config,
      dbus_service=resolved_dbus_service,
      keyboard_resolver=keyboard_resolver,
      client_factory=client_factory,
      emitter_factory=emitter_factory,
      rewrite_client_factory=rewrite_client_factory,
    )
  except Exception:
    await resolved_dbus_service.close()
    raise

  try:
    await service.run()
  finally:
    await resolved_dbus_service.close()


def build_client(config: ActiveListenerConfig) -> ActiveListenerClient:
  """Build the live transcriber client for the configured workstation.

  :param config: Validated runtime configuration.
  :type config: ActiveListenerConfig
  :returns: Configured live transcriber client.
  :rtype: ActiveListenerClient
  """

  return EavesdropClient.transcriber(
    host=config.host,
    port=config.port,
    audio_device=config.audio_device,
  )


def build_emitter(socket_path: str | None) -> TextEmitter:
  """Build and initialize the text emission boundary.

  :param socket_path: Optional custom ydotool daemon socket path.
  :type socket_path: str | None
  :returns: Initialized text emitter.
  :rtype: TextEmitter
  """

  emitter = PydotoolTextEmitter(socket_path=socket_path)
  emitter.initialize()
  return emitter


def build_rewrite_client(config: LlmRewriteConfig) -> ActiveListenerRewriteClient:
  return LlmRewriteClient(base_url=config.base_url, timeout_s=config.timeout_s)
