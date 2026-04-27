from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Protocol

from structlog.stdlib import BoundLogger

from active_listener.app.ports import (
  ActiveListenerClient,
  ActiveListenerRewriteClient,
  ActiveListenerTranscriptHistoryStore,
)
from active_listener.app.signals import ClientSignal, KeyboardEventSignal, RuntimeSignal
from active_listener.app.state import (
  AppAction,
  AppActionDecision,
  ConnectionDecision,
  ForegroundPhase,
  decide_app_action,
  decide_client_event,
)
from active_listener.config.models import ActiveListenerConfig
from active_listener.infra.dbus import AppStateService
from active_listener.infra.emitter import TextEmitter
from active_listener.infra.keyboard import KeyboardControlEvent, KeyboardEventKind, KeyboardInput
from active_listener.recording.finalizer import RecordingFinalizer
from active_listener.recording.session import RecordingAudioBuffer, RecordingSession
from active_listener.recording.spectrum import SpectrumAnalyzer
from eavesdrop.client import (
  ConnectedEvent,
  DisconnectedEvent,
  ReconnectedEvent,
  ReconnectingEvent,
  TranscriptionEvent,
)

UNKNOWN_DISCONNECT_REASON = "unknown disconnect reason"


class SpectrumRuntime(Protocol):
  def start(self) -> asyncio.Task[None]: ...

  async def stop(self) -> None: ...


async def _publish_noop_spectrum(_bars: bytes) -> None:
  return None


def _build_noop_spectrum_analyzer() -> SpectrumAnalyzer:
  return SpectrumAnalyzer(publish=_publish_noop_spectrum)


@dataclass
class ActiveListenerService:
  """Long-running active-listener service instance."""

  config: ActiveListenerConfig
  keyboard: KeyboardInput
  client: ActiveListenerClient
  emitter: TextEmitter
  logger: BoundLogger
  rewrite_client: ActiveListenerRewriteClient
  history_store: ActiveListenerTranscriptHistoryStore
  dbus_service: AppStateService
  recording_audio_buffer: RecordingAudioBuffer
  spectrum_analyzer: SpectrumRuntime = field(default_factory=_build_noop_spectrum_analyzer)
  phase: ForegroundPhase = ForegroundPhase.IDLE
  disconnect_generation: int = 0
  _llm_available: bool = field(default=False, init=False)
  _llm_active: bool = field(default=False, init=False)
  _recording_session: RecordingSession = field(init=False)
  _recording_finalizer: RecordingFinalizer = field(init=False)
  _background_tasks: set[asyncio.Task[None]] = field(default_factory=set)
  _spectrum_task: asyncio.Task[None] | None = field(default=None, init=False)

  def __post_init__(self) -> None:
    self._llm_available = self.config.llm_rewrite is not None
    self._llm_active = self._llm_available
    self._recording_session = RecordingSession(
      keyboard=self.keyboard,
      client=self.client,
      logger=self.logger,
      audio_buffer=self.recording_audio_buffer,
    )
    self._recording_finalizer = RecordingFinalizer(
      config=self.config,
      client=self.client,
      emitter=self.emitter,
      logger=self.logger,
      rewrite_client=self.rewrite_client,
      history_store=self.history_store,
      dbus_service=self.dbus_service,
      ingest_transcription_message=self._recording_session.ingest_transcription_message,
      current_llm_available=self.current_llm_available,
      current_llm_active=self.current_llm_active,
      current_disconnect_generation=self._current_disconnect_generation,
    )

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
      asyncio.create_task(self._pump_keyboard_events(signal_queue)),
      asyncio.create_task(self._pump_client_events(signal_queue)),
    }

    try:
      while True:
        signal = await signal_queue.get()
        if isinstance(signal, KeyboardEventSignal):
          await self.handle_keyboard_event(signal.event)
        else:
          await self.handle_client_event(signal.event)
    finally:
      for task in producer_tasks:
        _ = task.cancel()
      if producer_tasks:
        _ = await asyncio.gather(*producer_tasks, return_exceptions=True)
      await self.close()

  async def close(self) -> None:
    await self._stop_spectrum_analysis()

    for task in list(self._background_tasks):
      _ = task.cancel()
    if self._background_tasks:
      _ = await asyncio.gather(*self._background_tasks, return_exceptions=True)
      self._background_tasks.clear()

    cleanup_errors: list[Exception] = []

    try:
      if self.phase is ForegroundPhase.RECORDING:
        try:
          self.phase = ForegroundPhase.IDLE
          await self._recording_session.stop_recording()
        except Exception as exc:
          cleanup_errors.append(exc)

      try:
        await self.client.disconnect()
      except Exception as exc:
        cleanup_errors.append(exc)

      try:
        await self.rewrite_client.close()
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

  def current_llm_available(self) -> bool:
    return self._llm_available

  def current_llm_active(self) -> bool:
    return self._llm_active

  async def set_llm_active(self, active: bool) -> bool:
    if not self._llm_available:
      raise RuntimeError("llm unavailable")

    if self._llm_active == active:
      return self._llm_active

    self._llm_active = active
    self.logger.info("llm runtime toggled", active=active)
    return self._llm_active

  async def handle_action(self, action: AppAction) -> AppActionDecision:
    decision = decide_app_action(self.phase, action)

    if decision is AppActionDecision.IGNORE:
      if action is AppAction.CANCEL:
        self.logger.info("cancel ignored while idle")
      return decision

    if decision is AppActionDecision.SUPPRESS_RECONNECTING_START:
      self.logger.info("recording start suppressed while reconnecting")
      return decision

    if decision is AppActionDecision.START_RECORDING:
      self._start_spectrum_analysis()
      try:
        await self.client.start_streaming()
        await self._recording_session.start_recording()
        self.phase = ForegroundPhase.RECORDING
        await self.dbus_service.set_state(self.phase)
        self.logger.info("recording started")
        return decision
      except Exception:
        await self._stop_spectrum_analysis()
        raise

    if decision is AppActionDecision.CANCEL_RECORDING:
      self.phase = ForegroundPhase.IDLE
      await self._stop_spectrum_analysis()
      await self._recording_session.stop_recording()
      await self.dbus_service.set_state(self.phase)
      await self.client.cancel_utterance()
      self._recording_session.reset_connection_cursor()
      self.logger.info("recording cancelled")
      return decision

    self.phase = ForegroundPhase.IDLE
    finished_recording = await self._recording_session.finish_recording()
    await self._stop_spectrum_analysis()
    await self.dbus_service.set_state(self.phase)
    finalization_task = asyncio.create_task(
      self._recording_finalizer.finalize_recording(
        disconnect_generation=self.disconnect_generation,
        finished_recording=finished_recording,
      )
    )
    self._background_tasks.add(finalization_task)
    finalization_task.add_done_callback(self._background_tasks.discard)
    self.logger.info("recording finished", background_finalization=True)
    return decision

  async def handle_keyboard_event(self, event: KeyboardControlEvent) -> None:
    if event.kind is KeyboardEventKind.ESCAPE_DOWN:
      _ = await self.handle_action(AppAction.CANCEL)
      return

    if self.phase is not ForegroundPhase.RECORDING:
      if event.kind is KeyboardEventKind.CAPSLOCK_DOWN:
        _ = await self.handle_action(AppAction.START_OR_FINISH)
      return

    if event.kind is KeyboardEventKind.CAPSLOCK_DOWN:
      await self._recording_session.handle_capslock_down(event.received_monotonic_s)
      return

    if event.kind is KeyboardEventKind.CAPSLOCK_UP:
      should_finish_recording = await self._recording_session.handle_capslock_up(
        event.received_monotonic_s
      )
      if should_finish_recording:
        _ = await self.handle_action(AppAction.START_OR_FINISH)

  async def handle_client_event(
    self,
    event: (
      ConnectedEvent | DisconnectedEvent | ReconnectingEvent | ReconnectedEvent | TranscriptionEvent
    ),
  ) -> None:
    if isinstance(event, TranscriptionEvent):
      await self._handle_transcription_event(event)
      return

    if isinstance(event, ConnectedEvent | ReconnectedEvent):
      self._recording_session.reset_connection_cursor()

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
    disconnected_event = event
    assert isinstance(disconnected_event, DisconnectedEvent)
    disconnect_reason = disconnected_event.reason or UNKNOWN_DISCONNECT_REASON

    if decision is ConnectionDecision.ABORT_RECORDING:
      self.phase = ForegroundPhase.RECONNECTING
      await self._stop_spectrum_analysis()
      await self._recording_session.stop_recording()
      await self.dbus_service.set_state(self.phase)
      await self.dbus_service.recording_aborted(disconnect_reason)
      await self.dbus_service.reconnecting()
      self.logger.warning(
        "recording aborted by disconnect",
        stream=disconnected_event.stream,
        reason=disconnect_reason,
      )
      return

    await self.dbus_service.set_state(self.phase)

    self.logger.warning(
      "client disconnected",
      stream=disconnected_event.stream,
      reason=disconnect_reason,
    )

  async def _handle_transcription_event(self, event: TranscriptionEvent) -> None:
    message = event.message
    self.logger.debug(
      "live transcription event received",
      stream=message.stream,
      phase=self.phase.value,
      flush_complete=message.flush_complete is True,
      segment_count=len(message.segments),
    )

    if self.phase is not ForegroundPhase.RECORDING:
      self.logger.debug(
        "live transcription event ignored",
        stream=message.stream,
        phase=self.phase.value,
      )
      return

    transcription_update = self._recording_session.ingest_live_transcription_message(message)
    if transcription_update is None:
      self.logger.debug(
        "live transcription event reduced without overlay update",
        stream=message.stream,
      )
      return

    self.logger.debug(
      "publishing live transcription update",
      stream=message.stream,
      runs=transcription_update.runs,
    )
    await self.dbus_service.transcription_updated(transcription_update.runs)
    self.logger.debug(
      "live transcription update published",
      stream=message.stream,
      run_count=len(transcription_update.runs),
    )

  async def _pump_keyboard_events(self, signal_queue: asyncio.Queue[RuntimeSignal]) -> None:
    async for event in self.keyboard.events():
      await signal_queue.put(KeyboardEventSignal(event=event))

  async def _pump_client_events(self, signal_queue: asyncio.Queue[RuntimeSignal]) -> None:
    async for event in self.client:
      await signal_queue.put(ClientSignal(event=event))

  def _current_disconnect_generation(self) -> int:
    return self.disconnect_generation

  def _start_spectrum_analysis(self) -> None:
    spectrum_task = self.spectrum_analyzer.start()
    if spectrum_task is self._spectrum_task:
      return

    self._spectrum_task = spectrum_task
    self._background_tasks.add(spectrum_task)
    spectrum_task.add_done_callback(self._background_tasks.discard)
    spectrum_task.add_done_callback(self._clear_spectrum_task)

  async def _stop_spectrum_analysis(self) -> None:
    await self.spectrum_analyzer.stop()
    if self._spectrum_task is not None:
      self._background_tasks.discard(self._spectrum_task)
      self._spectrum_task = None

  def _clear_spectrum_task(self, task: asyncio.Task[None]) -> None:
    if self._spectrum_task is task:
      self._spectrum_task = None
