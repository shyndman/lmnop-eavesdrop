from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Protocol

from structlog.stdlib import BoundLogger

from active_listener.app.ports import (
  ActiveListenerClient,
  ActiveListenerRewriteClient,
)
from active_listener.app.signals import AppActionSignal, ClientSignal, RuntimeSignal
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
from active_listener.infra.keyboard import KeyboardInput
from active_listener.recording.finalizer import RecordingFinalizer
from active_listener.recording.session import RecordingSession
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
  dbus_service: AppStateService
  spectrum_analyzer: SpectrumRuntime = field(default_factory=_build_noop_spectrum_analyzer)
  phase: ForegroundPhase = ForegroundPhase.IDLE
  disconnect_generation: int = 0
  _recording_session: RecordingSession = field(init=False)
  _recording_finalizer: RecordingFinalizer = field(init=False)
  _background_tasks: set[asyncio.Task[None]] = field(default_factory=set)
  _spectrum_task: asyncio.Task[None] | None = field(default=None, init=False)

  def __post_init__(self) -> None:
    self._recording_session = RecordingSession(
      keyboard=self.keyboard,
      client=self.client,
      logger=self.logger,
    )
    self._recording_finalizer = RecordingFinalizer(
      config=self.config,
      client=self.client,
      emitter=self.emitter,
      logger=self.logger,
      rewrite_client=self.rewrite_client,
      dbus_service=self.dbus_service,
      ingest_transcription_message=self._recording_session.ingest_transcription_message,
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
      asyncio.create_task(self._pump_keyboard_actions(signal_queue)),
      asyncio.create_task(self._pump_client_events(signal_queue)),
    }

    try:
      while True:
        signal = await signal_queue.get()
        if isinstance(signal, AppActionSignal):
          _ = await self.handle_action(signal.action)
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
      self.logger.info("recording cancelled")
      return decision

    self.phase = ForegroundPhase.IDLE
    reducer_state = await self._recording_session.finish_recording()
    await self._stop_spectrum_analysis()
    await self.dbus_service.set_state(self.phase)
    finalization_task = asyncio.create_task(
      self._recording_finalizer.finalize_recording(
        disconnect_generation=self.disconnect_generation,
        reducer_state=reducer_state,
      )
    )
    self._background_tasks.add(finalization_task)
    finalization_task.add_done_callback(self._background_tasks.discard)
    self.logger.info("recording finished", background_finalization=True)
    return decision

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

    completed_segments = [
      (segment.id, segment.text) for segment in transcription_update.completed_segments
    ]
    incomplete_segment = transcription_update.incomplete_segment
    self.logger.debug(
      "publishing live transcription update",
      stream=message.stream,
      completed_segments=completed_segments,
      incomplete_segment=(incomplete_segment.id, incomplete_segment.text),
    )
    await self.dbus_service.transcription_updated(
      completed_segments=completed_segments,
      incomplete_segment=(incomplete_segment.id, incomplete_segment.text),
    )
    self.logger.debug(
      "live transcription update published",
      stream=message.stream,
      completed_segment_count=len(completed_segments),
      incomplete_segment_id=incomplete_segment.id,
    )

  async def _pump_keyboard_actions(self, signal_queue: asyncio.Queue[RuntimeSignal]) -> None:
    async for action in self.keyboard.actions():
      await signal_queue.put(AppActionSignal(action=action))

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
