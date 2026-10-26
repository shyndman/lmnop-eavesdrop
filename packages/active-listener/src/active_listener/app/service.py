from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass, field

from structlog.stdlib import BoundLogger

from active_listener.app.overlay import publish_live_transcription
from active_listener.app.ports import (
  ActiveListenerClient,
  ActiveListenerRewriteClient,
  ActiveListenerTranscriptHistoryStore,
  MediaPlaybackController,
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
from active_listener.app.workstation import (
  SpectrumRuntime,
  WorkstationEffects,
  build_noop_spectrum_analyzer,
)
from active_listener.config.models import ActiveListenerConfig
from active_listener.infra.corrections import ActiveListenerCorrectionStore, CorrectionStore
from active_listener.infra.dbus import AppStateService
from active_listener.infra.emitter import TextEmitter
from active_listener.infra.keyboard import KeyboardControlEvent, KeyboardEventKind, KeyboardInput
from active_listener.infra.langfuse import RecordingTrace, end_recording_observation
from active_listener.recording.finalizer import RecordingFinalizer
from active_listener.recording.session import RecordingAudioBuffer, RecordingSession
from eavesdrop.client import (
  ConnectedEvent,
  DisconnectedEvent,
  LanguageDetectionEvent,
  LiveClientEvent,
  ReconnectedEvent,
  ReconnectingEvent,
  TranscriptionEvent,
)

UNKNOWN_DISCONNECT_REASON = "unknown disconnect reason"


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
  media_playback_controller: MediaPlaybackController
  recording_audio_buffer: RecordingAudioBuffer
  correction_store: ActiveListenerCorrectionStore = field(default_factory=CorrectionStore.default)
  spectrum_analyzer: SpectrumRuntime = field(default_factory=build_noop_spectrum_analyzer)
  phase: ForegroundPhase = ForegroundPhase.IDLE
  disconnect_generation: int = 0
  _llm_available: bool = field(default=False, init=False)
  _llm_active: bool = field(default=False, init=False)
  _trace: RecordingTrace = field(init=False)
  _workstation: WorkstationEffects = field(init=False)
  _recording_session: RecordingSession = field(init=False)
  _recording_finalizer: RecordingFinalizer = field(init=False)
  _background_tasks: set[asyncio.Task[None]] = field(default_factory=set)

  def __post_init__(self) -> None:
    self._trace = RecordingTrace(self.logger)
    self._workstation = WorkstationEffects(
      self.media_playback_controller, self.spectrum_analyzer, self.logger
    )
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
      current_llm_available=self.current_llm_available,
      current_llm_active=self.current_llm_active,
      current_disconnect_generation=self._current_disconnect_generation,
      correction_store=self.correction_store,
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
    cleanup_errors: list[Exception] = []

    try:
      await self._workstation.close()
    except Exception as exc:
      cleanup_errors.append(exc)

    for task in list(self._background_tasks):
      _ = task.cancel()
    if self._background_tasks:
      _ = await asyncio.gather(*self._background_tasks, return_exceptions=True)
      self._background_tasks.clear()

    try:
      if self.phase is ForegroundPhase.RECORDING:
        try:
          self.phase = ForegroundPhase.IDLE
          recording_observation = self._trace.take()
          try:
            await self._recording_session.stop_recording()
          finally:
            end_recording_observation(
              recording_observation,
              logger=self.logger,
              status_message="recording stopped during shutdown",
            )
            await self._workstation.resume_media()
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
      await self._recording_finalizer.wait_for_pending_flushes()

      recording_id = secrets.token_hex(8)
      correction_load_task = asyncio.create_task(self.correction_store.load_async())
      self._trace.start(recording_id=recording_id)
      await asyncio.sleep(0)
      await self._workstation.start()
      try:
        await self._recording_session.start_recording(
          recording_id,
          correction_load_task=correction_load_task,
        )
        await self.client.start_streaming(recording_id)
        self.phase = ForegroundPhase.RECORDING
        await self.dbus_service.set_state(self.phase)
        self.logger.info("recording started")
        self._trace.recording_event(
          name="active-listener-recording-started",
          metadata={"stream": self._trace.stream, "recording_id": recording_id},
        )
        return decision
      except Exception:
        recording_observation = self._trace.take()
        try:
          if self._recording_session.is_recording:
            await self._recording_session.stop_recording()
          else:
            if not correction_load_task.done():
              _ = correction_load_task.cancel()
            _ = await asyncio.gather(correction_load_task, return_exceptions=True)
          await self._workstation.stop_spectrum()
        finally:
          end_recording_observation(
            recording_observation,
            logger=self.logger,
            level="ERROR",
            status_message="recording start failed",
          )
          await self._workstation.resume_media()
        raise

    if decision is AppActionDecision.CANCEL_RECORDING:
      recording_id = self._recording_session.active_recording_id
      if recording_id is None:
        raise RuntimeError("cancel requested without active recording id")

      self.phase = ForegroundPhase.IDLE
      recording_observation = self._trace.take()
      observation_level: str | None = None
      observation_status_message = "recording cancelled"
      try:
        await self._workstation.stop_spectrum()
        await self._recording_session.stop_recording()
        await self.dbus_service.set_state(self.phase)
        await self.client.cancel_utterance(recording_id)
        self._recording_session.reset_connection_cursor()
        self.logger.info("recording cancelled")
        self._trace.recording_event(
          recording_observation=recording_observation,
          name="active-listener-recording-cancelled",
          metadata={"stream": self._trace.stream, "recording_id": recording_id},
        )
        return decision
      except Exception:
        observation_level = "ERROR"
        observation_status_message = "recording cancel failed"
        raise
      finally:
        end_recording_observation(
          recording_observation,
          logger=self.logger,
          level=observation_level,
          status_message=observation_status_message,
        )
        await self._workstation.resume_media()

    self.phase = ForegroundPhase.IDLE
    recording_observation = self._trace.take()
    try:
      finished_recording = await self._recording_session.finish_recording()
      await self._workstation.stop_spectrum()
      await self.dbus_service.set_state(self.phase)
      self._recording_finalizer.reserve_flush()
      try:
        finalization_task = asyncio.create_task(
          self._recording_finalizer.finalize_recording(
            disconnect_generation=self.disconnect_generation,
            finished_recording=finished_recording,
            recording_observation=recording_observation,
          )
        )
        recording_observation = None
      except Exception:
        self._recording_finalizer.cancel_reserved_flush()
        raise
      self._background_tasks.add(finalization_task)
      finalization_task.add_done_callback(self._background_tasks.discard)
      self.logger.info("recording finished", background_finalization=True)
      return decision
    except Exception:
      end_recording_observation(
        recording_observation,
        logger=self.logger,
        level="ERROR",
        status_message="recording finish failed",
      )
      raise
    finally:
      await self._workstation.resume_media()

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

  async def handle_client_event(self, event: LiveClientEvent) -> None:
    if isinstance(event, LanguageDetectionEvent):
      # Language metadata does not change the recording lifecycle or transcript.
      return

    if isinstance(event, TranscriptionEvent):
      self._trace.stream = event.stream
      await publish_live_transcription(
        event,
        phase=self.phase,
        recording=self._recording_session,
        dbus=self.dbus_service,
        logger=self.logger,
      )
      return

    self._trace.stream = event.stream

    if isinstance(event, ConnectedEvent | ReconnectedEvent):
      self._recording_session.reset_connection_cursor()

    decision = decide_client_event(self.phase, event)

    if isinstance(event, ConnectedEvent):
      self._trace.session_event(
        session_id=event.stream,
        name="active-listener-client-connected",
        metadata={"stream": event.stream},
      )

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
        self._trace.session_event(
          session_id=reconnect_event.stream,
          name="active-listener-client-reconnecting",
          metadata={
            "stream": reconnect_event.stream,
            "attempt": reconnect_event.attempt,
            "retry_delay_s": reconnect_event.retry_delay_s,
          },
        )
      return

    if decision is ConnectionDecision.RECONNECTED:
      self.phase = ForegroundPhase.IDLE
      await self.dbus_service.set_state(self.phase)
      await self.dbus_service.reconnected()
      reconnected_event = event
      assert isinstance(reconnected_event, ReconnectedEvent)
      self.logger.info("client reconnected", stream=reconnected_event.stream)
      self._trace.session_event(
        session_id=reconnected_event.stream,
        name="active-listener-client-reconnected",
        metadata={"stream": reconnected_event.stream},
      )
      return

    self.disconnect_generation += 1
    self.phase = ForegroundPhase.RECONNECTING
    disconnected_event = event
    assert isinstance(disconnected_event, DisconnectedEvent)
    disconnect_reason = disconnected_event.reason or UNKNOWN_DISCONNECT_REASON
    aborted_recording_id = self._recording_session.active_recording_id

    if decision is ConnectionDecision.ABORT_RECORDING:
      self.phase = ForegroundPhase.RECONNECTING
      try:
        await self._workstation.stop_spectrum()
        await self._recording_session.stop_recording()
        await self.dbus_service.set_state(self.phase)
        await self.dbus_service.recording_aborted(disconnect_reason)
        await self.dbus_service.reconnecting()
        self.logger.warning(
          "recording aborted by disconnect",
          stream=disconnected_event.stream,
          reason=disconnect_reason,
        )
        self._trace.recording_event(
          name="active-listener-recording-aborted",
          metadata={
            "stream": disconnected_event.stream,
            "recording_id": aborted_recording_id,
            "reason": disconnect_reason,
          },
        )
        return
      finally:
        await self._workstation.resume_media()

    await self.dbus_service.set_state(self.phase)

    self.logger.warning(
      "client disconnected",
      stream=disconnected_event.stream,
      reason=disconnect_reason,
    )
    self._trace.session_event(
      session_id=disconnected_event.stream,
      name="active-listener-client-disconnected",
      metadata={"stream": disconnected_event.stream, "reason": disconnect_reason},
    )

  async def _pump_keyboard_events(self, signal_queue: asyncio.Queue[RuntimeSignal]) -> None:
    async for event in self.keyboard.events():
      await signal_queue.put(KeyboardEventSignal(event=event))

  async def _pump_client_events(self, signal_queue: asyncio.Queue[RuntimeSignal]) -> None:
    async for event in self.client:
      await signal_queue.put(ClientSignal(event=event))

  def _current_disconnect_generation(self) -> int:
    return self.disconnect_generation
