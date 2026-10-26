from __future__ import annotations

import asyncio
import secrets

from structlog.stdlib import BoundLogger

from active_listener.app.overlay import update_dbus_transcription
from active_listener.app.ports import (
  ActiveListenerClient,
  ActiveListenerRewriteClient,
  ActiveListenerTranscriptHistoryStore,
  MediaPlaybackController,
)
from active_listener.app.recording_keyboard import RecordingKeyboard
from active_listener.app.signals import (
  ClientSignal,
  ConnectedSignal,
  DisconnectedSignal,
  KeyboardEventSignal,
  ReconnectedSignal,
  ReconnectingSignal,
  RuntimeSignal,
  bind_client_callbacks,
)
from active_listener.app.state import (
  AppAction,
  AppActionDecision,
  ForegroundPhase,
  decide_app_action,
)
from active_listener.app.workstation import (
  RecordingWorkstation,
  SpectrumAnalyzer,
  build_noop_spectrum_analyzer,
)
from active_listener.config.models import ActiveListenerConfig
from active_listener.infra.corrections import ActiveListenerCorrectionStore, CorrectionStore
from active_listener.infra.dbus import AppStateService
from active_listener.infra.emitter import TextEmitter
from active_listener.infra.keyboard import KeyboardControlEvent, KeyboardEventKind, KeyboardInput
from active_listener.infra.langfuse import RecordingTrace, end_recording_observation
from active_listener.recording.finalizer import RecordingFinalizer
from active_listener.recording.session import RecordingAudioBuffer, RecordingTranscript
from eavesdrop.wire import TranscriptionMessage

UNKNOWN_DISCONNECT_REASON = "unknown disconnect reason"


class ActiveListenerService:
  """Coordinate hardware, foreground recording, and transferred finalization.

  Constructor dependencies are initialized workstation resources. Client callbacks
  enqueue into the owned application queue before connect; run consumes it in order.
  Finalization owns each finished snapshot independently of later foreground state.
  """

  def __init__(
    self,
    *,
    config: ActiveListenerConfig,
    keyboard: KeyboardInput,
    client: ActiveListenerClient,
    emitter: TextEmitter,
    logger: BoundLogger,
    rewrite_client: ActiveListenerRewriteClient,
    history_store: ActiveListenerTranscriptHistoryStore,
    dbus_service: AppStateService,
    media_playback_controller: MediaPlaybackController,
    recording_audio_buffer: RecordingAudioBuffer,
    correction_store: ActiveListenerCorrectionStore | None = None,
    spectrum_analyzer: SpectrumAnalyzer | None = None,
    phase: ForegroundPhase = ForegroundPhase.IDLE,
  ) -> None:
    self.config: ActiveListenerConfig = config
    self.keyboard: KeyboardInput = keyboard
    self.client: ActiveListenerClient = client
    self.emitter: TextEmitter = emitter
    self.logger: BoundLogger = logger
    self.rewrite_client: ActiveListenerRewriteClient = rewrite_client
    self.history_store: ActiveListenerTranscriptHistoryStore = history_store
    self.dbus_service: AppStateService = dbus_service
    self.phase: ForegroundPhase = phase
    self.correction_store: ActiveListenerCorrectionStore = (
      correction_store or CorrectionStore.default()
    )
    self._llm_available: bool = config.llm_rewrite is not None
    self._llm_active: bool = self._llm_available
    self._background_tasks: set[asyncio.Task[None]] = set()
    self._signal_queue: asyncio.Queue[RuntimeSignal] = asyncio.Queue()
    bind_client_callbacks(client, self._signal_queue)
    self._initialize_recording(
      media_playback_controller,
      spectrum_analyzer or build_noop_spectrum_analyzer(),
      recording_audio_buffer,
    )

  def _initialize_recording(
    self,
    media: MediaPlaybackController,
    spectrum: SpectrumAnalyzer,
    audio: RecordingAudioBuffer,
  ) -> None:
    self._trace: RecordingTrace = RecordingTrace(self.logger)
    self._workstation: RecordingWorkstation = RecordingWorkstation(media, spectrum, self.logger)
    self._recording_transcript: RecordingTranscript = RecordingTranscript(
      client=self.client, logger=self.logger, audio_buffer=audio
    )
    self._recording_keyboard: RecordingKeyboard = RecordingKeyboard(
      self.keyboard, self._recording_transcript, self.logger
    )
    self._recording_finalizer: RecordingFinalizer = RecordingFinalizer(
      config=self.config,
      client=self.client,
      emitter=self.emitter,
      logger=self.logger,
      rewrite_client=self.rewrite_client,
      history_store=self.history_store,
      dbus_service=self.dbus_service,
      correction_store=self.correction_store,
    )

  async def run(self) -> None:
    """Consume hardware and client callback signals until cancellation, then close."""
    self.logger.info(
      "active-listener ready",
      keyboard_name=self.config.keyboard_name,
      host=self.config.host,
      port=self.config.port,
      audio_device=self.config.audio_device,
    )
    keyboard_task = asyncio.create_task(self._pump_keyboard_events())
    try:
      while True:
        signal = await self._signal_queue.get()
        match signal:
          case KeyboardEventSignal(event=event):
            await self.handle_keyboard_event(event)
          case _:
            await self.handle_client_event(signal)
    finally:
      _ = keyboard_task.cancel()
      _ = await asyncio.gather(keyboard_task, return_exceptions=True)
      await self.close()

  async def close(self) -> None:
    """End owned observations and release workstation resources despite failures."""
    errors: list[Exception] = []
    try:
      if self._recording_transcript.is_recording:
        self.phase = ForegroundPhase.IDLE
        observation = self._trace.transfer_and_clear_observation()
        try:
          await self._stop_capture()
        except Exception as exc:
          self.logger.exception("recording shutdown failed")
          errors.append(exc)
        finally:
          end_recording_observation(
            observation, logger=self.logger, status_message="recording stopped during shutdown"
          )
          await self._workstation.resume_media()
      # Disconnect fails any outstanding server flush immediately. Already accepted
      # results remain owned by finalization, which is drained before rewrite closes.
      for cleanup in (self._workstation.close, self.client.disconnect):
        try:
          await cleanup()
        except Exception as exc:
          self.logger.exception("listener resource cleanup failed")
          errors.append(exc)
      await self.wait_for_background_tasks()
      try:
        await self.rewrite_client.close()
      except Exception as exc:
        self.logger.exception("rewrite client cleanup failed")
        errors.append(exc)
    finally:
      self.keyboard.close()
    if len(errors) == 1:
      raise errors[0]
    if errors:
      raise ExceptionGroup("active-listener close failed", errors)

  async def wait_for_background_tasks(self) -> None:
    """Drain finalization work already accepted from finished recordings."""
    if self._background_tasks:
      _ = await asyncio.gather(*list(self._background_tasks), return_exceptions=True)

  def current_llm_available(self) -> bool:
    """Return whether a rewrite provider is configured."""
    return self._llm_available

  def current_llm_active(self) -> bool:
    """Return the current user-controlled rewrite choice."""
    return self._llm_active

  async def set_llm_active(self, active: bool) -> bool:
    """Set rewriting for recordings that finish after this call.

    :param active: Requested rewrite choice.
    :return: Effective choice.
    :raises RuntimeError: If no rewrite provider is configured.
    """
    if not self._llm_available:
      self.logger.error("rewrite toggle requested without a configured provider")
      raise RuntimeError("llm unavailable")
    self._llm_active = active
    self.logger.info("llm runtime toggled", active=active)
    return active

  async def handle_action(self, action: AppAction) -> AppActionDecision:
    """Apply a public workstation action using the current foreground phase.

    :param action: Start/finish or cancel request.
    :return: Applied or ignored policy decision.
    """
    decision = decide_app_action(self.phase, action)
    match decision:
      case AppActionDecision.START_RECORDING:
        await self._start_recording()
      case AppActionDecision.CANCEL_RECORDING:
        await self._cancel_recording()
      case AppActionDecision.FINISH_RECORDING:
        await self._finish_recording()
      case AppActionDecision.SUPPRESS_RECONNECTING_START:
        self.logger.info("recording start suppressed while reconnecting")
      case AppActionDecision.IGNORE:
        self.logger.info("recording action ignored", action=action.value)
    return decision

  async def _start_recording(self) -> None:
    await self._recording_finalizer.wait_for_pending_flushes()
    recording_id = secrets.token_hex(8)
    correction_task = asyncio.create_task(self.correction_store.load_async())
    self._trace.start(recording_id=recording_id)
    try:
      await self._workstation.start()
      await self._recording_keyboard.start()
      await self._recording_transcript.start_recording(
        recording_id, correction_load_task=correction_task
      )
      await self.client.start_streaming(recording_id)
      self.phase = ForegroundPhase.RECORDING
      await self.dbus_service.set_state(self.phase)
      self.logger.info("recording started")
      self._trace.recording_event(
        name="active-listener-recording-started",
        metadata={
          "session_id": self._trace.session_id,
          "recording_id": recording_id,
        },
      )
    except Exception:
      self.logger.exception("recording start failed")
      observation = self._trace.transfer_and_clear_observation()
      self.phase = ForegroundPhase.IDLE
      try:
        await self._recording_keyboard.stop()
        if self._recording_transcript.is_recording:
          await self._recording_transcript.stop_recording()
        else:
          _ = correction_task.cancel()
          _ = await asyncio.gather(correction_task, return_exceptions=True)
        await self._workstation.stop_spectrum()
      finally:
        end_recording_observation(
          observation, logger=self.logger, level="ERROR", status_message="recording start failed"
        )
        await self._workstation.resume_media()
      raise

  async def _stop_capture(self) -> None:
    try:
      await self._recording_keyboard.stop()
    finally:
      await self._recording_transcript.stop_recording()

  async def _cancel_recording(self) -> None:
    recording_id = self._recording_transcript.active_recording_id
    if recording_id is None:
      self.logger.error("cancel requested without active recording id")
      raise RuntimeError("cancel requested without active recording id")
    self.phase = ForegroundPhase.IDLE
    observation = self._trace.transfer_and_clear_observation()
    try:
      await self._workstation.stop_spectrum()
      await self._stop_capture()
      await self.dbus_service.set_state(self.phase)
      await self.client.cancel_utterance(recording_id)
      self._recording_transcript.reset_connection_cursor()
      self._trace.recording_event(
        recording_observation=observation,
        name="active-listener-recording-cancelled",
        metadata={"session_id": self._trace.session_id, "recording_id": recording_id},
      )
      self.logger.info("recording cancelled")
    except Exception:
      self.logger.exception("recording cancel failed")
      raise
    finally:
      end_recording_observation(
        observation, logger=self.logger, status_message="recording cancelled"
      )
      await self._workstation.resume_media()

  async def _finish_recording(self) -> None:
    self.phase = ForegroundPhase.IDLE
    observation = self._trace.transfer_and_clear_observation()
    try:
      await self._recording_keyboard.stop()
      recording = await self._recording_transcript.finish_recording()
      await self._workstation.stop_spectrum()
      await self.dbus_service.set_state(self.phase)
      self._recording_finalizer.reserve_flush()
      try:
        task = asyncio.create_task(
          self._recording_finalizer.finalize_recording(
            finished_recording=recording,
            recording_observation=observation,
            llm_available=self._llm_available,
            llm_active=self._llm_active,
          )
        )
      except Exception:
        self._recording_finalizer.cancel_reserved_flush()
        raise
      observation = None
      self._background_tasks.add(task)
      task.add_done_callback(self._background_tasks.discard)
      self.logger.info("recording finished", background_finalization=True)
    except Exception:
      self.logger.exception("recording finish failed")
      end_recording_observation(
        observation, logger=self.logger, level="ERROR", status_message="recording finish failed"
      )
      raise
    finally:
      await self._workstation.resume_media()

  async def handle_keyboard_event(self, event: KeyboardControlEvent) -> None:
    """Translate hardware gestures into recording actions or command intervals.

    :param event: Normalized keyboard event and receipt timestamp.
    """
    match event.kind:
      case KeyboardEventKind.ESCAPE_DOWN:
        _ = await self.handle_action(AppAction.CANCEL)
      case KeyboardEventKind.CAPSLOCK_DOWN:
        if self.phase is ForegroundPhase.RECORDING:
          await self._recording_keyboard.capslock_down(event.received_monotonic_s)
        else:
          _ = await self.handle_action(AppAction.START_OR_FINISH)
      case KeyboardEventKind.CAPSLOCK_UP:
        if self.phase is ForegroundPhase.RECORDING:
          if await self._recording_keyboard.capslock_up(event.received_monotonic_s):
            _ = await self.handle_action(AppAction.START_OR_FINISH)

  async def handle_client_event(self, event: ClientSignal) -> None:
    """Apply an app-local lifecycle signal or original wire transcription.

    :param event: Signal enqueued by a synchronous client callback.
    """
    match event:
      case TranscriptionMessage():
        self._trace.session_id = event.stream
        await update_dbus_transcription(
          event,
          phase=self.phase,
          recording=self._recording_transcript,
          dbus=self.dbus_service,
          logger=self.logger,
        )
      case ConnectedSignal(session_id=session_id):
        self._trace.session_id = session_id
        self._recording_transcript.reset_connection_cursor()
        self._trace.session_event(
          name="active-listener-client-connected", metadata={"session_id": session_id}
        )
        if self.phase is ForegroundPhase.STARTING:
          self.phase = ForegroundPhase.IDLE
          await self.dbus_service.set_state(self.phase)
      case ReconnectingSignal(session_id=session_id, attempt=attempt, retry_delay_s=delay):
        self._trace.session_id = session_id
        self.phase = ForegroundPhase.RECONNECTING
        await self.dbus_service.set_state(self.phase)
        await self.dbus_service.reconnecting()
        self.logger.warning(
          "client reconnecting", session_id=session_id, attempt=attempt, retry_delay_s=delay
        )
      case ReconnectedSignal(session_id=session_id):
        self._trace.session_id = session_id
        self._recording_transcript.reset_connection_cursor()
        self.phase = ForegroundPhase.IDLE
        await self.dbus_service.set_state(self.phase)
        await self.dbus_service.reconnected()
        self._trace.session_event(
          name="active-listener-client-reconnected", metadata={"session_id": session_id}
        )
      case DisconnectedSignal(session_id=session_id, reason=reason):
        await self._handle_disconnect(session_id, reason or UNKNOWN_DISCONNECT_REASON)

  async def _handle_disconnect(self, session_id: str, reason: str) -> None:
    self._trace.session_id = session_id
    was_recording = self._recording_transcript.is_recording
    recording_id = self._recording_transcript.active_recording_id
    self.phase = ForegroundPhase.RECONNECTING
    if was_recording:
      observation = self._trace.transfer_and_clear_observation()
      try:
        await self._workstation.stop_spectrum()
        await self._stop_capture()
        await self.dbus_service.set_state(self.phase)
        await self.dbus_service.recording_aborted(reason)
        await self.dbus_service.reconnecting()
        self._trace.recording_event(
          recording_observation=observation,
          name="active-listener-recording-aborted",
          metadata={"session_id": session_id, "recording_id": recording_id, "reason": reason},
        )
      finally:
        end_recording_observation(
          observation,
          logger=self.logger,
          level="ERROR",
          status_message="recording aborted by disconnect",
        )
        await self._workstation.resume_media()
    else:
      await self.dbus_service.set_state(self.phase)
    self.logger.warning("client disconnected", session_id=session_id, reason=reason)
    self._trace.session_event(
      name="active-listener-client-disconnected",
      metadata={"session_id": session_id, "reason": reason},
    )

  async def _pump_keyboard_events(self) -> None:
    async for event in self.keyboard.events():
      await self._signal_queue.put(KeyboardEventSignal(event=event))
