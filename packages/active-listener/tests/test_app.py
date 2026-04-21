"""Application policy tests for active-listener."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast, final

import numpy as np
import pytest
from structlog.stdlib import BoundLogger
from typing_extensions import override

from active_listener.app.ports import ActiveListenerRuntimeError
from active_listener.app.service import ActiveListenerService
from active_listener.app.state import ForegroundPhase, KeyboardAction
from active_listener.bootstrap import build_capture_callback, create_service, run_service
from active_listener.config.models import (
  ActiveListenerConfig,
  LiteRtRewriteProvider,
  LlmRewriteConfig,
)
from active_listener.infra.keyboard import KeyboardInput
from active_listener.infra.rewrite import (
  LoadedRewritePrompt,
  LoadedRewritePromptFile,
  RewriteClientError,
  RewriteClientTimeoutError,
  RewritePromptError,
)
from active_listener.recording.spectrum import (
  SAMPLE_RATE_HZ,
  SPECTRUM_TICK_INTERVAL_SECONDS,
  WINDOW_SIZE,
)
from eavesdrop.client import (
  DisconnectedEvent,
  ReconnectedEvent,
  ReconnectingEvent,
  TranscriptionEvent,
)
from eavesdrop.wire import Segment, TranscriptionMessage


class RecordingLogger:
  """Structured logger stand-in for app-policy assertions."""

  def __init__(self) -> None:
    self.debug_messages: list[str] = []
    self.info_messages: list[str] = []
    self.warning_messages: list[str] = []
    self.exception_messages: list[str] = []
    self.debug_records: list[LogRecord] = []
    self.info_records: list[LogRecord] = []
    self.warning_records: list[LogRecord] = []
    self.exception_records: list[LogRecord] = []

  def debug(self, event: str, **kwargs: object) -> None:
    self.debug_messages.append(event)
    self.debug_records.append(LogRecord(event=event, fields=kwargs))

  def info(self, event: str, **kwargs: object) -> None:
    self.info_messages.append(event)
    self.info_records.append(LogRecord(event=event, fields=kwargs))

  def warning(self, event: str, **kwargs: object) -> None:
    self.warning_messages.append(event)
    self.warning_records.append(LogRecord(event=event, fields=kwargs))

  def exception(self, event: str, **kwargs: object) -> None:
    self.exception_messages.append(event)
    self.exception_records.append(LogRecord(event=event, fields=kwargs))


@dataclass(frozen=True)
class LogRecord:
  """Captured structured log entry for test assertions."""

  event: str
  fields: dict[str, object]


AppEvent = DisconnectedEvent | ReconnectedEvent | ReconnectingEvent | TranscriptionEvent
DbusOverlaySegment = tuple[int, str]


def _rewrite_config() -> LlmRewriteConfig:
  return LlmRewriteConfig(
    prompt_path="/tmp/rewrite/system.md",
    provider=LiteRtRewriteProvider(
      type="litert",
      model_path="/tmp/rewrite/model.litertlm",
    ),
  )


def _config(*, rewrite_enabled: bool = False) -> ActiveListenerConfig:
  return ActiveListenerConfig(
    keyboard_name="Exact Keyboard",
    host="localhost",
    port=9090,
    audio_device="default",
    llm_rewrite=_rewrite_config() if rewrite_enabled else None,
  )


def _prompt(
  *,
  instructions: str = "Rewrite this transcript.",
) -> LoadedRewritePrompt:
  return LoadedRewritePrompt(instructions=instructions)


def _loaded_prompt_file(
  prompt: LoadedRewritePrompt,
  *,
  prompt_path: str = "/tmp/rewrite/system.md",
) -> LoadedRewritePromptFile:
  return LoadedRewritePromptFile(prompt_path=Path(prompt_path), prompt=prompt)


async def _hold_open(signal: asyncio.Event) -> None:
  _ = await signal.wait()


def _prompt_loader(
  loaded_prompt: LoadedRewritePromptFile,
) -> Callable[[str], LoadedRewritePromptFile]:
  def load(_path: str) -> LoadedRewritePromptFile:
    return loaded_prompt

  return load


def _failing_prompt_loader(error: Exception) -> Callable[[str], LoadedRewritePromptFile]:
  def load(_path: str) -> LoadedRewritePromptFile:
    raise error

  return load


@dataclass
class FakeKeyboard:
  """Keyboard boundary stand-in used by app tests."""

  grab_calls: int = 0
  ungrab_calls: int = 0
  close_calls: int = 0
  grabbed: bool = False
  queued_actions: list[KeyboardAction] = field(default_factory=list)

  async def actions(self) -> AsyncIterator[KeyboardAction]:
    for action in self.queued_actions:
      yield action

  @asynccontextmanager
  async def recording_grab(self) -> AsyncIterator[Callable[[], None]]:
    self.grab()
    released = False

    def release() -> None:
      nonlocal released
      if released:
        return
      released = True
      self.ungrab()

    try:
      yield release
    finally:
      release()

  def grab(self) -> None:
    self.grab_calls += 1
    self.grabbed = True

  def ungrab(self) -> None:
    self.ungrab_calls += 1
    self.grabbed = False

  def close(self) -> None:
    self.close_calls += 1
    self.grabbed = False


@dataclass
class FakeEmitter:
  """Text emitter stand-in used by app tests."""

  initialize_calls: int = 0
  emitted: list[str] = field(default_factory=list)

  def initialize(self) -> None:
    self.initialize_calls += 1

  def emit_text(self, text: str) -> None:
    self.emitted.append(text)


@dataclass
class FakeDbusService:
  states: list[ForegroundPhase] = field(default_factory=list)
  signals: list[tuple[str, object | None]] = field(default_factory=list)
  events: list[tuple[str, object | None]] = field(default_factory=list)
  close_calls: int = 0

  async def set_state(self, state: ForegroundPhase) -> None:
    if self.states and self.states[-1] is state:
      return
    self.states.append(state)
    self.events.append(("State", state))

  async def transcription_updated(
    self,
    completed_segments: list[DbusOverlaySegment],
    incomplete_segment: DbusOverlaySegment,
  ) -> None:
    self.signals.append(("TranscriptionUpdated", (completed_segments, incomplete_segment)))
    self.events.append(("TranscriptionUpdated", (completed_segments, incomplete_segment)))

  async def spectrum_updated(self, bars: bytes) -> None:
    self.signals.append(("SpectrumUpdated", bars))
    self.events.append(("SpectrumUpdated", bars))

  async def recording_aborted(self, reason: str) -> None:
    self.signals.append(("RecordingAborted", reason))
    self.events.append(("RecordingAborted", reason))

  async def pipeline_failed(self, step: str, reason: str) -> None:
    self.signals.append(("PipelineFailed", (step, reason)))
    self.events.append(("PipelineFailed", (step, reason)))

  async def fatal_error(self, reason: str) -> None:
    self.signals.append(("FatalError", reason))
    self.events.append(("FatalError", reason))

  async def reconnecting(self) -> None:
    self.signals.append(("Reconnecting", None))
    self.events.append(("Reconnecting", None))

  async def reconnected(self) -> None:
    self.signals.append(("Reconnected", None))
    self.events.append(("Reconnected", None))

  async def close(self) -> None:
    self.close_calls += 1
    return None


@dataclass
class FakeClient:
  """Live-client stand-in used by app tests."""

  connect_calls: int = 0
  disconnect_calls: int = 0
  start_calls: int = 0
  stop_calls: int = 0
  cancel_calls: int = 0
  flush_calls: list[bool] = field(default_factory=list)
  flush_results: list[asyncio.Future[TranscriptionMessage] | TranscriptionMessage | Exception] = (
    field(default_factory=list)
  )
  events: list[AppEvent] = field(default_factory=list)

  async def connect(self) -> None:
    self.connect_calls += 1

  async def disconnect(self) -> None:
    self.disconnect_calls += 1

  async def start_streaming(self) -> None:
    self.start_calls += 1

  async def stop_streaming(self) -> None:
    self.stop_calls += 1

  async def cancel_utterance(self) -> None:
    self.cancel_calls += 1

  async def flush(self, *, force_complete: bool = True) -> TranscriptionMessage:
    self.flush_calls.append(force_complete)
    result = self.flush_results.pop(0)
    if isinstance(result, Exception):
      raise result
    if isinstance(result, asyncio.Future):
      return await result
    return result

  def __aiter__(self) -> AsyncIterator[AppEvent]:
    return self._iterate_events()

  async def _iterate_events(self) -> AsyncIterator[AppEvent]:
    for event in self.events:
      yield event


@dataclass
class FakeSpectrumAnalyzer:
  start_calls: int = 0
  stop_calls: int = 0
  ingested: list[bytes] = field(default_factory=list)
  _task: asyncio.Task[None] | None = None
  _release: asyncio.Event = field(default_factory=asyncio.Event)

  def start(self) -> asyncio.Task[None]:
    self.start_calls += 1
    if self._task is None or self._task.done():
      self._release = asyncio.Event()
      self._task = asyncio.create_task(_hold_open(self._release))
    return self._task

  async def stop(self) -> None:
    self.stop_calls += 1
    self._release.set()
    if self._task is not None:
      _ = await asyncio.gather(self._task, return_exceptions=True)
      self._task = None

  def ingest(self, chunk: bytes) -> None:
    self.ingested.append(chunk)


@dataclass
class ExplodingSpectrumAnalyzer:
  def ingest(self, chunk: bytes) -> None:
    _ = chunk
    raise RuntimeError("spectrum boom")


@final
class FailingConnectClient(FakeClient):
  """Client stand-in whose initial connect attempt fails."""

  @override
  async def connect(self) -> None:
    raise RuntimeError("server unavailable")


@final
class FailingDisconnectClient(FakeClient):
  """Client stand-in whose disconnect path fails."""

  @override
  async def disconnect(self) -> None:
    self.disconnect_calls += 1
    raise RuntimeError("disconnect failed")


@final
class FailingStopClient(FakeClient):
  """Client stand-in whose stop-streaming path fails."""

  @override
  async def stop_streaming(self) -> None:
    self.stop_calls += 1
    raise RuntimeError("stop failed")


@dataclass
class FakeRewriteClient:
  rewritten_text: str = "rewritten text"
  error: Exception | None = None
  calls: list[dict[str, str]] = field(default_factory=list)
  close_calls: int = 0

  async def rewrite_text(
    self,
    *,
    instructions: str,
    transcript: str,
  ) -> str:
    self.calls.append(
      {
        "instructions": instructions,
        "transcript": transcript,
      }
    )
    if self.error is not None:
      raise self.error
    return self.rewritten_text

  async def close(self) -> None:
    self.close_calls += 1


def _segment(segment_id: int, text: str, *, completed: bool) -> Segment:
  return Segment(
    id=segment_id,
    seek=0,
    start=0.0,
    end=0.1,
    text=text,
    tokens=[],
    temperature=0.0,
    avg_logprob=0.0,
    compression_ratio=1.0,
    words=None,
    completed=completed,
  )


def _message(
  *segments: Segment,
  stream: str = "stream-1",
  flush_complete: bool | None = True,
) -> TranscriptionMessage:
  return TranscriptionMessage(stream=stream, segments=list(segments), flush_complete=flush_complete)


def _transcription_event(message: TranscriptionMessage) -> TranscriptionEvent:
  return TranscriptionEvent(stream=message.stream, message=message)


def _sine_wave_bytes(frequency_hz: float, sample_count: int) -> bytes:
  sample_positions = np.arange(sample_count, dtype=np.float32) / SAMPLE_RATE_HZ
  samples = np.sin(2.0 * np.pi * frequency_hz * sample_positions).astype(np.float32)
  return samples.tobytes()


@dataclass(frozen=True)
class ServiceHarness:
  """Concrete service bundle used by app-policy tests."""

  service: ActiveListenerService
  client: FakeClient
  keyboard: FakeKeyboard
  emitter: FakeEmitter
  logger: RecordingLogger
  rewrite_client: FakeRewriteClient
  dbus_service: FakeDbusService


def _service(
  *,
  client: FakeClient | None = None,
  keyboard: FakeKeyboard | None = None,
  emitter: FakeEmitter | None = None,
  logger: RecordingLogger | None = None,
  rewrite_client: FakeRewriteClient | None = None,
  dbus_service: FakeDbusService | None = None,
  config: ActiveListenerConfig | None = None,
  spectrum_analyzer: FakeSpectrumAnalyzer | None = None,
) -> ServiceHarness:
  resolved_client = client or FakeClient()
  resolved_keyboard = keyboard or FakeKeyboard()
  resolved_emitter = emitter or FakeEmitter()
  resolved_logger = logger or RecordingLogger()
  resolved_rewrite_client = rewrite_client or FakeRewriteClient()
  resolved_dbus_service = dbus_service or FakeDbusService()
  service = ActiveListenerService(
    config=config or _config(),
    keyboard=resolved_keyboard,
    client=resolved_client,
    emitter=resolved_emitter,
    logger=cast(BoundLogger, cast(object, resolved_logger)),
    rewrite_client=resolved_rewrite_client,
    dbus_service=resolved_dbus_service,
    spectrum_analyzer=spectrum_analyzer or FakeSpectrumAnalyzer(),
  )
  return ServiceHarness(
    service,
    resolved_client,
    resolved_keyboard,
    resolved_emitter,
    resolved_logger,
    resolved_rewrite_client,
    resolved_dbus_service,
  )


@pytest.mark.asyncio
async def test_create_service_connects_client_and_initializes_emitter() -> None:
  keyboard = FakeKeyboard()
  emitter = FakeEmitter()
  client = FakeClient()
  dbus_service = FakeDbusService()

  service = await create_service(
    _config(),
    dbus_service=dbus_service,
    keyboard_resolver=lambda _name: keyboard,
    client_factory=lambda _config, _on_capture: client,
    emitter_factory=lambda: (emitter.initialize(), emitter)[1],
  )

  assert service.phase is ForegroundPhase.IDLE
  assert client.connect_calls == 1
  assert emitter.initialize_calls == 1
  assert dbus_service.states == [ForegroundPhase.IDLE]


@pytest.mark.asyncio
async def test_create_service_passes_capture_callback_into_client_factory() -> None:
  keyboard = FakeKeyboard()
  emitter = FakeEmitter()
  captured_callbacks: list[Callable[[bytes], None]] = []

  def client_factory(
    _config: ActiveListenerConfig,
    on_capture: Callable[[bytes], None],
  ) -> FakeClient:
    captured_callbacks.append(on_capture)
    return FakeClient()

  service = await create_service(
    _config(),
    dbus_service=FakeDbusService(),
    keyboard_resolver=lambda _name: keyboard,
    client_factory=client_factory,
    emitter_factory=lambda: (emitter.initialize(), emitter)[1],
  )

  assert service.phase is ForegroundPhase.IDLE
  assert len(captured_callbacks) == 1


@pytest.mark.asyncio
async def test_create_service_fails_fast_on_keyboard_resolution_error() -> None:
  with pytest.raises(ActiveListenerRuntimeError, match="missing keyboard"):
    _ = await create_service(
      _config(),
      keyboard_resolver=lambda _name: (_ for _ in ()).throw(RuntimeError("missing keyboard")),
    )


@pytest.mark.asyncio
async def test_create_service_fails_fast_on_connect_error() -> None:
  keyboard = FakeKeyboard()
  client = FailingConnectClient()
  rewrite_client = FakeRewriteClient()

  with pytest.raises(ActiveListenerRuntimeError, match="server unavailable"):
    _ = await create_service(
      _config(),
      keyboard_resolver=lambda _name: keyboard,
      client_factory=lambda _config, _on_capture: client,
      emitter_factory=lambda: FakeEmitter(),
      rewrite_client_factory=lambda _config: rewrite_client,
    )

  assert keyboard.close_calls == 1
  assert rewrite_client.close_calls == 1


@pytest.mark.asyncio
async def test_create_service_fails_fast_when_rewrite_client_cannot_initialize() -> None:
  keyboard = FakeKeyboard()

  with pytest.raises(ActiveListenerRuntimeError, match="bad model"):
    _ = await create_service(
      _config(rewrite_enabled=True),
      keyboard_resolver=lambda _name: keyboard,
      client_factory=lambda _config, _on_capture: FakeClient(),
      emitter_factory=lambda: FakeEmitter(),
      rewrite_client_factory=lambda _config: (_ for _ in ()).throw(RewriteClientError("bad model")),
    )

  assert keyboard.close_calls == 1


@pytest.mark.asyncio
async def test_start_and_cancel_recording_toggle_grab_and_streaming() -> None:
  spectrum_analyzer = FakeSpectrumAnalyzer()
  harness = _service(spectrum_analyzer=spectrum_analyzer)
  service = harness.service

  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await service.handle_keyboard_action(KeyboardAction.CANCEL)

  assert service.phase is ForegroundPhase.IDLE
  assert harness.client.start_calls == 1
  assert harness.client.stop_calls == 1
  assert harness.client.cancel_calls == 1
  assert harness.keyboard.grab_calls == 1
  assert harness.keyboard.ungrab_calls == 1
  assert spectrum_analyzer.start_calls == 1
  assert spectrum_analyzer.stop_calls == 1
  assert harness.logger.info_messages == ["recording started", "recording cancelled"]


@pytest.mark.asyncio
async def test_cancel_clears_foreground_state_before_stop_streaming_returns() -> None:
  harness = _service(client=FailingStopClient())
  service = harness.service

  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)

  with pytest.raises(RuntimeError, match="stop failed"):
    await service.handle_keyboard_action(KeyboardAction.CANCEL)

  assert service.phase is ForegroundPhase.IDLE
  assert harness.client.cancel_calls == 0
  assert harness.keyboard.grabbed is False
  assert harness.keyboard.ungrab_calls == 1


@pytest.mark.asyncio
async def test_escape_is_ignored_while_idle() -> None:
  harness = _service()
  service = harness.service

  await service.handle_keyboard_action(KeyboardAction.CANCEL)

  assert service.phase is ForegroundPhase.IDLE
  assert harness.client.stop_calls == 0
  assert harness.keyboard.ungrab_calls == 0
  assert harness.logger.info_messages == ["cancel ignored while idle"]


@pytest.mark.asyncio
async def test_caps_lock_is_suppressed_while_reconnecting() -> None:
  harness = _service()
  service = harness.service
  service.phase = ForegroundPhase.RECONNECTING

  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)

  assert harness.client.start_calls == 0
  assert harness.keyboard.grab_calls == 0
  assert harness.logger.info_messages == ["recording start suppressed while reconnecting"]


@pytest.mark.asyncio
async def test_disconnect_during_recording_forces_local_abort() -> None:
  spectrum_analyzer = FakeSpectrumAnalyzer()
  harness = _service(spectrum_analyzer=spectrum_analyzer)
  service = harness.service
  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)

  await service.handle_client_event(DisconnectedEvent(stream="stream-1", reason="socket closed"))

  assert service.phase is ForegroundPhase.RECONNECTING
  assert harness.client.stop_calls == 1
  assert harness.keyboard.ungrab_calls == 1
  assert harness.dbus_service.states == [ForegroundPhase.RECORDING, ForegroundPhase.RECONNECTING]
  assert harness.dbus_service.signals == [
    ("RecordingAborted", "socket closed"),
    ("Reconnecting", None),
  ]
  assert spectrum_analyzer.stop_calls == 1
  assert harness.logger.warning_messages == ["recording aborted by disconnect"]


@pytest.mark.asyncio
async def test_disconnect_without_reason_uses_truthful_dbus_fallback() -> None:
  harness = _service()
  await harness.service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)

  await harness.service.handle_client_event(DisconnectedEvent(stream="stream-1", reason=None))

  assert harness.dbus_service.signals == [
    ("RecordingAborted", "unknown disconnect reason"),
    ("Reconnecting", None),
  ]
  assert harness.logger.warning_records[-1] == LogRecord(
    event="recording aborted by disconnect",
    fields={"stream": "stream-1", "reason": "unknown disconnect reason"},
  )


@pytest.mark.asyncio
async def test_finish_clears_foreground_state_before_stop_streaming_returns() -> None:
  harness = _service(client=FailingStopClient())
  service = harness.service

  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)

  with pytest.raises(RuntimeError, match="stop failed"):
    await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)

  assert service.phase is ForegroundPhase.IDLE
  assert harness.keyboard.grabbed is False
  assert harness.keyboard.ungrab_calls == 1


@pytest.mark.asyncio
async def test_close_releases_keyboard_even_when_disconnect_raises() -> None:
  spectrum_analyzer = FakeSpectrumAnalyzer()
  harness = _service(client=FailingDisconnectClient(), spectrum_analyzer=spectrum_analyzer)
  service = harness.service

  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)

  with pytest.raises(RuntimeError, match="disconnect failed"):
    await service.close()

  assert harness.rewrite_client.close_calls == 1
  assert harness.keyboard.close_calls == 1
  assert harness.keyboard.grabbed is False
  assert spectrum_analyzer.stop_calls == 1


@pytest.mark.asyncio
async def test_capture_callback_failures_are_logged_locally() -> None:
  logger = RecordingLogger()
  on_capture = build_capture_callback(
    spectrum_analyzer=ExplodingSpectrumAnalyzer(),
    logger=cast(BoundLogger, cast(object, logger)),
  )

  on_capture(b"chunk")

  assert logger.exception_messages == ["spectrum capture callback failed"]


@pytest.mark.asyncio
async def test_spectrum_emission_flows_through_dbus_while_recording() -> None:
  keyboard = FakeKeyboard()
  emitter = FakeEmitter()
  dbus_service = FakeDbusService()
  captured_callbacks: list[Callable[[bytes], None]] = []

  def client_factory(
    _config: ActiveListenerConfig,
    on_capture: Callable[[bytes], None],
  ) -> FakeClient:
    captured_callbacks.append(on_capture)
    return FakeClient()

  service = await create_service(
    _config(),
    dbus_service=dbus_service,
    keyboard_resolver=lambda _name: keyboard,
    client_factory=client_factory,
    emitter_factory=lambda: (emitter.initialize(), emitter)[1],
    rewrite_client_factory=lambda _config: FakeRewriteClient(),
  )

  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  captured_callbacks[0](_sine_wave_bytes(440.0, WINDOW_SIZE * 2))
  await asyncio.sleep(SPECTRUM_TICK_INTERVAL_SECONDS * 3)
  await service.handle_keyboard_action(KeyboardAction.CANCEL)

  state_index = dbus_service.events.index(("State", ForegroundPhase.RECORDING))
  spectrum_event = next(event for event in dbus_service.events if event[0] == "SpectrumUpdated")

  assert state_index < dbus_service.events.index(spectrum_event)
  assert isinstance(spectrum_event[1], bytes)
  assert len(spectrum_event[1]) == 50


@pytest.mark.asyncio
async def test_close_still_disconnects_when_stop_streaming_fails() -> None:
  harness = _service(client=FailingStopClient())
  service = harness.service

  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)

  with pytest.raises(RuntimeError, match="stop failed"):
    await service.close()

  assert harness.client.disconnect_calls == 1
  assert harness.rewrite_client.close_calls == 1
  assert harness.keyboard.close_calls == 1
  assert harness.keyboard.grabbed is False


@pytest.mark.asyncio
async def test_run_service_emits_fatal_error_once_on_startup_failure() -> None:
  dbus_service = FakeDbusService()

  def fail_keyboard_resolver(_keyboard_name: str) -> KeyboardInput:
    raise ActiveListenerRuntimeError("keyboard missing")

  with pytest.raises(ActiveListenerRuntimeError, match="keyboard missing"):
    await run_service(
      _config(),
      dbus_service=dbus_service,
      keyboard_resolver=fail_keyboard_resolver,
      client_factory=lambda _config, _on_capture: FakeClient(),
      emitter_factory=lambda: FakeEmitter(),
      rewrite_client_factory=lambda _config: FakeRewriteClient(),
    )

  assert dbus_service.states == []
  assert dbus_service.signals == [("FatalError", "keyboard missing")]
  assert dbus_service.close_calls == 1


@pytest.mark.asyncio
async def test_run_service_emits_fatal_error_once_on_runtime_failure() -> None:
  keyboard = FakeKeyboard(queued_actions=[KeyboardAction.START_OR_FINISH, KeyboardAction.CANCEL])
  client = FailingStopClient()
  dbus_service = FakeDbusService()

  with pytest.raises(RuntimeError, match="stop failed"):
    await run_service(
      _config(),
      dbus_service=dbus_service,
      keyboard_resolver=lambda _name: keyboard,
      client_factory=lambda _config, _on_capture: client,
      emitter_factory=lambda: FakeEmitter(),
      rewrite_client_factory=lambda _config: FakeRewriteClient(),
    )

  assert dbus_service.states == [ForegroundPhase.IDLE, ForegroundPhase.RECORDING]
  assert dbus_service.signals == [("FatalError", "stop failed")]
  assert dbus_service.close_calls == 1


@pytest.mark.asyncio
async def test_reconnecting_and_reconnected_events_update_phase() -> None:
  harness = _service()
  service = harness.service

  await service.handle_client_event(
    ReconnectingEvent(stream="stream-1", attempt=2, retry_delay_s=10.0)
  )
  await service.handle_client_event(ReconnectedEvent(stream="stream-1"))

  assert service.phase is ForegroundPhase.IDLE
  assert harness.dbus_service.states == [ForegroundPhase.RECONNECTING, ForegroundPhase.IDLE]
  assert harness.dbus_service.signals == [
    ("Reconnecting", None),
    ("Reconnected", None),
  ]
  assert harness.logger.warning_messages == ["client reconnecting"]
  assert harness.logger.info_messages == ["client reconnected"]


@pytest.mark.asyncio
async def test_start_and_finish_publish_recording_and_idle_states() -> None:
  client = FakeClient(
    flush_results=[
      _message(
        _segment(1, "alpha", completed=True),
        _segment(2, "tail", completed=False),
      )
    ]
  )
  harness = _service(client=client)

  await harness.service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await harness.service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()

  assert harness.dbus_service.states == [ForegroundPhase.RECORDING, ForegroundPhase.IDLE]
  assert harness.dbus_service.signals == []
  assert client.cancel_calls == 0
  assert client.flush_calls == [True]


@pytest.mark.asyncio
async def test_idle_disconnect_sets_reconnecting_without_extra_dbus_signal() -> None:
  harness = _service()

  await harness.service.handle_client_event(
    DisconnectedEvent(stream="stream-1", reason="socket closed")
  )

  assert harness.service.phase is ForegroundPhase.RECONNECTING
  assert harness.dbus_service.states == [ForegroundPhase.RECONNECTING]
  assert harness.dbus_service.signals == []


@pytest.mark.asyncio
async def test_transcription_events_are_consumed_only_while_recording() -> None:
  client = FakeClient(
    flush_results=[
      _message(
        _segment(1, "alpha", completed=True),
        _segment(2, "bravo", completed=True),
        _segment(3, "tail", completed=False),
      )
    ]
  )
  emitter = FakeEmitter()
  harness = _service(client=client, emitter=emitter)
  service = harness.service

  await service.handle_client_event(
    _transcription_event(
      _message(
        _segment(90, "ignored", completed=True),
        _segment(91, "tail", completed=False),
        flush_complete=False,
      )
    )
  )
  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await service.handle_client_event(
    _transcription_event(
      _message(
        _segment(1, "alpha", completed=True),
        _segment(2, "draft", completed=False),
        flush_complete=False,
      )
    )
  )
  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await service.wait_for_background_tasks()

  assert service.phase is ForegroundPhase.IDLE
  assert client.stop_calls == 1
  assert client.flush_calls == [True]
  assert harness.dbus_service.signals == [
    ("TranscriptionUpdated", ([(1, "alpha")], (2, "draft"))),
  ]
  assert emitter.emitted == ["alpha bravo "]
  assert harness.keyboard.ungrab_calls == 1
  assert (
    LogRecord(
      event="live transcription event ignored",
      fields={"stream": "stream-1", "phase": "idle"},
    )
    in harness.logger.debug_records
  )
  assert (
    LogRecord(
      event="live transcription event received",
      fields={
        "stream": "stream-1",
        "phase": "recording",
        "flush_complete": False,
        "segment_count": 2,
      },
    )
    in harness.logger.debug_records
  )
  assert (
    LogRecord(
      event="publishing live transcription update",
      fields={
        "stream": "stream-1",
        "completed_segments": [(1, "alpha")],
        "incomplete_segment": (2, "draft"),
      },
    )
    in harness.logger.debug_records
  )
  assert (
    LogRecord(
      event="live transcription update published",
      fields={
        "stream": "stream-1",
        "completed_segment_count": 1,
        "incomplete_segment_id": 2,
      },
    )
    in harness.logger.debug_records
  )
  assert "recording finished" in harness.logger.info_messages
  assert harness.logger.info_messages[-1] == "text emitted"


@pytest.mark.asyncio
async def test_cancel_preserves_connection_cursor_for_next_recording() -> None:
  client = FakeClient(
    flush_results=[
      _message(
        _segment(1, "alpha", completed=True),
        _segment(2, "bravo", completed=True),
        _segment(3, "tail", completed=False),
      )
    ]
  )
  emitter = FakeEmitter()
  harness = _service(client=client, emitter=emitter)
  service = harness.service

  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await service.handle_client_event(
    _transcription_event(
      _message(
        _segment(1, "alpha", completed=True),
        _segment(2, "tail", completed=False),
        flush_complete=False,
      )
    )
  )
  await service.handle_keyboard_action(KeyboardAction.CANCEL)
  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await service.wait_for_background_tasks()

  assert client.cancel_calls == 1
  assert client.flush_calls == [True]
  assert emitter.emitted == ["bravo "]


@pytest.mark.asyncio
async def test_new_recording_ignores_completed_history_before_seeded_cursor() -> None:
  client = FakeClient(
    flush_results=[
      _message(
        _segment(1, "alpha", completed=True),
        _segment(2, "tail", completed=False),
      ),
      _message(
        _segment(1, "alpha", completed=True),
        _segment(2, "bravo", completed=True),
        _segment(3, "charlie", completed=True),
        _segment(4, "tail", completed=False),
      ),
    ]
  )
  emitter = FakeEmitter()
  harness = _service(client=client, emitter=emitter)
  service = harness.service

  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await service.wait_for_background_tasks()

  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await service.wait_for_background_tasks()

  assert client.flush_calls == [True, True]
  assert emitter.emitted == ["alpha ", "bravo charlie "]


@pytest.mark.asyncio
async def test_reconnected_session_resets_connection_cursor_before_next_recording() -> None:
  client = FakeClient(
    flush_results=[
      _message(
        _segment(1, "alpha", completed=True),
        _segment(2, "tail", completed=False),
      ),
      _message(
        _segment(1, "fresh", completed=True),
        _segment(2, "tail", completed=False),
      ),
    ]
  )
  emitter = FakeEmitter()
  harness = _service(client=client, emitter=emitter)
  service = harness.service

  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await service.wait_for_background_tasks()

  await service.handle_client_event(
    ReconnectingEvent(stream="stream-1", attempt=2, retry_delay_s=10.0)
  )
  await service.handle_client_event(ReconnectedEvent(stream="stream-1"))

  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await service.wait_for_background_tasks()

  assert client.flush_calls == [True, True]
  assert emitter.emitted == ["alpha ", "fresh "]


@pytest.mark.asyncio
async def test_new_recording_can_start_while_older_finalization_waits() -> None:
  pending_flush = asyncio.get_running_loop().create_future()
  client = FakeClient(
    flush_results=[
      pending_flush,
      _message(
        _segment(3, "second", completed=True),
        _segment(4, "tail", completed=False),
      ),
    ]
  )
  emitter = FakeEmitter()
  harness = _service(client=client, emitter=emitter)
  service = harness.service

  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  pending_flush.set_result(
    _message(
      _segment(2, "first", completed=True),
      _segment(3, "tail", completed=False),
    )
  )
  await service.wait_for_background_tasks()

  assert client.start_calls == 2
  assert client.stop_calls == 2
  assert client.flush_calls == [True, True]
  assert emitter.emitted == ["first ", "second "]
  assert harness.keyboard.grab_calls == 2
  assert harness.keyboard.ungrab_calls == 2


@pytest.mark.asyncio
async def test_disconnect_during_background_finalization_skips_emission() -> None:
  pending_flush = asyncio.get_running_loop().create_future()
  client = FakeClient(flush_results=[pending_flush])
  emitter = FakeEmitter()
  harness = _service(client=client, emitter=emitter)
  service = harness.service

  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await service.handle_client_event(DisconnectedEvent(stream="stream-1", reason="socket closed"))
  pending_flush.set_result(_message(_segment(1, "alpha", completed=True)))
  await service.wait_for_background_tasks()

  assert emitter.emitted == []
  assert service.phase is ForegroundPhase.RECONNECTING
  assert harness.logger.warning_messages[-2:] == [
    "client disconnected",
    "skipping emission after disconnect",
  ]


@pytest.mark.asyncio
async def test_finished_recording_emits_joined_text_from_live_updates_and_flush() -> None:
  client = FakeClient(
    flush_results=[
      _message(
        _segment(2, "bravo", completed=True),
        _segment(3, "charlie", completed=True),
        _segment(4, "tail", completed=False),
      )
    ]
  )
  emitter = FakeEmitter()
  harness = _service(client=client, emitter=emitter)
  service = harness.service

  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await service.handle_client_event(
    _transcription_event(
      _message(
        _segment(1, "alpha", completed=True),
        _segment(2, "draft", completed=False),
        flush_complete=False,
      )
    )
  )
  await service.handle_client_event(
    _transcription_event(
      _message(
        _segment(1, "alpha", completed=True),
        _segment(2, "bravo", completed=True),
        _segment(3, "draft", completed=False),
        flush_complete=False,
      )
    )
  )
  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await service.wait_for_background_tasks()

  assert client.flush_calls == [True]
  assert emitter.emitted == ["alpha bravo charlie "]


@pytest.mark.asyncio
async def test_missing_transcription_sentinel_warns_and_falls_back_to_window() -> None:
  client = FakeClient(
    flush_results=[
      _message(
        _segment(7, "restart", completed=True),
        _segment(8, "tail", completed=True),
        _segment(9, "draft", completed=False),
      )
    ]
  )
  emitter = FakeEmitter()
  logger = RecordingLogger()
  harness = _service(client=client, emitter=emitter, logger=logger)
  service = harness.service

  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await service.handle_client_event(
    _transcription_event(
      _message(
        _segment(1, "alpha", completed=True),
        _segment(2, "tail", completed=False),
        flush_complete=False,
      )
    )
  )
  await service.handle_client_event(
    _transcription_event(
      _message(
        _segment(7, "restart", completed=True),
        _segment(8, "draft", completed=False),
        flush_complete=False,
      )
    )
  )
  await service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await service.wait_for_background_tasks()

  assert emitter.emitted == ["alpha restart tail "]
  assert logger.warning_records[-1] == LogRecord(
    event="transcription reducer sentinel missing",
    fields={"stream": "stream-1", "last_id": 1},
  )


@pytest.mark.asyncio
async def test_finalize_recording_emits_raw_text_when_rewrite_is_disabled() -> None:
  client = FakeClient(
    flush_results=[
      _message(
        _segment(1, "alpha", completed=True),
        _segment(2, "tail", completed=False),
      )
    ]
  )
  harness = _service(client=client)

  await harness.service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await harness.service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()

  assert harness.emitter.emitted == ["alpha "]
  assert harness.rewrite_client.calls == []
  assert harness.logger.info_records[-1] == LogRecord(
    event="text emitted",
    fields={
      "stream": "stream-1",
      "emitted_text": "alpha ",
      "text_length": 6,
      "source": "raw",
    },
  )


@pytest.mark.asyncio
async def test_finalize_recording_rewrites_text_when_rewrite_succeeds(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  client = FakeClient(
    flush_results=[
      _message(
        _segment(1, "alpha", completed=True),
        _segment(2, "tail", completed=False),
      )
    ]
  )
  rewrite_client = FakeRewriteClient(rewritten_text="rewritten alpha")
  harness = _service(
    client=client,
    rewrite_client=rewrite_client,
    config=_config(rewrite_enabled=True),
  )
  monkeypatch.setattr(
    "active_listener.infra.rewrite.load_active_listener_rewrite_prompt",
    _prompt_loader(_loaded_prompt_file(_prompt())),
  )

  await harness.service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await harness.service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()

  assert harness.emitter.emitted == ["rewritten alpha "]
  assert rewrite_client.calls == [
    {
      "instructions": "Rewrite this transcript.",
      "transcript": "alpha",
    }
  ]
  assert harness.logger.info_records[-1] == LogRecord(
    event="text emitted",
    fields={
      "stream": "stream-1",
      "emitted_text": "rewritten alpha ",
      "text_length": 16,
      "source": "pipeline",
    },
  )


@pytest.mark.asyncio
async def test_finalize_recording_drops_text_and_signals_pipeline_failure_when_rewrite_fails(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  client = FakeClient(
    flush_results=[
      _message(
        _segment(1, "alpha", completed=True),
        _segment(2, "tail", completed=False),
      )
    ]
  )
  rewrite_client = FakeRewriteClient(error=RewriteClientError("model failed"))
  harness = _service(
    client=client,
    rewrite_client=rewrite_client,
    config=_config(rewrite_enabled=True),
  )
  monkeypatch.setattr(
    "active_listener.infra.rewrite.load_active_listener_rewrite_prompt",
    _prompt_loader(_loaded_prompt_file(_prompt())),
  )

  await harness.service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await harness.service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()

  assert harness.emitter.emitted == []
  assert harness.dbus_service.signals[-1] == (
    "PipelineFailed",
    ("rewrite_with_llm", "model failed"),
  )
  assert harness.logger.exception_records[-1] == LogRecord(
    event="dictation pipeline step failed",
    fields={
      "stream": "stream-1",
      "step": "rewrite_with_llm",
      "reason": "model failed",
    },
  )


@pytest.mark.asyncio
async def test_finalize_recording_skips_rewrite_after_disconnect(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  pending_flush = asyncio.get_running_loop().create_future()
  client = FakeClient(flush_results=[pending_flush])
  harness = _service(
    client=client,
    config=_config(rewrite_enabled=True),
  )
  prompt_load_calls: list[str] = []

  def fake_load_prompt(_path: str) -> LoadedRewritePromptFile:
    prompt_load_calls.append("called")
    return _loaded_prompt_file(_prompt())

  monkeypatch.setattr(
    "active_listener.infra.rewrite.load_active_listener_rewrite_prompt", fake_load_prompt
  )

  await harness.service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await harness.service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await harness.service.handle_client_event(
    DisconnectedEvent(stream="stream-1", reason="socket closed")
  )
  pending_flush.set_result(
    _message(
      _segment(1, "alpha", completed=True),
      _segment(2, "tail", completed=False),
    )
  )
  await harness.service.wait_for_background_tasks()

  assert prompt_load_calls == []
  assert harness.emitter.emitted == []


@pytest.mark.asyncio
async def test_finalize_recording_skips_rewrite_for_empty_text(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  client = FakeClient(
    flush_results=[
      _message(
        _segment(1, "   ", completed=True),
        _segment(2, "tail", completed=False),
      )
    ]
  )
  harness = _service(
    client=client,
    config=_config(rewrite_enabled=True),
  )
  prompt_load_calls: list[str] = []

  def fake_load_prompt(_path: str) -> LoadedRewritePromptFile:
    prompt_load_calls.append("called")
    return _loaded_prompt_file(_prompt())

  monkeypatch.setattr(
    "active_listener.infra.rewrite.load_active_listener_rewrite_prompt", fake_load_prompt
  )

  await harness.service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await harness.service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()

  assert prompt_load_calls == []
  assert harness.emitter.emitted == []
  assert harness.logger.info_messages[-1] == "recording finalized without committed text"


@pytest.mark.asyncio
async def test_rewrite_logging_captures_prompt_failure(monkeypatch: pytest.MonkeyPatch) -> None:
  client = FakeClient(
    flush_results=[
      _message(
        _segment(1, "alpha", completed=True),
        _segment(2, "tail", completed=False),
      )
    ]
  )
  harness = _service(client=client, config=_config(rewrite_enabled=True))
  monkeypatch.setattr(
    "active_listener.infra.rewrite.load_active_listener_rewrite_prompt",
    _failing_prompt_loader(
      RewritePromptError("bad prompt", prompt_path=Path("/tmp/override/system.md"))
    ),
  )

  await harness.service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await harness.service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()

  assert harness.emitter.emitted == []
  assert harness.dbus_service.signals[-1] == (
    "PipelineFailed",
    ("rewrite_with_llm", "bad prompt"),
  )
  assert harness.logger.exception_records[-1] == LogRecord(
    event="dictation pipeline step failed",
    fields={
      "stream": "stream-1",
      "step": "rewrite_with_llm",
      "reason": "bad prompt",
    },
  )


@pytest.mark.asyncio
async def test_rewrite_logging_uses_resolved_prompt_path(monkeypatch: pytest.MonkeyPatch) -> None:
  client = FakeClient(
    flush_results=[
      _message(
        _segment(1, "alpha", completed=True),
        _segment(2, "tail", completed=False),
      )
    ]
  )
  harness = _service(client=client, config=_config(rewrite_enabled=True))
  monkeypatch.setattr(
    "active_listener.infra.rewrite.load_active_listener_rewrite_prompt",
    _prompt_loader(_loaded_prompt_file(_prompt(), prompt_path="/tmp/override/system.md")),
  )

  await harness.service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await harness.service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()

  assert (
    LogRecord(
      event="rewrite prompt loaded",
      fields={
        "stream": "stream-1",
        "prompt_path": "/tmp/override/system.md",
        "instructions": "Rewrite this transcript.",
      },
    )
    in harness.logger.info_records
  )


@pytest.mark.asyncio
async def test_rewrite_logging_captures_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
  client = FakeClient(
    flush_results=[
      _message(
        _segment(1, "alpha", completed=True),
        _segment(2, "tail", completed=False),
      )
    ]
  )
  rewrite_client = FakeRewriteClient(error=RewriteClientTimeoutError("timed out"))
  harness = _service(
    client=client,
    rewrite_client=rewrite_client,
    config=_config(rewrite_enabled=True),
  )
  monkeypatch.setattr(
    "active_listener.infra.rewrite.load_active_listener_rewrite_prompt",
    _prompt_loader(_loaded_prompt_file(_prompt(), prompt_path="/tmp/override/system.md")),
  )

  await harness.service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await harness.service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()

  assert harness.emitter.emitted == []
  assert harness.dbus_service.signals[-1] == (
    "PipelineFailed",
    ("rewrite_with_llm", "timed out"),
  )
  assert harness.logger.exception_records[-1] == LogRecord(
    event="dictation pipeline step failed",
    fields={
      "stream": "stream-1",
      "step": "rewrite_with_llm",
      "reason": "timed out",
    },
  )


@pytest.mark.asyncio
async def test_rewrite_logging_captures_success_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
  client = FakeClient(
    flush_results=[
      _message(
        _segment(1, "alpha", completed=True),
        _segment(2, "tail", completed=False),
      )
    ]
  )
  rewrite_client = FakeRewriteClient(rewritten_text="rewritten alpha")
  harness = _service(
    client=client,
    rewrite_client=rewrite_client,
    config=_config(rewrite_enabled=True),
  )
  monkeypatch.setattr(
    "active_listener.infra.rewrite.load_active_listener_rewrite_prompt",
    _prompt_loader(
      _loaded_prompt_file(
        _prompt(),
        prompt_path="/tmp/override/system.md",
      )
    ),
  )

  await harness.service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await harness.service.handle_keyboard_action(KeyboardAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()

  assert (
    LogRecord(
      event="finalized raw transcript",
      fields={"stream": "stream-1", "raw_text": "alpha"},
    )
    in harness.logger.info_records
  )
  assert (
    LogRecord(
      event="rewrite succeeded",
      fields={
        "stream": "stream-1",
        "prompt_path": "/tmp/override/system.md",
        "raw_text": "alpha",
        "rewritten_text": "rewritten alpha",
      },
    )
    in harness.logger.info_records
  )
