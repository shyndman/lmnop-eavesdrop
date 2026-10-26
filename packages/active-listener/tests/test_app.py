"""Application policy tests for active-listener."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import cast, final

import numpy as np
import pytest
from pydantic import JsonValue
from structlog.stdlib import BoundLogger
from typing_extensions import override

from active_listener.app.ports import (
  ActiveListenerRuntimeError,
  FinalizedTranscriptRecord,
  RewriteResult,
)
from active_listener.app.service import ActiveListenerService
from active_listener.app.signals import (
  ConnectedSignal,
  DisconnectedSignal,
  ReconnectedSignal,
  ReconnectingSignal,
)
from active_listener.app.state import AppAction, AppActionDecision, ForegroundPhase
from active_listener.bootstrap import build_capture_callback, create_service, run_service
from active_listener.config.models import (
  ActiveListenerConfig,
  LiteRtRewriteProvider,
  LlmRewriteConfig,
)
from active_listener.infra.keyboard import KeyboardControlEvent, KeyboardEventKind, KeyboardInput
from active_listener.infra.langfuse import RecordingInput, RecordingOutput
from active_listener.infra.rewrite import (
  LoadedRewritePrompt,
  LoadedRewritePromptFile,
  RewriteClientError,
  RewriteClientTimeoutError,
  RewritePromptError,
)
from active_listener.recording.reducer import TextRun
from active_listener.recording.session import (
  RecordingAudioBuffer,
  RecordingTranscript,
)
from active_listener.recording.spectrum import (
  SAMPLE_RATE_HZ,
)
from eavesdrop.common import get_logger
from eavesdrop.wire import Segment, TranscriptionMessage, Word

RECORDING_CLOCK_START = 100.0


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
    ffmpeg_path=None,
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


def _keyboard_event(
  kind: KeyboardEventKind,
  *,
  received_monotonic_s: float = 0.0,
) -> KeyboardControlEvent:
  return KeyboardControlEvent(kind=kind, received_monotonic_s=received_monotonic_s)


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
  queued_events: list[KeyboardControlEvent] = field(default_factory=list)

  async def events(self) -> AsyncIterator[KeyboardControlEvent]:
    for event in self.queued_events:
      yield event

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
class FailingEmitter(FakeEmitter):
  @override
  def emit_text(self, text: str) -> None:
    _ = text
    raise RuntimeError("emit failed")


@dataclass
class FakeDbusService:
  states: list[ForegroundPhase] = field(default_factory=list)
  signals: list[tuple[str, str | bytes | list[TextRun] | tuple[str, str] | None]] = field(
    default_factory=list
  )
  events: list[tuple[str, str | bytes | list[TextRun] | tuple[str, str] | None]] = field(
    default_factory=list
  )
  close_calls: int = 0

  async def set_state(self, state: ForegroundPhase) -> None:
    if self.states and self.states[-1] is state:
      return
    self.states.append(state)
    self.events.append(("State", state))

  async def transcription_updated(self, runs: list[TextRun]) -> None:
    self.signals.append(("TranscriptionUpdated", runs))
    self.events.append(("TranscriptionUpdated", runs))

  async def spectrum_updated(self, bars: bytes) -> None:
    self.signals.append(("SpectrumUpdated", bars))
    self.events.append(("SpectrumUpdated", bars))

  async def recording_aborted(self, reason: str) -> None:
    self.signals.append(("RecordingAborted", reason))
    self.events.append(("RecordingAborted", reason))

  async def audio_archive_failed(self, reason: str) -> None:
    self.signals.append(("AudioArchiveFailed", reason))
    self.events.append(("AudioArchiveFailed", reason))

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
class FakeMediaPlaybackController:
  pause_results: list[bool] = field(default_factory=list)
  pause_calls: int = 0
  resume_calls: int = 0
  pause_error: Exception | None = None
  resume_error: Exception | None = None

  async def pause_if_playing(self) -> bool:
    self.pause_calls += 1
    if self.pause_error is not None:
      raise self.pause_error
    if not self.pause_results:
      return False
    return self.pause_results.pop(0)

  async def resume(self) -> None:
    self.resume_calls += 1
    if self.resume_error is not None:
      raise self.resume_error


@dataclass
class FakeClient:
  """Live-client stand-in used by app tests."""

  connect_calls: int = 0
  disconnect_calls: int = 0
  start_calls: int = 0
  stop_calls: int = 0
  cancel_calls: int = 0
  start_recording_ids: list[str] = field(default_factory=list)
  cancel_recording_ids: list[str] = field(default_factory=list)
  flush_recording_ids: list[str] = field(default_factory=list)
  flush_calls: list[bool] = field(default_factory=list)
  flush_results: list[asyncio.Future[TranscriptionMessage] | TranscriptionMessage | Exception] = (
    field(default_factory=list)
  )
  on_transcription: Callable[[TranscriptionMessage], None] | None = None
  on_connected: Callable[[str], None] | None = None
  on_disconnected: Callable[[str, str | None], None] | None = None
  on_reconnecting: Callable[[str, int, float], None] | None = None
  on_reconnected: Callable[[str], None] | None = None

  async def connect(self) -> None:
    self.connect_calls += 1

  async def disconnect(self) -> None:
    self.disconnect_calls += 1

  async def start_streaming(self, recording_id: str) -> None:
    self.start_calls += 1
    self.start_recording_ids.append(recording_id)

  async def stop_streaming(self) -> None:
    self.stop_calls += 1

  async def cancel_utterance(self, recording_id: str) -> None:
    self.cancel_calls += 1
    self.cancel_recording_ids.append(recording_id)

  async def flush(self, recording_id: str, *, force_complete: bool = True) -> TranscriptionMessage:
    self.flush_recording_ids.append(recording_id)
    self.flush_calls.append(force_complete)
    result = self.flush_results.pop(0)
    match result:
      case Exception():
        raise result
      case asyncio.Future():
        return await result
      case TranscriptionMessage():
        return result


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
class FailingStartClient(FakeClient):
  """Client stand-in whose start-streaming path fails."""

  @override
  async def start_streaming(self, recording_id: str) -> None:
    self.start_calls += 1
    self.start_recording_ids.append(recording_id)
    raise RuntimeError("start failed")


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
  corrections: dict[str, str] = field(default_factory=dict)
  error: Exception | None = None
  input_tokens: int | None = None
  output_tokens: int | None = None
  cost: Decimal | None = None
  calls: list[dict[str, str]] = field(default_factory=list)
  close_calls: int = 0

  async def rewrite_text(
    self,
    *,
    instructions: str,
    transcript: str,
  ) -> RewriteResult:
    self.calls.append(
      {
        "instructions": instructions,
        "transcript": transcript,
      }
    )
    if self.error is not None:
      raise self.error
    return RewriteResult(
      text=self.rewritten_text,
      model="fake-model",
      input_tokens=self.input_tokens,
      output_tokens=self.output_tokens,
      cost=self.cost,
      corrections=self.corrections,
    )

  async def close(self) -> None:
    self.close_calls += 1


@dataclass
class BlockingFirstRewriteClient(FakeRewriteClient):
  """Rewrite client whose first rewrite blocks until the test releases it."""

  first_call_started: asyncio.Event = field(default_factory=asyncio.Event)
  release_first: asyncio.Future[RewriteResult] | None = None
  call_count: int = 0

  @override
  async def rewrite_text(
    self,
    *,
    instructions: str,
    transcript: str,
  ) -> RewriteResult:
    self.call_count += 1
    if self.call_count == 1:
      self.calls.append(
        {
          "instructions": instructions,
          "transcript": transcript,
        }
      )
      self.first_call_started.set()
      if self.release_first is None:
        raise RuntimeError("first rewrite release future not configured")
      return await self.release_first
    return await super().rewrite_text(
      instructions=instructions,
      transcript=transcript,
    )


@dataclass
class FakeHistoryStore:
  recordings: list[RecordedFinalization] = field(default_factory=list)

  def record_finalized_recording(
    self,
    record: FinalizedTranscriptRecord,
    archived_audio: bytes | None,
  ) -> None:
    self.recordings.append(RecordedFinalization(record=record, archived_audio=archived_audio))


@dataclass(frozen=True)
class RecordedFinalization:
  record: FinalizedTranscriptRecord
  archived_audio: bytes | None


@dataclass
class FakeCorrectionStore:
  path: Path = Path("/tmp/active-listener.corrections.jsonc")
  loaded_corrections: dict[str, str] = field(default_factory=dict)
  load_started: asyncio.Event = field(default_factory=asyncio.Event)
  load_release: asyncio.Future[dict[str, str]] | None = None
  load_error: Exception | None = None
  merged_corrections: list[dict[str, str]] = field(default_factory=list)
  merge_error: Exception | None = None

  async def load_async(self) -> dict[str, str]:
    self.load_started.set()
    if self.load_release is not None:
      return await self.load_release
    if self.load_error is not None:
      raise self.load_error
    return self.loaded_corrections

  async def merge_async(self, corrections: Mapping[str, str]) -> None:
    if self.merge_error is not None:
      raise self.merge_error
    self.merged_corrections.append(dict(corrections))


def _segment(
  segment_id: int,
  text: str,
  *,
  completed: bool,
  start: float = 0.0,
  end: float = 0.1,
  words: list[Word] | None = None,
) -> Segment:
  return Segment(
    id=segment_id,
    start=start,
    end=end,
    text=text,
    words=(
      words if words is not None else [_word(token, start=start, end=end) for token in text.split()]
    ),
    completed=completed,
  )


def _word(text: str, *, start: float, end: float) -> Word:
  return Word(start=start, end=end, word=text)


@dataclass
class RecordingClock:
  """Control recording-relative hardware times without changing asyncio's clock."""

  now: float = RECORDING_CLOCK_START

  def monotonic(self) -> float:
    return self.now


@pytest.fixture
def recording_clock(monkeypatch: pytest.MonkeyPatch) -> RecordingClock:
  clock = RecordingClock()
  monkeypatch.setattr("active_listener.recording.session.time", clock)
  return clock


async def _command_hold(service: ActiveListenerService, start_s: float, end_s: float) -> None:
  await service.handle_keyboard_event(
    _keyboard_event(
      KeyboardEventKind.CAPSLOCK_DOWN, received_monotonic_s=RECORDING_CLOCK_START + start_s
    )
  )
  await service.handle_keyboard_event(
    _keyboard_event(
      KeyboardEventKind.CAPSLOCK_UP, received_monotonic_s=RECORDING_CLOCK_START + end_s
    )
  )


def _active_recording_id(service: ActiveListenerService) -> str:
  return cast(FakeClient, service.client).start_recording_ids[-1]


def _message(
  *segments: Segment,
  stream: str = "stream-1",
  recording_id: str | None = None,
  flush_complete: bool | None = True,
) -> TranscriptionMessage:
  return TranscriptionMessage(
    stream=stream,
    segments=list(segments),
    recording_id=recording_id,
    flush_complete=flush_complete,
  )


def _live_message(
  service: ActiveListenerService,
  *segments: Segment,
  stream: str = "stream-1",
  flush_complete: bool | None = True,
) -> TranscriptionMessage:
  return _message(
    *segments,
    stream=stream,
    recording_id=_active_recording_id(service),
    flush_complete=flush_complete,
  )


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
  logger: BoundLogger
  rewrite_client: FakeRewriteClient
  history_store: FakeHistoryStore
  dbus_service: FakeDbusService
  media_playback_controller: FakeMediaPlaybackController
  recording_audio_buffer: RecordingAudioBuffer
  correction_store: FakeCorrectionStore


def _service(
  *,
  client: FakeClient | None = None,
  keyboard: FakeKeyboard | None = None,
  emitter: FakeEmitter | None = None,
  logger: BoundLogger | None = None,
  rewrite_client: FakeRewriteClient | None = None,
  history_store: FakeHistoryStore | None = None,
  dbus_service: FakeDbusService | None = None,
  media_playback_controller: FakeMediaPlaybackController | None = None,
  config: ActiveListenerConfig | None = None,
  spectrum_analyzer: FakeSpectrumAnalyzer | None = None,
  recording_audio_buffer: RecordingAudioBuffer | None = None,
  correction_store: FakeCorrectionStore | None = None,
) -> ServiceHarness:
  resolved_client = client or FakeClient()
  resolved_keyboard = keyboard or FakeKeyboard()
  resolved_emitter = emitter or FakeEmitter()
  resolved_logger = logger or get_logger("test/listener")
  resolved_rewrite_client = rewrite_client or FakeRewriteClient()
  resolved_history_store = history_store or FakeHistoryStore()
  resolved_dbus_service = dbus_service or FakeDbusService()
  resolved_media_playback_controller = media_playback_controller or FakeMediaPlaybackController()
  resolved_recording_audio_buffer = recording_audio_buffer or RecordingAudioBuffer()
  resolved_correction_store = correction_store or FakeCorrectionStore()
  service = ActiveListenerService(
    config=config or _config(),
    keyboard=resolved_keyboard,
    client=resolved_client,
    emitter=resolved_emitter,
    logger=resolved_logger,
    rewrite_client=resolved_rewrite_client,
    history_store=resolved_history_store,
    dbus_service=resolved_dbus_service,
    media_playback_controller=resolved_media_playback_controller,
    recording_audio_buffer=resolved_recording_audio_buffer,
    correction_store=resolved_correction_store,
    spectrum_analyzer=spectrum_analyzer or FakeSpectrumAnalyzer(),
  )
  return ServiceHarness(
    service,
    resolved_client,
    resolved_keyboard,
    resolved_emitter,
    resolved_logger,
    resolved_rewrite_client,
    resolved_history_store,
    resolved_dbus_service,
    resolved_media_playback_controller,
    resolved_recording_audio_buffer,
    resolved_correction_store,
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
    history_store_factory=lambda _config, _logger, _dbus_service: FakeHistoryStore(),
  )

  assert service.phase is ForegroundPhase.IDLE
  assert service.current_llm_available() is False
  assert service.current_llm_active() is False
  assert client.connect_calls == 1
  assert emitter.initialize_calls == 1
  assert dbus_service.states == [ForegroundPhase.IDLE]


@pytest.mark.asyncio
async def test_create_service_defaults_llm_runtime_active_when_rewrite_is_configured() -> None:
  service = await create_service(
    _config(rewrite_enabled=True),
    dbus_service=FakeDbusService(),
    keyboard_resolver=lambda _name: FakeKeyboard(),
    client_factory=lambda _config, _on_capture: FakeClient(),
    emitter_factory=lambda: FakeEmitter(),
    rewrite_client_factory=lambda _config: FakeRewriteClient(),
    history_store_factory=lambda _config, _logger, _dbus_service: FakeHistoryStore(),
  )

  assert service.current_llm_available() is True
  assert service.current_llm_active() is True


@pytest.mark.asyncio
async def test_create_service_fails_fast_on_keyboard_resolution_error() -> None:
  with pytest.raises(ActiveListenerRuntimeError, match="missing keyboard"):
    _ = await create_service(
      _config(),
      keyboard_resolver=lambda _name: (_ for _ in ()).throw(RuntimeError("missing keyboard")),
      history_store_factory=lambda _config, _logger, _dbus_service: FakeHistoryStore(),
    )


@pytest.mark.asyncio
async def test_create_service_fails_fast_on_connect_error(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  logger = get_logger("test/listener")
  keyboard = FakeKeyboard()
  client = FailingConnectClient()
  rewrite_client = FakeRewriteClient()

  monkeypatch.setattr("active_listener.bootstrap.get_logger", lambda _name: logger)

  with pytest.raises(ActiveListenerRuntimeError):
    _ = await create_service(
      _config(),
      keyboard_resolver=lambda _name: keyboard,
      client_factory=lambda _config, _on_capture: client,
      emitter_factory=lambda: FakeEmitter(),
      rewrite_client_factory=lambda _config: rewrite_client,
      history_store_factory=lambda _config, _logger, _dbus_service: FakeHistoryStore(),
    )

  assert keyboard.close_calls == 1
  assert rewrite_client.close_calls == 1
  assert client.disconnect_calls == 1


@pytest.mark.asyncio
async def test_create_service_fails_fast_when_rewrite_client_cannot_initialize() -> None:
  keyboard = FakeKeyboard()
  client = FakeClient()

  with pytest.raises(ActiveListenerRuntimeError, match="bad model"):
    _ = await create_service(
      _config(rewrite_enabled=True),
      keyboard_resolver=lambda _name: keyboard,
      client_factory=lambda _config, _on_capture: client,
      emitter_factory=lambda: FakeEmitter(),
      rewrite_client_factory=lambda _config: (_ for _ in ()).throw(RewriteClientError("bad model")),
      history_store_factory=lambda _config, _logger, _dbus_service: FakeHistoryStore(),
    )

  assert keyboard.close_calls == 1
  assert client.disconnect_calls == 0


@pytest.mark.asyncio
async def test_start_and_cancel_recording_toggle_grab_and_streaming() -> None:
  spectrum_analyzer = FakeSpectrumAnalyzer()
  harness = _service(spectrum_analyzer=spectrum_analyzer)
  service = harness.service

  start_decision = await service.handle_action(AppAction.START_OR_FINISH)
  cancel_decision = await service.handle_action(AppAction.CANCEL)

  assert service.phase is ForegroundPhase.IDLE
  assert start_decision is AppActionDecision.START_RECORDING
  assert cancel_decision is AppActionDecision.CANCEL_RECORDING
  assert harness.client.start_calls == 1
  assert harness.client.stop_calls == 1
  assert harness.client.cancel_calls == 1
  assert len(harness.client.start_recording_ids) == 1
  assert harness.client.cancel_recording_ids == [harness.client.start_recording_ids[0]]
  assert harness.keyboard.grab_calls == 1
  assert harness.keyboard.ungrab_calls == 1
  assert spectrum_analyzer.start_calls == 1
  assert spectrum_analyzer.stop_calls == 1


@pytest.mark.asyncio
async def test_start_and_cancel_resume_prior_media() -> None:
  media_playback_controller = FakeMediaPlaybackController(pause_results=[True])
  harness = _service(media_playback_controller=media_playback_controller)
  service = harness.service

  _ = await service.handle_action(AppAction.START_OR_FINISH)
  _ = await service.handle_action(AppAction.CANCEL)

  assert media_playback_controller.pause_calls == 1
  assert media_playback_controller.resume_calls == 1


@pytest.mark.asyncio
async def test_start_and_finish_resume_prior_media() -> None:
  media_playback_controller = FakeMediaPlaybackController(pause_results=[True])
  client = FakeClient(
    flush_results=[
      _message(
        _segment(1, "alpha", completed=True),
        _segment(2, "tail", completed=False),
      )
    ]
  )
  harness = _service(
    client=client,
    media_playback_controller=media_playback_controller,
  )

  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()

  assert media_playback_controller.pause_calls == 1
  assert media_playback_controller.resume_calls == 1


@pytest.mark.asyncio
async def test_media_pause_failure_does_not_block_recording_start() -> None:
  media_playback_controller = FakeMediaPlaybackController(pause_error=RuntimeError("pause boom"))
  harness = _service(media_playback_controller=media_playback_controller)

  decision = await harness.service.handle_action(AppAction.START_OR_FINISH)

  assert decision is AppActionDecision.START_RECORDING
  assert harness.service.phase is ForegroundPhase.RECORDING
  assert harness.client.start_calls == 1
  assert media_playback_controller.pause_calls == 1
  assert media_playback_controller.resume_calls == 0


@pytest.mark.asyncio
async def test_start_failure_resumes_prior_media() -> None:
  media_playback_controller = FakeMediaPlaybackController(pause_results=[True])
  harness = _service(
    client=FailingStartClient(),
    media_playback_controller=media_playback_controller,
  )

  with pytest.raises(RuntimeError, match="start failed"):
    _ = await harness.service.handle_action(AppAction.START_OR_FINISH)

  assert media_playback_controller.pause_calls == 1
  assert media_playback_controller.resume_calls == 1
  assert harness.service.phase is ForegroundPhase.IDLE


@pytest.mark.asyncio
async def test_set_llm_active_fails_when_rewrite_is_unavailable() -> None:
  harness = _service()

  with pytest.raises(RuntimeError, match="llm unavailable"):
    _ = await harness.service.set_llm_active(True)


@pytest.mark.asyncio
async def test_set_llm_active_updates_runtime_flag() -> None:
  harness = _service(config=_config(rewrite_enabled=True))

  assert await harness.service.set_llm_active(False) is False
  assert harness.service.current_llm_active() is False


@pytest.mark.asyncio
async def test_cancel_clears_foreground_state_before_stop_streaming_returns() -> None:
  harness = _service(client=FailingStopClient())
  service = harness.service

  _ = await service.handle_action(AppAction.START_OR_FINISH)

  with pytest.raises(RuntimeError, match="stop failed"):
    _ = await service.handle_action(AppAction.CANCEL)

  assert service.phase is ForegroundPhase.IDLE
  assert harness.client.cancel_calls == 0
  assert harness.keyboard.grabbed is False
  assert harness.keyboard.ungrab_calls == 1


@pytest.mark.asyncio
async def test_escape_is_ignored_while_idle() -> None:
  harness = _service()
  service = harness.service

  decision = await service.handle_action(AppAction.CANCEL)

  assert service.phase is ForegroundPhase.IDLE
  assert decision is AppActionDecision.IGNORE
  assert harness.client.stop_calls == 0
  assert harness.keyboard.ungrab_calls == 0


@pytest.mark.asyncio
async def test_caps_lock_is_suppressed_while_reconnecting() -> None:
  harness = _service()
  service = harness.service
  service.phase = ForegroundPhase.RECONNECTING

  decision = await service.handle_action(AppAction.START_OR_FINISH)

  assert decision is AppActionDecision.SUPPRESS_RECONNECTING_START
  assert harness.client.start_calls == 0
  assert harness.keyboard.grab_calls == 0


@pytest.mark.asyncio
async def test_disconnect_during_recording_forces_local_abort() -> None:
  spectrum = FakeSpectrumAnalyzer()
  media = FakeMediaPlaybackController(pause_results=[True])
  harness = _service(spectrum_analyzer=spectrum, media_playback_controller=media)
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await harness.service.handle_client_event(
    DisconnectedSignal(session_id="stream-1", reason="closed")
  )
  assert harness.service.phase is ForegroundPhase.RECONNECTING
  assert harness.keyboard.grabbed is False
  assert harness.dbus_service.signals == [("RecordingAborted", "closed"), ("Reconnecting", None)]
  assert media.resume_calls == 1 and spectrum.stop_calls == 1


@pytest.mark.asyncio
async def test_disconnect_without_reason_uses_truthful_dbus_fallback() -> None:
  harness = _service()
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)

  await harness.service.handle_client_event(DisconnectedSignal(session_id="stream-1", reason=None))

  assert harness.dbus_service.signals == [
    ("RecordingAborted", "unknown disconnect reason"),
    ("Reconnecting", None),
  ]


@pytest.mark.asyncio
async def test_finish_clears_foreground_state_before_stop_streaming_returns() -> None:
  harness = _service(client=FailingStopClient())
  service = harness.service

  _ = await service.handle_action(AppAction.START_OR_FINISH)

  with pytest.raises(RuntimeError, match="stop failed"):
    _ = await service.handle_action(AppAction.START_OR_FINISH)

  assert service.phase is ForegroundPhase.IDLE
  assert harness.keyboard.grabbed is False
  assert harness.keyboard.ungrab_calls == 1


@pytest.mark.asyncio
async def test_close_releases_keyboard_even_when_disconnect_raises() -> None:
  spectrum_analyzer = FakeSpectrumAnalyzer()
  harness = _service(client=FailingDisconnectClient(), spectrum_analyzer=spectrum_analyzer)
  service = harness.service

  _ = await service.handle_action(AppAction.START_OR_FINISH)

  with pytest.raises(RuntimeError, match="disconnect failed"):
    await service.close()

  assert harness.rewrite_client.close_calls == 1
  assert harness.keyboard.close_calls == 1
  assert harness.keyboard.grabbed is False
  assert spectrum_analyzer.stop_calls == 1


@pytest.mark.asyncio
async def test_close_resumes_prior_media() -> None:
  media_playback_controller = FakeMediaPlaybackController(pause_results=[True])
  harness = _service(media_playback_controller=media_playback_controller)

  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await harness.service.close()

  assert media_playback_controller.pause_calls == 1
  assert media_playback_controller.resume_calls == 1


@pytest.mark.asyncio
async def test_resume_failure_does_not_block_recording_cancel() -> None:
  media_playback_controller = FakeMediaPlaybackController(
    pause_results=[True],
    resume_error=RuntimeError("resume boom"),
  )
  harness = _service(media_playback_controller=media_playback_controller)

  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  decision = await harness.service.handle_action(AppAction.CANCEL)

  assert decision is AppActionDecision.CANCEL_RECORDING
  assert harness.service.phase is ForegroundPhase.IDLE
  assert harness.client.cancel_calls == 1
  assert media_playback_controller.resume_calls == 1


@pytest.mark.asyncio
async def test_close_after_cancel_does_not_resume_twice() -> None:
  media_playback_controller = FakeMediaPlaybackController(pause_results=[True])
  harness = _service(media_playback_controller=media_playback_controller)

  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  _ = await harness.service.handle_action(AppAction.CANCEL)
  await harness.service.close()

  assert media_playback_controller.resume_calls == 1


@pytest.mark.asyncio
async def test_spectrum_failure_does_not_lose_recording_audio() -> None:
  logger = get_logger("test/listener")
  audio_buffer = RecordingAudioBuffer()
  audio_buffer.start()
  on_capture = build_capture_callback(
    spectrum_analyzer=ExplodingSpectrumAnalyzer(),
    audio_buffer=audio_buffer,
    logger=logger,
  )

  on_capture(b"chunk")

  assert audio_buffer.finish() == b"chunk"


def test_capture_callback_records_audio_only_while_buffer_is_active() -> None:
  logger = get_logger("test/listener")
  spectrum_analyzer = FakeSpectrumAnalyzer()
  audio_buffer = RecordingAudioBuffer()
  on_capture = build_capture_callback(
    spectrum_analyzer=spectrum_analyzer,
    audio_buffer=audio_buffer,
    logger=logger,
  )
  first_chunk = _sine_wave_bytes(220.0, 16)
  second_chunk = _sine_wave_bytes(330.0, 16)

  on_capture(first_chunk)
  audio_buffer.start()
  on_capture(first_chunk)
  on_capture(second_chunk)
  captured_snapshot = audio_buffer.finish()
  on_capture(first_chunk)

  assert spectrum_analyzer.ingested == [first_chunk, first_chunk, second_chunk, first_chunk]
  assert captured_snapshot == first_chunk + second_chunk
  assert audio_buffer.finish() == b""


@pytest.mark.asyncio
async def test_close_still_disconnects_when_stop_streaming_fails() -> None:
  harness = _service(client=FailingStopClient())
  service = harness.service

  _ = await service.handle_action(AppAction.START_OR_FINISH)

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
      history_store_factory=lambda _config, _logger, _dbus_service: FakeHistoryStore(),
    )

  assert dbus_service.states == []
  assert dbus_service.signals == [("FatalError", "keyboard missing")]
  assert dbus_service.close_calls == 1


@pytest.mark.asyncio
async def test_run_service_emits_fatal_error_once_on_runtime_failure() -> None:
  keyboard = FakeKeyboard(
    queued_events=[
      _keyboard_event(KeyboardEventKind.CAPSLOCK_DOWN),
      _keyboard_event(KeyboardEventKind.ESCAPE_DOWN),
    ]
  )
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
      history_store_factory=lambda _config, _logger, _dbus_service: FakeHistoryStore(),
    )

  assert dbus_service.states == [ForegroundPhase.IDLE, ForegroundPhase.RECORDING]
  assert dbus_service.signals == [("FatalError", "stop failed")]
  assert dbus_service.close_calls == 1


@pytest.mark.asyncio
async def test_create_service_resolves_configured_ffmpeg_path(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  logger = get_logger("test/listener")
  keyboard = FakeKeyboard()
  ffmpeg_path = tmp_path / "bin" / "ffmpeg"
  ffmpeg_path.parent.mkdir(parents=True)
  _ = ffmpeg_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
  ffmpeg_path.chmod(0o755)

  class CapturingHistoryStore:
    def __init__(
      self,
      *,
      logger: BoundLogger,
      dbus_service: FakeDbusService,
    ) -> None:
      _ = logger
      _ = dbus_service

    def record_finalized_recording(
      self,
      record: FinalizedTranscriptRecord,
      archived_audio: bytes | None,
    ) -> None:
      _ = record
      _ = archived_audio

  monkeypatch.setattr("active_listener.bootstrap.get_logger", lambda _name: logger)
  monkeypatch.setattr(
    "active_listener.bootstrap.SqliteTranscriptHistoryStore",
    CapturingHistoryStore,
  )

  service = await create_service(
    ActiveListenerConfig(
      keyboard_name="Exact Keyboard",
      host="localhost",
      port=9090,
      audio_device="default",
      ffmpeg_path=str(ffmpeg_path),
    ),
    dbus_service=FakeDbusService(),
    keyboard_resolver=lambda _name: keyboard,
    client_factory=lambda _config, _on_capture: FakeClient(),
    emitter_factory=lambda: FakeEmitter(),
    rewrite_client_factory=lambda _config: FakeRewriteClient(),
  )

  assert service.phase is ForegroundPhase.IDLE


@pytest.mark.asyncio
async def test_create_service_falls_back_to_path_for_ffmpeg_resolution(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  logger = get_logger("test/listener")
  keyboard = FakeKeyboard()
  ffmpeg_path = tmp_path / "path-bin" / "ffmpeg"
  ffmpeg_path.parent.mkdir(parents=True)
  _ = ffmpeg_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
  ffmpeg_path.chmod(0o755)

  class CapturingHistoryStore:
    def __init__(
      self,
      *,
      logger: BoundLogger,
      dbus_service: FakeDbusService,
    ) -> None:
      _ = logger
      _ = dbus_service

    def record_finalized_recording(
      self,
      record: FinalizedTranscriptRecord,
      archived_audio: bytes | None,
    ) -> None:
      _ = record
      _ = archived_audio

  monkeypatch.setattr("active_listener.bootstrap.get_logger", lambda _name: logger)
  monkeypatch.setattr("active_listener.infra.audio.shutil.which", lambda _name: str(ffmpeg_path))
  monkeypatch.setattr(
    "active_listener.bootstrap.SqliteTranscriptHistoryStore",
    CapturingHistoryStore,
  )

  service = await create_service(
    _config(),
    dbus_service=FakeDbusService(),
    keyboard_resolver=lambda _name: keyboard,
    client_factory=lambda _config, _on_capture: FakeClient(),
    emitter_factory=lambda: FakeEmitter(),
    rewrite_client_factory=lambda _config: FakeRewriteClient(),
  )

  assert service.phase is ForegroundPhase.IDLE


@pytest.mark.asyncio
async def test_create_service_fails_fast_when_ffmpeg_cannot_be_resolved(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  logger = get_logger("test/listener")
  keyboard = FakeKeyboard()

  monkeypatch.setattr("active_listener.bootstrap.get_logger", lambda _name: logger)
  monkeypatch.setattr("active_listener.infra.audio.shutil.which", lambda _name: None)

  with pytest.raises(
    ActiveListenerRuntimeError,
    match="ffmpeg executable not found; set ffmpeg_path or add ffmpeg to PATH",
  ):
    _ = await create_service(
      _config(),
      keyboard_resolver=lambda _name: keyboard,
      client_factory=lambda _config, _on_capture: FakeClient(),
      emitter_factory=lambda: FakeEmitter(),
      rewrite_client_factory=lambda _config: FakeRewriteClient(),
    )

  assert keyboard.close_calls == 1


@pytest.mark.asyncio
async def test_reconnecting_and_reconnected_events_update_phase() -> None:
  harness = _service()
  service = harness.service

  await service.handle_client_event(
    ReconnectingSignal(session_id="stream-1", attempt=2, retry_delay_s=10.0)
  )
  await service.handle_client_event(ReconnectedSignal(session_id="stream-1"))

  assert service.phase is ForegroundPhase.IDLE
  assert harness.dbus_service.states == [ForegroundPhase.RECONNECTING, ForegroundPhase.IDLE]
  assert harness.dbus_service.signals == [
    ("Reconnecting", None),
    ("Reconnected", None),
  ]


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

  start_decision = await harness.service.handle_action(AppAction.START_OR_FINISH)
  finish_decision = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()

  assert start_decision is AppActionDecision.START_RECORDING
  assert finish_decision is AppActionDecision.FINISH_RECORDING
  assert harness.dbus_service.states == [ForegroundPhase.RECORDING, ForegroundPhase.IDLE]
  assert harness.dbus_service.signals == []
  assert client.cancel_calls == 0
  assert client.flush_calls == [True]
  assert len(client.start_recording_ids) == 1
  assert client.flush_recording_ids == [client.start_recording_ids[0]]


@pytest.mark.asyncio
async def test_new_start_waits_for_finished_recording_flush_to_complete() -> None:
  pending_flush: asyncio.Future[TranscriptionMessage] = asyncio.Future()
  client = FakeClient(flush_results=[pending_flush])
  harness = _service(client=client)
  service = harness.service

  _ = await service.handle_action(AppAction.START_OR_FINISH)
  first_recording_id = client.start_recording_ids[0]

  finish_decision = await service.handle_action(AppAction.START_OR_FINISH)
  second_start = asyncio.create_task(service.handle_action(AppAction.START_OR_FINISH))
  await asyncio.sleep(0)

  assert finish_decision is AppActionDecision.FINISH_RECORDING
  assert second_start.done() is False
  assert client.flush_recording_ids == [first_recording_id]

  pending_flush.set_result(
    _message(
      _segment(1, "alpha", completed=True),
      _segment(2, "tail", completed=False),
      recording_id=first_recording_id,
    )
  )
  start_decision = await second_start

  assert start_decision is AppActionDecision.START_RECORDING
  assert len(client.start_recording_ids) == 2
  assert client.start_recording_ids[1] != first_recording_id

  _ = await service.handle_action(AppAction.CANCEL)


@pytest.mark.asyncio
async def test_idle_capslock_up_is_ignored_but_down_starts_recording() -> None:
  harness = _service()

  await harness.service.handle_keyboard_event(_keyboard_event(KeyboardEventKind.CAPSLOCK_UP))

  assert harness.service.phase is ForegroundPhase.IDLE
  assert harness.client.start_calls == 0

  await harness.service.handle_keyboard_event(_keyboard_event(KeyboardEventKind.CAPSLOCK_DOWN))

  assert harness.service.phase is ForegroundPhase.RECORDING
  assert harness.client.start_calls == 1


@pytest.mark.asyncio
async def test_idle_disconnect_sets_reconnecting_without_extra_dbus_signal() -> None:
  harness = _service()

  await harness.service.handle_client_event(
    DisconnectedSignal(session_id="stream-1", reason="socket closed")
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
    _message(
      _segment(90, "ignored", completed=True),
      _segment(91, "tail", completed=False),
      flush_complete=False,
    )
  )
  _ = await service.handle_action(AppAction.START_OR_FINISH)
  await service.handle_client_event(
    _live_message(
      service,
      _segment(1, "alpha", completed=True),
      _segment(2, "draft", completed=False),
      flush_complete=False,
    )
  )
  _ = await service.handle_action(AppAction.START_OR_FINISH)
  await service.wait_for_background_tasks()

  assert service.phase is ForegroundPhase.IDLE
  assert client.stop_calls == 1
  assert client.flush_calls == [True]
  assert harness.dbus_service.signals == [
    (
      "TranscriptionUpdated",
      [
        TextRun(text="alpha", is_command=False, is_complete=True),
        TextRun(text="draft", is_command=False, is_complete=False),
      ],
    ),
  ]
  assert emitter.emitted == ["alpha bravo "]
  assert harness.keyboard.ungrab_calls == 1


@pytest.mark.asyncio
async def test_cancel_clears_connection_cursor_for_next_recording() -> None:
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

  _ = await service.handle_action(AppAction.START_OR_FINISH)
  await service.handle_client_event(
    _live_message(
      service,
      _segment(1, "alpha", completed=True),
      _segment(2, "tail", completed=False),
      flush_complete=False,
    )
  )
  _ = await service.handle_action(AppAction.CANCEL)
  _ = await service.handle_action(AppAction.START_OR_FINISH)
  _ = await service.handle_action(AppAction.START_OR_FINISH)
  await service.wait_for_background_tasks()

  assert client.cancel_calls == 1
  assert client.flush_calls == [True]
  assert emitter.emitted == ["alpha bravo "]


@pytest.mark.asyncio
async def test_new_recording_ignores_completed_history_before_seeded_cursor() -> None:
  client = FakeClient(
    flush_results=[
      _message(
        _segment(1, "alpha", completed=True),
        _segment(2, "tail", completed=False),
      ),
      _message(
        _segment(1, "bravo", completed=True),
        _segment(2, "charlie", completed=True),
        _segment(3, "tail", completed=False),
      ),
    ]
  )
  emitter = FakeEmitter()
  harness = _service(client=client, emitter=emitter)
  service = harness.service

  _ = await service.handle_action(AppAction.START_OR_FINISH)
  _ = await service.handle_action(AppAction.START_OR_FINISH)
  await service.wait_for_background_tasks()

  _ = await service.handle_action(AppAction.START_OR_FINISH)
  _ = await service.handle_action(AppAction.START_OR_FINISH)
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

  _ = await service.handle_action(AppAction.START_OR_FINISH)
  _ = await service.handle_action(AppAction.START_OR_FINISH)
  await service.wait_for_background_tasks()

  await service.handle_client_event(
    ReconnectingSignal(session_id="stream-1", attempt=2, retry_delay_s=10.0)
  )
  await service.handle_client_event(ReconnectedSignal(session_id="stream-1"))

  _ = await service.handle_action(AppAction.START_OR_FINISH)
  _ = await service.handle_action(AppAction.START_OR_FINISH)
  await service.wait_for_background_tasks()

  assert client.flush_calls == [True, True]
  assert emitter.emitted == ["alpha ", "fresh "]


@pytest.mark.asyncio
async def test_new_recording_waits_while_older_finalization_flush_is_pending() -> None:
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

  _ = await service.handle_action(AppAction.START_OR_FINISH)
  _ = await service.handle_action(AppAction.START_OR_FINISH)
  blocked_start = asyncio.create_task(service.handle_action(AppAction.START_OR_FINISH))
  await asyncio.sleep(0)

  assert blocked_start.done() is False

  pending_flush.set_result(
    _message(
      _segment(2, "first", completed=True),
      _segment(3, "tail", completed=False),
    )
  )
  assert await blocked_start is AppActionDecision.START_RECORDING
  _ = await service.handle_action(AppAction.START_OR_FINISH)
  await service.wait_for_background_tasks()

  assert client.start_calls == 2
  assert client.stop_calls == 2
  assert client.flush_calls == [True, True]
  assert emitter.emitted == ["first ", "second "]
  assert harness.keyboard.grab_calls == 2
  assert harness.keyboard.ungrab_calls == 2


@pytest.mark.asyncio
async def test_new_commit_flushes_while_older_rewrite_is_still_running(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Commit must clear the server operation boundary before waiting on older rewrite work."""
  rewrite_client = BlockingFirstRewriteClient()
  first_rewrite_result = asyncio.get_running_loop().create_future()
  rewrite_client.release_first = first_rewrite_result
  client = FakeClient(
    flush_results=[
      _message(
        _segment(2, "first", completed=True),
        _segment(3, "tail", completed=False),
      ),
      _message(
        _segment(3, "second", completed=True),
        _segment(4, "tail", completed=False),
      ),
    ]
  )
  emitter = FakeEmitter()
  harness = _service(
    config=_config(rewrite_enabled=True),
    client=client,
    emitter=emitter,
    rewrite_client=rewrite_client,
  )
  _ = monkeypatch.setattr(
    "active_listener.infra.rewrite.load_active_listener_rewrite_prompt",
    _prompt_loader(_loaded_prompt_file(_prompt())),
  )
  service = harness.service

  _ = await service.handle_action(AppAction.START_OR_FINISH)
  _ = await service.handle_action(AppAction.START_OR_FINISH)
  _ = await asyncio.wait_for(rewrite_client.first_call_started.wait(), timeout=0.5)

  _ = await service.handle_action(AppAction.START_OR_FINISH)
  _ = await service.handle_action(AppAction.START_OR_FINISH)
  await asyncio.sleep(0)

  assert client.flush_calls == [True, True]
  assert emitter.emitted == []

  first_rewrite_result.set_result(
    RewriteResult(
      text="first rewritten",
      model="fake-model",
      input_tokens=None,
      output_tokens=None,
      cost=None,
      corrections={},
    )
  )
  await service.wait_for_background_tasks()

  assert emitter.emitted == ["first rewritten ", "rewritten text "]
  assert rewrite_client.call_count == 2


@pytest.mark.asyncio
async def test_disconnect_during_background_finalization_preserves_accepted_result() -> None:
  pending_flush: asyncio.Future[TranscriptionMessage] = asyncio.get_running_loop().create_future()
  harness = _service(client=FakeClient(flush_results=[pending_flush]))
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  pending_flush.set_result(_message(_segment(1, "alpha", completed=True)))
  await harness.service.handle_client_event(
    DisconnectedSignal(session_id="stream-1", reason="closed")
  )
  await harness.service.wait_for_background_tasks()
  assert harness.emitter.emitted == ["alpha "]
  assert harness.service.phase is ForegroundPhase.RECONNECTING


@pytest.mark.asyncio
async def test_provisional_revisions_remain_raw_until_force_flush() -> None:
  harness = _service(
    client=FakeClient(flush_results=[_message(_segment(1, "alpha bravo", completed=True))])
  )
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  for text in ("draft rough", "alpha bravo"):
    await harness.service.handle_client_event(
      _live_message(harness.service, _segment(1, text, completed=False), flush_complete=False)
    )
  published = [
    runs for name, runs in harness.dbus_service.signals if name == "TranscriptionUpdated"
  ]
  assert published == [
    [TextRun(text="draft rough", is_command=False, is_complete=False)],
    [TextRun(text="alpha bravo", is_command=False, is_complete=False)],
  ]
  assert harness.emitter.emitted == []
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()
  assert harness.emitter.emitted == ["alpha bravo "]


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

  _ = await service.handle_action(AppAction.START_OR_FINISH)
  await service.handle_client_event(
    _live_message(
      service,
      _segment(1, "alpha", completed=True),
      _segment(2, "draft", completed=False),
      flush_complete=False,
    )
  )
  await service.handle_client_event(
    _live_message(
      service,
      _segment(1, "alpha", completed=True),
      _segment(2, "bravo", completed=True),
      _segment(3, "draft", completed=False),
      flush_complete=False,
    )
  )
  _ = await service.handle_action(AppAction.START_OR_FINISH)
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
  logger = get_logger("test/listener")
  harness = _service(client=client, emitter=emitter, logger=logger)
  service = harness.service

  _ = await service.handle_action(AppAction.START_OR_FINISH)
  await service.handle_client_event(
    _live_message(
      service,
      _segment(1, "alpha", completed=True),
      _segment(2, "tail", completed=False),
      flush_complete=False,
    )
  )
  await service.handle_client_event(
    _live_message(
      service,
      _segment(7, "restart", completed=True),
      _segment(8, "draft", completed=False),
      flush_complete=False,
    )
  )
  _ = await service.handle_action(AppAction.START_OR_FINISH)
  await service.wait_for_background_tasks()

  assert emitter.emitted == ["alpha restart tail "]


@pytest.mark.asyncio
async def test_finalize_recording_emits_raw_text_when_rewrite_is_disabled(
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
  harness = _service(client=client)
  monkeypatch.setattr(
    "active_listener.recording.finalizer.encode_recording_audio",
    lambda *_args, **_kwargs: b"m4a-bytes",
  )
  captured_audio = _sine_wave_bytes(440.0, 32)

  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  harness.recording_audio_buffer.append(captured_audio)
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()

  assert harness.emitter.emitted == ["alpha "]
  assert harness.rewrite_client.calls == []

  assert harness.history_store.recordings == [
    RecordedFinalization(
      record=FinalizedTranscriptRecord(
        pre_finalization_text="alpha",
        post_finalization_text="alpha ",
        llm_model=None,
        tokens_in=None,
        tokens_out=None,
        cost=None,
        word_count=1,
        duration_seconds=0.1,
      ),
      archived_audio=b"m4a-bytes",
    )
  ]


@pytest.mark.asyncio
async def test_finalize_recording_normalizes_hillary_before_emitting_text() -> None:
  client = FakeClient(
    flush_results=[
      _message(
        _segment(1, "hillary", completed=True),
        _segment(2, "tail", completed=False),
      )
    ]
  )
  harness = _service(client=client)

  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()

  assert harness.emitter.emitted == ["hilary "]


@pytest.mark.asyncio
async def test_finalize_recording_normalizes_hashtag_before_emitting_text() -> None:
  client = FakeClient(
    flush_results=[
      _message(
        _segment(1, "release hashtag 42", completed=True),
        _segment(2, "tail", completed=False),
      )
    ]
  )
  harness = _service(client=client)

  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()

  assert harness.emitter.emitted == ["release#42 "]


@pytest.mark.asyncio
async def test_finalize_recording_drops_command_runs_when_llm_is_disabled(
  recording_clock: RecordingClock,
) -> None:
  client = FakeClient(
    flush_results=[
      _message(
        _segment(
          1,
          "Hello,",
          completed=True,
          start=0.0,
          end=0.2,
          words=[_word("Hello,", start=0.0, end=0.2)],
        ),
        _segment(
          2,
          "scratch that.",
          completed=True,
          start=0.3,
          end=0.6,
          words=[
            _word("scratch", start=0.3, end=0.45),
            _word("that.", start=0.45, end=0.6),
          ],
        ),
        _segment(
          3,
          "thank you, Bye.",
          completed=True,
          start=0.7,
          end=1.0,
          words=[
            _word("thank", start=0.7, end=0.8),
            _word("you,", start=0.8, end=0.9),
            _word("Bye.", start=0.9, end=1.0),
          ],
        ),
        _segment(4, "", completed=False),
      )
    ]
  )
  harness = _service(client=client, config=_config(rewrite_enabled=True))

  assert await harness.service.set_llm_active(False) is False

  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await _command_hold(harness.service, 0.3, 0.6)

  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()

  assert harness.emitter.emitted == ["Hello, thank you, Bye. "]
  assert harness.rewrite_client.calls == []


@pytest.mark.asyncio
async def test_live_transcription_publishes_shaped_runs(recording_clock: RecordingClock) -> None:
  harness = _service()
  service = harness.service

  _ = await service.handle_action(AppAction.START_OR_FINISH)
  await _command_hold(service, 0.6, 0.9)

  await service.handle_client_event(
    _live_message(
      service,
      _segment(
        1,
        "hillary release hashtag 42 thank you",
        completed=True,
        start=0.0,
        end=0.5,
        words=[
          _word("hillary", start=0.0, end=0.1),
          _word("release", start=0.1, end=0.2),
          _word("hashtag", start=0.2, end=0.3),
          _word("42", start=0.3, end=0.4),
          _word("thank", start=0.4, end=0.45),
          _word("you", start=0.45, end=0.5),
        ],
      ),
      _segment(
        2,
        "scratch that.",
        completed=False,
        start=0.6,
        end=0.9,
        words=[
          _word("scratch", start=0.6, end=0.75),
          _word("that.", start=0.75, end=0.9),
        ],
      ),
      flush_complete=False,
    )
  )
  _ = await service.handle_action(AppAction.CANCEL)
  await service.wait_for_background_tasks()

  published = [
    runs for name, runs in harness.dbus_service.signals if name == "TranscriptionUpdated"
  ]
  assert published == [
    [
      TextRun(text="hilary release#42 thank you", is_command=False, is_complete=True),
      TextRun(text="scratch that.", is_command=True, is_complete=False),
    ]
  ]


@pytest.mark.asyncio
async def test_live_overlay_leaves_incomplete_tail_unshaped() -> None:
  # Flicker guard: the incomplete tail is unstable as the transcriber revises it, so it is
  # shown raw. Spoken symbols should not fuse before the segment is committed.
  harness = _service()
  service = harness.service

  _ = await service.handle_action(AppAction.START_OR_FINISH)

  await service.handle_client_event(
    _live_message(
      service,
      _segment(
        1,
        "hillary",
        completed=True,
        start=0.0,
        end=0.2,
        words=[_word("hillary", start=0.0, end=0.2)],
      ),
      _segment(
        2,
        "dot slash",
        completed=False,
        start=0.3,
        end=0.5,
        words=[
          _word("dot", start=0.3, end=0.4),
          _word("slash", start=0.4, end=0.5),
        ],
      ),
      flush_complete=False,
    )
  )
  _ = await service.handle_action(AppAction.CANCEL)
  await service.wait_for_background_tasks()

  published = [
    runs for name, runs in harness.dbus_service.signals if name == "TranscriptionUpdated"
  ]
  assert published == [
    [
      TextRun(text="hilary", is_command=False, is_complete=True),
      TextRun(text="dot slash", is_command=False, is_complete=False),
    ]
  ]


@pytest.mark.asyncio
async def test_finalize_recording_applies_loaded_corrections_before_rewrite(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  client = FakeClient(
    flush_results=[
      _message(
        _segment(1, "I use pie torch", completed=True),
        _segment(2, "tail", completed=False),
      )
    ]
  )
  correction_store = FakeCorrectionStore(loaded_corrections={"pie torch": "PyTorch"})
  rewrite_client = FakeRewriteClient(rewritten_text="I use PyTorch")
  harness = _service(
    client=client,
    correction_store=correction_store,
    rewrite_client=rewrite_client,
    config=_config(rewrite_enabled=True),
  )
  monkeypatch.setattr(
    "active_listener.infra.rewrite.load_active_listener_rewrite_prompt",
    _prompt_loader(_loaded_prompt_file(_prompt())),
  )

  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()

  assert rewrite_client.calls == [
    {"instructions": "Rewrite this transcript.", "transcript": "I use PyTorch"}
  ]
  assert harness.emitter.emitted == ["I use PyTorch "]


@pytest.mark.asyncio
async def test_finalize_recording_waits_for_started_correction_load_before_rewrite(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  correction_load = asyncio.get_running_loop().create_future()
  correction_store = FakeCorrectionStore(load_release=correction_load)
  rewrite_client = FakeRewriteClient(rewritten_text="rewritten alpha")
  client = FakeClient(
    flush_results=[
      _message(
        _segment(1, "alfa", completed=True),
        _segment(2, "tail", completed=False),
      )
    ]
  )
  harness = _service(
    client=client,
    correction_store=correction_store,
    rewrite_client=rewrite_client,
    config=_config(rewrite_enabled=True),
  )
  monkeypatch.setattr(
    "active_listener.infra.rewrite.load_active_listener_rewrite_prompt",
    _prompt_loader(_loaded_prompt_file(_prompt())),
  )

  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await asyncio.sleep(0)

  assert rewrite_client.calls == []

  correction_load.set_result({"alfa": "alpha"})
  await harness.service.wait_for_background_tasks()

  assert rewrite_client.calls == [
    {"instructions": "Rewrite this transcript.", "transcript": "alpha"}
  ]


@pytest.mark.asyncio
async def test_finalize_recording_continues_when_correction_load_fails(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  correction_store = FakeCorrectionStore(load_error=RuntimeError("bad jsonc"))
  rewrite_client = FakeRewriteClient(rewritten_text="rewritten alpha")
  client = FakeClient(
    flush_results=[
      _message(
        _segment(1, "alpha", completed=True),
        _segment(2, "tail", completed=False),
      )
    ]
  )
  harness = _service(
    client=client,
    correction_store=correction_store,
    rewrite_client=rewrite_client,
    config=_config(rewrite_enabled=True),
  )
  monkeypatch.setattr(
    "active_listener.infra.rewrite.load_active_listener_rewrite_prompt",
    _prompt_loader(_loaded_prompt_file(_prompt())),
  )

  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()

  assert harness.emitter.emitted == ["rewritten alpha "]


@pytest.mark.asyncio
async def test_finalize_recording_persists_rewrite_corrections(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  correction_store = FakeCorrectionStore()
  rewrite_client = FakeRewriteClient(
    rewritten_text="I use PyTorch",
    corrections={"pie torch": "PyTorch"},
  )
  client = FakeClient(
    flush_results=[
      _message(
        _segment(1, "I use pie torch", completed=True),
        _segment(2, "tail", completed=False),
      )
    ]
  )
  harness = _service(
    client=client,
    correction_store=correction_store,
    rewrite_client=rewrite_client,
    config=_config(rewrite_enabled=True),
  )
  monkeypatch.setattr(
    "active_listener.infra.rewrite.load_active_listener_rewrite_prompt",
    _prompt_loader(_loaded_prompt_file(_prompt())),
  )

  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()

  assert correction_store.merged_corrections == [{"pie torch": "PyTorch"}]
  assert harness.emitter.emitted == ["I use PyTorch "]


@pytest.mark.asyncio
async def test_finalize_recording_continues_when_correction_persistence_fails(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  correction_store = FakeCorrectionStore(merge_error=RuntimeError("disk full"))
  rewrite_client = FakeRewriteClient(
    rewritten_text="I use PyTorch",
    corrections={"pie torch": "PyTorch"},
  )
  client = FakeClient(
    flush_results=[
      _message(
        _segment(1, "I use pie torch", completed=True),
        _segment(2, "tail", completed=False),
      )
    ]
  )
  harness = _service(
    client=client,
    correction_store=correction_store,
    rewrite_client=rewrite_client,
    config=_config(rewrite_enabled=True),
  )
  monkeypatch.setattr(
    "active_listener.infra.rewrite.load_active_listener_rewrite_prompt",
    _prompt_loader(_loaded_prompt_file(_prompt())),
  )

  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()

  assert harness.emitter.emitted == ["I use PyTorch "]


@pytest.mark.asyncio
async def test_finalize_recording_skips_only_rewrite_when_runtime_disabled(
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
  harness = _service(client=client, config=_config(rewrite_enabled=True))
  monkeypatch.setattr(
    "active_listener.recording.finalizer.encode_recording_audio",
    lambda *_args, **_kwargs: b"m4a-bytes",
  )

  assert await harness.service.set_llm_active(False) is False

  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  harness.recording_audio_buffer.append(_sine_wave_bytes(440.0, 32))
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()

  assert harness.emitter.emitted == ["alpha "]
  assert harness.rewrite_client.calls == []

  assert harness.history_store.recordings == [
    RecordedFinalization(
      record=FinalizedTranscriptRecord(
        pre_finalization_text="alpha",
        post_finalization_text="alpha ",
        llm_model=None,
        tokens_in=None,
        tokens_out=None,
        cost=None,
        word_count=1,
        duration_seconds=0.1,
      ),
      archived_audio=b"m4a-bytes",
    )
  ]


@pytest.mark.asyncio
async def test_finalize_recording_does_not_archive_when_text_emission_fails() -> None:
  client = FakeClient(
    flush_results=[
      _message(
        _segment(1, "alpha", completed=True),
        _segment(2, "tail", completed=False),
      )
    ]
  )
  harness = _service(client=client, emitter=FailingEmitter())

  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  harness.recording_audio_buffer.append(_sine_wave_bytes(440.0, 32))
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()

  assert harness.history_store.recordings == []


@pytest.mark.asyncio
async def test_finalize_recording_rewrites_text_when_rewrite_succeeds(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  client = FakeClient(flush_results=[_message(_segment(1, "alpha", completed=True))])
  rewrite = FakeRewriteClient(
    rewritten_text="rewritten alpha", input_tokens=12, output_tokens=3, cost=Decimal("0.002")
  )
  harness = _service(client=client, rewrite_client=rewrite, config=_config(rewrite_enabled=True))
  monkeypatch.setattr(
    "active_listener.infra.rewrite.load_active_listener_rewrite_prompt",
    _prompt_loader(_loaded_prompt_file(_prompt())),
  )
  await harness.service.handle_client_event(ConnectedSignal(session_id="stream-1"))
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()
  assert harness.emitter.emitted == ["rewritten alpha "]
  record = harness.history_store.recordings[0].record
  assert record.pre_finalization_text == "alpha"
  assert record.post_finalization_text == "rewritten alpha "
  assert record.llm_model == "fake-model"
  assert record.tokens_in == 12 and record.tokens_out == 3
  assert record.cost == Decimal("0.002")


@pytest.mark.asyncio
async def test_finalize_recording_serializes_command_text_runs_for_rewrite(
  monkeypatch: pytest.MonkeyPatch,
  recording_clock: RecordingClock,
) -> None:
  client = FakeClient(
    flush_results=[
      _message(
        _segment(
          1,
          "Hello,",
          completed=True,
          start=0.0,
          end=0.2,
          words=[_word("Hello,", start=0.0, end=0.2)],
        ),
        _segment(
          2,
          "scratch that.",
          completed=True,
          start=0.3,
          end=0.6,
          words=[
            _word("scratch", start=0.3, end=0.45),
            _word("that.", start=0.45, end=0.6),
          ],
        ),
        _segment(
          3,
          "Bye.",
          completed=True,
          start=0.7,
          end=0.8,
          words=[_word("Bye.", start=0.7, end=0.8)],
        ),
        _segment(4, "", completed=False),
      )
    ]
  )
  rewrite_client = FakeRewriteClient(rewritten_text="rewritten output")
  harness = _service(
    client=client,
    rewrite_client=rewrite_client,
    config=_config(rewrite_enabled=True),
  )
  monkeypatch.setattr(
    "active_listener.infra.rewrite.load_active_listener_rewrite_prompt",
    _prompt_loader(_loaded_prompt_file(_prompt())),
  )

  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await _command_hold(harness.service, 0.3, 0.6)

  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()

  assert rewrite_client.calls == [
    {
      "instructions": "Rewrite this transcript.",
      "transcript": "Hello, <instruction>scratch that.</instruction> Bye.",
    }
  ]


@pytest.mark.asyncio
async def test_finalize_recording_emits_shaped_text_and_signals_failure_when_rewrite_fails(
  monkeypatch: pytest.MonkeyPatch,
  recording_clock: RecordingClock,
) -> None:
  client = FakeClient(
    flush_results=[
      _message(
        _segment(
          1,
          "release hashtag 42 thank you",
          completed=True,
          end=0.3,
          words=[
            _word("release", start=0.0, end=0.05),
            _word("hashtag", start=0.05, end=0.1),
            _word("42", start=0.1, end=0.15),
            _word("thank", start=0.15, end=0.2),
            _word("you", start=0.2, end=0.3),
          ],
        ),
        _segment(
          2,
          "scratch that.",
          completed=True,
          start=0.3,
          end=0.6,
          words=[_word("scratch", start=0.3, end=0.45), _word("that.", start=0.45, end=0.6)],
        ),
      )
    ]
  )
  harness = _service(
    client=client,
    config=_config(rewrite_enabled=True),
    rewrite_client=FakeRewriteClient(error=RewriteClientError("model failed")),
  )
  monkeypatch.setattr(
    "active_listener.infra.rewrite.load_active_listener_rewrite_prompt",
    _prompt_loader(_loaded_prompt_file(_prompt())),
  )
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await _command_hold(harness.service, 0.3, 0.6)
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()
  assert harness.emitter.emitted == ["release#42 thank you "]
  record = harness.history_store.recordings[0].record
  assert record.post_finalization_text == "release#42 thank you "
  assert record.llm_model is None and record.cost is None
  assert harness.dbus_service.signals[-1] == (
    "PipelineFailed",
    ("rewrite_with_llm", "model failed"),
  )


@pytest.mark.asyncio
async def test_disconnected_pending_flush_prevents_incomplete_recording_emission() -> None:
  pending_flush: asyncio.Future[TranscriptionMessage] = asyncio.get_running_loop().create_future()
  harness = _service(
    client=FakeClient(flush_results=[pending_flush]), config=_config(rewrite_enabled=True)
  )
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await asyncio.sleep(0)
  pending_flush.set_exception(RuntimeError("socket closed during flush"))
  await harness.service.handle_client_event(
    DisconnectedSignal(session_id="stream-1", reason="closed")
  )
  await harness.service.wait_for_background_tasks()
  assert harness.rewrite_client.calls == []
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

  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()

  assert prompt_load_calls == []
  assert harness.emitter.emitted == []


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

  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()

  assert harness.emitter.emitted == ["alpha "]
  assert harness.dbus_service.signals[-1] == (
    "PipelineFailed",
    ("rewrite_with_llm", "bad prompt"),
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

  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await harness.service.wait_for_background_tasks()

  assert harness.emitter.emitted == ["alpha "]
  assert harness.dbus_service.signals[-1] == (
    "PipelineFailed",
    ("rewrite_with_llm", "timed out"),
  )


@pytest.mark.asyncio
async def test_recording_caps_tap_finishes_and_emits(recording_clock: RecordingClock) -> None:
  harness = _service(
    client=FakeClient(flush_results=[_message(_segment(1, "alpha", completed=True))])
  )
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await _command_hold(harness.service, 0.1, 0.105)
  await harness.service.wait_for_background_tasks()
  assert harness.service.phase is ForegroundPhase.IDLE
  assert harness.keyboard.grabbed is False
  assert harness.emitter.emitted == ["alpha "]


@pytest.mark.asyncio
async def test_scheduled_hold_marks_command_text_and_release_keeps_recording(
  monkeypatch: pytest.MonkeyPatch,
  recording_clock: RecordingClock,
) -> None:
  monkeypatch.setattr("active_listener.app.recording_keyboard.HOLD_THRESHOLD_S", 0.001)
  harness = _service()
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await harness.service.handle_keyboard_event(
    _keyboard_event(
      KeyboardEventKind.CAPSLOCK_DOWN, received_monotonic_s=RECORDING_CLOCK_START + 0.3
    )
  )
  await asyncio.sleep(0.01)
  await harness.service.handle_client_event(
    _live_message(
      harness.service,
      _segment(1, "instruction", completed=False, start=0.35, end=0.4),
      flush_complete=False,
    )
  )
  assert harness.dbus_service.signals[-1] == (
    "TranscriptionUpdated",
    [TextRun(text="instruction", is_command=True, is_complete=False)],
  )
  await harness.service.handle_keyboard_event(
    _keyboard_event(KeyboardEventKind.CAPSLOCK_UP, received_monotonic_s=RECORDING_CLOCK_START + 0.6)
  )
  assert harness.service.phase is ForegroundPhase.RECORDING
  _ = await harness.service.handle_action(AppAction.CANCEL)


@pytest.mark.asyncio
async def test_disconnected_hold_cannot_mark_the_next_recording_as_a_command(
  monkeypatch: pytest.MonkeyPatch,
  recording_clock: RecordingClock,
) -> None:
  monkeypatch.setattr("active_listener.app.recording_keyboard.HOLD_THRESHOLD_S", 0.01)
  harness = _service()
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await harness.service.handle_keyboard_event(
    _keyboard_event(
      KeyboardEventKind.CAPSLOCK_DOWN, received_monotonic_s=RECORDING_CLOCK_START + 0.3
    )
  )
  await harness.service.handle_client_event(DisconnectedSignal("stream-1", "closed"))
  await harness.service.handle_client_event(ReconnectedSignal("stream-1"))
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await asyncio.sleep(0.02)
  await harness.service.handle_client_event(
    _live_message(
      harness.service,
      _segment(1, "dictation", completed=False, start=0.35, end=0.4),
      flush_complete=False,
    )
  )
  assert harness.dbus_service.signals[-1] == (
    "TranscriptionUpdated",
    [TextRun(text="dictation", is_command=False, is_complete=False)],
  )
  _ = await harness.service.handle_action(AppAction.CANCEL)


@pytest.mark.asyncio
async def test_recording_audio_discards_aborted_capture_and_freezes_finished_capture() -> None:
  audio = RecordingAudioBuffer()
  transcript = RecordingTranscript(
    client=FakeClient(), logger=get_logger("test/recording"), audio_buffer=audio
  )
  await transcript.start_recording("aborted")
  audio.append(b"aborted audio")
  await transcript.stop_recording()
  await transcript.start_recording("kept")
  audio.append(b"kept audio")
  finished = await transcript.finish_recording()
  audio.append(b"late audio")
  assert finished.recording_id == "kept"
  assert finished.captured_audio.pcm_f32le == b"kept audio"


class RecordingObservationProbe:
  """Stand in for the external tracing SDK while observing parent lifetime."""

  def __init__(self) -> None:
    self.end_count: int = 0

  def update(
    self,
    *,
    input: RecordingInput | str | None = None,
    output: RecordingOutput | str | None = None,
    metadata: dict[str, JsonValue] | None = None,
    level: str | None = None,
    status_message: str | None = None,
    usage_details: dict[str, int] | None = None,
    cost_details: dict[str, float] | None = None,
  ) -> RecordingObservationProbe:
    return self

  def end(self, *, end_time: int | None = None) -> None:
    self.end_count += 1

  @contextmanager
  def start_as_current_observation(
    self,
    *,
    name: str,
    input: str | None = None,
    output: str | None = None,
    metadata: dict[str, JsonValue] | None = None,
    level: str | None = None,
    status_message: str | None = None,
    usage_details: dict[str, int] | None = None,
    cost_details: dict[str, float] | None = None,
  ) -> Iterator[RecordingObservationProbe]:
    yield self


@pytest.mark.asyncio
async def test_observations_end_once_across_abort_reconnect_cancel_and_shutdown(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  observations: list[RecordingObservationProbe] = []

  def start_observation(*, session_id: str, recording_id: str) -> RecordingObservationProbe:
    observation = RecordingObservationProbe()
    observations.append(observation)
    return observation

  monkeypatch.setattr(
    "active_listener.infra.langfuse.start_recording_observation", start_observation
  )
  harness = _service()
  await harness.service.handle_client_event(ConnectedSignal("stream-1"))
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await harness.service.handle_client_event(DisconnectedSignal("stream-1", "closed"))
  assert [observation.end_count for observation in observations] == [1]
  await harness.service.handle_client_event(ReconnectedSignal("stream-1"))
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  _ = await harness.service.handle_action(AppAction.CANCEL)
  _ = await harness.service.handle_action(AppAction.START_OR_FINISH)
  await harness.service.close()
  assert [observation.end_count for observation in observations] == [1, 1, 1]
