from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import Protocol

from structlog.stdlib import BoundLogger

from active_listener.app.ports import (
  ActiveListenerClient,
  ActiveListenerRewriteClient,
  ActiveListenerRuntimeError,
  ActiveListenerTranscriptHistoryStore,
  MediaPlaybackController,
)
from active_listener.app.service import ActiveListenerService
from active_listener.app.state import ForegroundPhase
from active_listener.config.models import (
  ActiveListenerConfig,
  LiteRtRewriteProvider,
  LlmRewriteConfig,
)
from active_listener.infra.audio import resolve_ffmpeg_executable
from active_listener.infra.dbus import AppStateService, NoopDbusService, SdbusDbusService
from active_listener.infra.emitter import GnomeShellExtensionTextEmitter, TextEmitter
from active_listener.infra.keyboard import KeyboardInput, resolve_keyboard
from active_listener.infra.langfuse import flush_langfuse
from active_listener.infra.mpris import PlayerctldMediaPlaybackController
from active_listener.infra.rewrite import (
  DisabledRewriteClient,
  LiteRtRewriteClient,
  PydanticAiRewriteClient,
)
from active_listener.infra.transcript_history import SqliteTranscriptHistoryStore
from active_listener.recording.session import RecordingAudioBuffer
from active_listener.recording.spectrum import (
  Float32PcmChunk,
  QuantizedSpectrumFrame,
  SpectrumAnalyzer,
)
from eavesdrop.client import EavesdropClient
from eavesdrop.common import get_logger


class SpectrumCaptureSink(Protocol):
  def ingest(self, chunk: Float32PcmChunk) -> None: ...


class RecordingAudioCaptureSink(Protocol):
  def append(self, chunk: Float32PcmChunk) -> None: ...


async def create_service(
  config: ActiveListenerConfig,
  *,
  dbus_service: AppStateService | None = None,
  keyboard_resolver: Callable[[str], KeyboardInput] = resolve_keyboard,
  client_factory: Callable[
    [ActiveListenerConfig, Callable[[Float32PcmChunk], None]], ActiveListenerClient
  ]
  | None = None,
  emitter_factory: Callable[[], TextEmitter] | None = None,
  media_playback_controller_factory: Callable[[BoundLogger], MediaPlaybackController] | None = None,
  rewrite_client_factory: Callable[[LlmRewriteConfig | None], ActiveListenerRewriteClient]
  | None = None,
  history_store_factory: Callable[
    [ActiveListenerConfig, BoundLogger, AppStateService], ActiveListenerTranscriptHistoryStore
  ]
  | None = None,
) -> ActiveListenerService:
  """Build resources and connect after binding push callbacks to the app queue.

  :param config: Validated workstation configuration.
  :param dbus_service: Optional D-Bus destination.
  :param keyboard_resolver: Keyboard hardware resolver.
  :param client_factory: Capture-aware transcription client factory.
  :param emitter_factory: Initialized text emitter factory.
  :param media_playback_controller_factory: Media controller factory.
  :param rewrite_client_factory: Rewrite provider factory.
  :param history_store_factory: Transcript archive factory.
  :return: Connected service ready to consume queued callback signals.
  :raises ActiveListenerRuntimeError: If dependency construction or connection fails.
  """
  logger = get_logger("al/app")
  dbus = dbus_service or NoopDbusService()
  service = await _construct_service(
    config,
    dbus,
    logger,
    keyboard_resolver,
    client_factory or build_client,
    emitter_factory or build_emitter,
    media_playback_controller_factory or build_media_playback_controller,
    rewrite_client_factory or build_rewrite_client,
    history_store_factory,
  )
  try:
    await service.client.connect()
  except Exception as exc:
    logger.exception("server connection failed", host=config.host, port=config.port)
    await service.close()
    raise ActiveListenerRuntimeError(f"could not connect to eavesdrop server: {exc}") from exc
  match dbus:
    case SdbusDbusService():
      dbus.attach_recording_control(service)
      dbus.attach_llm_runtime_control(service)
    case _:
      pass
  await dbus.set_state(ForegroundPhase.IDLE)
  return service


async def _construct_service(
  config: ActiveListenerConfig,
  dbus: AppStateService,
  logger: BoundLogger,
  keyboard_resolver: Callable[[str], KeyboardInput],
  client_factory: Callable[
    [ActiveListenerConfig, Callable[[Float32PcmChunk], None]], ActiveListenerClient
  ],
  emitter_factory: Callable[[], TextEmitter],
  media_factory: Callable[[BoundLogger], MediaPlaybackController],
  rewrite_factory: Callable[[LlmRewriteConfig | None], ActiveListenerRewriteClient],
  history_factory: Callable[
    [ActiveListenerConfig, BoundLogger, AppStateService], ActiveListenerTranscriptHistoryStore
  ]
  | None,
) -> ActiveListenerService:
  try:
    keyboard = keyboard_resolver(config.keyboard_name)
  except Exception as exc:
    logger.exception("keyboard resolution failed", keyboard_name=config.keyboard_name)
    raise ActiveListenerRuntimeError(str(exc)) from exc
  client: ActiveListenerClient | None = None
  rewrite: ActiveListenerRewriteClient | None = None
  try:
    if history_factory is None:
      _ = resolve_ffmpeg_path(config.ffmpeg_path, logger=logger)
    history = (history_factory or build_history_store)(config, logger, dbus)
    emitter = emitter_factory()
    spectrum = SpectrumAnalyzer(
      publish=lambda bars: publish_spectrum_frame(dbus_service=dbus, logger=logger, bars=bars)
    )
    audio = RecordingAudioBuffer()
    on_capture = build_capture_callback(
      spectrum_analyzer=spectrum, audio_buffer=audio, logger=logger
    )
    client = client_factory(config, on_capture)
    media = media_factory(logger)
    rewrite = rewrite_factory(config.llm_rewrite)
    return ActiveListenerService(
      config=config,
      keyboard=keyboard,
      client=client,
      emitter=emitter,
      logger=logger,
      rewrite_client=rewrite,
      history_store=history,
      dbus_service=dbus,
      media_playback_controller=media,
      recording_audio_buffer=audio,
      spectrum_analyzer=spectrum,
    )
  except Exception as exc:
    logger.exception("startup dependency construction failed")
    await cleanup_startup_prerequisites(
      keyboard=keyboard,
      client=client,
      rewrite_client=rewrite,
      disconnect_client=False,
      logger=logger,
    )
    raise ActiveListenerRuntimeError(f"failed to build startup dependencies: {exc}") from exc


async def run_service(
  config: ActiveListenerConfig,
  *,
  dbus_service: AppStateService | None = None,
  keyboard_resolver: Callable[[str], KeyboardInput] = resolve_keyboard,
  client_factory: (
    Callable[[ActiveListenerConfig, Callable[[Float32PcmChunk], None]], ActiveListenerClient] | None
  ) = None,
  emitter_factory: Callable[[], TextEmitter] | None = None,
  media_playback_controller_factory: Callable[[BoundLogger], MediaPlaybackController] | None = None,
  rewrite_client_factory: Callable[[LlmRewriteConfig | None], ActiveListenerRewriteClient]
  | None = None,
  history_store_factory: Callable[
    [ActiveListenerConfig, BoundLogger, AppStateService],
    ActiveListenerTranscriptHistoryStore,
  ]
  | None = None,
) -> None:
  """Create and run the service, reporting fatal failures through D-Bus.

  :param config: Validated runtime configuration.
  :param dbus_service: Optional D-Bus destination.
  :param keyboard_resolver: Hardware resolver.
  :param client_factory: Capture-aware client factory.
  :param emitter_factory: Initialized emitter factory.
  :param media_playback_controller_factory: Media controller factory.
  :param rewrite_client_factory: Rewrite provider factory.
  :param history_store_factory: Transcript archive factory.
  :return: None after service shutdown.
  :raises Exception: If startup or service execution fails after cleanup.
  """

  logger = get_logger("al/app")
  resolved_dbus_service = dbus_service or NoopDbusService()
  try:
    service = await create_service(
      config,
      dbus_service=resolved_dbus_service,
      keyboard_resolver=keyboard_resolver,
      client_factory=client_factory,
      emitter_factory=emitter_factory,
      media_playback_controller_factory=media_playback_controller_factory,
      rewrite_client_factory=rewrite_client_factory,
      history_store_factory=history_store_factory,
    )
  except Exception as exc:
    await emit_fatal_error_if_possible(
      dbus_service=resolved_dbus_service,
      reason=str(exc),
      logger=logger,
      failure_kind="startup",
    )
    await resolved_dbus_service.close()
    raise
  await _run_connected_service(service, resolved_dbus_service, logger)


async def _run_connected_service(
  service: ActiveListenerService, dbus: AppStateService, logger: BoundLogger
) -> None:
  try:
    await service.run()
  except Exception as exc:
    logger.exception("active-listener runtime failed")
    await emit_fatal_error_if_possible(
      dbus_service=dbus,
      reason=str(exc),
      logger=logger,
      failure_kind="runtime",
    )
    raise
  finally:
    await dbus.close()
    try:
      flush_langfuse()
    except Exception:
      logger.exception("langfuse flush failed")


async def emit_fatal_error_if_possible(
  *,
  dbus_service: AppStateService,
  reason: str,
  logger: BoundLogger,
  failure_kind: str,
) -> None:
  """Publish a one-shot fatal event when DBus is live.

  Fatal publication is only truthful after the process has an exported
  DBus service. ``NoopDbusService`` means DBus was disabled or unavailable, so
  there is no bus consumer that could observe the event.
  """

  match dbus_service:
    case NoopDbusService():
      return
    case _:
      pass

  try:
    await dbus_service.fatal_error(reason)
  except Exception:
    logger.exception(f"{failure_kind} fatal publication failed", reason=reason)


def build_client(
  config: ActiveListenerConfig,
  on_capture: Callable[[Float32PcmChunk], None],
) -> ActiveListenerClient:
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
    on_capture=on_capture,
  )


def build_capture_callback(
  *,
  spectrum_analyzer: SpectrumCaptureSink,
  audio_buffer: RecordingAudioCaptureSink,
  logger: BoundLogger,
) -> Callable[[Float32PcmChunk], None]:
  def on_capture(chunk: Float32PcmChunk) -> None:
    try:
      spectrum_analyzer.ingest(chunk)
    except Exception:
      logger.exception("spectrum capture callback failed", byte_count=len(chunk))

    try:
      audio_buffer.append(chunk)
    except Exception:
      logger.exception("recording audio capture failed", byte_count=len(chunk))

  return on_capture


async def publish_spectrum_frame(
  *,
  dbus_service: AppStateService,
  logger: BoundLogger,
  bars: QuantizedSpectrumFrame,
) -> None:
  try:
    await dbus_service.spectrum_updated(bars)
  except Exception:
    logger.exception("spectrum publication failed", bar_count=len(bars))


def build_emitter() -> TextEmitter:
  """Build and initialize the text emission boundary.

  :returns: Initialized text emitter.
  :rtype: TextEmitter
  """

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()
  return emitter


def build_media_playback_controller(logger: BoundLogger) -> MediaPlaybackController:
  return PlayerctldMediaPlaybackController(logger=logger)


def build_rewrite_client(config: LlmRewriteConfig | None) -> ActiveListenerRewriteClient:
  if config is None:
    return DisabledRewriteClient()

  match config.provider:
    case LiteRtRewriteProvider(model_path=path):
      return LiteRtRewriteClient(model_path=path)
    case provider:
      return PydanticAiRewriteClient(model=provider.model)


def build_history_store(
  config: ActiveListenerConfig,
  logger: BoundLogger,
  dbus_service: AppStateService,
) -> ActiveListenerTranscriptHistoryStore:
  """Build the local transcript history store.

  :param config: Validated runtime configuration.
  :type config: ActiveListenerConfig
  :param logger: Structured logger used for best-effort insert failures.
  :type logger: BoundLogger
  :param dbus_service: DBus publisher used for archive-failure notifications.
  :type dbus_service: AppStateService
  :returns: SQLite-backed transcript history store.
  :rtype: ActiveListenerTranscriptHistoryStore
  """
  _ = config

  return SqliteTranscriptHistoryStore(
    logger=logger,
    dbus_service=dbus_service,
  )


def resolve_ffmpeg_path(configured_path: str | None, *, logger: BoundLogger) -> str:
  """Resolve and validate the ffmpeg binary at startup (fail-fast).

  Shares the use-time lookup with
  :func:`active_listener.infra.audio.resolve_ffmpeg_executable`, so the
  validated and logged path is exactly the one encoding uses later, then probes
  it with ``-version`` to fail fast on a present-but-broken binary.

  :param configured_path: Optional explicit ffmpeg path override.
  :type configured_path: str | None
  :param logger: Structured logger for reporting an unusable binary.
  :type logger: BoundLogger
  :returns: Path to a runnable ffmpeg binary.
  :rtype: str
  :raises ActiveListenerRuntimeError: If no runnable ffmpeg can be resolved.
  """
  resolved = resolve_ffmpeg_executable(configured_path)
  if not _ffmpeg_runs(resolved):
    logger.warning("resolved ffmpeg binary is not runnable", ffmpeg_path=resolved)
    raise ActiveListenerRuntimeError(f"ffmpeg executable is not runnable: {resolved}")
  return resolved


def _ffmpeg_runs(path: str) -> bool:
  try:
    result = subprocess.run([path, "-version"], capture_output=True, check=False)
  except OSError:
    return False
  return result.returncode == 0


async def cleanup_startup_prerequisites(
  *,
  keyboard: KeyboardInput,
  client: ActiveListenerClient | None,
  rewrite_client: ActiveListenerRewriteClient | None,
  disconnect_client: bool,
  logger: BoundLogger,
) -> None:
  if disconnect_client and client is not None:
    try:
      await client.disconnect()
    except Exception:
      logger.exception("startup client cleanup failed")

  if rewrite_client is not None:
    try:
      await rewrite_client.close()
    except Exception:
      logger.exception("startup rewrite cleanup failed")

  keyboard.close()
