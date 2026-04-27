from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from structlog.stdlib import BoundLogger

from active_listener.app.ports import (
  ActiveListenerClient,
  ActiveListenerRewriteClient,
  ActiveListenerRuntimeError,
  ActiveListenerTranscriptHistoryStore,
)
from active_listener.app.service import ActiveListenerService
from active_listener.app.state import ForegroundPhase
from active_listener.config.models import (
  ActiveListenerConfig,
  LiteRtRewriteProvider,
  LlmRewriteConfig,
)
from active_listener.infra.dbus import AppStateService, NoopDbusService, SdbusDbusService
from active_listener.infra.emitter import GnomeShellExtensionTextEmitter, TextEmitter
from active_listener.infra.keyboard import KeyboardInput, resolve_keyboard
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

from .infra.langfuse import flush_langfuse


class SpectrumCaptureSink(Protocol):
  def ingest(self, chunk: Float32PcmChunk) -> None: ...


class RecordingAudioCaptureSink(Protocol):
  def append(self, chunk: Float32PcmChunk) -> None: ...


async def create_service(
  config: ActiveListenerConfig,
  *,
  dbus_service: AppStateService | None = None,
  keyboard_resolver: Callable[[str], KeyboardInput] = resolve_keyboard,
  client_factory: (
    Callable[[ActiveListenerConfig, Callable[[Float32PcmChunk], None]], ActiveListenerClient] | None
  ) = None,
  emitter_factory: Callable[[], TextEmitter] | None = None,
  rewrite_client_factory: Callable[[LlmRewriteConfig | None], ActiveListenerRewriteClient]
  | None = None,
  history_store_factory: Callable[
    [ActiveListenerConfig, BoundLogger, AppStateService],
    ActiveListenerTranscriptHistoryStore,
  ]
  | None = None,
) -> ActiveListenerService:
  """Construct a fully initialized service instance.

  :param config: Validated runtime configuration.
  :type config: ActiveListenerConfig
  :param keyboard_resolver: Resolver for the exact-name keyboard dependency.
  :type keyboard_resolver: Callable[[str], KeyboardInput]
  :param client_factory: Factory for the live transcriber dependency.
  :type client_factory: Callable[[ActiveListenerConfig], ActiveListenerClient] | None
  :param emitter_factory: Factory for the text emitter dependency.
  :type emitter_factory: Callable[[], TextEmitter] | None
  :param history_store_factory: Factory for the transcript history dependency.
  :type history_store_factory:
      Callable[[ActiveListenerConfig, BoundLogger, AppStateService],
      ActiveListenerTranscriptHistoryStore] | None
  :returns: Ready-to-run service instance.
  :rtype: ActiveListenerService
  :raises ActiveListenerRuntimeError: If startup prerequisites cannot be satisfied.
  """

  logger = get_logger("al/app")
  resolved_dbus_service = dbus_service or NoopDbusService()
  resolved_client_factory = client_factory or build_client
  resolved_emitter_factory = emitter_factory or build_emitter
  resolved_rewrite_client_factory = rewrite_client_factory or build_rewrite_client
  resolved_history_store_factory = history_store_factory or build_history_store
  spectrum_analyzer = SpectrumAnalyzer(
    publish=lambda bars: publish_spectrum_frame(
      dbus_service=resolved_dbus_service,
      logger=logger,
      bars=bars,
    )
  )
  recording_audio_buffer = RecordingAudioBuffer()
  on_capture = build_capture_callback(
    spectrum_analyzer=spectrum_analyzer,
    audio_buffer=recording_audio_buffer,
    logger=logger,
  )
  client: ActiveListenerClient | None = None
  rewrite_client: ActiveListenerRewriteClient | None = None
  history_store: ActiveListenerTranscriptHistoryStore | None = None
  connect_started = False

  try:
    keyboard = keyboard_resolver(config.keyboard_name)
  except Exception as exc:
    logger.exception("keyboard resolution failed", keyboard_name=config.keyboard_name)
    raise ActiveListenerRuntimeError(str(exc)) from exc

  try:
    history_store = resolved_history_store_factory(config, logger, resolved_dbus_service)
    emitter = resolved_emitter_factory()
    client = resolved_client_factory(config, on_capture)
    rewrite_client = resolved_rewrite_client_factory(config.llm_rewrite)
    connect_started = True
    await client.connect()
  except Exception as exc:
    await cleanup_startup_prerequisites(
      keyboard=keyboard,
      client=client,
      rewrite_client=rewrite_client,
      disconnect_client=connect_started,
      logger=logger,
    )
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
  assert history_store is not None
  service = ActiveListenerService(
    config=config,
    keyboard=keyboard,
    client=client,
    emitter=emitter,
    logger=logger,
    rewrite_client=rewrite_client,
    history_store=history_store,
    dbus_service=resolved_dbus_service,
    recording_audio_buffer=recording_audio_buffer,
    spectrum_analyzer=spectrum_analyzer,
  )
  if isinstance(resolved_dbus_service, SdbusDbusService):
    resolved_dbus_service.attach_recording_control(service)
    resolved_dbus_service.attach_llm_runtime_control(service)
  await resolved_dbus_service.set_state(ForegroundPhase.IDLE)
  return service


async def run_service(
  config: ActiveListenerConfig,
  *,
  dbus_service: AppStateService | None = None,
  keyboard_resolver: Callable[[str], KeyboardInput] = resolve_keyboard,
  client_factory: (
    Callable[[ActiveListenerConfig, Callable[[Float32PcmChunk], None]], ActiveListenerClient] | None
  ) = None,
  emitter_factory: Callable[[], TextEmitter] | None = None,
  rewrite_client_factory: Callable[[LlmRewriteConfig | None], ActiveListenerRewriteClient]
  | None = None,
  history_store_factory: Callable[
    [ActiveListenerConfig, BoundLogger, AppStateService],
    ActiveListenerTranscriptHistoryStore,
  ]
  | None = None,
) -> None:
  """Create and run the long-lived active-listener service.

  :param config: Validated runtime configuration.
  :type config: ActiveListenerConfig
  :param keyboard_resolver: Resolver for the exact-name keyboard dependency.
  :type keyboard_resolver: Callable[[str], KeyboardInput]
  :param client_factory: Factory for the live transcriber dependency.
  :type client_factory: Callable[[ActiveListenerConfig], ActiveListenerClient] | None
  :param emitter_factory: Factory for the text emitter dependency.
  :type emitter_factory: Callable[[], TextEmitter] | None
  :param history_store_factory: Factory for the transcript history dependency.
  :type history_store_factory:
      Callable[[ActiveListenerConfig, BoundLogger, AppStateService],
      ActiveListenerTranscriptHistoryStore] | None
  :returns: None
  :rtype: None
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

  try:
    await service.run()
  except Exception as exc:
    await emit_fatal_error_if_possible(
      dbus_service=resolved_dbus_service,
      reason=str(exc),
      logger=logger,
      failure_kind="runtime",
    )
    raise
  finally:
    await resolved_dbus_service.close()
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

  if isinstance(dbus_service, NoopDbusService):
    return

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
    word_timestamps=True,
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


def build_rewrite_client(config: LlmRewriteConfig | None) -> ActiveListenerRewriteClient:
  if config is None:
    return DisabledRewriteClient()

  provider = config.provider
  if isinstance(provider, LiteRtRewriteProvider):
    return LiteRtRewriteClient(model_path=provider.model_path)

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

  ffmpeg_path = resolve_ffmpeg_path(config.ffmpeg_path, logger=logger)
  logger.info("resolved ffmpeg binary", ffmpeg_path=ffmpeg_path)
  return SqliteTranscriptHistoryStore(
    logger=logger,
    ffmpeg_path=ffmpeg_path,
    dbus_service=dbus_service,
  )


def resolve_ffmpeg_path(configured_path: str | None, *, logger: BoundLogger) -> str:
  resolved_configured_path = _resolve_usable_executable_path(configured_path)
  if resolved_configured_path is not None:
    return resolved_configured_path

  if configured_path is not None:
    logger.warning("configured ffmpeg path unusable", ffmpeg_path=configured_path)

  resolved_path = shutil.which("ffmpeg")
  resolved_path_from_path = _resolve_usable_executable_path(resolved_path)
  if resolved_path_from_path is not None:
    return resolved_path_from_path

  raise ActiveListenerRuntimeError(
    "ffmpeg executable not found; set ffmpeg_path or add ffmpeg to PATH"
  )


def _resolve_usable_executable_path(raw_path: str | None) -> str | None:
  if raw_path is None:
    return None

  candidate_path = Path(raw_path).expanduser().resolve()
  if not candidate_path.is_file():
    return None
  if not os.access(candidate_path, os.X_OK):
    return None

  try:
    result = subprocess.run(
      [str(candidate_path), "-version"],
      capture_output=True,
      check=False,
    )
  except FileNotFoundError:
    return None
  if result.returncode != 0:
    return None

  return str(candidate_path)


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
