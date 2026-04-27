"""CLI contract tests for the active-listener entrypoint."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from active_listener import main
from active_listener.app.ports import ActiveListenerRuntimeError
from active_listener.app.state import ForegroundPhase
from active_listener.cli import ActiveListenerCommand, build_app_state_service, env_int, require_env
from active_listener.config.models import (
  ActiveListenerConfig,
  LiteRtRewriteProvider,
  LlmRewriteConfig,
)
from active_listener.infra.dbus import (
  DbusDuplicateInstanceError,
  DbusServiceError,
  NoopDbusService,
)
from active_listener.recording.reducer import TextRun


class RecordingLogger:
  """Collects log events emitted by the CLI entrypoint."""

  def __init__(self) -> None:
    self.info_messages: list[str] = []
    self.exception_messages: list[str] = []

  def info(self, event: str, **_kwargs: object) -> None:
    self.info_messages.append(event)

  def exception(self, event: str, **_kwargs: object) -> None:
    self.exception_messages.append(event)


@dataclass
class StubCommand:
  """Synchronous command stand-in for main() tests."""

  start_error: Exception | None = None
  started: bool = False

  def start(self) -> None:
    self.started = True
    if self.start_error is not None:
      raise self.start_error


@dataclass
class FakeDbusService:
  states: list[ForegroundPhase] = field(default_factory=lambda: [ForegroundPhase.STARTING])
  signals: list[tuple[str, object | None]] = field(default_factory=list)
  close_calls: int = 0

  async def set_state(self, state: ForegroundPhase) -> None:
    self.states.append(state)

  async def transcription_updated(self, runs: list[TextRun]) -> None:
    _ = runs

  async def recording_aborted(self, reason: str) -> None:
    _ = reason

  async def audio_archive_failed(self, reason: str) -> None:
    self.signals.append(("AudioArchiveFailed", reason))

  async def pipeline_failed(self, step: str, reason: str) -> None:
    _ = step
    _ = reason

  async def fatal_error(self, reason: str) -> None:
    self.signals.append(("FatalError", reason))

  async def reconnecting(self) -> None:
    return None

  async def reconnected(self) -> None:
    return None

  async def close(self) -> None:
    self.close_calls += 1


def _write_config(
  path: Path,
  *,
  keyboard_name: str = "Config Keyboard",
  host: str = "config.local",
  port: int = 9090,
  audio_device: str = "config-device",
  llm_rewrite_block: str | None = None,
) -> None:
  rewrite_block = llm_rewrite_block or (
    "llm_rewrite:\n"
    '  prompt_path: "prompts/rewrite_prompt.md"\n'
    "  provider:\n"
    "    type: litert\n"
    '    model_path: "models/rewrite.litertlm"\n'
  )
  _ = path.write_text(
    "\n".join(
      [
        f'keyboard_name: "{keyboard_name}"',
        f'host: "{host}"',
        f"port: {port}",
        f'audio_device: "{audio_device}"',
        "",
        rewrite_block.rstrip(),
        "",
      ]
    ),
    encoding="utf-8",
  )


def test_main_starts_command_and_configures_logging(monkeypatch: pytest.MonkeyPatch) -> None:
  logger = RecordingLogger()
  setup_calls: list[str] = []
  command = StubCommand()
  httpx_logger = logging.getLogger("httpx")
  original_httpx_level = httpx_logger.level

  def fake_get_logger(_name: str) -> RecordingLogger:
    return logger

  def parse_command(_cls: type[ActiveListenerCommand]) -> StubCommand:
    return command

  monkeypatch.setattr(
    "active_listener.cli.setup_logging_from_env",
    lambda: setup_calls.append("called"),
  )
  monkeypatch.setattr("active_listener.cli.get_logger", fake_get_logger)
  monkeypatch.setattr(ActiveListenerCommand, "parse", classmethod(parse_command))

  try:
    exit_code = main()

    assert exit_code == 0
    assert command.started is True
    assert setup_calls == ["called"]
    assert httpx_logger.level == logging.WARNING
    assert logger.info_messages == ["starting active-listener"]
  finally:
    httpx_logger.setLevel(original_httpx_level)


def test_main_returns_non_zero_when_startup_raises(monkeypatch: pytest.MonkeyPatch) -> None:
  logger = RecordingLogger()
  command = StubCommand(start_error=ActiveListenerRuntimeError("boom"))

  def fake_get_logger(_name: str) -> RecordingLogger:
    return logger

  def parse_command(_cls: type[ActiveListenerCommand]) -> StubCommand:
    return command

  monkeypatch.setattr("active_listener.cli.setup_logging_from_env", lambda: None)
  monkeypatch.setattr("active_listener.cli.get_logger", fake_get_logger)
  monkeypatch.setattr(ActiveListenerCommand, "parse", classmethod(parse_command))

  exit_code = main()

  assert exit_code == 1
  assert logger.exception_messages == ["active-listener failed"]


@pytest.mark.asyncio
async def test_command_run_uses_config_file_values(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  captured: list[tuple[ActiveListenerConfig, object]] = []
  config_path = tmp_path / "config.yaml"
  _write_config(config_path)
  dbus_service = FakeDbusService()

  async def fake_run_service(config: ActiveListenerConfig, *, dbus_service: object) -> None:
    captured.append((config, dbus_service))

  async def fake_build_app_state_service(*, no_dbus: bool) -> object:
    assert no_dbus is False
    return dbus_service

  monkeypatch.setattr("active_listener.cli.build_app_state_service", fake_build_app_state_service)
  monkeypatch.setattr("active_listener.cli.run_service", fake_run_service)

  command = ActiveListenerCommand(config_path=str(config_path))

  await command.run()

  assert captured == [
    (
      ActiveListenerConfig(
        keyboard_name="Config Keyboard",
        host="config.local",
        port=9090,
        audio_device="config-device",
        llm_rewrite=LlmRewriteConfig(
          prompt_path=str(config_path.parent / "prompts" / "rewrite_prompt.md"),
          provider=LiteRtRewriteProvider(
            type="litert",
            model_path=str(config_path.parent / "models" / "rewrite.litertlm"),
          ),
        ),
      ),
      dbus_service,
    )
  ]


@pytest.mark.asyncio
async def test_command_run_uses_default_xdg_config_path_when_flag_is_not_set(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  captured: list[ActiveListenerConfig] = []
  config_home = tmp_path / "xdg-config"
  config_path = config_home / "eavesdrop" / "active-listener.yaml"
  config_path.parent.mkdir(parents=True)
  _write_config(config_path, host="default-host", port=9191)

  async def fake_run_service(config: ActiveListenerConfig, *, dbus_service: object) -> None:
    _ = dbus_service
    captured.append(config)

  async def fake_build_app_state_service(*, no_dbus: bool) -> NoopDbusService:
    _ = no_dbus
    return NoopDbusService()

  monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
  monkeypatch.setattr("active_listener.cli.build_app_state_service", fake_build_app_state_service)
  monkeypatch.setattr("active_listener.cli.run_service", fake_run_service)

  command = ActiveListenerCommand()

  await command.run()

  assert captured[0].host == "default-host"
  assert captured[0].port == 9191
  assert captured[0].llm_rewrite == LlmRewriteConfig(
    prompt_path=str(config_path.parent / "prompts" / "rewrite_prompt.md"),
    provider=LiteRtRewriteProvider(
      type="litert",
      model_path=str(config_path.parent / "models" / "rewrite.litertlm"),
    ),
  )


@pytest.mark.asyncio
async def test_command_run_explicit_config_path_overrides_default_xdg_location(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  captured: list[ActiveListenerConfig] = []
  config_home = tmp_path / "xdg-config"
  default_config_path = config_home / "eavesdrop" / "active-listener.yaml"
  explicit_config_path = tmp_path / "explicit-config.yaml"
  default_config_path.parent.mkdir(parents=True)
  _write_config(default_config_path, host="default-host", port=9191)
  _write_config(explicit_config_path, host="explicit-host", port=9292)

  async def fake_run_service(config: ActiveListenerConfig, *, dbus_service: object) -> None:
    _ = dbus_service
    captured.append(config)

  async def fake_build_app_state_service(*, no_dbus: bool) -> NoopDbusService:
    _ = no_dbus
    return NoopDbusService()

  monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
  monkeypatch.setattr("active_listener.cli.build_app_state_service", fake_build_app_state_service)
  monkeypatch.setattr("active_listener.cli.run_service", fake_run_service)

  command = ActiveListenerCommand(config_path=str(explicit_config_path))

  await command.run()

  assert captured[0].host == "explicit-host"
  assert captured[0].port == 9292


@pytest.mark.asyncio
async def test_command_run_overrides_config_file_values(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  captured: list[ActiveListenerConfig] = []
  config_path = tmp_path / "config.yaml"
  _write_config(config_path, host="config-host", port=7000)

  async def fake_run_service(config: ActiveListenerConfig, *, dbus_service: object) -> None:
    _ = dbus_service
    captured.append(config)

  async def fake_build_app_state_service(*, no_dbus: bool) -> NoopDbusService:
    _ = no_dbus
    return NoopDbusService()

  monkeypatch.setattr("active_listener.cli.build_app_state_service", fake_build_app_state_service)
  monkeypatch.setattr("active_listener.cli.run_service", fake_run_service)

  command = ActiveListenerCommand(
    config_path=str(config_path),
    host="override-host",
    port=8080,
  )

  await command.run()

  assert captured[0].host == "override-host"
  assert captured[0].port == 8080
  assert captured[0].keyboard_name == "Config Keyboard"


@pytest.mark.asyncio
async def test_command_run_overrides_default_path_config_values(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  captured: list[ActiveListenerConfig] = []
  config_home = tmp_path / "xdg-config"
  config_path = config_home / "eavesdrop" / "active-listener.yaml"
  config_path.parent.mkdir(parents=True)
  _write_config(config_path, host="default-host", port=7000)

  async def fake_run_service(config: ActiveListenerConfig, *, dbus_service: object) -> None:
    _ = dbus_service
    captured.append(config)

  async def fake_build_app_state_service(*, no_dbus: bool) -> NoopDbusService:
    _ = no_dbus
    return NoopDbusService()

  monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
  monkeypatch.setattr("active_listener.cli.build_app_state_service", fake_build_app_state_service)
  monkeypatch.setattr("active_listener.cli.run_service", fake_run_service)

  command = ActiveListenerCommand(host="override-host", port=8080)

  await command.run()

  assert captured[0].host == "override-host"
  assert captured[0].port == 8080
  assert captured[0].keyboard_name == "Config Keyboard"


@pytest.mark.asyncio
async def test_command_run_raises_for_missing_required_rewrite_fields(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  config_path = tmp_path / "config.yaml"
  _write_config(
    config_path,
    llm_rewrite_block='llm_rewrite:\n  prompt_path: "prompts/rewrite_prompt.md"\n',
  )
  dbus_service = FakeDbusService()

  async def fake_run_service(_config: ActiveListenerConfig, *, dbus_service: object) -> None:
    _ = dbus_service
    raise AssertionError("run_service should not be called")

  async def fake_build_app_state_service(*, no_dbus: bool) -> FakeDbusService:
    assert no_dbus is False
    return dbus_service

  monkeypatch.setattr("active_listener.cli.build_app_state_service", fake_build_app_state_service)
  monkeypatch.setattr("active_listener.cli.run_service", fake_run_service)

  command = ActiveListenerCommand(config_path=str(config_path))

  with pytest.raises(ValueError, match="llm_rewrite") as exc_info:
    await command.run()

  assert dbus_service.signals == [("FatalError", str(exc_info.value))]
  assert dbus_service.close_calls == 1


@pytest.mark.asyncio
async def test_command_run_rejects_removed_rewrite_fields(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  config_path = tmp_path / "config.yaml"
  _write_config(
    config_path,
    llm_rewrite_block=(
      "llm_rewrite:\n"
      '  prompt_path: "prompts/rewrite_prompt.md"\n'
      "  provider:\n"
      "    type: litert\n"
      '    model_path: "models/rewrite.litertlm"\n'
      '    base_url: "http://localhost:11434/v1"\n'
      "    timeout_s: 30\n"
    ),
  )
  dbus_service = FakeDbusService()

  async def fake_run_service(_config: ActiveListenerConfig, *, dbus_service: object) -> None:
    _ = dbus_service
    raise AssertionError("run_service should not be called")

  async def fake_build_app_state_service(*, no_dbus: bool) -> FakeDbusService:
    assert no_dbus is False
    return dbus_service

  monkeypatch.setattr("active_listener.cli.build_app_state_service", fake_build_app_state_service)
  monkeypatch.setattr("active_listener.cli.run_service", fake_run_service)

  command = ActiveListenerCommand(config_path=str(config_path))

  with pytest.raises(ValueError, match="base_url|timeout_s") as exc_info:
    await command.run()

  assert dbus_service.signals == [("FatalError", str(exc_info.value))]
  assert dbus_service.close_calls == 1


@pytest.mark.asyncio
async def test_command_run_preserves_file_value_when_cli_flag_is_not_set(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  captured: list[ActiveListenerConfig] = []
  config_path = tmp_path / "config.yaml"
  _write_config(config_path, audio_device="alsa-input")

  async def fake_run_service(config: ActiveListenerConfig, *, dbus_service: object) -> None:
    _ = dbus_service
    captured.append(config)

  async def fake_build_app_state_service(*, no_dbus: bool) -> NoopDbusService:
    _ = no_dbus
    return NoopDbusService()

  monkeypatch.setattr("active_listener.cli.build_app_state_service", fake_build_app_state_service)
  monkeypatch.setattr("active_listener.cli.run_service", fake_run_service)

  command = ActiveListenerCommand(
    config_path=str(config_path),
    host="override-host",
  )

  await command.run()

  assert captured[0].host == "override-host"
  assert captured[0].audio_device == "alsa-input"


@pytest.mark.asyncio
async def test_command_run_supports_no_dbus_mode(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  captured: list[object] = []
  config_path = tmp_path / "config.yaml"
  _write_config(config_path)

  async def fake_run_service(config: ActiveListenerConfig, *, dbus_service: object) -> None:
    _ = config
    captured.append(dbus_service)

  async def fake_build_app_state_service(*, no_dbus: bool) -> NoopDbusService | FakeDbusService:
    if no_dbus:
      return NoopDbusService()
    return FakeDbusService()

  monkeypatch.setattr("active_listener.cli.build_app_state_service", fake_build_app_state_service)
  monkeypatch.setattr("active_listener.cli.run_service", fake_run_service)

  command = ActiveListenerCommand(config_path=str(config_path), no_dbus=True)

  await command.run()

  assert isinstance(captured[0], NoopDbusService)


@pytest.mark.asyncio
async def test_command_run_builds_dbus_before_service_startup(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  config_path = tmp_path / "config.yaml"
  _write_config(config_path)
  dbus_service = FakeDbusService()

  async def fake_run_service(config: ActiveListenerConfig, *, dbus_service: object) -> None:
    _ = config
    assert dbus_service is expected_dbus_service

  expected_dbus_service = dbus_service

  async def fake_build_app_state_service(*, no_dbus: bool) -> object:
    assert no_dbus is False
    return dbus_service

  monkeypatch.setattr("active_listener.cli.build_app_state_service", fake_build_app_state_service)
  monkeypatch.setattr("active_listener.cli.run_service", fake_run_service)

  command = ActiveListenerCommand(config_path=str(config_path))

  await command.run()

  assert dbus_service.states == [ForegroundPhase.STARTING]


@pytest.mark.asyncio
async def test_build_app_state_service_maps_duplicate_instance_error() -> None:
  async def fake_connect() -> FakeDbusService:
    raise DbusDuplicateInstanceError("another active-listener instance is already running")

  monkeypatch = pytest.MonkeyPatch()
  monkeypatch.setattr("active_listener.cli.SdbusDbusService.connect", fake_connect)

  with pytest.raises(ActiveListenerRuntimeError, match="already running"):
    _ = await build_app_state_service(no_dbus=False)

  monkeypatch.undo()


@pytest.mark.asyncio
async def test_build_app_state_service_suggests_no_dbus_for_session_bus_failure() -> None:
  async def fake_connect() -> FakeDbusService:
    raise DbusServiceError("session bus unavailable")

  monkeypatch = pytest.MonkeyPatch()
  monkeypatch.setattr("active_listener.cli.SdbusDbusService.connect", fake_connect)

  with pytest.raises(ActiveListenerRuntimeError, match="--no-dbus"):
    _ = await build_app_state_service(no_dbus=False)

  monkeypatch.undo()


def test_require_env_raises_for_missing_value(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.delenv("SOME_REQUIRED_ENV", raising=False)

  with pytest.raises(RuntimeError, match="SOME_REQUIRED_ENV"):
    _ = require_env("SOME_REQUIRED_ENV")


def test_env_int_raises_for_invalid_integer(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("EAVESDROP_PORT", "not-an-int")

  with pytest.raises(RuntimeError, match="EAVESDROP_PORT"):
    _ = env_int("EAVESDROP_PORT", 9090)
