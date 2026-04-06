"""CLI contract tests for the active-listener entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from active_listener import main
from active_listener.app import (
  ActiveListenerConfig,
  ActiveListenerRuntimeError,
  LlmRewriteConfig,
)
from active_listener.cli import ActiveListenerCommand, env_int, require_env


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


def _write_config(
  path: Path,
  *,
  keyboard_name: str = "Config Keyboard",
  host: str = "config.local",
  port: int = 9090,
  audio_device: str = "config-device",
  ydotool_socket: str | None = "/tmp/config.sock",
  llm_rewrite_block: str | None = None,
) -> None:
  rewrite_block = llm_rewrite_block or (
    "llm_rewrite:\n"
    "  enabled: true\n"
    '  base_url: "http://localhost:11434/v1"\n'
    "  timeout_s: 30\n"
    '  prompt_path: "packages/active-listener/src/active_listener/rewrite_prompt.md"\n'
  )
  ydotool_value = "null" if ydotool_socket is None else f'"{ydotool_socket}"'
  _ = path.write_text(
    "\n".join(
      [
        f'keyboard_name: "{keyboard_name}"',
        f'host: "{host}"',
        f"port: {port}",
        f'audio_device: "{audio_device}"',
        f"ydotool_socket: {ydotool_value}",
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

  exit_code = main()

  assert exit_code == 0
  assert command.started is True
  assert setup_calls == ["called"]
  assert logger.info_messages == ["starting active-listener"]


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
  captured: list[ActiveListenerConfig] = []
  config_path = tmp_path / "config.yaml"
  _write_config(config_path)

  async def fake_run_service(config: ActiveListenerConfig) -> None:
    captured.append(config)

  monkeypatch.setattr("active_listener.cli.run_service", fake_run_service)

  command = ActiveListenerCommand(config_path=str(config_path))

  await command.run()

  assert captured == [
    ActiveListenerConfig(
      keyboard_name="Config Keyboard",
      host="config.local",
      port=9090,
      audio_device="config-device",
      ydotool_socket="/tmp/config.sock",
      llm_rewrite=LlmRewriteConfig(
        enabled=True,
        base_url="http://localhost:11434/v1",
        timeout_s=30,
        prompt_path="packages/active-listener/src/active_listener/rewrite_prompt.md",
      ),
    )
  ]


@pytest.mark.asyncio
async def test_command_run_overrides_config_file_values(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  captured: list[ActiveListenerConfig] = []
  config_path = tmp_path / "config.yaml"
  _write_config(config_path, host="config-host", port=7000)

  async def fake_run_service(config: ActiveListenerConfig) -> None:
    captured.append(config)

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
async def test_command_run_raises_for_missing_required_rewrite_fields(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  config_path = tmp_path / "config.yaml"
  _write_config(
    config_path,
    llm_rewrite_block="llm_rewrite:\n  enabled: true\n",
  )

  async def fake_run_service(_config: ActiveListenerConfig) -> None:
    raise AssertionError("run_service should not be called")

  monkeypatch.setattr("active_listener.cli.run_service", fake_run_service)

  command = ActiveListenerCommand(config_path=str(config_path))

  with pytest.raises(ValueError, match="llm_rewrite"):
    await command.run()


@pytest.mark.asyncio
async def test_command_run_preserves_file_value_when_cli_flag_is_not_set(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  captured: list[ActiveListenerConfig] = []
  config_path = tmp_path / "config.yaml"
  _write_config(config_path, audio_device="alsa-input")

  async def fake_run_service(config: ActiveListenerConfig) -> None:
    captured.append(config)

  monkeypatch.setattr("active_listener.cli.run_service", fake_run_service)

  command = ActiveListenerCommand(
    config_path=str(config_path),
    host="override-host",
  )

  await command.run()

  assert captured[0].host == "override-host"
  assert captured[0].audio_device == "alsa-input"


def test_require_env_raises_for_missing_value(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.delenv("SOME_REQUIRED_ENV", raising=False)

  with pytest.raises(RuntimeError, match="SOME_REQUIRED_ENV"):
    _ = require_env("SOME_REQUIRED_ENV")


def test_env_int_raises_for_invalid_integer(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("EAVESDROP_PORT", "not-an-int")

  with pytest.raises(RuntimeError, match="EAVESDROP_PORT"):
    _ = env_int("EAVESDROP_PORT", 9090)
