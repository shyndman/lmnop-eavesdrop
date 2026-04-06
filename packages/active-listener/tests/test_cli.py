"""CLI contract tests for the active-listener entrypoint."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from active_listener import main
from active_listener.app import ActiveListenerConfig, ActiveListenerRuntimeError
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
async def test_command_run_builds_validated_config(monkeypatch: pytest.MonkeyPatch) -> None:
  captured: list[ActiveListenerConfig] = []

  async def fake_run_service(config: ActiveListenerConfig) -> None:
    captured.append(config)

  monkeypatch.setattr("active_listener.cli.run_service", fake_run_service)

  command = ActiveListenerCommand(
    keyboard_name="Exact Keyboard",
    host="server.local",
    port=9090,
    audio_device="default",
    ydotool_socket="/tmp/ydotool.sock",
  )

  await command.run()

  assert captured == [
    ActiveListenerConfig(
      keyboard_name="Exact Keyboard",
      host="server.local",
      port=9090,
      audio_device="default",
      ydotool_socket="/tmp/ydotool.sock",
    )
  ]


def test_keyboard_name_defaults_to_local_keyboard(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.delenv("ACTIVE_LISTENER_KEYBOARD_NAME", raising=False)

  command = ActiveListenerCommand()

  assert command.keyboard_name == "AT Translated Set 2 keyboard"


def test_keyboard_name_reads_env_at_command_construction(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("ACTIVE_LISTENER_KEYBOARD_NAME", "External Keyboard")

  command = ActiveListenerCommand()

  assert command.keyboard_name == "External Keyboard"


def test_require_env_raises_for_missing_value(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.delenv("SOME_REQUIRED_ENV", raising=False)

  with pytest.raises(RuntimeError, match="SOME_REQUIRED_ENV"):
    _ = require_env("SOME_REQUIRED_ENV")


def test_env_int_raises_for_invalid_integer(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("EAVESDROP_PORT", "not-an-int")

  with pytest.raises(RuntimeError, match="EAVESDROP_PORT"):
    _ = env_int("EAVESDROP_PORT", 9090)
