"""CLI entrypoint for the active-listener service."""

from __future__ import annotations

import os

from clypi import Command, arg
from typing_extensions import override

from active_listener.bootstrap import emit_fatal_error_if_possible, run_service
from active_listener.config import load_active_listener_config
from active_listener.dbus_service import (
  AppStateService,
  DbusDuplicateInstanceError,
  DbusServiceError,
  NoopDbusService,
  SdbusDbusService,
)
from active_listener.service import ActiveListenerRuntimeError
from eavesdrop.common import get_logger, setup_logging_from_env


def require_env(name: str) -> str:
  """Read a required environment variable.

  :param name: Environment variable name.
  :type name: str
  :returns: Non-empty environment variable value.
  :rtype: str
  :raises RuntimeError: If the variable is missing or empty.
  """

  value = os.getenv(name)
  if value is None or value == "":
    raise RuntimeError(f"Missing required environment variable: {name}")
  return value


def env_int(name: str, default: int) -> int:
  """Read an integer environment variable with a typed default.

  :param name: Environment variable name.
  :type name: str
  :param default: Default value used when the variable is unset.
  :type default: int
  :returns: Parsed integer value.
  :rtype: int
  :raises RuntimeError: If the environment value is not an integer.
  """

  value = os.getenv(name)
  if value is None:
    return default

  try:
    return int(value)
  except ValueError as exc:
    raise RuntimeError(f"Environment variable {name} must be an integer") from exc


async def build_app_state_service(*, no_dbus: bool) -> AppStateService:
  if no_dbus:
    return NoopDbusService()

  try:
    return await SdbusDbusService.connect()
  except DbusDuplicateInstanceError as exc:
    raise ActiveListenerRuntimeError(str(exc)) from exc
  except DbusServiceError as exc:
    message = (
      "failed to start DBus app-state publishing: "
      f"{exc}. Try --no-dbus if this environment has no session bus"
    )
    raise ActiveListenerRuntimeError(message) from exc


class ActiveListenerCommand(Command):
  """Run the long-lived active-listener hotkey service."""

  config_path: str | None = arg(
    default=None,
    help="Path to the active-listener YAML config file.",
  )

  keyboard_name: str | None = arg(
    default=None,
    help="Exact evdev keyboard device name override.",
  )
  host: str | None = arg(
    default=None,
    help="Eavesdrop server hostname override.",
  )
  port: int | None = arg(
    default=None,
    help="Eavesdrop server port override.",
  )
  audio_device: str | None = arg(
    default=None,
    help="PortAudio capture device name override.",
  )
  ydotool_socket: str | None = arg(
    default=None,
    help="Optional ydotool daemon socket path override.",
  )
  no_dbus: bool = arg(
    default=False,
    help="Disable DBus app-state publishing.",
  )

  @override
  async def run(self) -> None:
    """Load validated config and start the long-running service.

    :returns: None
    :rtype: None
    """

    logger = get_logger("al/cli")
    dbus_service = await build_app_state_service(no_dbus=self.no_dbus)

    try:
      config = load_active_listener_config(
        config_path=self.config_path,
        overrides={
          "keyboard_name": self.keyboard_name,
          "host": self.host,
          "port": self.port,
          "audio_device": self.audio_device,
          "ydotool_socket": self.ydotool_socket,
        },
      )
    except Exception as exc:
      await emit_fatal_error_if_possible(
        dbus_service=dbus_service,
        reason=str(exc),
        logger=logger,
        failure_kind="startup",
      )
      await dbus_service.close()
      raise

    await run_service(config, dbus_service=dbus_service)


def main() -> int:
  """Start the bare active-listener CLI command.

  :returns: Shell exit status code.
  :rtype: int
  """

  setup_logging_from_env()
  logger = get_logger("al/main")

  try:
    command = ActiveListenerCommand.parse()
    logger.info("starting active-listener")
    _ = command.start()
  except KeyboardInterrupt:
    logger.info("active-listener interrupted")
    return 0
  except Exception:
    logger.exception("active-listener failed")
    return 1

  return 0
