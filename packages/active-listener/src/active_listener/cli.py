"""CLI entrypoint for the active-listener service."""

from __future__ import annotations

import os

from clypi import Command, arg
from typing_extensions import override

from active_listener.app import ActiveListenerConfig, run_service
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


class ActiveListenerCommand(Command):
  """Run the long-lived active-listener hotkey service."""

  keyboard_name: str = arg(
    default_factory=lambda: os.getenv(
      "ACTIVE_LISTENER_KEYBOARD_NAME", "AT Translated Set 2 keyboard"
    ),
    help=(
      "Exact evdev keyboard device name. "
      '(Env: ACTIVE_LISTENER_KEYBOARD_NAME, default: "AT Translated Set 2 keyboard")'
    ),
  )
  host: str = arg(
    default=os.getenv("EAVESDROP_HOST", "localhost"),
    help="Eavesdrop server hostname. (Env: EAVESDROP_HOST)",
  )
  port: int = arg(
    default=env_int("EAVESDROP_PORT", 9090),
    help="Eavesdrop server port. (Env: EAVESDROP_PORT)",
  )
  audio_device: str = arg(
    default=os.getenv("EAVESDROP_AUDIO_DEVICE", "default"),
    help="PortAudio capture device name. (Env: EAVESDROP_AUDIO_DEVICE)",
  )
  ydotool_socket: str | None = arg(
    default=os.getenv("YDOTOOL_SOCKET"),
    help="Optional ydotool daemon socket path. (Env: YDOTOOL_SOCKET)",
  )

  @override
  async def run(self) -> None:
    """Build validated config and start the long-running service.

    :returns: None
    :rtype: None
    """

    config = ActiveListenerConfig.model_validate(
      {
        "keyboard_name": self.keyboard_name,
        "host": self.host,
        "port": self.port,
        "audio_device": self.audio_device,
        "ydotool_socket": self.ydotool_socket,
      },
      strict=True,
    )
    await run_service(config)


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
