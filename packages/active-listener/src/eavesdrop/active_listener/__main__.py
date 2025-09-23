"""Main entry point for eavesdrop active listener application."""

import os
import sys
from pathlib import Path
from typing import NamedTuple, TypedDict

import sounddevice as sd
from clypi import Command, arg

from eavesdrop.active_listener.app import App
from eavesdrop.common import get_logger, setup_logging_from_env


class AudioDevice(TypedDict):
  """Type definition for sounddevice query_devices() return value."""

  name: str
  hostapi: int
  max_input_channels: int
  max_output_channels: int
  default_low_input_latency: float
  default_low_output_latency: float
  default_high_input_latency: float
  default_high_output_latency: float
  default_samplerate: float


class ServerHostPort(NamedTuple):
  """Named tuple for server host and port."""

  host: str
  port: int


def parse_ui_binary(value: str | list[str]) -> Path:
  """Parse and validate UI binary path.

  Validates that the path exists, is a file, and has executable permissions.
  The path points to the active-listener-ui unpacked binary that will be
  used for the user interface application.

  :param value: Path to the UI binary executable
  :type value: str
  :return: Validated absolute path to the executable
  :rtype: Path
  :raises ValueError: If path doesn't exist, isn't executable, or is invalid
  """
  assert isinstance(value, str), "UI binary path must be a string"

  if not isinstance(value, str):
    raise ValueError("--ui-bin must be a string")

  if not value.strip():
    raise ValueError("--ui-bin cannot be empty")

  # Resolve to absolute path
  ui_path = Path(value).resolve()

  # Check if path exists
  if not ui_path.exists():
    raise ValueError(f"--ui-bin does not exist: {ui_path}")

  # Check if it's a file (not a directory)
  if not ui_path.is_file():
    raise ValueError(f"--ui-bin is not a file: {ui_path}")

  # Check if it's executable
  if not os.access(ui_path, os.X_OK):
    raise ValueError(f"--ui-bin is not executable: {ui_path}")

  return ui_path


def parse_server(value: str | list[str]) -> ServerHostPort:
  assert isinstance(value, str), "Server must be a string in hostname:port format"

  """Parse server argument in hostname:port format."""
  if not isinstance(value, str):
    raise ValueError("Server must be a string")

  if not value:
    raise ValueError("Invalid server format: cannot be empty")

  if ":" not in value:
    raise ValueError("Invalid server format: must be hostname:port")

  host, port_str = value.rsplit(":", 1)

  if not host:
    raise ValueError("Invalid server format: missing hostname")

  if not port_str:
    raise ValueError("Invalid server format: missing port after colon")

  try:
    port = int(port_str)
  except ValueError:
    raise ValueError("Invalid port: must be numeric")

  if port < 1 or port > 65535:
    raise ValueError("Invalid port: must be between 1 and 65535")

  return ServerHostPort(host=host, port=port)


class ListDevices(Command):
  """List available audio devices.

  Shows all available audio input and output devices with their IDs,
  names, channel counts, and host API information.
  """

  async def run(self) -> None:
    """List all available audio input devices."""
    try:
      all_devices = sd.query_devices()
      print("Audio Input Devices (for use with --audio-device):")
      print("=" * 55)

      # Get default device info
      default_input = sd.default.device[0] if isinstance(sd.default.device, tuple) else None

      input_devices = []
      for i, device in enumerate(all_devices):
        device_typed: AudioDevice = device  # Type annotation for the device
        if device_typed["max_input_channels"] > 0:
          input_devices.append((i, device_typed))

      if not input_devices:
        print("No input devices found.")
        return

      for device_id, device in input_devices:
        # Check if this is the default input device
        is_default = device_id == default_input

        # Show device name prominently for easy copying
        device_name = device["name"]
        default_marker = " [DEFAULT INPUT]" if is_default else ""

        print(f"Device: {device_name}{default_marker}")
        print(f"  ID: {device_id}")
        print(f"  Input channels: {device['max_input_channels']}")
        print(f"  Low input latency: {device['default_low_input_latency']:.4f}s")
        print(f"  High input latency: {device['default_high_input_latency']:.4f}s")
        print()

    except Exception as e:
      print(f"Error querying audio devices: {e}")
      raise


class ActiveListener(Command):
  """Eavesdrop active listener - real-time voice-to-text dictation.

  Connects to an eavesdrop transcription server, captures audio from the
  specified input device, and automatically types transcription results
  into the currently focused desktop application.
  """

  subcommand: ListDevices | None

  server: ServerHostPort = arg(default=ServerHostPort("localhost", 9090), parser=parse_server)
  audio_device: str = arg(default="default")
  ui_bin: Path = arg(parser=parse_ui_binary)

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.logger = get_logger("cli")
    self._ui_bin_path = self.ui_bin

  async def run(self) -> None:
    """Main entry point for the command."""
    app = App.create(
      server_host=self.server.host,
      server_port=self.server.port,
      audio_device=self.audio_device,
      ui_bin_path=self._ui_bin_path,
    )
    await app.run()


def main() -> None:
  setup_logging_from_env()
  logger = get_logger("main")

  """Main entry point for active-listener command."""
  try:
    cli = ActiveListener.parse()
    cli.start()
  except KeyboardInterrupt:
    logger.info("\nReceived interrupt signal, shutting down...")
    sys.exit(0)
  except Exception:
    logger.exception("Fatal error")
    sys.exit(1)


if __name__ == "__main__":
  main()
