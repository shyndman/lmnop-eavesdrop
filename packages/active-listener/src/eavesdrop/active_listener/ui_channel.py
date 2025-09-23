"""UI subprocess communication channel for Active Listener.

Provides subprocess management and communication capabilities for the UI application,
including process lifecycle management, message sending, and error detection.
"""

from pathlib import Path
from typing import Any

from eavesdrop.common import get_logger


class UIChannel:
  """Communication channel for UI subprocess management and messaging.

  Handles launching, monitoring, and communicating with the Active Listener UI
  subprocess through stdin/stdout pipes with JSON-line message protocol.

  :param ui_bin_path: Path to the UI executable binary
  :type ui_bin_path: Path
  :param working_dir: Working directory for the UI subprocess
  :type working_dir: Path
  """

  def __init__(self, ui_bin_path: Path, working_dir: Path) -> None:
    """Initialize the UI communication channel.

    :param ui_bin_path: Absolute path to the UI executable binary
    :type ui_bin_path: Path
    :param working_dir: Working directory to set for the UI subprocess
    :type working_dir: Path
    """
    self._ui_bin_path = ui_bin_path
    self._working_dir = working_dir
    self.logger = get_logger("ui_channel")

  async def start(self) -> None:
    """Launch the UI subprocess and wait for ready signal.

    Spawns the UI process, monitors stdout for the ready signal
    "ACTIVE_LISTENER_UI_READY", and sets up communication pipes.

    :raises RuntimeError: If UI process fails to start or ready signal not received
    :raises FileNotFoundError: If UI executable is not found
    :raises PermissionError: If UI executable is not executable
    """
    ...

  def send_message(self, message: dict[str, Any]) -> None:
    """Send a JSON-line message to the UI subprocess stdin.

    Serializes the message as JSON and sends it to the UI process with
    newline termination. This is a fire-and-forget operation.

    :param message: Dictionary message to send to UI
    :type message: dict[str, Any]
    :raises BrokenPipeError: If UI subprocess stdin pipe is broken
    :raises RuntimeError: If UI subprocess is not running
    """
    ...

  async def shutdown(self) -> None:
    """Terminate the UI subprocess gracefully.

    Sends SIGTERM to the UI process and waits for clean termination.
    Forces termination if the process doesn't exit within reasonable time.

    :raises RuntimeError: If subprocess termination fails
    """
    ...

  def is_healthy(self) -> bool:
    """Check if the UI subprocess is running and responsive.

    :return: True if subprocess is running, False if crashed or terminated
    :rtype: bool
    """
    ...
