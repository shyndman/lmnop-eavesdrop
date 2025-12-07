"""UI subprocess communication channel for Active Listener.

Provides subprocess management and communication capabilities for the UI application,
including process lifecycle management, message sending, and error detection.
"""

import asyncio
from pathlib import Path

from eavesdrop.active_listener.ui_messages import UIMessage, serialize_ui_message
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
    self._process: asyncio.subprocess.Process | None = None
    self._ready_signal_received = False
    self.logger = get_logger("ui_channel")

  def is_healthy(self) -> bool:
    """Check if the UI subprocess is running and responsive.

    :return: True if subprocess is running, False if crashed or terminated
    :rtype: bool
    """
    if not self._process:
      return False

    # Check if process is still running (None means still running)
    return self._process.returncode is None

  async def start(self) -> None:
    """Launch the UI subprocess and wait for ready signal.

    Spawns the UI process, monitors stdout for the ready signal
    "ACTIVE_LISTENER_UI_READY", and sets up communication pipes.

    :raises RuntimeError: If UI process fails to start or ready signal not received
    :raises FileNotFoundError: If UI executable is not found
    :raises PermissionError: If UI executable is not executable
    """
    self.logger.info(
      "Starting UI subprocess", executable=self._ui_bin_path, working_dir=self._working_dir
    )

    try:
      # Launch the UI subprocess with pipes for communication
      # stdout is DEVNULL because Textual apps render terminal UI to stdout
      self._process = await asyncio.create_subprocess_exec(
        str(self._ui_bin_path),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
        cwd=self._working_dir,
      )

      self.logger.info("UI subprocess launched", pid=self._process.pid)

      # Wait for ready signal, then continue monitoring stderr in background
      await self._monitor_ready_signal()

    except FileNotFoundError:
      self.logger.exception("UI executable not found", path=self._ui_bin_path)
      raise RuntimeError(f"UI executable not found: {self._ui_bin_path}")
    except PermissionError:
      self.logger.exception("UI executable not executable", path=self._ui_bin_path)
      raise RuntimeError(f"UI executable not executable: {self._ui_bin_path}")
    except Exception as e:
      self.logger.exception("Failed to start UI subprocess")
      raise RuntimeError(f"Failed to start UI subprocess: {e}")

  async def _monitor_ready_signal(self) -> None:
    """Monitor UI subprocess stderr for ready signal.

    Textual apps render to stdout, so the ready signal comes via stderr.
    After receiving the ready signal, this method continues monitoring
    stderr for debug/error output.

    :raises RuntimeError: If process exits before ready signal received
    """
    if not self._process or not self._process.stderr:
      raise RuntimeError("UI subprocess not properly initialized")

    warning_interval = 10.0  # Log warning every 10 seconds
    last_warning_time = asyncio.get_event_loop().time()

    while not self._ready_signal_received:
      # Check if process has exited
      if self._process.returncode is not None:
        raise RuntimeError(
          "UI subprocess exited before sending ready signal (exit code: {self._process.returncode})"
        )

      try:
        # Read stderr line with timeout
        line_bytes = await asyncio.wait_for(self._process.stderr.readline(), timeout=1.0)

        if not line_bytes:  # EOF - process closed stderr
          raise RuntimeError("UI subprocess closed stderr before sending ready signal")

        line = line_bytes.decode().strip()
        self.logger.debug("UI subprocess stderr", line=line)

        # Check for ready signal
        if "ACTIVE_LISTENER_UI_READY" in line:
          self.logger.info("UI subprocess ready signal received")
          self._ready_signal_received = True
          # Continue monitoring stderr in background after ready signal
          asyncio.create_task(self._monitor_stderr())
          return

      except asyncio.TimeoutError:
        # No output in the last second, check if we should log a warning
        current_time = asyncio.get_event_loop().time()
        if current_time - last_warning_time >= warning_interval:
          self.logger.warning(
            "Still waiting for UI ready signal", waiting_time=current_time - last_warning_time
          )
          last_warning_time = current_time
        continue

  async def _monitor_stderr(self) -> None:
    """Monitor UI subprocess stderr for debug/error output.

    Called after ready signal is received. Runs continuously in background
    to capture and log stderr output for debugging.
    """
    if not self._process or not self._process.stderr:
      return

    try:
      while True:
        line_bytes = await self._process.stderr.readline()
        if not line_bytes:  # EOF
          break

        line = line_bytes.decode().strip()
        if line:  # Only log non-empty lines
          self.logger.debug("UI subprocess stderr", line=line)

    except Exception:
      # Stderr monitoring is non-critical, don't let errors break the flow
      self.logger.exception("Error monitoring UI subprocess stderr")

  def send_message(self, message: UIMessage) -> None:
    """Send a JSON-line message to the UI subprocess stdin.

    Serializes the message as JSON and sends it to the UI process with
    newline termination. This is a fire-and-forget operation.

    :param message: Dictionary message to send to UI
    :type message: Message
    :raises BrokenPipeError: If UI subprocess stdin pipe is broken
    :raises RuntimeError: If UI subprocess is not running
    """
    if not self.is_healthy():
      raise RuntimeError("UI subprocess is not running")

    if not self._process or not self._process.stdin:
      raise RuntimeError("UI subprocess stdin not available")

    try:
      # Serialize message as JSON with newline termination
      json_line = serialize_ui_message(message) + "\n"
      json_bytes = json_line.encode("utf-8")

      # Write to subprocess stdin
      self._process.stdin.write(json_bytes)

      self.logger.debug("Sent message to UI", message_type=type(message).__name__)

    except BrokenPipeError:
      self.logger.exception("UI subprocess stdin pipe is broken")
      raise RuntimeError("UI subprocess stdin pipe is broken")
    except Exception as e:
      self.logger.exception("Failed to send message to UI subprocess")
      raise RuntimeError(f"Failed to send message to UI subprocess: {e}")

  async def shutdown(self) -> None:
    """Terminate the UI subprocess gracefully.

    Sends SIGTERM to the UI process and waits for clean termination.
    Forces termination if the process doesn't exit within reasonable time.

    :raises RuntimeError: If subprocess termination fails
    """
    if not self._process:
      self.logger.debug("No UI subprocess to shutdown")
      return

    if self._process.returncode is not None:
      self.logger.debug("UI subprocess already terminated", exit_code=self._process.returncode)
      return

    try:
      self.logger.info("Shutting down UI subprocess", pid=self._process.pid)

      # Send SIGTERM for graceful termination
      self._process.terminate()

      # Wait for process to exit with timeout
      await self._force_shutdown(self._process)

    except Exception as e:
      self.logger.exception("Error during UI subprocess shutdown")
      raise RuntimeError(f"Failed to shutdown UI subprocess: {e}")

  async def _force_shutdown(self, process: asyncio.subprocess.Process):
    try:
      await asyncio.wait_for(process.wait(), timeout=5.0)
      self.logger.info("UI subprocess terminated gracefully", exit_code=process.returncode)

    except asyncio.TimeoutError:
      self.logger.warning("UI subprocess did not exit gracefully, forcing termination")
      process.kill()
      await process.wait()
      self.logger.info("UI subprocess force terminated", exit_code=process.returncode)
