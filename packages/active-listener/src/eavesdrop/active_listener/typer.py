"""Desktop typing functionality using python-ydotool.

Provides desktop automation capabilities for typing transcribed text
into the currently focused application with error recovery and validation.
"""

import os

import pydotool
import structlog

from eavesdrop.active_listener.text_manager import TypingOperation


class DesktopTyper:
  """Handles desktop typing operations using ydotool."""

  def __init__(self):
    self._initialized: bool = False
    self._available: bool = False
    self.logger = structlog.get_logger("🤸👂")

  def initialize(self) -> None:
    """Initialize ydotool system integration."""
    socket_paths = [
      f"/run/user/{os.getuid()}/.ydotool_socket",
      "/tmp/.ydotool_socket",
      f"/tmp/.ydotool_socket_{os.getuid()}",
    ]

    for socket_path in socket_paths:
      if os.path.exists(socket_path):
        self.logger.info("Found ydotool socket", path=socket_path)
        try:
          pydotool.init(socket_path)
          self._initialized = True
          self._available = True
          self.logger.info("ydotool initialized successfully", socket=socket_path)
          return
        except Exception as e:
          self.logger.warning("Failed to initialize with socket", path=socket_path, error=str(e))
          continue

    # No working socket found
    self.logger.error("No working ydotool socket found", searched_paths=socket_paths)
    self._initialized = False
    self._available = False
    raise Exception(f"Failed to find working ydotool socket. Searched: {socket_paths}")

  def is_available(self) -> bool:
    """Check if ydotool is available and properly initialized."""
    return self._available and self._initialized

  def type_text(self, text: str) -> None:
    """Type the given text using ydotool."""
    if not self.is_available():
      raise Exception("ydotool is not available")

    try:
      pydotool.type_string(text)
    except Exception as e:
      self._available = False
      raise Exception(f"Failed to type text: {e}")

  def delete_characters(self, count: int) -> None:
    """Delete the specified number of characters using backspace."""
    if not self.is_available():
      raise Exception("pydotool is not available")

    if count <= 0:
      return

    try:
      # Use key_combination for backspace operations
      for _ in range(count):
        pydotool.key_combination([pydotool.KEY_BACKSPACE])
    except Exception as e:
      self._available = False
      raise Exception(f"Failed to delete characters: {e}")

  def execute_typing_operation(self, operation: TypingOperation) -> bool:
    """Execute a complete typing operation with delete + type sequence."""
    if not self.is_available():
      return False

    try:
      # Delete characters first
      if operation.chars_to_delete > 0:
        self.delete_characters(operation.chars_to_delete)

      # Type new text
      if operation.text_to_type:
        self.type_text(operation.text_to_type)

      # Mark operation as completed
      operation.completed = True
      return True

    except Exception:
      operation.completed = False
      return False

  def execute_with_retry(self, operation: TypingOperation, max_attempts: int = 3) -> bool:
    """Execute typing operation with retry logic."""
    for attempt in range(max_attempts):
      try:
        if self.execute_typing_operation(operation):
          return True

        # If pydotool became unavailable, try to reinitialize
        if not self.is_available():
          self.initialize()

      except Exception:
        if attempt == max_attempts - 1:
          return False
        continue

    return False

  def check_permissions(self) -> bool:
    """Check if system permissions are properly configured for ydotool."""
    try:
      # Check if /dev/uinput exists and is accessible
      if not os.path.exists("/dev/uinput"):
        return False

      # Check if we can read the uinput device
      return os.access("/dev/uinput", os.R_OK | os.W_OK)

    except Exception:
      return False

  def get_permission_error_message(self) -> str:
    """Get helpful error message for permission issues."""
    return (
      "ydotool requires proper permissions to access /dev/uinput. "
      "Please ensure:\n"
      "1. Your user is in the 'input' group: sudo usermod -a -G input $USER\n"
      "2. Log out and back in for group changes to take effect\n"
      "3. The uinput module is loaded: sudo modprobe uinput\n"
      "4. ydotool is installed: sudo apt install ydotool"
    )
