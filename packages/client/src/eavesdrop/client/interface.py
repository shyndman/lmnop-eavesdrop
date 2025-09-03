"""
Terminal interface and user interaction for Eavesdrop client.
"""

import select
import sys
import termios
import threading
import tty
from collections.abc import Callable


class TerminalInterface:
  """Handles terminal control and user interaction."""

  def __init__(
    self,
    on_start_session: Callable[[], None],
    on_stop_session: Callable[[], None],
    on_exit: Callable[[], None],
  ):
    self.on_start_session = on_start_session
    self.on_stop_session = on_stop_session
    self.on_exit = on_exit

    self.old_settings: list | None = None
    self.running = True
    self.session_active = False

  def setup_terminal(self):
    """Setup terminal for raw input to detect spacebar presses."""
    try:
      self.old_settings = termios.tcgetattr(sys.stdin)
      tty.setraw(sys.stdin.fileno())
    except Exception:
      pass

  def restore_terminal(self):
    """Restore normal terminal settings."""
    try:
      if self.old_settings:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    except Exception:
      pass

  def safe_print(self, message: str):
    """Print with proper terminal handling."""
    self.restore_terminal()
    print(message)
    sys.stdout.flush()
    self.setup_terminal()

  def set_session_active(self, active: bool):
    """Update session state for keyboard handling."""
    self.session_active = active

  def keyboard_listener(self):
    """Listen for spacebar presses and control keys."""
    try:
      while self.running:
        if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
          char = sys.stdin.read(1)
          if char == " ":  # Spacebar
            if self.session_active:
              self.on_stop_session()
            else:
              self.on_start_session()
          elif char == "\x03":  # Ctrl+C
            self.running = False
            self.on_exit()
            break
    except KeyboardInterrupt:
      self.running = False
      self.on_exit()

  def start_keyboard_listener(self):
    """Start keyboard listener in background thread."""
    keyboard_thread = threading.Thread(target=self.keyboard_listener)
    keyboard_thread.daemon = True
    keyboard_thread.start()

  def show_welcome_message(self, host: str, port: int):
    """Display welcome message and instructions."""
    print("Eavesdrop Microphone Client")
    print(f"Server: {host}:{port}")
    print("\nPress SPACEBAR to start recording...")
    print("Press Ctrl+C to exit\n")

  def cleanup(self):
    """Clean up terminal state."""
    self.running = False
    self.restore_terminal()
