"""Clipboard-based text output for commit operations.

Provides functionality to copy transcribed text to the system clipboard
and trigger a paste operation into the currently focused application.
"""

import os
import subprocess

import pydotool

from eavesdrop.common import get_logger

logger = get_logger("output")

# Module-level state for ydotool initialization
_ydotool_initialized = False


class ClipboardOutputError(Exception):
  """Raised when clipboard or paste operations fail."""


def _ensure_ydotool_initialized() -> None:
  """Initialize ydotool if not already done.

  Searches for the ydotool socket in common locations and initializes
  the pydotool library.

  :raises ClipboardOutputError: If no working ydotool socket is found
  """
  global _ydotool_initialized

  if _ydotool_initialized:
    return

  socket_paths = [
    f"/run/user/{os.getuid()}/.ydotool_socket",
    "/tmp/.ydotool_socket",
    f"/tmp/.ydotool_socket_{os.getuid()}",
  ]

  for socket_path in socket_paths:
    if os.path.exists(socket_path):
      try:
        pydotool.init(socket_path)
        _ydotool_initialized = True
        logger.info("ydotool initialized", socket=socket_path)
        return
      except Exception as e:
        logger.warning("Failed to initialize ydotool with socket", path=socket_path, error=str(e))
        continue

  raise ClipboardOutputError(
    f"No working ydotool socket found. Searched: {socket_paths}. Ensure ydotoold is running."
  )


def _copy_to_clipboard(text: str) -> None:
  """Copy text to the system clipboard using xclip.

  :param text: Text to copy to clipboard
  :type text: str
  :raises ClipboardOutputError: If the clipboard operation fails
  """
  try:
    _ = subprocess.run(
      ["xclip", "-selection", "clipboard"],
      input=text.encode("utf-8"),
      check=True,
      capture_output=True,
    )
  except subprocess.CalledProcessError as e:
    stderr_msg = e.stderr.decode() if e.stderr else "unknown error"
    raise ClipboardOutputError(f"Failed to copy to clipboard: {stderr_msg}") from e
  except FileNotFoundError as e:
    raise ClipboardOutputError("xclip not found. Please install it: sudo apt install xclip") from e


def _trigger_paste() -> None:
  """Trigger a Ctrl+V paste operation via ydotool.

  :raises ClipboardOutputError: If the paste operation fails
  """
  _ensure_ydotool_initialized()

  try:
    pydotool.key_combination([pydotool.KEY_LEFTCTRL, pydotool.KEY_V])
  except Exception as e:
    raise ClipboardOutputError(f"Failed to trigger paste: {e}") from e


def commit_text_to_focused_app(text: str) -> None:
  """Copy text to clipboard and paste into the currently focused application.

  This is the primary output mechanism for the active listener. When the user
  commits their transcription session, this function copies the final text
  to the clipboard and triggers a paste operation.

  :param text: The text to output
  :type text: str
  :raises ClipboardOutputError: If clipboard copy or paste fails
  """
  logger.info("Committing text to focused application", text=text)

  _copy_to_clipboard(text)
  logger.debug("Text copied to clipboard")

  _trigger_paste()
  logger.debug("Paste triggered")
