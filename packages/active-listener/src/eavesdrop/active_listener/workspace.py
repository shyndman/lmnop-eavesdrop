"""Voice-driven text transcription workspace for Active Listener.

Provides a REPL-like environment for iterative text building and refinement
through voice transcription and command modes.
"""

from eavesdrop.active_listener.messages import Mode
from eavesdrop.active_listener.ui_channel import UIChannel
from eavesdrop.common import get_logger
from eavesdrop.wire import TranscriptionMessage


class TextTranscriptionWorkspace:
  """Voice-driven text workspace with dual-mode operation.

  Manages a text buffer that users can build and refine through voice
  transcription (TRANSCRIBE mode) and voice commands (COMMAND mode).
  Coordinates state changes with the UI through the communication channel.

  :param ui_channel: Communication channel for UI subprocess messaging
  :type ui_channel: UIChannel
  """

  def __init__(self, ui_channel: UIChannel) -> None:
    """Initialize the text transcription workspace.

    :param ui_channel: Active UI communication channel for sending updates
    :type ui_channel: UIChannel
    """
    self._ui_channel = ui_channel
    self._current_text: str = ""
    self._current_mode: Mode = Mode.TRANSCRIBE
    self.logger = get_logger("workspace")

  def process_transcription_message(self, message: TranscriptionMessage) -> None:
    """Process incoming transcription message and update workspace state.

    Analyzes the transcription segments, updates internal text state,
    and sends appropriate UI update messages based on current mode.

    :param message: Transcription message from eavesdrop server
    :type message: TranscriptionMessage
    """
    ...

  def set_mode(self, mode: Mode) -> None:
    """Switch workspace between transcription and command modes.

    Changes the workspace mode and notifies the UI of the mode change.
    TRANSCRIBE mode: voice input builds the text buffer
    COMMAND mode: voice input edits/transforms the text buffer

    :param mode: Target mode to switch to
    :type mode: Mode
    """
    ...

  def get_current_text(self) -> str:
    """Get the current text content of the workspace buffer.

    :return: Complete text content currently in the workspace
    :rtype: str
    """
    ...

  def get_current_mode(self) -> Mode:
    """Get the current operating mode of the workspace.

    :return: Current mode (TRANSCRIBE or COMMAND)
    :rtype: Mode
    """
    ...
