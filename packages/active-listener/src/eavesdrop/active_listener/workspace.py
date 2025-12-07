"""Voice-driven text transcription workspace for Active Listener.

Provides a REPL-like environment for iterative text building and refinement
through voice transcription and command modes.
"""

from collections import OrderedDict
from io import StringIO

from pydantic.dataclasses import dataclass

from eavesdrop.active_listener.ui_channel import UIChannel
from eavesdrop.active_listener.ui_messages import (
  AppendSegmentsMessage,
  ChangeModeMessage,
  Mode,
)
from eavesdrop.common import get_logger
from eavesdrop.wire import TranscriptionMessage
from eavesdrop.wire.transcription import Segment


@dataclass
class SegmentChanges:
  """Result of extracting changes from a transcription message.

  Represents the semantic changes that occurred in the transcription stream,
  independent of UI concerns or other side effects.
  """

  newly_completed: list[Segment]
  """Segments that transitioned from in-progress to completed"""

  in_progress: Segment
  """Current incomplete segment being transcribed (if any)"""


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

    self._current_mode: Mode = Mode.TRANSCRIBE
    self._text_by_mode: dict[Mode, StringIO] = {
      Mode.TRANSCRIBE: StringIO(),
      Mode.COMMAND: StringIO(),
    }

    self._completed_by_mode: dict[Mode, OrderedDict[int, Segment]] = {
      Mode.TRANSCRIBE: OrderedDict(),
      Mode.COMMAND: OrderedDict(),
    }
    """Track completed segments per mode to avoid duplication and cross-mode contamination"""

    self._in_progress_start_pos: dict[Mode, int] = {}
    """Track where in-progress segment starts in each buffer for efficient replacement"""

    self._in_progress: Segment | None = None
    """Track the current incomplete segment being processed"""

    self.logger = get_logger("workspace")

  def process_transcription_message(self, message: TranscriptionMessage) -> None:
    """Process incoming transcription message and update workspace state.

    Three-stage processing pipeline:
    1. Extract semantic changes from the message
    2. Update workspace text representation with changes
    3. Notify UI of changes as a side effect

    :param message: Transcription message from eavesdrop server
    :type message: TranscriptionMessage
    """
    self.logger.debug(
      "Processing transcription message", segment_count=len(message.segments), stream=message.stream
    )

    try:
      # Stage 1: Extract what actually changed in the transcription
      changes = self._extract_segment_changes(message)

      # Stage 2: Update our authoritative text state with the changes
      self._update_workspace_text(changes)

      # Stage 3: Notify UI of the changes as a side effect
      self._notify_ui(changes)

    except Exception as e:
      self.logger.exception("Error processing transcription message")
      raise RuntimeError(f"Failed to process transcription message: {e}") from e

  def _extract_segment_changes(self, message: TranscriptionMessage) -> SegmentChanges:
    """Extract semantic changes from transcription message.

    Analyzes the message segments to identify what actually changed:
    - Which segments newly transitioned from in-progress to completed
    - What the current in-progress segment is (if any)

    This is pure business logic focused on understanding the transcription
    stream, independent of UI concerns or text processing.

    :param message: Transcription message from server
    :type message: TranscriptionMessage
    :return: Structured changes detected in the message
    :rtype: SegmentChanges
    """
    newly_completed = [
      s
      for s in message.segments
      if s.completed and s.id not in self._completed_by_mode[self._current_mode]
    ]
    return SegmentChanges(newly_completed=newly_completed, in_progress=message.segments[-1])

  def _update_workspace_text(self, changes: SegmentChanges) -> None:
    """Update the authoritative text representation with extracted changes.

    This is where the core text processing happens. Updates the workspace
    text buffer and internal state tracking. Future text processing features
    like keyword detection would be added here.

    :param changes: Semantic changes detected from transcription
    :type changes: SegmentChanges
    """
    # Update completed segments tracking
    completed = self._completed_by_mode[self._current_mode]
    for s in changes.newly_completed:
      completed[s.id] = s
      self.logger.debug("Added completed segment to workspace", segment_id=s.id)

    # Update current in-progress segment
    previous_id = self._in_progress.id if self._in_progress is not None else None
    if changes.in_progress.id != previous_id:
      self.logger.debug(
        "In-progress segment changed",
        old_id=previous_id,
        new_id=changes.in_progress.id,
      )
    self._in_progress = changes.in_progress

    # Rebuild complete text from segments and write to current mode buffer
    completed_text = " ".join([segment.text for segment in completed.values()])
    if self._in_progress:
      current_text = f"{completed_text} {self._in_progress.text}".strip()
    else:
      current_text = completed_text.strip()

    # Write to the StringIO buffer for the current mode
    current_buffer = self._text_by_mode[self._current_mode]
    current_buffer.seek(0)
    current_buffer.truncate()
    current_buffer.write(current_text)

    self.logger.debug("Updated workspace text", mode=self._current_mode, text=current_text)

  def _notify_ui(self, changes: SegmentChanges) -> None:
    """Notify UI of workspace changes as a side effect.

    Sends UI messages based on the changes that were already processed
    and applied to the workspace. This is a pure side effect - the
    workspace state is already updated.

    :param changes: Changes that were applied to workspace
    :type changes: SegmentChanges
    """
    if changes.newly_completed or changes.in_progress:
      self._send_append_segments_message(changes.newly_completed, changes.in_progress)

  def _send_append_segments_message(
    self, newly_completed: list[Segment], in_progress: Segment | None
  ) -> None:
    """Send AppendSegmentsMessage to UI subprocess.

    :param newly_completed: Segments that just became completed
    :type newly_completed: list[Segment]
    :param in_progress: Current incomplete segment (if any)
    :type in_progress: Segment | None
    """

    # Don't send message if there's no in-progress segment and no newly completed segments
    if not newly_completed and in_progress is None:
      self.logger.debug("No changes to send to UI, skipping message")
      return

    # The protocol requires in_progress_segment to be non-null, but this is a design flaw
    # For now, we have to provide a fallback, but this should be fixed in the protocol
    if in_progress is None:
      self.logger.warning("No in-progress segment but protocol requires one - using empty segment")
      in_progress = Segment(
        id=0,
        seek=0,
        start=0.0,
        end=0.0,
        text="",
        tokens=[],
        avg_logprob=0.0,
        compression_ratio=0.0,
        words=None,
        temperature=None,
      )

    message = AppendSegmentsMessage(
      target_mode=self._current_mode,
      completed_segments=newly_completed,
      in_progress_segment=in_progress,
    )

    try:
      # TODO The channel should be typed to take in Message. Perform the serialization internally.
      self._ui_channel.send_message(message)
      self.logger.debug(
        "Sent UI update",
        newly_completed_count=len(newly_completed),
        has_in_progress=in_progress.text != "",
      )
    except Exception as e:
      self.logger.error("Failed to send UI message", error=str(e))
      # Don't let UI communication failures break the core processing

  def set_mode(self, mode: Mode) -> None:
    """Switch workspace between transcription and command modes.

    Changes the workspace mode and notifies the UI of the mode change.
    TRANSCRIBE mode: voice input builds the text buffer
    COMMAND mode: voice input edits/transforms the text buffer

    :param mode: Target mode to switch to
    :type mode: Mode
    """
    if mode == self._current_mode:
      self.logger.debug("Mode unchanged", mode=mode)
      return

    previous_mode = self._current_mode
    self._current_mode = mode

    self.logger.info("Mode switched", previous=previous_mode, new=mode)

    # Notify UI of mode change as side effect
    self._send_mode_change_message(mode)

  def get_text(self) -> str:
    """Get the current text content of the workspace buffer.

    :return: Complete text content currently in the workspace
    :rtype: str
    """
    return self._text_by_mode[self._current_mode].getvalue()

  def get_mode(self) -> Mode:
    """Get the current operating mode of the workspace.

    :return: Current mode (TRANSCRIBE or COMMAND)
    :rtype: Mode
    """
    return self._current_mode

  def _send_mode_change_message(self, mode: Mode) -> None:
    """Send ChangeModeMessage to UI subprocess.

    :param mode: New mode to send to UI
    :type mode: Mode
    """
    try:
      # TODO The channel should be typed to take in Message. Perform the serialization internally.
      self._ui_channel.send_message(ChangeModeMessage(target_mode=mode))
      self.logger.debug("Sent mode change to UI", mode=mode)
    except Exception as e:
      self.logger.error("Failed to send mode change message", error=str(e))
      # Don't let UI communication failures break the core processing
