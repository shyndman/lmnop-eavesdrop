"""Voice-driven text transcription workspace for Active Listener.

Provides a REPL-like environment for iterative text building and refinement
through voice transcription and command modes.
"""

from collections import OrderedDict
from dataclasses import dataclass

from eavesdrop.active_listener.messages import Mode
from eavesdrop.active_listener.ui_channel import UIChannel
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

  in_progress: Segment | None
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
    self._current_text: str = ""
    self._current_mode: Mode = Mode.TRANSCRIBE

    # Segment state tracking for UI synchronization
    self._completed_segments_sent: OrderedDict[int, Segment] = OrderedDict()
    """Track completed segments that have been sent to UI to avoid duplication"""

    self._current_in_progress_segment: Segment | None = None
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
      raise RuntimeError(f"Failed to process transcription message: {e}")

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
    newly_completed = []
    in_progress = None

    for segment in message.segments:
      if segment.completed:
        # Check if this is a newly completed segment
        if segment.id not in self._completed_segments_sent:
          newly_completed.append(segment)
          self.logger.debug(
            "Detected newly completed segment", segment_id=segment.id, text=segment.text
          )
      else:
        # This is an in-progress segment - there should only be one
        if in_progress is not None:
          self.logger.warning(
            "Multiple in-progress segments in message",
            first_id=in_progress.id,
            second_id=segment.id,
          )
        in_progress = segment
        self.logger.debug("Detected in-progress segment", segment_id=segment.id, text=segment.text)

    return SegmentChanges(newly_completed=newly_completed, in_progress=in_progress)

  def _update_workspace_text(self, changes: SegmentChanges) -> None:
    """Update the authoritative text representation with extracted changes.

    This is where the core text processing happens. Updates the workspace
    text buffer and internal state tracking. Future text processing features
    like keyword detection would be added here.

    :param changes: Semantic changes detected from transcription
    :type changes: SegmentChanges
    """
    # Update completed segments tracking
    for segment in changes.newly_completed:
      self._completed_segments_sent[segment.id] = segment
      self.logger.debug("Added completed segment to workspace", segment_id=segment.id)

    # Update current in-progress segment
    if changes.in_progress:
      if (
        self._current_in_progress_segment
        and changes.in_progress.id != self._current_in_progress_segment.id
      ):
        self.logger.debug(
          "In-progress segment changed",
          old_id=self._current_in_progress_segment.id,
          new_id=changes.in_progress.id,
        )
      self._current_in_progress_segment = changes.in_progress
    elif self._current_in_progress_segment:
      # No in-progress segment in message, but we had one - it might have completed
      self.logger.debug("No in-progress segment in message, clearing current")
      self._current_in_progress_segment = None

    # Rebuild complete text from segments
    completed_text = " ".join([segment.text for segment in self._completed_segments_sent.values()])
    if self._current_in_progress_segment:
      self._current_text = f"{completed_text} {self._current_in_progress_segment.text}".strip()
    else:
      self._current_text = completed_text.strip()

    self.logger.debug("Updated workspace text", text_length=len(self._current_text))

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

    # Create the message with appropriate fallback for in_progress
    message_data = {
      "type": "append_segments",
      "target_mode": self._current_mode,
      "completed_segments": newly_completed,
      "in_progress_segment": in_progress
      or Segment(
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
      ),
    }

    try:
      self._ui_channel.send_message(message_data)
      self.logger.debug(
        "Sent UI update",
        newly_completed_count=len(newly_completed),
        has_in_progress=in_progress is not None,
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

  def get_current_text(self) -> str:
    """Get the current text content of the workspace buffer.

    :return: Complete text content currently in the workspace
    :rtype: str
    """
    return self._current_text

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
    message_data = {"type": "change_mode", "target_mode": mode}

    try:
      self._ui_channel.send_message(message_data)
      self.logger.debug("Sent mode change to UI", mode=mode)
    except Exception as e:
      self.logger.error("Failed to send mode change message", error=str(e))
      # Don't let UI communication failures break the core processing
