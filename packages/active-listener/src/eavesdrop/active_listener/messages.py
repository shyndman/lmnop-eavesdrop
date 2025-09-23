"""
Message types for Python <-> Electron communication via stdin/stdout.

These messages handle real-time transcription display, mode switching between
transcription and command recognition, and operation lifecycle management.
"""

from enum import StrEnum
from typing import Literal

from pydantic import Field
from pydantic.dataclasses import dataclass

from eavesdrop.wire.transcription import Segment


class Mode(StrEnum):
  """Application modes that determine which DOM element receives transcription updates.

  - TRANSCRIBE: Normal speech-to-text mode, updates #transcription element
  - COMMAND: Voice command dictation mode, updates #command element
  """

  TRANSCRIBE = "TRANSCRIBE"
  COMMAND = "COMMAND"


class MessageType(StrEnum):
  """Message type discriminator enum for type-safe message handling."""

  APPEND_SEGMENTS = "append_segments"
  CHANGE_MODE = "change_mode"
  SET_SEGMENTS = "set_segments"
  SET_STRING = "set_string"
  COMMAND_EXECUTING = "command_executing"
  COMMIT_OPERATION = "commit_operation"


@dataclass(kw_only=True)
class AppendSegmentsMessage:
  """Sends real-time transcription updates to the Electron UI.

  This message incrementally updates the transcription display with new completed
  segments and the current in-progress segment. Used for high-frequency updates
  as speech is being processed and transcribed.

  The target_mode determines whether segments are displayed in the transcription
  buffer (TRANSCRIBE mode) or as command input (COMMAND mode).

  :param type: Message type discriminator
  :type type: Literal["append_segments"]
  :param target_mode: Whether segments go to transcription buffer or command input
  :type target_mode: Mode
  :param completed_segments: Newly finalized segments to append
  :type completed_segments: list[Segment]
  :param in_progress_segment: Current partial segment that may change
  :type in_progress_segment: Segment
  """

  type: Literal["append_segments"] = "append_segments"
  target_mode: Mode = Field(
    description="Which mode/DOM element to update (transcription vs command)"
  )
  completed_segments: list[Segment] = Field(
    description="Newly finalized segments to permanently append"
  )
  in_progress_segment: Segment = Field(
    description="Current partial transcription text that may change on next update"
  )


@dataclass(kw_only=True)
class ChangeModeMessage:
  """Switches between transcription buffer and command input modes.

  TRANSCRIBE mode: Speech is transcribed into a text buffer for eventual typing.
  COMMAND mode: Speech is transcribed as commands to describing edits to the transcription buffer.

  This message instructs the UI to switch between these semantic modes, affecting
  where subsequent transcription updates are displayed and how they're interpreted.

  :param type: Message type discriminator
  :type type: Literal["change_mode"]
  :param target_mode: The mode to switch to
  :type target_mode: Mode
  """

  type: Literal["change_mode"] = "change_mode"
  target_mode: Mode = Field(description="The mode to switch to")


@dataclass(kw_only=True)
class SetStringMessage:
  """Replaces content with a string that will be preprocessed by the UI.

  Used when command modifications are extensive and it's easier to send
  the result as processed text rather than reconstructing segments.
  The UI will handle formatting and display of the string content.

  This may become the primary method for sending command results due to
  its simplicity compared to segment reconstruction.

  :param type: Message type discriminator
  :type type: Literal["set_string"]
  :param target_mode: Whether updating transcription buffer or command input
  :type target_mode: Mode
  :param content: Processed text content to display
  :type content: str
  """

  type: Literal["set_string"] = "set_string"
  target_mode: Mode = Field(description="Which mode/DOM element to update")
  content: str = Field(description="Raw content string to preprocess and display")


@dataclass(kw_only=True)
class CommandExecutingMessage:
  """Signals that a command is being processed and provides status feedback.

  Sent when Python begins executing a recognized voice command (not when it
  completes). The waiting_messages are displayed to provide user feedback
  during potentially long-running command execution - could be status updates
  or fun phrases to keep the user engaged while waiting.

  :param type: Message type discriminator
  :type type: Literal["command_executing"]
  :param waiting_messages: Status messages or fun phrases to display while processing
  :type waiting_messages: list[str]
  """

  type: Literal["command_executing"] = "command_executing"
  waiting_messages: list[str] = Field(
    description="Messages providing user feedback during command processing"
  )


@dataclass(kw_only=True)
class CommitOperationMessage:
  """Signals completion of a transcription session.

  Indicates that the user has finished their transcription session and the
  final text should be typed/committed. This is triggered by an explicit
  user action and prepares the application for the next transcription session.

  The cancelled flag indicates whether the session completed normally or was
  interrupted/cancelled by the user.

  :param type: Message type discriminator
  :type type: Literal["commit_operation"]
  :param cancelled: True if the session was cancelled, false if completed normally
  :type cancelled: bool
  """

  type: Literal["commit_operation"] = "commit_operation"
  cancelled: bool = Field(
    description="True if the operation was cancelled, false if completed normally"
  )
