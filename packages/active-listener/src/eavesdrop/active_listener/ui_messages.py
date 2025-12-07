"""
Message types for Python -> Electron communication via stdin/stdout.

These messages send workspace state updates to the UI for display purposes only.
The workspace maintains authoritative state; the UI is purely a view that reflects
workspace data without any bidirectional communication or state management.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter
from pydantic.dataclasses import dataclass

from eavesdrop.wire.transcription import Segment


class Mode(StrEnum):
  """Workspace operation modes that determine the purpose of voice input.

  - TRANSCRIBE: Voice builds up a text buffer for composition and content creation
  - COMMAND: Voice provides transformation instructions to an agent for processing the
    transcribed content

  Each mode maintains its own separate text buffer in the workspace.
  """

  TRANSCRIBE = "TRANSCRIBE"
  COMMAND = "COMMAND"


class UIMessageType(StrEnum):
  """Message type discriminator enum for type-safe message handling."""

  APPEND_SEGMENTS = "append_segments"
  CHANGE_MODE = "change_mode"
  SET_STRING = "set_string"
  COMMAND_EXECUTING = "command_executing"
  COMMIT_OPERATION = "commit_operation"


@dataclass(kw_only=True)
class AppendSegmentsMessage:
  """Sends workspace transcription state updates to the UI for display.

  Notifies the UI to display new completed segments and the current in-progress segment
  from the workspace's authoritative state. The UI renders this data without maintaining
  any state of its own.

  The target_mode indicates which workspace buffer (TRANSCRIBE or COMMAND) these
  segments belong to, determining which UI element should display them.

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
  """Notifies the UI that the workspace has switched operation modes.

  TRANSCRIBE mode: Speech builds up a text buffer for composition and content creation.
  COMMAND mode: Speech provides transformation instructions to an agent.

  This message informs the UI to update its display to reflect the workspace's
  current mode, affecting which UI element will receive subsequent updates.

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


# Union type for all UI messages
UIMessage = (
  AppendSegmentsMessage
  | ChangeModeMessage
  | SetStringMessage
  | CommandExecutingMessage
  | CommitOperationMessage
)


class _UIMessageCodec(BaseModel):
  """Private message wrapper type for deserializing the discriminated union of UI message types."""

  message: UIMessage = Field(discriminator="type")


def serialize_ui_message(message: UIMessage) -> str:
  """Serialize a UI message to JSON string.

  :param message: Any UI message instance
  :type message: UIMessage
  :return: JSON string representation of the message
  :rtype: str
  """
  message_type = type(message)
  adapter = TypeAdapter(message_type)
  return adapter.dump_json(message).decode("utf-8")


def deserialize_ui_message(json_str: str) -> UIMessage:
  """Deserialize a JSON string to UI message.

  :param json_str: JSON string containing the message
  :type json_str: str
  :return: Deserialized message instance of the appropriate type
  :rtype: UIMessage
  """
  # Wrap the incoming message in the expected MessageCodec structure
  wrapped_json = f'{{"message": {json_str}}}'
  codec = _UIMessageCodec.model_validate_json(wrapped_json)
  return codec.message
