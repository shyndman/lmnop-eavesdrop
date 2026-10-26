"""Shared audio representation and live transcription output contract."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic.dataclasses import dataclass

from eavesdrop.wire import Segment

Float32Audio = NDArray[np.float32]


@dataclass
class TranscriptionResult:
  """Structured transcription result containing segments and metadata."""

  segments: list[Segment]
  """List of transcription segments with timing and text information."""

  language: str | None = None
  """Configured transcription language code."""

  recording_id: str | None = None
  """Recording identifier associated with this result."""

  flush_complete: bool | None = None
  """True only for the single response that satisfies an accepted flush."""


class TranscriptionSink(Protocol):
  """
  Protocol for transcription result outputs.

  Implementations handle the delivery of transcription results to clients.
  """

  async def send_result(self, result: TranscriptionResult) -> None:
    """
    Send transcription result to the output destination.

    :param
        result: The transcription result to send.
    """
    ...

  async def send_error(self, error: str) -> None:
    """
    Send error message to the output destination.

    :param
        error: Error message to send.
    """
    ...

  async def send_server_ready(self, backend: str) -> None:
    """
    Send server ready notification to the output destination.

    :param
        backend: Name of the transcription backend being used.
    """
    ...

  async def disconnect(self) -> None:
    """Send disconnect notification and clean up resources."""
    ...
