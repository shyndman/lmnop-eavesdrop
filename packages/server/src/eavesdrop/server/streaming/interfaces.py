"""
Protocol interfaces for streaming transcription components.

Defines the contracts for transcription output sinks using Python's Protocol
system for structural typing.
"""

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
  """Specified language code."""

  recording_id: str | None = None
  """Live recording epoch identifier."""

  flush_complete: bool | None = None
  """True only for the single response that satisfies an accepted flush."""


class TranscriptionSink(Protocol):
  """Protocol for transcription result outputs."""

  async def send_result(self, result: TranscriptionResult) -> None:
    """Send transcription result to the output destination."""
    ...

  async def send_error(self, error: str) -> None:
    """Send error message to the output destination."""
    ...

  async def send_server_ready(self, backend: str) -> None:
    """Send server ready notification to the output destination."""
    ...

  async def disconnect(self) -> None:
    """Send disconnect notification and clean up resources."""
    ...
