"""
Protocol interfaces for streaming transcription components.

Defines the contracts for audio input sources and transcription output sinks
using Python's Protocol system for structural typing.
"""

from typing import Protocol

import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass

from eavesdrop.wire import Segment


@dataclass
class TranscriptionResult:
  """Structured transcription result containing segments and metadata."""

  segments: list[Segment]
  """List of transcription segments with timing and text information."""

  language: str | None = None
  """Detected or specified language code."""

  language_probability: float | None = Field(default=None, ge=0.0, le=1.0)
  """Confidence score for language detection."""


class AudioSource(Protocol):
  """
  Protocol for audio input sources.

  Implementations provide audio data to the streaming transcription system.
  """

  async def read_audio(self) -> np.ndarray | None:
    """
    Read the next chunk of audio data.

    :returns:
        Audio data as numpy array, or None for end-of-stream.
    """
    ...

  def close(self) -> None:
    """Close the audio source and clean up resources."""
    ...


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

  async def send_language_detection(self, language: str, probability: float) -> None:
    """
    Send language detection result to the output destination.

    :param
        language: Detected language code.
        probability: Confidence score for the detection.
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
