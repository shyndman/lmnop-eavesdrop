"""
Protocol interfaces for streaming transcription components.

Defines the contracts for audio input sources and transcription output sinks
using Python's Protocol system for structural typing.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class TranscriptionResult:
  """Structured transcription result containing segments and metadata."""

  segments: list[dict]
  """List of transcription segments with timing and text information."""

  language: str | None = None
  """Detected or specified language code."""

  language_probability: float | None = None
  """Confidence score for language detection."""


class AudioSource(Protocol):
  """
  Protocol for audio input sources.

  Implementations provide audio data to the streaming transcription system.
  """

  async def read_audio(self) -> np.ndarray | None:
    """
    Read the next chunk of audio data.

    Returns:
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

    Args:
        result: The transcription result to send.
    """
    ...

  async def send_error(self, error: str) -> None:
    """
    Send error message to the output destination.

    Args:
        error: Error message to send.
    """
    ...

  async def send_language_detection(self, language: str, probability: float) -> None:
    """
    Send language detection result to the output destination.

    Args:
        language: Detected language code.
        probability: Confidence score for the detection.
    """
    ...

  async def send_server_ready(self, backend: str) -> None:
    """
    Send server ready notification to the output destination.

    Args:
        backend: Name of the transcription backend being used.
    """
    ...

  async def disconnect(self) -> None:
    """Send disconnect notification and clean up resources."""
    ...
