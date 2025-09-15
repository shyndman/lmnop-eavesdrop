"""
Transcription data types for wire protocol communication.

Contains the core data structures used for transcription results and user configuration.
"""

from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass


@dataclass
class Word:
  start: float
  end: float
  word: str
  probability: float


@dataclass
class Segment:
  id: int
  seek: int
  start: float
  end: float
  text: str
  tokens: list[int]
  avg_logprob: float
  compression_ratio: float
  words: list[Word] | None
  temperature: float | None
  completed: bool = False
  time_offset: float = 0.0

  def absolute_start_time(self) -> float:
    """Return the absolute start time in the audio stream."""
    return self.time_offset + self.start

  def absolute_end_time(self) -> float:
    """Return the absolute end time in the audio stream."""
    return self.time_offset + self.end

  def duration(self) -> float:
    """Return the duration of this segment in seconds."""
    return self.end - self.start


class UserTranscriptionOptions(BaseModel):
  """Transcription options that clients can specify."""

  # Transcription behavior
  send_last_n_segments: int = Field(default=1, gt=0)
  """Number of most recent segments to send to the client."""

  initial_prompt: str | None = None
  """Initial prompt for whisper inference."""

  hotwords: list[str] | None = None
  """List of hotwords for whisper inference to improve recognition of specific terms."""

  word_timestamps: bool = False
  """Whether to include word timestamps in the transcription output."""
