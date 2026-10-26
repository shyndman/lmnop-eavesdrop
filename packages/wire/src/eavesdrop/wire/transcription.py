"""
Transcription data types for wire protocol communication.

Contains the core data structures used for transcription results and user configuration.
"""

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


def compute_segment_chain_id(previous_id: int, text: str) -> int:
  """Generate chain-based segment ID using CRC64.

  Creates a deterministic, collision-resistant ID by chaining the previous
  segment's ID with the current segment's text content.

  :param previous_id: ID of the preceding segment in the chain
  :param text: Text content of the current segment
  :returns: CRC64 hash as positive integer
  """
  from fastcrc import crc64

  chain_input = f"{previous_id}{text}"
  return crc64.ecma_182(chain_input.encode())


class Word(BaseModel):
  """Native word text and recording-relative timestamps emitted with segments."""

  start: float
  """Start timestamp of the word in seconds on the recording timeline."""

  end: float
  """End timestamp of the word in seconds on the recording timeline."""

  word: str
  """The transcribed text content of the word."""


class Segment(BaseModel):
  """Transcript text, native timings, and stable completion identity for one segment."""

  id: int
  """Unique segment identifier computed using chain-based CRC64 hash."""

  start: float
  """Segment start time in seconds on the recording timeline."""

  end: float
  """Segment end time in seconds on the recording timeline."""

  text: str
  """Transcribed text content of the audio segment."""

  words: list[Word]
  """Native word-level text and recording-relative timing breakdown."""

  time_offset: float = Field(default=0.0, exclude=True)
  """Internal segment-local offset retained only for Python-side compatibility helpers."""

  completed: bool = False
  """Whether the segment transcription has been finalized and assigned a chain ID."""

  @property
  def in_progress(self) -> bool:
    """Whether the segment is still being processed and not yet finalized."""
    return not self.completed

  @property
  def absolute_start_time(self) -> float:
    """Return the start time in the recording timeline.

    Segments created by current wire-facing pipeline stages now already use recording-
    relative `start` values, so no offset is added when `start` already appears to be on
    the recording timeline. For older compatibility where `start` is still window-relative,
    `time_offset` is added as a fallback.
    """
    if self.time_offset > 0 and self.start < self.time_offset:
      return self.time_offset + self.start
    return self.start

  @property
  def absolute_end_time(self) -> float:
    """Return the end time in the recording timeline.

    Mirrors `absolute_start_time` so callers depending on compatibility helpers continue
    to receive timeline-correct values after the flattening cutover.
    """
    if self.time_offset > 0 and self.end < self.time_offset:
      return self.time_offset + self.end
    return self.end

  @computed_field
  @property
  def duration(self) -> float:
    """Return the duration of this segment in seconds."""
    return self.end - self.start

  def mark_completed(self, preceding_segment: "Segment | None") -> None:
    """Mark segment as completed and assign chain-based ID.

    Computes a deterministic ID based on the preceding segment's ID and this
    segment's text content. This creates a stable chain where the same sequence
    of text always produces the same sequence of IDs.

    :param preceding_segment: The segment that comes before this one in time order,
                             or None if this is the first segment
    """
    self.completed = True

    # Get previous ID or use baseline for first segment
    if preceding_segment is not None:
      previous_id = preceding_segment.id
    else:
      # Baseline ID for chain start
      previous_id = compute_segment_chain_id(0, "")

    # Assign chain-based ID
    self.id = compute_segment_chain_id(previous_id, self.text)


class UserTranscriptionOptions(BaseModel):
  """Supported session options for Orukeet's English transcription contract."""

  model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

  send_last_n_segments: int | None = Field(default=None, gt=0)
  """Number of most recent segments to send to the client."""

  language: Literal["en"] | None = None
  """English-only compatibility hint; None also selects English, without detection."""
