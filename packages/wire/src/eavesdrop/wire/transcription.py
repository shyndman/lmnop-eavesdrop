"""
Transcription data types for wire protocol communication.

Contains the core data structures used for transcription results and user configuration.
"""

import math

from pydantic import BaseModel, Field, computed_field


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
  """Lists of these objects are sent accomanpanying segments if word-level timestamps are
  requested."""

  start: float
  """Start timestamp of the word in seconds within the audio segment."""

  end: float
  """End timestamp of the word in seconds within the audio segment."""

  word: str
  """The transcribed text content of the word."""

  probability: float
  """Confidence score from forced alignment, ranging from 0.0 to 1.0."""


class Segment(BaseModel):
  id: int
  """Unique segment identifier computed using chain-based CRC64 hash."""

  seek: int
  """Frame position in audio features where segment processing started."""

  start: float
  """Segment start time in seconds relative to the current audio window."""

  end: float
  """Segment end time in seconds relative to the current audio window."""

  text: str
  """Transcribed text content of the audio segment."""

  tokens: list[int]
  """List of token IDs from the model's vocabulary used to generate this segment."""

  avg_logprob: float
  """Average log probability across all tokens, indicating generation confidence."""

  compression_ratio: float
  """Ratio of text length to token count, used for hallucination detection."""

  words: list[Word] | None
  """Word-level timing breakdown when word timestamps are enabled, None otherwise."""

  temperature: float | None
  """Generation temperature used when creating this segment, None if not tracked."""

  completed: bool = False
  """Whether the segment transcription has been finalized and assigned a chain ID."""

  time_offset: float = 0.0
  """Absolute time offset to convert relative segment times to stream timestamps."""

  @computed_field
  @property
  def avg_probability(self) -> float:
    """Return the segment probability by exponentiating the average log probability."""
    return math.exp(self.avg_logprob)

  @computed_field
  @property
  def absolute_start_time(self) -> float:
    """Return the absolute start time in the audio stream."""
    return self.time_offset + self.start

  @computed_field
  @property
  def absolute_end_time(self) -> float:
    """Return the absolute end time in the audio stream."""
    return self.time_offset + self.end

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
  """Transcription options that clients can specify."""

  # Transcription behavior
  send_last_n_segments: int = Field(default=3, gt=0)
  """Number of most recent segments to send to the client."""

  initial_prompt: str | None = None
  """Initial prompt for whisper inference."""

  hotwords: list[str] | None = None
  """List of hotwords for whisper inference to improve recognition of specific terms."""

  word_timestamps: bool = False
  """Whether to include word timestamps in the transcription output."""
