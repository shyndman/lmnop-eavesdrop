"""
Transcription data types for wire protocol communication.

Contains the core data structures used for transcription results and user configuration.
"""

from pydantic import BaseModel, Field


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
  end: float
  word: str
  probability: float


class Segment(BaseModel):
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
