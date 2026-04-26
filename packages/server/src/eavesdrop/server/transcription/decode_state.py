from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from faster_whisper.audio import pad_or_trim

from eavesdrop.common import get_logger
from eavesdrop.server.transcription.models import SegmentDict, TranscriptionOptions
from eavesdrop.wire import Segment, Word


class _TranscribeSegmentsResult(NamedTuple):
  """Result from transcription with generation metrics."""

  segments: Iterable[Segment]
  total_attempts: int


@dataclass
class _TranscribeContext:
  """Context manager for transcription loop state and calculations."""

  # Input parameters
  features: np.ndarray
  time_per_frame: float
  max_segment_frames: int  # nb_max_frames from feature extractor
  frames_per_second: float
  initial_tokens: list[int] = field(default_factory=list)

  # Computed on init
  total_frames: int = field(init=False)
  """Total number of frames in the audio features."""

  total_duration: float = field(init=False)
  """Total duration of audio in seconds."""

  # Loop state

  seek: int = field(default=0, init=False)
  """Current position in audio features (frame index)."""

  at_beginning: bool = field(default=True, init=False)
  """True if at the beginning of the audio."""

  all_tokens: list[int] = field(init=False)
  """Accumulated token history for context conditioning."""

  prompt_reset_since: int = field(default=0, init=False)
  """Token position where prompt context was last reset."""

  last_speech_timestamp: float = field(default=0.0, init=False)
  """Timestamp of last detected speech for word alignment."""

  all_segments: list[Segment] = field(default_factory=list, init=False)
  """Collected transcription segments."""

  total_attempts: int = field(default=0, init=False)
  """Total generation attempts across all segments."""

  # Computed fields (updated on commit)
  time_offset: float = field(default=0.0, init=False)
  """Current time offset in seconds."""

  window_end_time: float = field(default=0.0, init=False)
  """End time of current processing window."""

  segment_size: int = field(default=0, init=False)
  """Size of current segment in frames."""

  segment_duration: float = field(default=0.0, init=False)
  """Duration of current segment in seconds."""

  context_tokens: list[int] = field(default_factory=list, init=False)
  """Tokens for current context window."""

  def __post_init__(self):
    self.total_frames = self.features.shape[-1] - 1
    self.total_duration = self.total_frames * self.time_per_frame
    self.all_tokens = self.initial_tokens.copy()

  def advance(self) -> bool:
    """Recompute all derived values from current seek position."""
    if self.done():
      return False

    self.at_beginning = self.seek == 0
    self.time_offset = self.seek * self.time_per_frame
    self.window_end_time = (self.seek + self.max_segment_frames) * self.time_per_frame
    self.segment_size = min(self.max_segment_frames, self.total_frames - self.seek)
    self.segment_duration = self.segment_size * self.time_per_frame
    self.context_tokens = self.all_tokens[self.prompt_reset_since :]
    return True

  def done(self) -> bool:
    """Check if we've processed all frames."""
    return self.seek >= self.total_frames

  def extract_segment(self) -> np.ndarray:
    """Extract and pad current audio segment."""
    segment = self.features[:, self.seek : self.seek + self.segment_size]
    return pad_or_trim(segment)

  def seek_next_to(self, new_seek: int):
    """Move to next position and commit changes."""
    self.seek = new_seek

  def add_segment(
    self,
    segment_data: SegmentDict,
    text: str,
    tokens: list[int],
    temperature: float,
    avg_logprob: float,
    compression_ratio: float,
    word_timestamps: bool,
  ):
    """Add completed segment to results."""
    text = text.strip()
    if segment_data["start"] == segment_data["end"] or not text:
      return

    self.all_tokens.extend(tokens)

    # Assign baseline ID for incomplete segments (will get chain ID when completed)
    from eavesdrop.wire.transcription import compute_segment_chain_id

    segment_id = compute_segment_chain_id(0, "")  # Baseline ID for incomplete segments
    start = segment_data["start"]
    end = segment_data["end"]

    id_logger = get_logger("seg-id")
    id_logger.debug(
      "Segment created (incomplete, will get chain ID when completed)",
      baseline_id=segment_id,
      relative_start=start,
      relative_end=end,
      seek=self.seek,
      text=text[:50] + "..." if len(text) > 50 else text,
    )

    self.all_segments.append(
      Segment(
        id=segment_id,
        seek=self.seek,
        start=start,
        end=end,
        text=text,
        tokens=tokens,
        temperature=temperature,
        avg_logprob=avg_logprob,
        compression_ratio=compression_ratio,
        words=(
          [
            Word(
              start=word["start"],
              end=word["end"],
              word=word["word"],
              probability=word.get("probability", 0.0),
            )
            for word in segment_data.get("words", [])
          ]
          if word_timestamps
          else None
        ),
      )
    )

  def should_reset_prompt(
    self, temperature: float, transcription_options: TranscriptionOptions
  ) -> bool:
    """Check if prompt context should be reset."""
    return (
      not transcription_options.condition_on_previous_text
      or temperature > transcription_options.prompt_reset_on_temperature
    )

  def reset_prompt(self):
    """Reset prompt context to current position."""
    self.prompt_reset_since = len(self.all_tokens)
