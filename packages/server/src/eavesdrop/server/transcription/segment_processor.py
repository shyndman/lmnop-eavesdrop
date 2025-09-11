"""Segment processing and anomaly detection for Whisper transcription.

This module handles the splitting of tokens into segments by timestamps and provides
anomaly detection for identifying potentially hallucinated or problematic segments.
"""

from typing import TypedDict

from faster_whisper.tokenizer import Tokenizer

from eavesdrop.server.transcription.models import SegmentDict


class SegmentTimingResult(TypedDict):
  """Result from segment timestamp processing."""

  segments: list[SegmentDict]
  seek_position: int
  single_timestamp_ending: bool


class SegmentProcessorResult(TypedDict):
  """Result from segment processing operations."""

  segments: list[SegmentDict]
  seek_position: int


class SegmentProcessor:
  """Processes token sequences into properly timed segments with anomaly detection."""

  def __init__(self, time_precision: float, input_stride: int):
    """Initialize the segment processor.

    Args:
        time_precision: Time precision for timestamp calculations.
        input_stride: Input stride for seeking calculations.
    """
    self.time_precision = time_precision
    self.input_stride = input_stride

  def split_segments_by_timestamps(
    self,
    tokenizer: Tokenizer,
    tokens: list[int],
    time_offset: float,
    segment_size: int,
    segment_duration: float,
    seek: int,
  ) -> SegmentTimingResult:
    """Split tokens into segments based on timestamp boundaries.

    Args:
        tokenizer: The Whisper tokenizer for timestamp detection.
        tokens: List of token IDs to process.
        time_offset: Time offset for the current segment.
        segment_size: Size of the audio segment in frames.
        segment_duration: Duration of the segment in seconds.
        seek: Current seek position in frames.

    Returns:
        SegmentTimingResult containing processed segments and updated seek position.
    """
    current_segments: list[SegmentDict] = []
    single_timestamp_ending = (
      len(tokens) >= 2 and tokens[-2] < tokenizer.timestamp_begin <= tokens[-1]
    )

    consecutive_timestamps = [
      i
      for i in range(len(tokens))
      if i > 0
      and tokens[i] >= tokenizer.timestamp_begin
      and tokens[i - 1] >= tokenizer.timestamp_begin
    ]

    if len(consecutive_timestamps) > 0:
      # Process segments with consecutive timestamps
      result = self._process_consecutive_timestamps(
        consecutive_timestamps,
        tokens,
        tokenizer,
        time_offset,
        seek,
        segment_size,
        single_timestamp_ending,
      )
      current_segments = result["segments"]
      seek = result["seek_position"]
    else:
      # Process segment without consecutive timestamps
      result = self._process_single_segment(
        tokens, tokenizer, time_offset, segment_duration, seek, segment_size
      )
      current_segments = result["segments"]
      seek = result["seek_position"]

    return {
      "segments": current_segments,
      "seek_position": seek,
      "single_timestamp_ending": single_timestamp_ending,
    }

  def _process_consecutive_timestamps(
    self,
    consecutive_timestamps: list[int],
    tokens: list[int],
    tokenizer: Tokenizer,
    time_offset: float,
    seek: int,
    segment_size: int,
    single_timestamp_ending: bool,
  ) -> SegmentProcessorResult:
    """Process tokens with consecutive timestamps."""
    current_segments: list[SegmentDict] = []
    slices = list(consecutive_timestamps)

    if single_timestamp_ending:
      slices.append(len(tokens))

    last_slice = 0
    for current_slice in slices:
      sliced_tokens = tokens[last_slice:current_slice]
      start_timestamp_position = sliced_tokens[0] - tokenizer.timestamp_begin
      end_timestamp_position = sliced_tokens[-1] - tokenizer.timestamp_begin
      start_time = time_offset + start_timestamp_position * self.time_precision
      end_time = time_offset + end_timestamp_position * self.time_precision

      current_segments.append(
        {
          "seek": seek,
          "start": start_time,
          "end": end_time,
          "tokens": sliced_tokens,
        }
      )
      last_slice = current_slice

    if single_timestamp_ending:
      # Single timestamp at the end means no speech after the last timestamp
      seek += segment_size
    else:
      # Seek to the last timestamp position
      last_timestamp_position = tokens[last_slice - 1] - tokenizer.timestamp_begin
      seek += last_timestamp_position * self.input_stride

    return SegmentProcessorResult(segments=current_segments, seek_position=seek)

  def _process_single_segment(
    self,
    tokens: list[int],
    tokenizer: Tokenizer,
    time_offset: float,
    segment_duration: float,
    seek: int,
    segment_size: int,
  ) -> SegmentProcessorResult:
    """Process tokens as a single segment without consecutive timestamps."""
    duration = segment_duration
    timestamps = [token for token in tokens if token >= tokenizer.timestamp_begin]

    if len(timestamps) > 0 and timestamps[-1] != tokenizer.timestamp_begin:
      last_timestamp_position = timestamps[-1] - tokenizer.timestamp_begin
      duration = last_timestamp_position * self.time_precision

    current_segments: list[SegmentDict] = [
      {
        "seek": seek,
        "start": time_offset,
        "end": time_offset + duration,
        "tokens": tokens,
      }
    ]

    seek += segment_size
    return SegmentProcessorResult(segments=current_segments, seek_position=seek)
