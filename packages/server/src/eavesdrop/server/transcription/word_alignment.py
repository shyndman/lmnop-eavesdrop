"""Word-timestamp alignment and timing calculation for Whisper transcription."""

import itertools

import ctranslate2
import numpy as np
from faster_whisper.tokenizer import Tokenizer

from eavesdrop.server.transcription.models import SegmentDict, WordDict, WordTimingDict
from eavesdrop.server.transcription.utils import merge_punctuations

# Private module constants for timing thresholds and calculations
_DEFAULT_MEDIAN_FILTER_WIDTH = 7
_MAX_MEDIAN_DURATION = 0.7
_DURATION_MULTIPLIER = 2
_SENTENCE_END_MARKS = ".。!！?？"
_PAUSE_THRESHOLD_MULTIPLIER = 4
_SEGMENT_BOUNDARY_TOLERANCE = 0.5
_TIMESTAMP_PRECISION = 2


class WordAlignmentProcessor:
  """Handles word-to-timestamp alignment using Whisper model alignment capabilities.

  This class determines word timing by implementing forced alignment between text tokens and audio
  frames using Whisper's cross-attention mechanism. The process involves:

  1. MODEL ALIGNMENT: Use Whisper's align() method to get token-to-frame alignments
      using cross-attention weights between encoder (audio) and decoder (text)

  2. TOKEN SPLITTING: Split each text token into individual words and their sub-tokens
      to handle cases where one token represents multiple words or word fragments

  3. BOUNDARY DETECTION: Calculate word boundaries by finding cumulative token lengths,
      creating indices that map from individual words back to their token positions

  4. TIMING CALCULATION:
      - Detect "jumps" in alignment indices to find significant attention shifts
      - Convert frame indices to timestamps using tokens_per_second conversion
      - Map word boundaries to jump times to assign start/end times per word

  5. PROBABILITY ASSIGNMENT: Average token probabilities across word boundaries
      to estimate confidence for each individual word
  """

  def __init__(
    self,
    frames_per_second: float,
    tokens_per_second: float,
    median_filter_width: int = _DEFAULT_MEDIAN_FILTER_WIDTH,
  ):
    """Initialize the word alignment processor.

    :param frames_per_second: Number of frames per second in the audio
    :type frames_per_second: float
    :param tokens_per_second: Number of tokens per second for time conversion
    :type tokens_per_second: float
    :param median_filter_width: Width for median filtering in alignment
    :type median_filter_width: int
    """
    self.frames_per_second = frames_per_second
    self.tokens_per_second = tokens_per_second
    self.median_filter_width = median_filter_width

  def find_alignment(
    self,
    model: ctranslate2.models.Whisper,
    tokenizer: Tokenizer,
    text_tokens: list[int],
    encoder_output: ctranslate2.StorageView,
    num_frames: int,
  ) -> list[list[WordTimingDict]]:
    """Find word alignments using the Whisper model's alignment capabilities.

    :param model: The Whisper model instance used for forced alignment
    :type model: ctranslate2.models.Whisper
    :param tokenizer: Tokenizer for processing tokens and splitting into words
    :type tokenizer: Tokenizer
    :param text_tokens: List of integer token IDs to be aligned with audio frames.
        Each token typically represents a word or sub-word unit from the transcription.
    :type text_tokens: list[int]
    :param encoder_output: Pre-computed encoder features from the Whisper model.
        This is the mel-spectrogram audio processed by the encoder into a feature
        representation with shape [batch_size, sequence_length, d_model]. Contains
        the audio context needed for cross-attention alignment.
    :type encoder_output: ctranslate2.StorageView
    :param num_frames: Total number of audio frames in the original audio segment.
        Used by the alignment algorithm to properly scale timestamp indices back
        to the original audio timeline. Critical for accurate timing calculations.
    :type num_frames: int
    :returns: List of word timing dictionaries for each token. The return structure is
        two-dimensional: the outer list corresponds to each input text token, and
        the inner list contains the individual words extracted from that token
        with their timing information.
    :rtype: list[list[WordTimingDict]]
    """
    if len(text_tokens) == 0:
      return []

    # Call Whisper's forced alignment: cross-attention between encoder (audio) and decoder (text)
    # Returns alignment results containing text_token_probs and alignments (text_idx, time_idx)
    # pairs
    results = model.align(
      encoder_output,
      tokenizer.sot_sequence,
      text_tokens,
      num_frames,
      median_filter_width=self.median_filter_width,
    )

    return_list: list[list[WordTimingDict]] = []
    for result, text_token in zip(results, text_tokens):
      text_token_probs = result.text_token_probs
      alignments = result.alignments
      text_indices = np.array([pair[0] for pair in alignments])
      time_indices = np.array([pair[1] for pair in alignments])

      words, word_tokens = tokenizer.split_to_word_tokens([text_token] + [tokenizer.eot])
      if len(word_tokens) <= 1:
        # Return empty list for tokens with no words (e.g., end-of-text only)
        # This prevents crashes when looking up jump_times with float indices
        return_list.append([])
        continue

      word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
      if len(word_boundaries) <= 1:
        return_list.append([])
        continue

      jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
      jump_times = time_indices[jumps] / self.tokens_per_second
      start_times = jump_times[word_boundaries[:-1]]
      end_times = jump_times[word_boundaries[1:]]
      word_probabilities = [
        np.mean(text_token_probs[i:j]) for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
      ]

      return_list.append(
        [
          {
            "word": word,
            "tokens": tokens,
            "start": start,
            "end": end,
            "probability": probability,
          }
          for word, tokens, start, end, probability in zip(
            words, word_tokens, start_times, end_times, word_probabilities
          )
        ]
      )

    return return_list


class TimingProcessor:
  """Handles timing calculations and corrections for word alignments."""

  def __init__(self, frames_per_second: float):
    """Initialize the timing processor.

    :param frames_per_second: Number of frames per second in the audio
    :type frames_per_second: float
    """
    self.frames_per_second = frames_per_second

  def calculate_duration_statistics(self, alignment: list[WordTimingDict]) -> tuple[float, float]:
    """Calculate median and maximum duration thresholds for word timing.

    :param alignment: List of word timing dictionaries
    :type alignment: list[WordTimingDict]
    :returns: Tuple of (median_duration, max_duration) thresholds
    :rtype: tuple[float, float]
    """
    word_durations = np.array([word["end"] - word["start"] for word in alignment])
    word_durations = word_durations[word_durations.nonzero()]

    median_duration = np.median(word_durations) if len(word_durations) > 0 else 0.0
    median_duration = min(_MAX_MEDIAN_DURATION, float(median_duration))
    max_duration = median_duration * _DURATION_MULTIPLIER

    return median_duration, max_duration

  def apply_sentence_boundary_corrections(
    self, alignment: list[WordTimingDict], max_duration: float
  ) -> None:
    """Apply corrections to word timings at sentence boundaries.

    This is a heuristic to truncate long words at sentence boundaries,
    as a better segmentation algorithm based on VAD should eventually replace this.

    :param alignment: List of word timing dictionaries to modify in-place
    :type alignment: list[WordTimingDict]
    :param max_duration: Maximum allowed duration for words at boundaries
    :type max_duration: float
    """
    if len(alignment) == 0:
      return

    # Ensure words at sentence boundaries are not longer than twice the median word duration
    for i in range(1, len(alignment)):
      if alignment[i]["end"] - alignment[i]["start"] > max_duration:
        if alignment[i]["word"] in _SENTENCE_END_MARKS:
          alignment[i]["end"] = alignment[i]["start"] + max_duration
        elif alignment[i - 1]["word"] in _SENTENCE_END_MARKS:
          alignment[i]["start"] = alignment[i]["end"] - max_duration

  def apply_segment_boundary_corrections(
    self,
    words: list[WordDict],
    subsegment: SegmentDict,
    last_speech_timestamp: float,
    median_duration: float,
    max_duration: float,
  ) -> None:
    """Apply timing corrections at segment boundaries.

    This handles edge cases where words are too long after pauses or at segment boundaries.

    :param words: List of word dictionaries to modify in-place
    :type words: list[WordDict]
    :param subsegment: Segment dictionary to potentially modify
    :type subsegment: SegmentDict
    :param last_speech_timestamp: Timestamp of the last speech segment
    :type last_speech_timestamp: float
    :param median_duration: Median duration for timing corrections
    :type median_duration: float
    :param max_duration: Maximum allowed duration for words
    :type max_duration: float
    """
    if len(words) == 0:
      return

    # Handle first and second word after a pause
    if words[0]["end"] - last_speech_timestamp > median_duration * _PAUSE_THRESHOLD_MULTIPLIER and (
      words[0]["end"] - words[0]["start"] > max_duration
      or (
        len(words) > 1 and words[1]["end"] - words[0]["start"] > max_duration * _DURATION_MULTIPLIER
      )
    ):
      if len(words) > 1 and words[1]["end"] - words[1]["start"] > max_duration:
        boundary = max(words[1]["end"] / 2, words[1]["end"] - max_duration)
        words[0]["end"] = words[1]["start"] = boundary
      words[0]["start"] = max(0, words[0]["end"] - max_duration)

    # Prefer segment-level start timestamp if the first word is too long
    if (
      subsegment["start"] < words[0]["end"]
      and subsegment["start"] - _SEGMENT_BOUNDARY_TOLERANCE > words[0]["start"]
    ):
      words[0]["start"] = max(
        0,
        min(words[0]["end"] - median_duration, subsegment["start"]),
      )
    else:
      subsegment["start"] = words[0]["start"]

    # Prefer segment-level end timestamp if the last word is too long
    if (
      subsegment["end"] > words[-1]["start"]
      and subsegment["end"] + _SEGMENT_BOUNDARY_TOLERANCE < words[-1]["end"]
    ):
      words[-1]["end"] = max(words[-1]["start"] + median_duration, subsegment["end"])
    else:
      subsegment["end"] = words[-1]["end"]


class WordTimestampAligner:
  """Main class for adding word timestamps to transcription segments."""

  def __init__(self, frames_per_second: float, tokens_per_second: float):
    """Initialize the word timestamp aligner.

    :param frames_per_second: Number of frames per second in the audio
    :type frames_per_second: float
    :param tokens_per_second: Number of tokens per second for time conversion
    :type tokens_per_second: float
    """
    self.alignment_processor = WordAlignmentProcessor(frames_per_second, tokens_per_second)
    self.timing_processor = TimingProcessor(frames_per_second)

  def add_word_timestamps(
    self,
    segments: list[list[SegmentDict]],
    model: ctranslate2.models.Whisper,
    tokenizer: Tokenizer,
    encoder_output: ctranslate2.StorageView,
    num_frames: int,
    prepend_punctuations: str,
    append_punctuations: str,
    last_speech_timestamp: float,
  ) -> float:
    """Add word-level timestamps to transcription segments.

    :param segments: List of segment groups to add timestamps to
    :type segments: list[list[SegmentDict]]
    :param model: The Whisper model instance
    :type model: ctranslate2.models.Whisper
    :param tokenizer: Tokenizer for processing tokens
    :type tokenizer: Tokenizer
    :param encoder_output: Encoder output from the model
    :type encoder_output: ctranslate2.StorageView
    :param num_frames: Number of audio frames
    :type num_frames: int
    :param prepend_punctuations: Characters to merge with following words
    :type prepend_punctuations: str
    :param append_punctuations: Characters to merge with preceding words
    :type append_punctuations: str
    :param last_speech_timestamp: Timestamp of the last speech segment
    :type last_speech_timestamp: float
    :returns: Updated last speech timestamp
    :rtype: float
    """
    if len(segments) == 0:
      return 0.0

    # Extract text tokens from segments
    text_tokens_per_segment = self._extract_text_tokens(segments, tokenizer)

    # Flatten all tokens for alignment
    text_tokens = list(
      itertools.chain.from_iterable(itertools.chain.from_iterable(text_tokens_per_segment))
    )

    # Find alignments for all tokens
    alignments = self.alignment_processor.find_alignment(
      model, tokenizer, text_tokens, encoder_output, num_frames
    )

    # Process timing statistics and apply corrections
    median_max_durations = self._process_alignments(
      alignments, prepend_punctuations, append_punctuations
    )

    # Apply word timestamps to segments
    return self._apply_timestamps_to_segments(
      segments,
      alignments,
      text_tokens_per_segment,
      median_max_durations,
      last_speech_timestamp,
    )

  def _extract_text_tokens(
    self, segments: list[list[SegmentDict]], tokenizer: Tokenizer
  ) -> list[list[list[int]]]:
    """Extract text tokens from segments, filtering out EOT tokens."""
    text_tokens_per_segment: list[list[list[int]]] = []
    for segment_group in segments:
      segment_tokens = [
        [token for token in segment["tokens"] if token < tokenizer.eot] for segment in segment_group
      ]
      text_tokens_per_segment.append(segment_tokens)
    return text_tokens_per_segment

  def _process_alignments(
    self,
    alignments: list[list[WordTimingDict]],
    prepend_punctuations: str,
    append_punctuations: str,
  ) -> list[tuple[float, float]]:
    """Process alignments to calculate duration statistics and apply corrections."""
    median_max_durations: list[tuple[float, float]] = []

    for alignment in alignments:
      median_duration, max_duration = self.timing_processor.calculate_duration_statistics(alignment)

      # Apply sentence boundary corrections if we have word durations
      if len(alignment) > 0:
        self.timing_processor.apply_sentence_boundary_corrections(alignment, max_duration)

      # Merge punctuations with adjacent words
      merge_punctuations(alignment, prepend_punctuations, append_punctuations)
      median_max_durations.append((median_duration, max_duration))

    return median_max_durations

  def _apply_timestamps_to_segments(
    self,
    segments: list[list[SegmentDict]],
    alignments: list[list[WordTimingDict]],
    text_tokens_per_segment: list[list[list[int]]],
    median_max_durations: list[tuple[float, float]],
    last_speech_timestamp: float,
  ) -> float:
    """Apply calculated timestamps to the segment words."""
    for segment_idx, segment in enumerate(segments):
      word_index = 0
      time_offset = segment[0]["seek"] / self.alignment_processor.frames_per_second
      median_duration, max_duration = median_max_durations[segment_idx]

      for subsegment_idx, subsegment in enumerate(segment):
        saved_tokens = 0
        words: list[WordDict] = []

        # Extract words for this subsegment
        while word_index < len(alignments[segment_idx]) and saved_tokens < len(
          text_tokens_per_segment[segment_idx][subsegment_idx]
        ):
          timing = alignments[segment_idx][word_index]

          if timing["word"]:
            words.append(
              {
                "word": timing["word"],
                "start": round(time_offset + timing["start"], _TIMESTAMP_PRECISION),
                "end": round(time_offset + timing["end"], _TIMESTAMP_PRECISION),
                "probability": timing["probability"],
              }
            )

          saved_tokens += len(timing["tokens"])
          word_index += 1

        # Apply segment boundary corrections
        if len(words) > 0:
          self.timing_processor.apply_segment_boundary_corrections(
            words, subsegment, last_speech_timestamp, median_duration, max_duration
          )
          last_speech_timestamp = subsegment["end"]

        # Assign words to subsegment
        segments[segment_idx][subsegment_idx]["words"] = words

    return last_speech_timestamp
