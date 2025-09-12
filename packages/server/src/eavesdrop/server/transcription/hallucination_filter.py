"""Hallucination detection and filtering for Whisper transcription.

This module provides sophisticated filtering logic to detect and remove likely
hallucinated segments based on silence gap analysis and anomaly detection.
"""

from eavesdrop.server.transcription.language_detection import AnomalyDetector
from eavesdrop.server.transcription.models import SegmentDict


class HallucinationFilter:
  """Filters out likely hallucinated segments using silence gap analysis."""

  def __init__(self, anomaly_detector: AnomalyDetector):
    """Initialize the hallucination filter.

    :param anomaly_detector: Detector for identifying anomalous segments.
    :type anomaly_detector: AnomalyDetector
    """
    self.anomaly_detector = anomaly_detector

  def filter_segments(
    self,
    segments: list[SegmentDict],
    threshold: float,
    time_offset: float,
    segment_duration: float,
    window_end_time: float,
    last_speech_timestamp: float,
    total_duration: float,
    total_frames: int,
    previous_seek: int,
    frames_per_second: float,
  ) -> tuple[list[SegmentDict], int | None]:
    """Filter segments to remove likely hallucinations.

    Uses silence gap analysis to identify and remove segments that are likely
    hallucinations. Returns filtered segments and new seek position if modified.

    :param segments: Segments to filter for hallucinations.
    :type segments: list[SegmentDict]
    :param threshold: Silence threshold for hallucination detection.
    :type threshold: float
    :param time_offset: Current time offset in seconds.
    :type time_offset: float
    :param segment_duration: Duration of current segment in seconds.
    :type segment_duration: float
    :param window_end_time: End time of current processing window.
    :type window_end_time: float
    :param last_speech_timestamp: Timestamp of last detected speech.
    :type last_speech_timestamp: float
    :param total_duration: Total duration of audio in seconds.
    :type total_duration: float
    :param total_frames: Total number of frames in audio.
    :type total_frames: int
    :param previous_seek: Previous seek position before current segment.
    :type previous_seek: int
    :param frames_per_second: Audio frames per second for seek calculations.
    :type frames_per_second: float
    :returns: Tuple of (filtered_segments, new_seek_position_or_None).
    :rtype: tuple[list[SegmentDict], int | None]
    """
    # Check for leading silence skip first
    leading_seek = self._skip_leading_hallucination(
      segments, threshold, time_offset, previous_seek, frames_per_second
    )
    if leading_seek is not None:
      return segments, leading_seek

    # Filter surrounded hallucinations
    filtered_segments, surrounded_seek = self._filter_surrounded_hallucinations(
      segments,
      threshold,
      time_offset,
      segment_duration,
      window_end_time,
      last_speech_timestamp,
      total_duration,
      total_frames,
      frames_per_second,
    )

    return filtered_segments, surrounded_seek

  def _skip_leading_hallucination(
    self,
    segments: list[SegmentDict],
    threshold: float,
    time_offset: float,
    previous_seek: int,
    frames_per_second: float,
  ) -> int | None:
    """Skip silence before first hallucinated segment.

    If the very first segment is anomalous and has a large gap before it,
    return a new seek position to skip past the silence.

    :param segments: Current segments to check.
    :type segments: list[SegmentDict]
    :param threshold: Silence threshold.
    :type threshold: float
    :param time_offset: Current time offset in seconds.
    :type time_offset: float
    :param previous_seek: Previous seek position.
    :type previous_seek: int
    :param frames_per_second: Frames per second.
    :type frames_per_second: float
    :returns: New seek position if modified, None otherwise.
    :rtype: int | None
    """
    first_segment = self.anomaly_detector.next_words_segment(segments)
    if first_segment is not None and self.anomaly_detector.is_segment_anomaly(first_segment):
      gap = first_segment["start"] - time_offset
      if gap > threshold:
        return previous_seek + round(gap * frames_per_second)
    return None

  def _filter_surrounded_hallucinations(
    self,
    segments: list[SegmentDict],
    threshold: float,
    time_offset: float,
    segment_duration: float,
    window_end_time: float,
    last_speech_timestamp: float,
    total_duration: float,
    total_frames: int,
    frames_per_second: float,
  ) -> tuple[list[SegmentDict], int | None]:
    """Filter out anomalous segments that are isolated by silence.

    Identifies hallucinated segments that are surrounded by silence or other
    hallucinations and removes them from the segment list.

    :param segments: Segments to filter.
    :type segments: list[SegmentDict]
    :param threshold: Silence threshold.
    :type threshold: float
    :param time_offset: Current time offset in seconds.
    :type time_offset: float
    :param segment_duration: Duration of current segment in seconds.
    :type segment_duration: float
    :param window_end_time: End time of current processing window.
    :type window_end_time: float
    :param last_speech_timestamp: Timestamp of last detected speech.
    :type last_speech_timestamp: float
    :param total_duration: Total duration of audio in seconds.
    :type total_duration: float
    :param total_frames: Total number of frames in audio.
    :type total_frames: int
    :param frames_per_second: Frames per second.
    :type frames_per_second: float
    :returns: Tuple of (filtered_segments, new_seek_position_or_None).
    :rtype: tuple[list[SegmentDict], int | None]
    """
    hal_last_end = last_speech_timestamp
    filtered_segments = segments.copy()

    for si in range(len(filtered_segments)):
      segment = filtered_segments[si]

      # Skip segments without words (no content to analyze)
      if not segment.get("words"):
        continue

      # Check if current segment is anomalous (potential hallucination)
      if self.anomaly_detector.is_segment_anomaly(segment):
        # Determine timing of next real segment for silence calculation
        next_segment = self.anomaly_detector.next_words_segment(filtered_segments[si + 1 :])
        hal_next_start = self._get_next_segment_start(next_segment, time_offset, segment_duration)

        # Check for silence before and after the anomalous segment
        silence_before = self._has_silence_before(segment, hal_last_end, threshold, time_offset)
        silence_after = self._has_silence_after(
          segment, hal_next_start, threshold, next_segment, window_end_time
        )

        # Remove hallucination if surrounded by silence
        if silence_before and silence_after:
          seek = self._calculate_hallucination_seek(
            segment, time_offset, total_duration, total_frames, threshold, frames_per_second
          )
          # Remove this and all subsequent segments in current batch
          return filtered_segments[:si], seek

      # Track end time of non-anomalous segments for next iteration
      hal_last_end = segment["end"]

    return filtered_segments, None

  def _get_next_segment_start(
    self, next_segment: SegmentDict | None, time_offset: float, segment_duration: float
  ) -> float:
    """Get the start time of the next real segment.

    :param next_segment: Next segment with words, or None.
    :type next_segment: SegmentDict | None
    :param time_offset: Current window time offset.
    :type time_offset: float
    :param segment_duration: Current segment duration.
    :type segment_duration: float
    :returns: Start time of next segment or estimated end time.
    :rtype: float
    """
    if next_segment is not None:
      next_words = next_segment.get("words", [])
      if next_words:
        return next_words[0]["start"]
    return time_offset + segment_duration

  def _has_silence_before(
    self, segment: SegmentDict, hal_last_end: float, threshold: float, time_offset: float
  ) -> bool:
    """Check if there's silence before the anomalous segment.

    :param segment: Segment to check.
    :type segment: SegmentDict
    :param hal_last_end: End time of last legitimate speech.
    :type hal_last_end: float
    :param threshold: Silence threshold.
    :type threshold: float
    :param time_offset: Current window time offset.
    :type time_offset: float
    :returns: True if silence detected before segment.
    :rtype: bool
    """
    return (
      segment["start"] - hal_last_end > threshold  # Gap from last speech
      or segment["start"] < threshold  # Very early in audio
      or segment["start"] - time_offset < 2.0  # Too close to window start
    )

  def _has_silence_after(
    self,
    segment: SegmentDict,
    hal_next_start: float,
    threshold: float,
    next_segment: SegmentDict | None,
    window_end_time: float,
  ) -> bool:
    """Check if there's silence after the anomalous segment.

    :param segment: Segment to check.
    :type segment: SegmentDict
    :param hal_next_start: Start time of next segment.
    :type hal_next_start: float
    :param threshold: Silence threshold.
    :type threshold: float
    :param next_segment: Next segment, or None.
    :type next_segment: SegmentDict | None
    :param window_end_time: End time of current processing window.
    :type window_end_time: float
    :returns: True if silence detected after segment.
    :rtype: bool
    """
    return (
      hal_next_start - segment["end"] > threshold  # Gap to next speech
      or self.anomaly_detector.is_segment_anomaly(next_segment)  # Next is also anomaly
      or window_end_time - segment["end"] < 2.0  # Too close to window end
    )

  def _calculate_hallucination_seek(
    self,
    segment: SegmentDict,
    time_offset: float,
    total_duration: float,
    total_frames: int,
    threshold: float,
    frames_per_second: float,
  ) -> int:
    """Calculate the seek position to skip past a hallucination.

    :param segment: Hallucinated segment to skip.
    :type segment: SegmentDict
    :param time_offset: Current time offset in seconds.
    :type time_offset: float
    :param total_duration: Total duration of audio in seconds.
    :type total_duration: float
    :param total_frames: Total number of frames in audio.
    :type total_frames: int
    :param threshold: Silence threshold.
    :type threshold: float
    :param frames_per_second: Frames per second.
    :type frames_per_second: float
    :returns: New seek position in frames.
    :rtype: int
    """
    seek = round(max(time_offset + 1, segment["start"]) * frames_per_second)

    # Handle end-of-audio case
    if total_duration - segment["end"] < threshold:
      seek = total_frames

    return seek
