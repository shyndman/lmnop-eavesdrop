"""Audio preprocessing, VAD filtering, and segmentation module.

This module handles voice activity detection, audio validation, preprocessing,
and feature extraction coordination for the transcription pipeline.
"""

import time
from typing import NamedTuple, cast

import numpy as np
from numpy.typing import NDArray
from structlog.stdlib import BoundLogger

from eavesdrop.common import get_logger
from eavesdrop.server.transcription.models import SpeechChunk
from eavesdrop.server.transcription.vendor_types import (
  FeatureExtractorLike,
  VadOptionsLike,
  load_collect_chunks,
  load_format_timestamp,
  load_get_speech_timestamps,
  load_vad_options,
)

# Private module constants
_VAD_LOG_THROTTLE_SECONDS = 60.0  # Throttle VAD complete silence logging to 1 minute
_MINIMUM_AUDIO_DURATION = 0.0  # Minimum audio duration to process

FloatAudio = NDArray[np.float32] | NDArray[np.float64]
WhisperFeatures = NDArray[np.float32]
VadSpeechChunk = dict[str, int]
format_timestamp = load_format_timestamp()
collect_chunks = load_collect_chunks()
get_speech_timestamps = load_get_speech_timestamps()
VadOptions = load_vad_options()


class AudioValidationResult(NamedTuple):
  """Result of audio validation and preprocessing."""

  audio: FloatAudio
  duration: float
  duration_after_vad: float
  speech_chunks: list[SpeechChunk] | None
  is_complete_silence: bool


class AudioProcessor:
  """Handles audio preprocessing, VAD filtering, and segmentation for transcription."""

  def __init__(self, feature_extractor: FeatureExtractorLike):
    """Initialize the audio processor.

    :param feature_extractor: The feature extractor instance for audio processing.
    :type feature_extractor: FeatureExtractor
    """
    self._feature_extractor: FeatureExtractorLike = feature_extractor
    self._logger: BoundLogger = get_logger("snd/proc")
    self._last_vad_log_time: float = 0.0

  @property
  def sampling_rate(self) -> int:
    """Get the sampling rate from the feature extractor."""
    return self._feature_extractor.sampling_rate

  def validate_and_preprocess_audio(
    self,
    audio: FloatAudio,
    vad_filter: bool = False,
    vad_parameters: VadOptionsLike | None = None,
  ) -> AudioValidationResult:
    """Validate and preprocess audio data for transcription.

    :param audio: Audio waveform as numpy array (16kHz sample rate).
    :type audio: np.ndarray
    :param vad_filter: Enable voice activity detection to filter silent parts.
    :type vad_filter: bool
    :param vad_parameters: VAD configuration options.
    :type vad_parameters: VadOptions
    :returns: AudioValidationResult with processed audio and metadata.
    :rtype: AudioValidationResult
    """
    if audio.ndim != 1:
      raise ValueError(f"Audio must be 1D array, got {audio.ndim}D")

    sampling_rate = self.sampling_rate
    original_duration = audio.shape[0] / sampling_rate
    processed_audio = audio
    duration_after_vad = original_duration
    speech_chunks: list[SpeechChunk] | None = None
    is_complete_silence = False

    # Apply VAD filtering if requested
    if vad_filter:
      resolved_vad_parameters = vad_parameters or VadOptions()
      raw_speech_chunks = get_speech_timestamps(audio, resolved_vad_parameters)
      speech_chunks = cast(list[SpeechChunk], raw_speech_chunks)
      audio_chunks, _ = collect_chunks(audio, raw_speech_chunks)
      processed_audio = cast(FloatAudio, np.concatenate(audio_chunks, axis=0))
      duration_after_vad = processed_audio.shape[0] / sampling_rate
      is_complete_silence = duration_after_vad <= _MINIMUM_AUDIO_DURATION

      # Log VAD filtering results (throttled for complete silence)
      self._log_vad_processing(original_duration, duration_after_vad, is_complete_silence)
    else:
      # No VAD filtering applied
      processed_audio = audio

    return AudioValidationResult(
      audio=processed_audio,
      duration=original_duration,
      duration_after_vad=duration_after_vad,
      speech_chunks=speech_chunks,
      is_complete_silence=is_complete_silence,
    )

  def extract_features(self, audio: FloatAudio) -> WhisperFeatures:
    """Extract features from audio using the configured feature extractor.

    :param audio: Preprocessed audio array.
    :type audio: np.ndarray
    :returns: Feature array ready for transcription.
    :rtype: np.ndarray
    """
    if audio.shape[0] == 0:
      # DEBUG: Empty audio passed to feature extraction
      self._logger.debug("Feature extraction skipped - audio is empty", audio_shape=audio.shape)
      # Return empty features for empty audio
      # Use n_mels as the feature dimension
      return np.empty((80, 0), dtype=np.float32)  # Default Whisper n_mels is 80

    return self._feature_extractor(audio)

  def prepare_segments_for_detection(self, audio: FloatAudio, max_segments: int = 1) -> FloatAudio:
    """Prepare audio segments for language detection.

    :param audio: Input audio array.
    :type audio: np.ndarray
    :param max_segments: Maximum number of segments to prepare.
    :type max_segments: int
    :returns: Truncated audio for language detection.
    :rtype: np.ndarray
    """
    max_samples = max_segments * self._feature_extractor.n_samples
    return audio[:max_samples]

  def _log_vad_processing(
    self,
    original_duration: float,
    duration_after_vad: float,
    is_complete_silence: bool,
  ) -> None:
    """Log VAD processing results with throttling for complete silence.

    :param original_duration: Original audio duration in seconds.
    :type original_duration: float
    :param duration_after_vad: Duration after VAD filtering in seconds.
    :type duration_after_vad: float
    :param is_complete_silence: Whether the audio is completely silent after VAD.
    :type is_complete_silence: bool
    """
    current_time = time.time()

    # For complete silence, throttle logging to avoid spam
    if is_complete_silence:
      if current_time - self._last_vad_log_time >= _VAD_LOG_THROTTLE_SECONDS:
        self._logger.info(
          "Processing audio with duration %s (complete silence detected)",
          format_timestamp(original_duration),
        )
        self._last_vad_log_time = current_time
    else:
      # Always log when there's actual speech content
      audio_removed = original_duration - duration_after_vad
      self._logger.info(
        "VAD processing",
        duration=format_timestamp(original_duration),
        duration_post_vad=format_timestamp(duration_after_vad),
        duration_removed=format_timestamp(audio_removed),
      )


def validate_audio_input(audio: object) -> None:
  """Validate audio input meets basic requirements.

  :param audio: Audio array to validate.
  :type audio: np.ndarray
  :raises ValueError: If audio doesn't meet requirements.
  """
  if not isinstance(audio, np.ndarray):
    raise ValueError(f"Audio must be numpy array, got {type(audio)}")

  if audio.ndim != 1:
    raise ValueError(f"Audio must be 1D array, got {audio.ndim}D")

  if audio.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
    raise ValueError(f"Audio must be float32 or float64, got {audio.dtype}")
