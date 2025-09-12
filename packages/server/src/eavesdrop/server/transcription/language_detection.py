"""Language detection and anomaly detection for Whisper transcription.

This module provides focused functionality for:
- Language detection using Whisper model capabilities
- Segment anomaly detection (identifying problematic transcription segments)
- Word anomaly scoring
- Language probability analysis
"""

from typing import NamedTuple, TypedDict

import ctranslate2
import numpy as np
from faster_whisper.audio import pad_or_trim
from faster_whisper.feature_extractor import FeatureExtractor
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.vad import (
  VadOptions,
  collect_chunks,
  get_speech_timestamps,
)

from eavesdrop.server.logs import get_logger
from eavesdrop.server.transcription.models import LanguageProbability, SegmentDict, WordDict

# Private module constants for anomaly detection thresholds
_WORD_PROBABILITY_THRESHOLD = 0.15
_SHORT_WORD_DURATION_THRESHOLD = 0.133
_SHORT_WORD_DURATION_MULTIPLIER = 15.0
_LONG_WORD_DURATION_THRESHOLD = 2.0
_SEGMENT_ANOMALY_SCORE_THRESHOLD = 3.0
_SEGMENT_ANOMALY_SCORE_OFFSET = 0.01
_MAX_WORDS_FOR_ANOMALY_CHECK = 8
_DEFAULT_PUNCTUATION = '"\'"¿([{-"\'.。,，!！?？:：")]}、'


class LanguageAnalysisResult(TypedDict):
  """Result of comprehensive language probability analysis."""

  top_language: tuple[str, float]
  confidence_level: str
  entropy: float
  top_3_languages: list[tuple[str, float]]
  total_languages: int


class LanguageDetectionResult(NamedTuple):
  """Result from language detection with probability and alternatives."""

  language: str
  probability: float
  all_probabilities: list[LanguageProbability]


class SegmentAnomalyResult(TypedDict):
  """Result from segment anomaly analysis."""

  segment_index: int
  is_anomaly: bool
  anomaly_score: float


class LanguageDetector:
  """Handles language detection using Whisper model capabilities."""

  def __init__(
    self,
    model: ctranslate2.models.Whisper,
    feature_extractor: FeatureExtractor,
  ):
    """Initialize the language detector.

    :param model: The CTranslate2 Whisper model instance
    :type model: ctranslate2.models.Whisper
    :param feature_extractor: Feature extractor for processing audio
    :type feature_extractor: FeatureExtractor
    """
    self.model = model
    self.feature_extractor = feature_extractor
    self.logger = get_logger("lang")

  def resolve_language(
    self, language: str | None, features: np.ndarray
  ) -> tuple[LanguageProbability, list[LanguageProbability]]:
    """Resolve the language for transcription based on input and model capabilities.

    :param language: Language code specified by caller (e.g., "en", "fr") or None for
      auto-detection.
    :type language: str | None
    :param features: Audio feature array for language detection if needed.
    :type features: np.ndarray
    :returns: Tuple containing the determined language with probability and all language
      probabilities.
    :rtype: tuple[LanguageProbability, list[LanguageProbability]]
    """
    # 1. Language specified by caller: use it if it's valid
    if language is not None:
      invalid_lang = not self.model.is_multilingual and language != "en"
      if invalid_lang:
        self.logger.warning(
          "Model is English-only but the language arg is not. Default is 'en'", lang=language
        )
        language = "en"
      return ((language, 1.0), [(language, 1.0)])

    # 2. No language specified for a unilingual model: automatically English, because all
    #    single-language models are English
    if not self.model.is_multilingual:
      language = "en"
      return ((language, 1.0), [(language, 1.0)])

    # 3. No language specified, multilingual model: detect the language

    # Determine the samples in question
    start_timestamp = 0.0
    content_frames = features.shape[-1] - 1
    frames_per_second = self.feature_extractor.sampling_rate / self.feature_extractor.hop_length
    seek = (
      int(start_timestamp * frames_per_second)
      if start_timestamp * frames_per_second < content_frames
      else 0
    )

    # Detect the language
    detection_result = self.detect_language(
      features=features[..., seek:],
      language_detection_segments=1,
      language_detection_threshold=0.5,
    )

    self.logger.info(
      "Detected language '%s' with probability %.2f",
      detection_result.language,
      detection_result.probability,
    )

    return (
      (detection_result.language, detection_result.probability),
      detection_result.all_probabilities,
    )

  def detect_language(
    self,
    audio: np.ndarray | None = None,
    features: np.ndarray | None = None,
    vad_filter: bool = False,
    vad_parameters: VadOptions = VadOptions(),
    language_detection_segments: int = 1,
    language_detection_threshold: float = 0.5,
  ) -> LanguageDetectionResult:
    """Use Whisper to detect the language of the input audio or features.

    :param audio: Input audio signal, must be a 1D float array sampled at 16khz.
    :type audio: np.ndarray | None
    :param features: Input Mel spectrogram features, must be a float array with
      shape (n_mels, n_frames), if `audio` is provided, the features will be ignored.
      Either `audio` or `features` must be provided.
    :type features: np.ndarray | None
    :param vad_filter: Enable the voice activity detection (VAD) to filter out parts of the audio
      without speech. This step is using the Silero VAD model.
    :type vad_filter: bool
    :param vad_parameters: VadOptions class instance (see available
      parameters and default values in the class `VadOptions`).
    :type vad_parameters: VadOptions
    :param language_detection_threshold: If the maximum probability of the language tokens is
      higher than this value, the language is detected.
    :type language_detection_threshold: float
    :param language_detection_segments: Number of segments to consider for the language detection.
    :type language_detection_segments: int
    :returns: LanguageDetectionResult containing detected language, probability, and all
      probabilities.
    :rtype: LanguageDetectionResult
    """
    self.logger.debug(
      f"Detecting language: vad_filter={vad_filter}, segments={language_detection_segments}, "
      f"threshold={language_detection_threshold}"
    )

    if not (audio is not None or features is not None):
      raise ValueError("Either `audio` or `features` must be provided.")

    if audio is not None:
      if vad_filter:
        speech_chunks = get_speech_timestamps(audio, vad_parameters)
        audio_chunks, _ = collect_chunks(audio, speech_chunks)
        audio = np.concatenate(audio_chunks, axis=0)

      audio = audio[: language_detection_segments * self.feature_extractor.n_samples]
      features = self.feature_extractor(audio)

    assert features is not None  # Should be guaranteed by logic above
    features = features[..., : language_detection_segments * self.feature_extractor.nb_max_frames]

    detected_language_info: dict[str, list[float]] = {}
    all_language_probs: list[tuple[str, float]] = []  # Initialize to avoid unbound variable

    for i in range(0, features.shape[-1], self.feature_extractor.nb_max_frames):
      self.logger.debug(
        f"Processing language detection segment {i // self.feature_extractor.nb_max_frames + 1}"
      )
      encoder_output = self._encode(
        pad_or_trim(features[..., i : i + self.feature_extractor.nb_max_frames])
      )
      # results is a list of tuple[str, float] with language names and probabilities.
      results = self.model.detect_language(encoder_output)[0]
      self.logger.debug(f"Language detection results: {results[:3]}...")  # Show top 3 results

      # Parse language names to strip out markers
      all_language_probs = [(token[2:-2], prob) for (token, prob) in results]
      # Get top language token and probability
      language, language_probability = all_language_probs[0]
      self.logger.debug(
        f"Top language candidate: {language} (probability: {language_probability:.3f})"
      )
      if language_probability > language_detection_threshold:
        self.logger.debug(
          "Language detection threshold met: "
          f"{language_probability:.3f} > {language_detection_threshold}"
        )
        break
      detected_language_info.setdefault(language, []).append(language_probability)
    else:
      # If no language detected for all segments, the majority vote of the highest
      # projected languages for all segments is used to determine the language.
      language = max(
        detected_language_info,
        key=lambda lang: len(detected_language_info[lang]),
      )
      language_probability = max(detected_language_info[language])

    self.logger.debug(
      f"Final detected language: {language} with probability {language_probability:.3f}"
    )
    return LanguageDetectionResult(
      language=language,
      probability=language_probability,
      all_probabilities=all_language_probs,
    )

  def _encode(self, features: np.ndarray) -> ctranslate2.StorageView:
    """Encode features using the Whisper model.

    :param features: Audio features to encode
    :type features: np.ndarray
    :returns: Encoded features as CTranslate2 storage view
    :rtype: ctranslate2.StorageView
    """
    # When the model is running on multiple GPUs, the encoder output should be moved
    # to the CPU since we don't know which GPU will handle the next job.
    to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1

    if features.ndim == 2:
      features = np.expand_dims(features, 0)

    # Convert to CTranslate2 storage format
    from eavesdrop.server.transcription.utils import get_ctranslate2_storage

    features = get_ctranslate2_storage(features)

    return self.model.encode(features, to_cpu=to_cpu)

  def update_tokenizer_language(
    self, tokenizer: Tokenizer, encoder_output: ctranslate2.StorageView
  ) -> str:
    """Detect and update tokenizer language for current segment.

    This method is used in multilingual transcription to re-detect the language
    for each segment, ensuring optimal transcription quality when the language
    may change between segments.

    :param tokenizer: The tokenizer instance to update with detected language.
    :type tokenizer: Tokenizer
    :param encoder_output: Encoded audio features for language detection.
    :type encoder_output: ctranslate2.StorageView
    :returns: The detected language code (e.g., "en", "fr").
    :rtype: str
    """
    language_token, _probability = self.model.detect_language(encoder_output)[0][0]
    language = language_token[2:-2]  # Remove <| and |> markers
    tokenizer.language = tokenizer.tokenizer.token_to_id(language_token)
    tokenizer.language_code = language
    return language


class AnomalyDetector:
  """Handles anomaly detection for transcription segments and words."""

  def __init__(self, punctuation: str = _DEFAULT_PUNCTUATION):
    """Initialize the anomaly detector.

    :param punctuation: String of punctuation characters to ignore in anomaly detection
    :type punctuation: str
    """
    self.punctuation = punctuation
    self.logger = get_logger("anomaly")

  def word_anomaly_score(self, word: WordDict) -> float:
    """Calculate anomaly score for a word based on probability and duration.

    Anomalous words are very long/short/improbable. Higher scores indicate
    more anomalous words.

    :param word: Word dictionary containing timing and probability information
    :type word: WordDict
    :returns: Anomaly score (higher = more anomalous)
    :rtype: float
    """
    probability = word.get("probability", 0.0)
    duration = word["end"] - word["start"]
    score = 0.0

    # Low probability words are anomalous
    if probability < _WORD_PROBABILITY_THRESHOLD:
      score += 1.0

    # Very short words are anomalous
    if duration < _SHORT_WORD_DURATION_THRESHOLD:
      score += (_SHORT_WORD_DURATION_THRESHOLD - duration) * _SHORT_WORD_DURATION_MULTIPLIER

    # Very long words are anomalous
    if duration > _LONG_WORD_DURATION_THRESHOLD:
      score += duration - _LONG_WORD_DURATION_THRESHOLD

    return score

  def next_words_segment(self, segments: list[SegmentDict]) -> SegmentDict | None:
    """Find the next segment that contains words.

    :param segments: List of segment dictionaries to search
    :type segments: list[SegmentDict]
    :returns: First segment containing words, or None if none found
    :rtype: SegmentDict | None
    """
    return next((s for s in segments if s.get("words")), None)

  def is_segment_anomaly(self, segment: SegmentDict | None) -> bool:
    """Determine if a segment is anomalous based on its words.

    :param segment: Segment dictionary containing words and timing information
    :type segment: SegmentDict | None
    :returns: True if the segment is considered anomalous, False otherwise
    :rtype: bool
    """
    if segment is None or not segment.get("words"):
      return False

    # Filter out punctuation and limit to first 8 words for analysis
    words = [w for w in segment.get("words", []) if w["word"] not in self.punctuation]
    words = words[:_MAX_WORDS_FOR_ANOMALY_CHECK]

    if not words:
      return False

    # Calculate total anomaly score for the segment
    total_score = sum(self.word_anomaly_score(w) for w in words)

    # Segment is anomalous if score is high relative to word count
    return (
      total_score >= _SEGMENT_ANOMALY_SCORE_THRESHOLD
      or total_score + _SEGMENT_ANOMALY_SCORE_OFFSET >= len(words)
    )

  def analyze_segment_anomalies(self, segments: list[SegmentDict]) -> list[SegmentAnomalyResult]:
    """Analyze multiple segments for anomalies.

    :param segments: List of segment dictionaries to analyze
    :type segments: list[SegmentDict]
    :returns: List of tuples containing (segment_index, is_anomaly, total_score)
    :rtype: list[SegmentAnomalyResult]
    """
    results = []
    for i, segment in enumerate(segments):
      is_anomaly = self.is_segment_anomaly(segment)

      # Calculate total score for reporting
      if segment.get("words"):
        words = [w for w in segment.get("words", []) if w["word"] not in self.punctuation]
        words = words[:_MAX_WORDS_FOR_ANOMALY_CHECK]
        total_score = sum(self.word_anomaly_score(w) for w in words)
      else:
        total_score = 0.0

      results.append(
        {
          "segment_index": i,
          "is_anomaly": is_anomaly,
          "anomaly_score": total_score,
        }
      )

    return results


class LanguageProbabilityAnalyzer:
  """Analyzes language probabilities and provides insights."""

  def __init__(self):
    """Initialize the language probability analyzer."""
    self.logger = get_logger("lang")

  def get_top_languages(
    self, all_language_probs: list[tuple[str, float]], top_n: int = 5
  ) -> list[tuple[str, float]]:
    """Get the top N languages by probability.

    :param all_language_probs: List of (language, probability) tuples
    :type all_language_probs: list[tuple[str, float]]
    :param top_n: Number of top languages to return
    :type top_n: int
    :returns: List of top N (language, probability) tuples
    :rtype: list[tuple[str, float]]
    """
    return sorted(all_language_probs, key=lambda x: x[1], reverse=True)[:top_n]

  def get_confidence_level(self, language_probability: float) -> str:
    """Categorize confidence level based on probability.

    :param language_probability: Probability of the detected language
    :type language_probability: float
    :returns: Confidence level as string: "high", "medium", "low"
    :rtype: str
    """
    if language_probability >= 0.8:
      return "high"
    elif language_probability >= 0.5:
      return "medium"
    else:
      return "low"

  def calculate_entropy(self, all_language_probs: list[tuple[str, float]]) -> float:
    """Calculate entropy of language probability distribution.

    Higher entropy indicates more uncertainty in language detection.

    :param all_language_probs: List of (language, probability) tuples
    :type all_language_probs: list[tuple[str, float]]
    :returns: Entropy value (higher = more uncertain)
    :rtype: float
    """
    if not all_language_probs:
      return 0.0

    entropy = 0.0
    for _, prob in all_language_probs:
      if prob > 0:
        entropy -= prob * np.log2(prob)

    return entropy

  def analyze_language_distribution(
    self, all_language_probs: list[tuple[str, float]]
  ) -> LanguageAnalysisResult:
    """Comprehensive analysis of language probability distribution.

    :param all_language_probs: List of (language, probability) tuples
    :type all_language_probs: list[tuple[str, float]]
    :returns: LanguageAnalysisResult with analysis results including top languages, entropy,
      and confidence
    :rtype: LanguageAnalysisResult
    """
    if not all_language_probs:
      return LanguageAnalysisResult(
        top_language=("unknown", 0.0),
        confidence_level="low",
        entropy=0.0,
        top_3_languages=[],
        total_languages=0,
      )

    top_language = all_language_probs[0]
    top_3 = self.get_top_languages(all_language_probs, 3)
    confidence = self.get_confidence_level(top_language[1])
    entropy = self.calculate_entropy(all_language_probs)

    return LanguageAnalysisResult(
      top_language=top_language,
      confidence_level=confidence,
      entropy=entropy,
      top_3_languages=top_3,
      total_languages=len(all_language_probs),
    )
