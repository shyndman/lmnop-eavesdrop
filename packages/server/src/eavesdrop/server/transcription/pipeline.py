from collections.abc import Iterable
from typing import TYPE_CHECKING

from structlog.stdlib import BoundLogger

if TYPE_CHECKING:
  from eavesdrop.server.transcription.session import TranscriptionSessionProtocol

from eavesdrop.common import get_logger
from eavesdrop.server.transcription.audio_processing import AudioProcessor, FloatAudio
from eavesdrop.server.transcription.generation_strategies import GenerationStrategies
from eavesdrop.server.transcription.hallucination_filter import HallucinationFilter
from eavesdrop.server.transcription.language_detection import (
  AnomalyDetector,
  LanguageDetector,
)
from eavesdrop.server.transcription.model_setup import (
  ModelFiles,
  WhisperModelConfig,
  load_whisper_model,
)
from eavesdrop.server.transcription.models import (
  FeatureExtractorConfig,
  TranscriptionInfo,
  VadParameters,
)
from eavesdrop.server.transcription.prompt_builder import PromptBuilder
from eavesdrop.server.transcription.request_runner import RequestRunner
from eavesdrop.server.transcription.segment_decoder import SegmentDecoder
from eavesdrop.server.transcription.segment_processor import SegmentProcessor
from eavesdrop.server.transcription.vendor_types import (
  FeatureExtractorLike,
  TokenizerLike,
  WhisperModelLike,
  load_language_codes,
)
from eavesdrop.server.transcription.word_alignment import WordTimestampAligner
from eavesdrop.wire import Segment

# Combined punctuation string for word alignment processing
# Contains both prepended ('"\'"¿([{-') and appended ('"\'.。,，!！?？:：")]}、') punctuation marks
# Used for merging punctuation with adjacent words during timestamp alignment
_PUNCTUATION = '"\'"¿([{-"\'.。,，!！?？:：")]}、'
LANGUAGE_CODES = load_language_codes()


class WhisperModel:
  def __init__(
    self,
    model_size_or_path: str,
    device: str = "auto",
    device_index: int | list[int] = 0,
    compute_type: str = "default",
    cpu_threads: int = 0,
    num_workers: int = 1,
    download_root: str | None = None,
    local_files_only: bool = False,
    files: ModelFiles | None = None,
    **model_kwargs: object,
  ):
    """Initializes the Whisper model.

    :param
      model_size_or_path: Size of the model to use (tiny, tiny.en, base, base.en,
        small, small.en, distil-small.en, medium, medium.en, distil-medium.en, large-v1,
        large-v2, large-v3, large, distil-large-v2, distil-large-v3, large-v3-turbo, or turbo),
        a path to a converted model directory, or a CTranslate2-converted Whisper model ID from
        the HF Hub. When a size or a model ID is configured, the converted model is downloaded
        from the Hugging Face Hub.
      device: Device to use for computation ("cpu", "cuda", "auto").
      device_index: Device ID to use.
        The model can also be loaded on multiple GPUs by passing a list of IDs
        (e.g. [0, 1, 2, 3]). In that case, multiple transcriptions can run in parallel
        when transcribe() is called from multiple Python threads (see also num_workers).
      compute_type: Type to use for computation.
        See https://opennmt.net/CTranslate2/quantization.html.
      cpu_threads: Number of threads to use when running on CPU (4 by default).
        A non zero value overrides the OMP_NUM_THREADS environment variable.
      num_workers: When transcribe() is called from multiple Python threads,
        having multiple workers enables true parallelism when running the model
        (concurrent calls to self.model.generate() will run in parallel).
        This can improve the global throughput at the cost of increased memory usage.
      download_root: Directory where the models should be saved. If not set, the models
        are saved in the standard Hugging Face cache directory.
      local_files_only:  If True, avoid downloading the file and return the path to the
        local cached file if it exists.
      files: Load model files from the memory. This argument is a dictionary mapping file names
        to file contents as file-like or bytes objects. If this is set, model_path acts as an
        identifier for this model.
    """
    self.logger: BoundLogger = get_logger("shh")
    self._last_vad_log_time: int = 0  # Track when we last logged VAD filtering

    # Use our new whisper module for model loading
    config = WhisperModelConfig(
      model_size_or_path=model_size_or_path,
      device=device,
      device_index=device_index,
      compute_type=compute_type,
      cpu_threads=cpu_threads,
      num_workers=num_workers,
      download_root=download_root,
      local_files_only=local_files_only,
      files=files,
      **model_kwargs,
    )

    # Load model using our new clean module
    model_bundle = load_whisper_model(config)

    # Extract components for backward compatibility
    self.model: WhisperModelLike = model_bundle.model
    self.hf_tokenizer: TokenizerLike = model_bundle.hf_tokenizer
    self.feature_extractor: FeatureExtractorLike = model_bundle.feature_extractor
    self.feat_kwargs: FeatureExtractorConfig = model_bundle.feature_kwargs

    # Copy computed properties for backward compatibility
    self.input_stride: int = model_bundle.input_stride
    self.num_samples_per_token: int = model_bundle.num_samples_per_token
    self.frames_per_second: int = model_bundle.frames_per_second
    self.tokens_per_second: int = model_bundle.tokens_per_second
    self.time_precision: float = model_bundle.time_precision
    self.max_length: int = model_bundle.max_length

    # Initialize processors with our new modules
    self.audio_processor: AudioProcessor = AudioProcessor(self.feature_extractor)
    self.language_detector: LanguageDetector = LanguageDetector(self.model, self.feature_extractor)
    self.word_aligner: WordTimestampAligner = WordTimestampAligner(
      self.frames_per_second, self.tokens_per_second
    )
    self.prompt_builder: PromptBuilder = PromptBuilder(self.max_length)
    self.segment_processor: SegmentProcessor = SegmentProcessor(
      self.time_precision, self.input_stride
    )
    self.generation_strategies: GenerationStrategies = GenerationStrategies(
      self.max_length, self.time_precision
    )

    # Initialize anomaly detector and hallucination filter
    self.hallucination_filter: HallucinationFilter = HallucinationFilter(
      AnomalyDetector(_PUNCTUATION)
    )

    self.segment_decoder: SegmentDecoder = SegmentDecoder(
      model=self.model,
      feature_extractor=self.feature_extractor,
      frames_per_second=self.frames_per_second,
      logger=self.logger,
      language_detector=self.language_detector,
      word_aligner=self.word_aligner,
      prompt_builder=self.prompt_builder,
      segment_processor=self.segment_processor,
      generation_strategies=self.generation_strategies,
      hallucination_filter=self.hallucination_filter,
    )
    self.request_runner: RequestRunner = RequestRunner(
      logger=self.logger,
      model=self.model,
      hf_tokenizer=self.hf_tokenizer,
      audio_processor=self.audio_processor,
      language_detector=self.language_detector,
      segment_decoder=self.segment_decoder,
    )

  @property
  def supported_languages(self) -> list[str]:
    """The languages supported by the model."""
    return list(LANGUAGE_CODES) if self.model.is_multilingual else ["en"]

  def transcribe(
    self,
    audio: FloatAudio,
    session: "TranscriptionSessionProtocol | None" = None,
    language: str | None = None,
    initial_prompt: str | None = None,
    vad_filter: bool = False,
    vad_parameters: VadParameters | None = None,
    hotwords: str | None = None,
    multilingual: bool = False,
    start_offset: float = 0.0,
    absolute_stream_start: float = 0.0,
    beam_size: int | None = None,
    word_timestamps: bool | None = None,
  ) -> tuple[Iterable[Segment], TranscriptionInfo]:
    """Transcribes audio data for live transcription.

    :param audio: Audio waveform as numpy array (16kHz sample rate).
    :type audio: np.ndarray
    :param language: The language spoken in the audio. It should be a language code such
        as "en" or "fr". If not set, the language will be detected in the first 30 seconds
        of audio.
    :type language: str | None
    :param task: Task to execute (transcribe or translate).
    :type task: str
    :param initial_prompt: Optional text string to provide as a prompt for the first window.
    :type initial_prompt: str | None
    :param vad_filter: Enable the voice activity detection (VAD) to filter out parts of the audio
        without speech. This step is using the Silero VAD model
        https://github.com/snakers4/silero-vad.
    :type vad_filter: bool
    :param vad_parameters: Repo-owned VAD configuration options.
    :type vad_parameters: VadParameters | None
    :param hotwords: Optional hotwords to provide as context to improve recognition of
      specific terms.
    :type hotwords: str | None
    :param start_offset: Start time offset in seconds from stream/connection start.
    :type start_offset: float
    :param beam_size: Optional override for beam search width during deterministic decoding.
    :type beam_size: int | None
    :param word_timestamps: Whether to compute detailed word-level timestamps.
    :type word_timestamps: bool | None
    :returns: A tuple with:

        - a generator over transcribed segments
        - an instance of TranscriptionInfo
    :rtype: tuple[Iterable[Segment], TranscriptionInfo]
    """

    # Use noop session if none provided
    if session is None:
      from eavesdrop.server.transcription.session import noop_session

      session = noop_session

    # Update session timing context with real buffer timing
    session.update_audio_context(
      start_offset=start_offset, duration=audio.shape[0] / self.audio_processor.sampling_rate
    )

    with session.trace_pipeline():
      return self.request_runner.run(
        audio=audio,
        session=session,
        language=language,
        initial_prompt=initial_prompt,
        vad_filter=vad_filter,
        vad_parameters=vad_parameters,
        hotwords=hotwords,
        multilingual=multilingual,
        absolute_stream_start=absolute_stream_start,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
      )
