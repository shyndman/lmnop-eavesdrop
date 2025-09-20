from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, cast

if TYPE_CHECKING:
  from eavesdrop.server.transcription.session import TranscriptionSessionProtocol

import ctranslate2
import numpy as np
from faster_whisper.audio import pad_or_trim
from faster_whisper.tokenizer import _LANGUAGE_CODES, Tokenizer
from faster_whisper.utils import get_end
from faster_whisper.vad import VadOptions

from eavesdrop.common import get_logger
from eavesdrop.server.transcription.audio_processing import AudioProcessor
from eavesdrop.server.transcription.generation_strategies import GenerationStrategies
from eavesdrop.server.transcription.hallucination_filter import HallucinationFilter
from eavesdrop.server.transcription.language_detection import (
  AnomalyDetector,
  LanguageDetector,
)
from eavesdrop.server.transcription.model_setup import (
  WhisperModelConfig,
  load_whisper_model,
)
from eavesdrop.server.transcription.models import (
  SegmentDict,
  TranscriptionInfo,
  TranscriptionOptions,
)
from eavesdrop.server.transcription.prompt_builder import PromptBuilder
from eavesdrop.server.transcription.segment_processor import SegmentProcessor
from eavesdrop.server.transcription.utils import (
  get_ctranslate2_storage,
  get_suppressed_tokens,
  restore_speech_timestamps,
)
from eavesdrop.server.transcription.word_alignment import WordTimestampAligner
from eavesdrop.wire import Segment, Word

# Combined punctuation string for word alignment processing
# Contains both prepended ('"\'"¿([{-') and appended ('"\'.。,，!！?？:：")]}、') punctuation marks
# Used for merging punctuation with adjacent words during timestamp alignment
_PUNCTUATION = '"\'"¿([{-"\'.。,，!！?？:：")]}、'

# Whisper has a couple of modes of operation. For now, we only use transcription.
_TRANSCRIBE_TASK = "transcribe"


class _TranscribeSegmentsResult(NamedTuple):
  """Result from transcription with generation metrics."""

  segments: Iterable[Segment]
  total_attempts: int


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
    files: dict | None = None,
    **model_kwargs,
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
    self.logger = get_logger("shh")
    self._last_vad_log_time = 0  # Track when we last logged VAD filtering

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
    self.model = model_bundle.model
    self.hf_tokenizer = model_bundle.hf_tokenizer
    self.feature_extractor = model_bundle.feature_extractor
    self.feat_kwargs = model_bundle.feature_kwargs

    # Copy computed properties for backward compatibility
    self.input_stride = model_bundle.input_stride
    self.num_samples_per_token = model_bundle.num_samples_per_token
    self.frames_per_second = model_bundle.frames_per_second
    self.tokens_per_second = model_bundle.tokens_per_second
    self.time_precision = model_bundle.time_precision
    self.max_length = model_bundle.max_length

    # Initialize processors with our new modules
    self.audio_processor = AudioProcessor(self.feature_extractor)
    self.language_detector = LanguageDetector(self.model, self.feature_extractor)
    self.word_aligner = WordTimestampAligner(self.frames_per_second, self.tokens_per_second)
    self.prompt_builder = PromptBuilder(self.max_length)
    self.segment_processor = SegmentProcessor(self.time_precision, self.input_stride)
    self.generation_strategies = GenerationStrategies(self.max_length, self.time_precision)

    # Initialize anomaly detector and hallucination filter
    self.hallucination_filter = HallucinationFilter(AnomalyDetector(_PUNCTUATION))

  @property
  def supported_languages(self) -> list[str]:
    """The languages supported by the model."""
    return list(_LANGUAGE_CODES) if self.model.is_multilingual else ["en"]

  def transcribe(
    self,
    audio: np.ndarray,
    session: "TranscriptionSessionProtocol | None" = None,
    language: str | None = None,
    initial_prompt: str | None = None,
    vad_filter: bool = False,
    vad_parameters: VadOptions = VadOptions(),
    hotwords: str | None = None,
    multilingual: bool = False,
    start_offset: float = 0.0,
    absolute_stream_start: float = 0.0,
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
    :param vad_parameters: VAD configuration options (VadOptions instance).
    :type vad_parameters: VadOptions
    :param hotwords: Optional hotwords to provide as context to improve recognition of
      specific terms.
    :type hotwords: str | None
    :param start_offset: Start time offset in seconds from stream/connection start.
    :type start_offset: float
    :returns: A tuple with:

        - a generator over transcribed segments
        - an instance of TranscriptionInfo
    :rtype: tuple[Iterable[Segment], TranscriptionInfo]
    """

    # Use noop session if none provided
    if session is None:
      from eavesdrop.server.transcription.session import _noop_session

      session = _noop_session

    # Update session timing context with real buffer timing
    session.update_audio_context(
      start_offset=start_offset, duration=audio.shape[0] / self.audio_processor.sampling_rate
    )

    with session.trace_pipeline():
      return self._transcribe(
        audio=audio,
        session=session,
        language=language,
        initial_prompt=initial_prompt,
        vad_filter=vad_filter,
        vad_parameters=vad_parameters,
        hotwords=hotwords,
        multilingual=multilingual,
        absolute_stream_start=absolute_stream_start,
      )

  def _transcribe(
    self,
    audio: np.ndarray,
    session: "TranscriptionSessionProtocol",
    language: str | None = None,
    initial_prompt: str | None = None,
    vad_filter: bool = False,
    vad_parameters: VadOptions = VadOptions(),
    hotwords: str | None = None,
    multilingual: bool = False,
    absolute_stream_start: float = 0.0,
  ) -> tuple[Iterable[Segment], TranscriptionInfo]:
    # Use our new audio processing module for validation and VAD
    with session.trace_vad_stage() as tracer:
      audio, duration, duration_after_vad, speech_chunks, will_be_complete_silence = (
        self.audio_processor.validate_and_preprocess_audio(audio, vad_filter, vad_parameters)
      )
      tracer(
        speech_chunks if vad_filter else None, self.audio_processor.sampling_rate, audio.shape[0]
      )

    # VAD Gatekeeper: Skip transcription if no speech detected
    if vad_filter and will_be_complete_silence:
      # VAD detected no speech - don't waste compute on Whisper
      return [], TranscriptionInfo(
        transcription_options=TranscriptionOptions(
          multilingual=multilingual,
          initial_prompt=initial_prompt,
        ),
        vad_options=vad_parameters,
        duration=duration,
        duration_after_vad=0.0,  # No speech detected
      )

    # Update audio with processed version
    no_sound_to_process = audio.shape[0] == 0
    if no_sound_to_process:
      # Return empty segments and minimal transcription info for empty audio
      return [], TranscriptionInfo(
        transcription_options=TranscriptionOptions(
          multilingual=multilingual,
          initial_prompt=initial_prompt,
        ),
        vad_options=vad_parameters,
      )

    # Extract features using our audio processor
    with session.trace_feature_stage():
      features = self.audio_processor.extract_features(audio)
      (language, probability), all_language_probs = self.language_detector.resolve_language(
        language, features
      )

    # Model inference and segment processing
    with session.trace_inference_stage() as tracer:
      segments, total_attempts = self._transcribe_segments(
        features,
        tokenizer=(
          tokenizer := Tokenizer(
            self.hf_tokenizer,
            self.model.is_multilingual,
            task=_TRANSCRIBE_TASK,
            language=language,
          )
        ),
        transcription_options=(
          options := TranscriptionOptions(
            multilingual=multilingual,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            suppress_tokens=get_suppressed_tokens(tokenizer, [-1, 0]),
          )
        ),
        log_progress=False,
        encoder_output=None,
        session=session,
        absolute_stream_start=absolute_stream_start,
      )
      tracer(total_attempts, 0.0)

    # Final segment processing and results
    with session.trace_segment_stage() as tracer:
      if speech_chunks:
        segments = restore_speech_timestamps(
          segments, speech_chunks, self.audio_processor.sampling_rate
        )
      tracer(segments)

    info = TranscriptionInfo(
      language=language,
      language_probability=probability,
      duration=duration,
      duration_after_vad=duration_after_vad,
      transcription_options=options,
      vad_options=vad_parameters,
      all_language_probs=all_language_probs,
      speech_chunks=speech_chunks,
    )

    return segments, info

  def _transcribe_segments(
    self,
    features: np.ndarray,
    tokenizer: Tokenizer,
    transcription_options: TranscriptionOptions,
    log_progress: bool,
    encoder_output: ctranslate2.StorageView | None = None,
    session: "TranscriptionSessionProtocol | None" = None,
    absolute_stream_start: float = 0.0,
  ) -> _TranscribeSegmentsResult:
    """A lower-level transcription function that generates the individual segments for a given
    audio clip.

    :returns: _TranscribeSegmentsResult containing segments and total generation attempts.
    :rtype: _TranscribeSegmentsResult
    """

    # Prepare initial tokens from prompt using prompt builder
    initial_tokens = self.prompt_builder.encode_initial_prompt(
      tokenizer, transcription_options.initial_prompt
    )

    # Initialize transcription context
    ctx = _TranscribeContext(
      features=features,
      time_per_frame=self.feature_extractor.time_per_frame,
      max_segment_frames=self.feature_extractor.nb_max_frames,
      frames_per_second=self.frames_per_second,
      initial_tokens=initial_tokens,
    )

    while ctx.advance():
      segment = ctx.extract_segment()

      if not ctx.at_beginning or encoder_output is None:
        encoder_output = self.encode(segment)

      # You may be thinking, "didn't they just detect a language earlier?", and you'd be quite
      # correct. That language was only used to initialize the tokenizer for the first pass. The
      # multilingual case is a little more interesting, so we re-detect the language here to
      # ensure we have the best possible language for the current segment.
      if transcription_options.multilingual:
        self.language_detector.update_tokenizer_language(tokenizer, encoder_output)

      # Generate the transcription
      result, avg_logprob, temperature, compression_ratio, attempts = (
        self.generation_strategies.generate_with_fallback(
          model=self.model,
          encoder_output=encoder_output,
          prompt=self.prompt_builder.build_prompt(
            tokenizer,
            ctx.context_tokens,
            without_timestamps=transcription_options.without_timestamps,
            prefix=transcription_options.prefix if ctx.at_beginning else None,
            hotwords=transcription_options.hotwords,
          ),
          tokenizer=tokenizer,
          transcription_options=transcription_options,
        )
      )
      tokens = result.sequences_ids[0]
      ctx.total_attempts += attempts

      # Split segments
      previous_seek = ctx.seek
      current_segments, new_seek, single_timestamp_ending = (
        self.segment_processor.split_segments_by_timestamps(
          tokenizer=tokenizer,
          tokens=tokens,
          time_offset=ctx.time_offset,
          segment_size=ctx.segment_size,
          segment_duration=ctx.segment_duration,
          seek=ctx.seek,
        )
      )

      # Update seek position from segment processor
      ctx.seek_next_to(new_seek)

      # Process word timestamps if requested
      if transcription_options.word_timestamps:
        self._process_word_timestamps(
          current_segments,
          ctx,
          transcription_options,
          encoder_output,
          tokenizer,
          single_timestamp_ending,
        )

        # Filter hallucinations using silence gap analysis
        if transcription_options.hallucination_silence_threshold is not None:
          filtered_segments, new_seek = self.hallucination_filter.filter_segments(
            segments=current_segments,
            threshold=transcription_options.hallucination_silence_threshold,
            time_offset=ctx.time_offset,
            segment_duration=ctx.segment_duration,
            window_end_time=ctx.window_end_time,
            last_speech_timestamp=ctx.last_speech_timestamp,
            total_duration=ctx.total_duration,
            total_frames=ctx.total_frames,
            previous_seek=previous_seek,
            frames_per_second=self.frames_per_second,
          )
          current_segments = filtered_segments
          if new_seek is not None:
            ctx.seek_next_to(new_seek)
            continue

      for segment in current_segments:
        tokens = segment["tokens"]
        text = tokenizer.decode(tokens)

        ctx.add_segment(
          segment_data=segment,
          text=text,
          tokens=tokens,
          temperature=temperature,
          avg_logprob=avg_logprob,
          compression_ratio=compression_ratio,
          word_timestamps=transcription_options.word_timestamps,
          time_offset=absolute_stream_start,
        )

      if (
        not transcription_options.condition_on_previous_text
        or temperature > transcription_options.prompt_reset_on_temperature
      ):
        if transcription_options.condition_on_previous_text:
          self.logger.debug(
            "Reset prompt. prompt_reset_on_temperature threshold is met %f > %f",
            temperature,
            transcription_options.prompt_reset_on_temperature,
          )

        ctx.prompt_reset_since = len(ctx.all_tokens)

    return _TranscribeSegmentsResult(segments=ctx.all_segments, total_attempts=ctx.total_attempts)

  def encode(self, features: np.ndarray) -> ctranslate2.StorageView:
    # When the model is running on multiple GPUs, the encoder output should be moved
    # to the CPU since we don't know which GPU will handle the next job.
    to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1

    if features.ndim == 2:
      features = np.expand_dims(features, 0)
    features = get_ctranslate2_storage(features)

    return self.model.encode(features, to_cpu=to_cpu)

  def _process_word_timestamps(
    self,
    current_segments: list[SegmentDict],
    ctx: "_TranscribeContext",
    transcription_options: TranscriptionOptions,
    encoder_output: ctranslate2.StorageView,
    tokenizer: Tokenizer,
    single_timestamp_ending: bool,
  ) -> None:
    """Process word-level timestamps and update transcription context.

    Performs forced alignment to get precise word timings, adjusts seek position
    based on actual speech boundaries, and updates speech timestamp tracking.

    :param current_segments: Segments to add word timestamps to.
    :type current_segments: list[SegmentDict]
    :param ctx: Transcription context containing seek position and timing state.
    :type ctx: _TranscribeContext
    :param transcription_options: Options containing word timestamp settings.
    :type transcription_options: TranscriptionOptions
    :param encoder_output: Encoded audio features for alignment.
    :type encoder_output: ctranslate2.StorageView
    :param tokenizer: Tokenizer for text processing.
    :type tokenizer: Tokenizer
    :param single_timestamp_ending: Whether segment ended with single timestamp.
    :type single_timestamp_ending: bool
    """
    # 1. FORCED ALIGNMENT: Use cross-attention to align words with audio frames
    self.word_aligner.add_word_timestamps(
      segments=[current_segments],
      model=self.model,
      tokenizer=tokenizer,
      encoder_output=encoder_output,
      num_frames=ctx.segment_size,
      prepend_punctuations=transcription_options.prepend_punctuations,
      append_punctuations=transcription_options.append_punctuations,
      last_speech_timestamp=ctx.last_speech_timestamp,
    )

    # 2. SEEK ADJUSTMENT: Move transcription window to end of actual speech
    # (Skip this if segment ended with single timestamp, indicating silence)
    if not single_timestamp_ending:
      last_word_end: float | None = get_end(cast(list[dict], current_segments))
      if last_word_end is not None and last_word_end > ctx.time_offset:
        ctx.seek_next_to(round(last_word_end * self.frames_per_second))

    # 3. SPEECH TIMESTAMP TRACKING: Update last known speech time
    # This is used for future silence gap calculations and seek positioning
    if last_word_end := get_end(cast(list[dict], current_segments)):
      ctx.last_speech_timestamp = last_word_end


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

  anomaly_detector: AnomalyDetector = field(init=False)
  """Detector for hallucination filtering."""

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
    self.anomaly_detector = AnomalyDetector(_PUNCTUATION)
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
    time_offset: float,
  ):
    """Add completed segment to results."""
    text = text.strip()
    if segment_data["start"] == segment_data["end"] or not text:
      return

    self.all_tokens.extend(tokens)

    # Assign baseline ID for incomplete segments (will get chain ID when completed)
    from eavesdrop.wire.transcription import compute_segment_chain_id

    segment_id = compute_segment_chain_id(0, "")  # Baseline ID for incomplete segments
    absolute_start_time = time_offset + segment_data["start"]

    id_logger = get_logger("seg-id")
    id_logger.debug(
      "Segment created (incomplete, will get chain ID when completed)",
      baseline_id=segment_id,
      relative_start=segment_data["start"],
      time_offset=time_offset,
      absolute_start_time=absolute_start_time,
      seek=self.seek,
      text=text[:50] + "..." if len(text) > 50 else text,
    )

    self.all_segments.append(
      Segment(
        id=segment_id,
        seek=self.seek,
        start=segment_data["start"],
        end=segment_data["end"],
        text=text,
        tokens=tokens,
        temperature=temperature,
        avg_logprob=avg_logprob,
        compression_ratio=compression_ratio,
        words=(
          [Word(**word) for word in segment_data.get("words", [])] if word_timestamps else None
        ),
        time_offset=time_offset,
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
