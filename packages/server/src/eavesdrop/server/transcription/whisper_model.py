from collections.abc import Iterable
from typing import cast

import ctranslate2
import numpy as np
from faster_whisper.audio import pad_or_trim
from faster_whisper.tokenizer import _LANGUAGE_CODES, Tokenizer
from faster_whisper.utils import get_end
from faster_whisper.vad import VadOptions
from tqdm import tqdm

from eavesdrop.server.logs import get_logger
from eavesdrop.server.transcription.audio_processing import AudioProcessor
from eavesdrop.server.transcription.generation_strategies import GenerationStrategies
from eavesdrop.server.transcription.language_detection import (
  AnomalyDetector,
  LanguageDetectionResult,
  LanguageDetector,
)
from eavesdrop.server.transcription.models import (
  SegmentDict,
  TranscriptionInfo,
  TranscriptionOptions,
)
from eavesdrop.server.transcription.prompt_builder import PromptBuilder
from eavesdrop.server.transcription.segment_processor import SegmentProcessor, SegmentTimingResult
from eavesdrop.server.transcription.utils import (
  get_ctranslate2_storage,
  get_suppressed_tokens,
  restore_speech_timestamps,
)
from eavesdrop.server.transcription.whisper import (
  WhisperModelConfig,
  load_whisper_model,
)
from eavesdrop.server.transcription.word_alignment import WordTimestampAligner
from eavesdrop.wire import Segment, Word

_PUNCTUATION = '"\'"¿([{-"\'.。,，!！?？:：")]}、'


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
    self.logger = get_logger("whispr")
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

  @property
  def supported_languages(self) -> list[str]:
    """The languages supported by the model."""
    return list(_LANGUAGE_CODES) if self.model.is_multilingual else ["en"]

  def transcribe(
    self,
    audio: np.ndarray,
    language: str | None = None,
    task: str = "transcribe",
    initial_prompt: str | None = None,
    vad_filter: bool = False,
    vad_parameters: VadOptions = VadOptions(),
    hotwords: str | None = None,
  ) -> tuple[Iterable[Segment], TranscriptionInfo]:
    """Transcribes audio data for live transcription.

    :param
      audio: Audio waveform as numpy array (16kHz sample rate).
      language: The language spoken in the audio. It should be a language code such
        as "en" or "fr". If not set, the language will be detected in the first 30 seconds
        of audio.
      task: Task to execute (transcribe or translate).
      initial_prompt: Optional text string to provide as a prompt for the first window.
      vad_filter: Enable the voice activity detection (VAD) to filter out parts of the audio
        without speech. This step is using the Silero VAD model
        https://github.com/snakers4/silero-vad.
      vad_parameters: VAD configuration options (VadOptions instance).
      hotwords: Optional hotwords to provide as context to improve recognition of specific terms.
    :returns:
      A tuple with:

        - a generator over transcribed segments
        - an instance of TranscriptionInfo
    """
    multilingual = False

    # Use our new audio processing module for validation and VAD
    audio_result = self.audio_processor.validate_and_preprocess_audio(
      audio, vad_filter, vad_parameters
    )

    # Update audio with processed version
    audio = audio_result["audio"]
    duration = audio_result["duration"]
    duration_after_vad = audio_result["duration_after_vad"]
    speech_chunks = audio_result["speech_chunks"]
    # Note: will_be_complete_silence available if needed for future logging
    # will_be_complete_silence = audio_result["is_complete_silence"]

    if audio.shape[0] == 0:
      # Return empty segments and minimal transcription info for empty audio
      empty_info = TranscriptionInfo(
        transcription_options=TranscriptionOptions(
          multilingual=multilingual,
          initial_prompt=initial_prompt,
        ),
        vad_options=vad_parameters,
      )
      return [], empty_info

    # Extract features using our audio processor
    features = self.audio_processor.extract_features(audio)
    sampling_rate = self.audio_processor.sampling_rate

    encoder_output: ctranslate2.StorageView | None = None
    all_language_probs: list[tuple[str, float]] | None = None

    # detecting the language if not provided
    if language is None:
      if not self.model.is_multilingual:
        language = "en"
        language_probability = 1
      else:
        start_timestamp = 0.0
        content_frames = features.shape[-1] - 1
        seek = (
          int(start_timestamp * self.frames_per_second)
          if start_timestamp * self.frames_per_second < content_frames
          else 0
        )
        detection_result = self.detect_language(
          features=features[..., seek:],
          language_detection_segments=1,
          language_detection_threshold=0.5,
        )
        language = detection_result["language"]
        language_probability = detection_result["probability"]
        all_language_probs = detection_result["all_probabilities"]

        self.logger.info(
          "Detected language '%s' with probability %.2f",
          language,
          language_probability,
        )
    else:
      if not self.model.is_multilingual and language != "en":
        self.logger.warning(
          "The current model is English-only but the language parameter is set to '%s'; "
          "using 'en' instead." % language
        )
        language = "en"

      language_probability = 1

    assert language is not None, "Language should be determined by this point"
    tokenizer = Tokenizer(
      self.hf_tokenizer,
      self.model.is_multilingual,
      task=task,
      language=language,
    )

    options = TranscriptionOptions(
      multilingual=multilingual,
      initial_prompt=initial_prompt,
      hotwords=hotwords,
    )
    # Set dynamic defaults
    if options.suppress_tokens is None:
      options.suppress_tokens = get_suppressed_tokens(tokenizer, [-1])

    segments = self.generate_segments(features, tokenizer, options, False, encoder_output)

    if speech_chunks:
      segments = restore_speech_timestamps(segments, speech_chunks, sampling_rate)

    info = TranscriptionInfo(
      language=language,
      language_probability=language_probability,
      duration=duration,
      duration_after_vad=duration_after_vad,
      transcription_options=options,
      vad_options=vad_parameters,
      all_language_probs=all_language_probs,
    )

    return segments, info

  def _split_segments_by_timestamps(
    self,
    tokenizer: Tokenizer,
    tokens: list[int],
    time_offset: float,
    segment_size: int,
    segment_duration: float,
    seek: int,
  ) -> SegmentTimingResult:
    """Split tokens into segments using our segment processor module."""
    return self.segment_processor.split_segments_by_timestamps(
      tokenizer, tokens, time_offset, segment_size, segment_duration, seek
    )

  def generate_segments(
    self,
    features: np.ndarray,
    tokenizer: Tokenizer,
    options: TranscriptionOptions,
    log_progress: bool,
    encoder_output: ctranslate2.StorageView | None = None,
  ) -> Iterable[Segment]:
    content_frames = features.shape[-1] - 1
    content_duration = float(content_frames * self.feature_extractor.time_per_frame)
    # In this implementation, we process the entire audio as a single clip.
    seek_clips: list[tuple[int, int]] = [(0, content_frames)]
    # Initialize anomaly detector for hallucination detection
    anomaly_detector = AnomalyDetector(_PUNCTUATION)

    idx = 0
    clip_idx = 0
    seek = seek_clips[clip_idx][0]
    all_tokens: list[int] = []
    prompt_reset_since = 0

    if options.initial_prompt is not None:
      initial_prompt = " " + options.initial_prompt.strip()
      initial_prompt_tokens = tokenizer.encode(initial_prompt)
      all_tokens.extend(initial_prompt_tokens)

    pbar = tqdm(total=content_duration, unit="seconds", disable=not log_progress)
    last_speech_timestamp = 0.0
    all_segments: list[Segment] = []
    # NOTE: This loop is obscurely flattened to make the diff readable.
    # A later commit should turn this into a simpler nested loop.
    # for seek_clip_start, seek_clip_end in seek_clips:
    #     while seek < seek_clip_end
    while clip_idx < len(seek_clips):
      seek_clip_start, seek_clip_end = seek_clips[clip_idx]
      seek_clip_end = min(seek_clip_end, content_frames)
      seek = max(seek, seek_clip_start)
      if seek >= seek_clip_end:
        clip_idx += 1
        if clip_idx < len(seek_clips):
          seek = seek_clips[clip_idx][0]
        continue
      time_offset = seek * self.feature_extractor.time_per_frame
      window_end_time = float(
        (seek + self.feature_extractor.nb_max_frames) * self.feature_extractor.time_per_frame
      )
      segment_size = min(
        self.feature_extractor.nb_max_frames,
        content_frames - seek,
        seek_clip_end - seek,
      )
      segment = features[:, seek : seek + segment_size]
      segment_duration = segment_size * self.feature_extractor.time_per_frame
      segment = pad_or_trim(segment)

      previous_tokens = all_tokens[prompt_reset_since:]

      if seek > 0 or encoder_output is None:
        encoder_output = self.encode(segment)

      if options.multilingual:
        results = self.model.detect_language(encoder_output)
        language_token, language_probability = results[0][0]
        language = language_token[2:-2]

        tokenizer.language = tokenizer.tokenizer.token_to_id(language_token)
        tokenizer.language_code = language

      prompt = self.prompt_builder.build_prompt(
        tokenizer,
        previous_tokens,
        without_timestamps=options.without_timestamps,
        prefix=options.prefix if seek == 0 else None,
        hotwords=options.hotwords,
      )

      generation_result = self.generation_strategies.generate_with_fallback(
        self.model, encoder_output, prompt, tokenizer, options
      )
      result = generation_result["result"]
      avg_logprob = generation_result["avg_logprob"]
      temperature = generation_result["temperature"]
      compression_ratio = generation_result["compression_ratio"]

      if options.no_speech_threshold is not None:
        # no voice activity check
        should_skip = result.no_speech_prob > options.no_speech_threshold

        if options.log_prob_threshold is not None and avg_logprob > options.log_prob_threshold:
          # don't skip if the logprob is high enough, despite the no_speech_prob
          should_skip = False

        if should_skip:
          self.logger.debug(
            "No speech threshold is met (%f > %f)",
            result.no_speech_prob,
            options.no_speech_threshold,
          )

          # fast-forward to the next segment boundary
          seek += segment_size
          continue

      tokens = result.sequences_ids[0]

      previous_seek = seek

      timing_result = self._split_segments_by_timestamps(
        tokenizer=tokenizer,
        tokens=tokens,
        time_offset=time_offset,
        segment_size=segment_size,
        segment_duration=segment_duration,
        seek=seek,
      )
      current_segments = timing_result["segments"]
      seek = timing_result["seek_position"]
      single_timestamp_ending = timing_result["single_timestamp_ending"]

      if options.word_timestamps:
        self.add_word_timestamps(
          [current_segments],
          tokenizer,
          encoder_output,
          segment_size,
          options.prepend_punctuations,
          options.append_punctuations,
          last_speech_timestamp=last_speech_timestamp,
        )
        if not single_timestamp_ending:
          last_word_end = get_end(cast(list[dict], current_segments))
          if last_word_end is not None and last_word_end > time_offset:
            seek = round(last_word_end * self.frames_per_second)

        # skip silence before possible hallucinations
        if options.hallucination_silence_threshold is not None:
          threshold = options.hallucination_silence_threshold

          # if first segment might be a hallucination, skip leading silence
          first_segment = anomaly_detector.next_words_segment(current_segments)
          if first_segment is not None and anomaly_detector.is_segment_anomaly(first_segment):
            gap = first_segment["start"] - time_offset
            if gap > threshold:
              seek = previous_seek + round(gap * self.frames_per_second)
              continue

          # skip silence before any possible hallucination that is surrounded
          # by silence or more hallucinations
          hal_last_end = last_speech_timestamp
          for si in range(len(current_segments)):
            segment = current_segments[si]
            if not segment.get("words"):
              continue
            if anomaly_detector.is_segment_anomaly(segment):
              next_segment = anomaly_detector.next_words_segment(current_segments[si + 1 :])
              if next_segment is not None:
                next_words = next_segment.get("words", [])
                if next_words:
                  hal_next_start = next_words[0]["start"]
                else:
                  hal_next_start = time_offset + segment_duration
              else:
                hal_next_start = time_offset + segment_duration
              silence_before = (
                segment["start"] - hal_last_end > threshold
                or segment["start"] < threshold
                or segment["start"] - time_offset < 2.0
              )
              silence_after = (
                hal_next_start - segment["end"] > threshold
                or anomaly_detector.is_segment_anomaly(next_segment)
                or window_end_time - segment["end"] < 2.0
              )
              if silence_before and silence_after:
                seek = round(max(time_offset + 1, segment["start"]) * self.frames_per_second)
                if content_duration - segment["end"] < threshold:
                  seek = content_frames
                current_segments[si:] = []
                break
            hal_last_end = segment["end"]

        last_word_end = get_end(cast(list[dict], current_segments))
        if last_word_end is not None:
          last_speech_timestamp = last_word_end
      for segment in current_segments:
        tokens = segment["tokens"]
        text = tokenizer.decode(tokens)

        if segment["start"] == segment["end"] or not text.strip():
          continue

        all_tokens.extend(tokens)
        idx += 1

        all_segments.append(
          Segment(
            id=idx,
            seek=previous_seek,
            start=segment["start"],
            end=segment["end"],
            text=text,
            tokens=tokens,
            temperature=temperature,
            avg_logprob=avg_logprob,
            compression_ratio=compression_ratio,
            no_speech_prob=result.no_speech_prob,
            words=(
              [Word(**word) for word in segment.get("words", [])]
              if options.word_timestamps
              else None
            ),
          )
        )

      if (
        not options.condition_on_previous_text or temperature > options.prompt_reset_on_temperature
      ):
        if options.condition_on_previous_text:
          self.logger.debug(
            "Reset prompt. prompt_reset_on_temperature threshold is met %f > %f",
            temperature,
            options.prompt_reset_on_temperature,
          )

        prompt_reset_since = len(all_tokens)

      pbar.update(
        (min(content_frames, seek) - previous_seek) * self.feature_extractor.time_per_frame,
      )
    pbar.close()
    return all_segments

  def encode(self, features: np.ndarray) -> ctranslate2.StorageView:
    # When the model is running on multiple GPUs, the encoder output should be moved
    # to the CPU since we don't know which GPU will handle the next job.
    to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1

    if features.ndim == 2:
      features = np.expand_dims(features, 0)
    features = get_ctranslate2_storage(features)

    return self.model.encode(features, to_cpu=to_cpu)

  def add_word_timestamps(
    self,
    segments: list[list[SegmentDict]],
    tokenizer: Tokenizer,
    encoder_output: ctranslate2.StorageView,
    num_frames: int,
    prepend_punctuations: str,
    append_punctuations: str,
    last_speech_timestamp: float,
  ) -> float:
    """Add word-level timestamps to transcription segments using our word alignment module."""
    # Use our new word alignment module - this replaces 110+ lines of complex logic
    return self.word_aligner.add_word_timestamps(
      segments=segments,
      model=self.model,
      tokenizer=tokenizer,
      encoder_output=encoder_output,
      num_frames=num_frames,
      prepend_punctuations=prepend_punctuations,
      append_punctuations=append_punctuations,
      last_speech_timestamp=last_speech_timestamp,
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

    :param
        audio: Input audio signal, must be a 1D float array sampled at 16khz.
        features: Input Mel spectrogram features, must be a float array with
            shape (n_mels, n_frames), if `audio` is provided, the features will be ignored.
            Either `audio` or `features` must be provided.
        vad_filter: Enable the voice activity detection (VAD) to filter out parts of the audio
            without speech. This step is using the Silero VAD model.
        vad_parameters: VadOptions class instance (see available
            parameters and default values in the class `VadOptions`).
        language_detection_threshold: If the maximum probability of the language tokens is
            higher than this value, the language is detected.
        language_detection_segments: Number of segments to consider for the language detection.

    :returns:
        language: Detected language.
        languege_probability: Probability of the detected language.
        all_language_probs: list of tuples with all language names and probabilities.
    """
    # Use our new language detection module
    return self.language_detector.detect_language(
      audio=audio,
      features=features,
      vad_filter=vad_filter,
      vad_parameters=vad_parameters,
      language_detection_segments=language_detection_segments,
      language_detection_threshold=language_detection_threshold,
    )
