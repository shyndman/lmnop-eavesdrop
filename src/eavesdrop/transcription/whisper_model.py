import itertools
import json
import os
from collections.abc import Iterable
from inspect import signature
from typing import cast

import ctranslate2
import numpy as np
import tokenizers
from faster_whisper.audio import pad_or_trim
from faster_whisper.feature_extractor import FeatureExtractor
from faster_whisper.tokenizer import _LANGUAGE_CODES, Tokenizer
from faster_whisper.utils import download_model, format_timestamp, get_end
from faster_whisper.vad import (
  VadOptions,
  collect_chunks,
  get_speech_timestamps,
)
from tqdm import tqdm

from ..logs import get_logger
from .models import (
  FeatureExtractorConfig,
  Segment,
  SegmentDict,
  TranscriptionInfo,
  TranscriptionOptions,
  Word,
  WordDict,
  WordTimingDict,
)
from .utils import (
  get_compression_ratio,
  get_ctranslate2_storage,
  merge_punctuations,
  restore_speech_timestamps,
)


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

    Args:
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
    self.logger = get_logger()

    tokenizer_bytes: bytes | None
    preprocessor_bytes: bytes | None
    tokenizer_bytes, preprocessor_bytes = None, None
    if files:
      model_path = model_size_or_path
      tokenizer_bytes = files.pop("tokenizer.json", None)
      preprocessor_bytes = files.pop("preprocessor_config.json", None)
    elif os.path.isdir(model_size_or_path):
      model_path = model_size_or_path
    else:
      model_path = download_model(
        model_size_or_path,
        local_files_only=local_files_only,
        cache_dir=download_root,
      )

    self.model: ctranslate2.models.Whisper = ctranslate2.models.Whisper(
      model_path,
      device=device,
      device_index=device_index,
      compute_type=compute_type,
      intra_threads=cpu_threads,
      inter_threads=num_workers,
      files=files,
      **model_kwargs,
    )

    self.logger.info(
      "Initialized FasterWhisper model: path='%s', device='%s', device_index=%s, "
      "compute_type='%s', cpu_threads=%d, num_workers=%d, is_multilingual=%s",
      model_path,
      device,
      device_index,
      compute_type,
      cpu_threads,
      num_workers,
      self.model.is_multilingual,
    )

    tokenizer_file = os.path.join(model_path, "tokenizer.json")
    if tokenizer_bytes:
      self.hf_tokenizer = tokenizers.Tokenizer.from_buffer(tokenizer_bytes)
    elif os.path.isfile(tokenizer_file):
      self.hf_tokenizer = tokenizers.Tokenizer.from_file(tokenizer_file)
    else:
      self.hf_tokenizer = tokenizers.Tokenizer.from_pretrained(
        "openai/whisper-tiny" + ("" if self.model.is_multilingual else ".en")
      )
    self.feat_kwargs = self._get_feature_kwargs(model_path, preprocessor_bytes)
    self.feature_extractor = FeatureExtractor(**self.feat_kwargs)

    self.logger.info(
      "Initialized FasterWhisper feature extractor: sampling_rate=%d, n_fft=%d, "
      "hop_length=%d, chunk_length=%d",
      self.feature_extractor.sampling_rate,
      self.feature_extractor.n_fft,
      self.feature_extractor.hop_length,
      self.feature_extractor.chunk_length,
    )
    self.input_stride = 2
    self.num_samples_per_token = self.feature_extractor.hop_length * self.input_stride
    self.frames_per_second = (
      self.feature_extractor.sampling_rate // self.feature_extractor.hop_length
    )
    self.tokens_per_second = self.feature_extractor.sampling_rate // self.num_samples_per_token
    self.time_precision = 0.02
    self.max_length = 448

  @property
  def supported_languages(self) -> list[str]:
    """The languages supported by the model."""
    return list(_LANGUAGE_CODES) if self.model.is_multilingual else ["en"]

  def _get_feature_kwargs(
    self, model_path: str, preprocessor_bytes: bytes | None = None
  ) -> FeatureExtractorConfig:
    config = {}
    try:
      config_path = os.path.join(model_path, "preprocessor_config.json")
      if preprocessor_bytes:
        config = json.loads(preprocessor_bytes)
      elif os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as file:
          config = json.load(file)
      else:
        return cast(FeatureExtractorConfig, config)
      valid_keys = signature(FeatureExtractor.__init__).parameters.keys()
      return cast(FeatureExtractorConfig, {k: v for k, v in config.items() if k in valid_keys})
    except json.JSONDecodeError as e:
      self.logger.warning("Could not load preprocessor config: %s", e)

    return cast(FeatureExtractorConfig, config)

  def transcribe(
    self,
    audio: np.ndarray,
    language: str | None = None,
    task: str = "transcribe",
    log_progress: bool = False,
    beam_size: int = 5,
    best_of: int = 5,
    patience: float = 1,
    length_penalty: float = 1,
    repetition_penalty: float = 1,
    no_repeat_ngram_size: int = 0,
    temperature: float | list[float] | tuple[float, ...] = [
      0.0,
      0.2,
      0.4,
      0.6,
      0.8,
      1.0,
    ],
    compression_ratio_threshold: float | None = 2.4,
    log_prob_threshold: float | None = -1.0,
    no_speech_threshold: float | None = 0.6,
    condition_on_previous_text: bool = True,
    prompt_reset_on_temperature: float = 0.5,
    initial_prompt: str | None = None,
    prefix: str | None = None,
    suppress_blank: bool = True,
    suppress_tokens: list[int] | None = [-1],
    without_timestamps: bool = False,
    max_initial_timestamp: float = 1.0,
    word_timestamps: bool = False,
    prepend_punctuations: str = '"\'"¿([{-',
    append_punctuations: str = '"\'.。,，!！?？:：")]}、',
    multilingual: bool = False,
    vad_filter: bool = False,
    vad_parameters: VadOptions = VadOptions(),
    max_new_tokens: int | None = None,
    chunk_length: int | None = None,
    hallucination_silence_threshold: float | None = None,
    hotwords: str | None = None,
    language_detection_threshold: float | None = 0.5,
    language_detection_segments: int = 1,
  ) -> tuple[Iterable[Segment], TranscriptionInfo]:
    """Transcribes audio data for live transcription.

    Arguments:
      audio: Audio waveform as numpy array (16kHz sample rate).
      language: The language spoken in the audio. It should be a language code such
        as "en" or "fr". If not set, the language will be detected in the first 30 seconds
        of audio.
      task: Task to execute (transcribe or translate).
      log_progress: whether to show progress bar or not.
      beam_size: Beam size to use for decoding.
      best_of: Number of candidates when sampling with non-zero temperature.
      patience: Beam search patience factor.
      length_penalty: Exponential length penalty constant.
      repetition_penalty: Penalty applied to the score of previously generated tokens
        (set > 1 to penalize).
      no_repeat_ngram_size: Prevent repetitions of ngrams with this size (set 0 to disable).
      temperature: Temperature for sampling. It can be a tuple of temperatures,
        which will be successively used upon failures according to either
        `compression_ratio_threshold` or `log_prob_threshold`.
      compression_ratio_threshold: If the gzip compression ratio is above this value,
        treat as failed.
      log_prob_threshold: If the average log probability over sampled tokens is
        below this value, treat as failed.
      no_speech_threshold: If the no_speech probability is higher than this value AND
        the average log probability over sampled tokens is below `log_prob_threshold`,
        consider the segment as silent.
      condition_on_previous_text: If True, the previous output of the model is provided
        as a prompt for the next window; disabling may make the text inconsistent across
        windows, but the model becomes less prone to getting stuck in a failure loop,
        such as repetition looping or timestamps going out of sync.
      prompt_reset_on_temperature: Resets prompt if temperature is above this value.
        Arg has effect only if condition_on_previous_text is True.
      initial_prompt: Optional text string to provide as a
        prompt for the first window.
      prefix: Optional text to provide as a prefix for the first window.
      suppress_blank: Suppress blank outputs at the beginning of the sampling.
      suppress_tokens: list of token IDs to suppress. -1 will suppress a default set
        of symbols as defined in `tokenizer.non_speech_tokens()`.
      without_timestamps: Only sample text tokens.
      max_initial_timestamp: The initial timestamp cannot be later than this.
      word_timestamps: Extract word-level timestamps using the cross-attention pattern
        and dynamic time warping, and include the timestamps for each word in each segment.
      prepend_punctuations: If word_timestamps is True, merge these punctuation symbols
        with the next word
      append_punctuations: If word_timestamps is True, merge these punctuation symbols
        with the previous word
      multilingual: Perform language detection on every segment.
      vad_filter: Enable the voice activity detection (VAD) to filter out parts of the audio
        without speech. This step is using the Silero VAD model
        https://github.com/snakers4/silero-vad.
      vad_parameters: VadOptions class instance (see available
        parameters and default values in the class `VadOptions`).
      max_new_tokens: Maximum number of new tokens to generate per-chunk. If not set,
        the maximum will be set by the default max_length.
      chunk_length: The length of audio segments. If it is not None, it will overwrite the
        default chunk_length of the FeatureExtractor.
      hallucination_silence_threshold:
        When word_timestamps is True, skip silent periods longer than this threshold
         (in seconds) when a possible hallucination is detected
      hotwords:
        Hotwords/hint phrases to provide the model with. Has no effect if prefix is not None.
      language_detection_threshold: If the maximum probability of the language tokens is higher
       than this value, the language is detected.
      language_detection_segments: Number of segments to consider for the language detection.
    Returns:
      A tuple with:

        - a generator over transcribed segments
        - an instance of TranscriptionInfo
    """
    sampling_rate: int = self.feature_extractor.sampling_rate

    if multilingual and not self.model.is_multilingual:
      self.logger.warning(
        "The current model is English-only but the multilingual parameter is set to"
        "True; setting to False instead."
      )
      multilingual = False

    # Audio is guaranteed to be np.ndarray for live transcription
    duration: float = audio.shape[0] / sampling_rate
    duration_after_vad: float = duration

    self.logger.info("Processing audio with duration %s", format_timestamp(duration))

    if vad_filter:
      speech_chunks = get_speech_timestamps(audio, vad_parameters)
      audio_chunks: list[np.ndarray]
      chunks_metadata: list[dict[str, int]]
      audio_chunks, chunks_metadata = collect_chunks(audio, speech_chunks)
      audio = np.concatenate(audio_chunks, axis=0)
      duration_after_vad = audio.shape[0] / sampling_rate

      self.logger.info(
        "VAD filter removed %s of audio",
        format_timestamp(duration - duration_after_vad),
      )

    else:
      speech_chunks = None
    if audio.shape[0] == 0:
      # Return empty segments and minimal transcription info for empty audio
      empty_info = TranscriptionInfo(
        language="en",
        language_probability=0.0,
        duration=0.0,
        duration_after_vad=0.0,
        all_language_probs=None,
        transcription_options=TranscriptionOptions(
          beam_size=beam_size,
          best_of=best_of,
          patience=patience,
          length_penalty=length_penalty,
          repetition_penalty=repetition_penalty,
          no_repeat_ngram_size=no_repeat_ngram_size,
          log_prob_threshold=log_prob_threshold,
          no_speech_threshold=no_speech_threshold,
          compression_ratio_threshold=compression_ratio_threshold,
          condition_on_previous_text=condition_on_previous_text,
          prompt_reset_on_temperature=prompt_reset_on_temperature,
          temperatures=list(temperature)
          if isinstance(temperature, (list, tuple))
          else [temperature],
          initial_prompt=initial_prompt,
          prefix=prefix,
          suppress_blank=suppress_blank,
          suppress_tokens=suppress_tokens,
          without_timestamps=without_timestamps,
          max_initial_timestamp=max_initial_timestamp,
          word_timestamps=word_timestamps,
          prepend_punctuations=prepend_punctuations,
          append_punctuations=append_punctuations,
          multilingual=multilingual,
          max_new_tokens=max_new_tokens,
          hallucination_silence_threshold=hallucination_silence_threshold,
          hotwords=hotwords,
        ),
        vad_options=vad_parameters,
      )
      return [], empty_info
    features = self.feature_extractor(audio, chunk_length=chunk_length)

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
        (
          language,
          language_probability,
          all_language_probs,
        ) = self.detect_language(
          features=features[..., seek:],
          language_detection_segments=language_detection_segments,
          language_detection_threshold=language_detection_threshold or 0.5,
        )

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

    tokenizer = Tokenizer(
      self.hf_tokenizer,
      self.model.is_multilingual,
      task=task,
      language=language,
    )

    from .utils import get_suppressed_tokens

    options = TranscriptionOptions(
      beam_size=beam_size,
      best_of=best_of,
      patience=patience,
      length_penalty=length_penalty,
      repetition_penalty=repetition_penalty,
      no_repeat_ngram_size=no_repeat_ngram_size,
      log_prob_threshold=log_prob_threshold,
      no_speech_threshold=no_speech_threshold,
      compression_ratio_threshold=compression_ratio_threshold,
      condition_on_previous_text=condition_on_previous_text,
      prompt_reset_on_temperature=prompt_reset_on_temperature,
      temperatures=list(temperature) if isinstance(temperature, (list, tuple)) else [temperature],
      initial_prompt=initial_prompt,
      prefix=prefix,
      suppress_blank=suppress_blank,
      suppress_tokens=(
        get_suppressed_tokens(tokenizer, suppress_tokens) if suppress_tokens else suppress_tokens
      ),
      without_timestamps=without_timestamps,
      max_initial_timestamp=max_initial_timestamp,
      word_timestamps=word_timestamps,
      prepend_punctuations=prepend_punctuations,
      append_punctuations=append_punctuations,
      multilingual=multilingual,
      max_new_tokens=max_new_tokens,
      hallucination_silence_threshold=hallucination_silence_threshold,
      hotwords=hotwords,
    )

    segments = self.generate_segments(features, tokenizer, options, log_progress, encoder_output)

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
  ) -> tuple[list[SegmentDict], int, bool]:
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
        # single timestamp at the end means no speech after the last timestamp.
        seek += segment_size
      else:
        # otherwise, ignore the unfinished segment and seek to the last timestamp
        last_timestamp_position = tokens[last_slice - 1] - tokenizer.timestamp_begin
        seek += last_timestamp_position * self.input_stride

    else:
      duration = segment_duration
      timestamps = [token for token in tokens if token >= tokenizer.timestamp_begin]
      if len(timestamps) > 0 and timestamps[-1] != tokenizer.timestamp_begin:
        last_timestamp_position = timestamps[-1] - tokenizer.timestamp_begin
        duration = last_timestamp_position * self.time_precision

      current_segments.append(
        {
          "seek": seek,
          "start": time_offset,
          "end": time_offset + duration,
          "tokens": tokens,
        }
      )

      seek += segment_size

    return current_segments, seek, single_timestamp_ending

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

    seek_points: list[int] = []
    if len(seek_points) == 0:
      seek_points.append(0)
    if len(seek_points) % 2 == 1:
      seek_points.append(content_frames)
    seek_clips: list[tuple[int, int]] = list(zip(seek_points[::2], seek_points[1::2]))

    punctuation = '"\'"¿([{-"\'.。,，!！?？:：")]}、'

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
      if seek_clip_end > content_frames:
        seek_clip_end = content_frames
      if seek < seek_clip_start:
        seek = seek_clip_start
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

      prompt = self.get_prompt(
        tokenizer,
        previous_tokens,
        without_timestamps=options.without_timestamps,
        prefix=options.prefix if seek == 0 else None,
        hotwords=options.hotwords,
      )

      (
        result,
        avg_logprob,
        temperature,
        compression_ratio,
      ) = self.generate_with_fallback(encoder_output, prompt, tokenizer, options)

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

      # anomalous words are very long/short/improbable
      def word_anomaly_score(word: WordDict) -> float:
        probability = word.get("probability", 0.0)
        duration = word["end"] - word["start"]
        score = 0.0
        if probability < 0.15:
          score += 1.0
        if duration < 0.133:
          score += (0.133 - duration) * 15
        if duration > 2.0:
          score += duration - 2.0
        return score

      def is_segment_anomaly(segment: SegmentDict | None) -> bool:
        if segment is None or not segment.get("words"):
          return False
        words = [w for w in segment.get("words", []) if w["word"] not in punctuation]
        words = words[:8]
        score = sum(word_anomaly_score(w) for w in words)
        return score >= 3 or score + 0.01 >= len(words)

      def next_words_segment(segments: list[SegmentDict]) -> SegmentDict | None:
        return next((s for s in segments if s.get("words")), None)

      (
        current_segments,
        seek,
        single_timestamp_ending,
      ) = self._split_segments_by_timestamps(
        tokenizer=tokenizer,
        tokens=tokens,
        time_offset=time_offset,
        segment_size=segment_size,
        segment_duration=segment_duration,
        seek=seek,
      )

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
          first_segment = next_words_segment(current_segments)
          if first_segment is not None and is_segment_anomaly(first_segment):
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
            if is_segment_anomaly(segment):
              next_segment = next_words_segment(current_segments[si + 1 :])
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
                or is_segment_anomaly(next_segment)
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

  def generate_with_fallback(
    self,
    encoder_output: ctranslate2.StorageView,
    prompt: list[int],
    tokenizer: Tokenizer,
    options: TranscriptionOptions,
  ) -> tuple[ctranslate2.models.WhisperGenerationResult, float, float, float]:
    decode_result = None
    all_results = []
    below_cr_threshold_results = []

    max_initial_timestamp_index = int(round(options.max_initial_timestamp / self.time_precision))
    if options.max_new_tokens is not None:
      max_length = len(prompt) + options.max_new_tokens
    else:
      max_length = self.max_length

    if max_length > self.max_length:
      raise ValueError(
        f"The length of the prompt is {len(prompt)}, and the `max_new_tokens` "
        f"{max_length - len(prompt)}. Thus, the combined length of the prompt "
        f"and `max_new_tokens` is: {max_length}. This exceeds the "
        f"`max_length` of the Whisper model: {self.max_length}. "
        "You should either reduce the length of your prompt, or "
        "reduce the value of `max_new_tokens`, "
        f"so that their combined length is less that {self.max_length}."
      )

    temperature = 0.0  # Initialize to avoid unbound variable warning
    for temperature in options.temperatures:
      self.logger.debug(f"Trying temperature: {temperature}")
      if temperature > 0:
        kwargs = {
          "beam_size": 1,
          "num_hypotheses": options.best_of,
          "sampling_topk": 0,
          "sampling_temperature": temperature,
        }
        self.logger.debug(f"Using sampling with temperature {temperature}")
      else:
        kwargs = {
          "beam_size": options.beam_size,
          "patience": options.patience,
        }
        self.logger.debug(f"Using beam search with beam_size {options.beam_size}")

      result = self.model.generate(
        encoder_output,
        [prompt],
        length_penalty=options.length_penalty,
        repetition_penalty=options.repetition_penalty,
        no_repeat_ngram_size=options.no_repeat_ngram_size,
        max_length=max_length,
        return_scores=True,
        return_no_speech_prob=True,
        suppress_blank=options.suppress_blank,
        suppress_tokens=options.suppress_tokens,
        max_initial_timestamp_index=max_initial_timestamp_index,
        **kwargs,
      )[0]

      tokens = result.sequences_ids[0]

      # Recover the average log prob from the returned score.
      seq_len = len(tokens)
      cum_logprob = result.scores[0] * (seq_len**options.length_penalty)
      avg_logprob = cum_logprob / (seq_len + 1)

      text = tokenizer.decode(tokens).strip()
      compression_ratio = get_compression_ratio(text)
      self.logger.debug(
        f"Generated text (temp={temperature}): '{text[:50]}...', "
        f"compression_ratio: {compression_ratio:.3f}, avg_logprob: {avg_logprob:.3f}"
      )

      decode_result = (
        result,
        avg_logprob,
        temperature,
        compression_ratio,
      )
      all_results.append(decode_result)

      needs_fallback = False

      if options.compression_ratio_threshold is not None:
        if compression_ratio > options.compression_ratio_threshold:
          needs_fallback = True  # too repetitive

          self.logger.debug(
            "Compression ratio threshold is not met with temperature %.1f (%f > %f)",
            temperature,
            compression_ratio,
            options.compression_ratio_threshold,
          )
        else:
          below_cr_threshold_results.append(decode_result)

      if options.log_prob_threshold is not None and avg_logprob < options.log_prob_threshold:
        needs_fallback = True  # average log probability is too low

        self.logger.debug(
          "Log probability threshold is not met with temperature %.1f (%f < %f)",
          temperature,
          avg_logprob,
          options.log_prob_threshold,
        )

      if (
        options.no_speech_threshold is not None
        and result.no_speech_prob > options.no_speech_threshold
        and options.log_prob_threshold is not None
        and avg_logprob < options.log_prob_threshold
      ):
        needs_fallback = False  # silence

      if not needs_fallback:
        break
    else:
      # all failed, select the result with the highest average log probability
      decode_result = max(below_cr_threshold_results or all_results, key=lambda x: x[1])
      # to pass final temperature for prompt_reset_on_temperature
      decode_result = (
        decode_result[0],
        decode_result[1],
        temperature,
        decode_result[3],
      )

    return decode_result

  def get_prompt(
    self,
    tokenizer: Tokenizer,
    previous_tokens: list[int],
    without_timestamps: bool = False,
    prefix: str | None = None,
    hotwords: str | None = None,
  ) -> list[int]:
    prompt = []

    if previous_tokens or (hotwords and not prefix):
      prompt.append(tokenizer.sot_prev)
      if hotwords and not prefix:
        hotwords_tokens = tokenizer.encode(" " + hotwords.strip())
        if len(hotwords_tokens) >= self.max_length // 2:
          hotwords_tokens = hotwords_tokens[: self.max_length // 2 - 1]
        prompt.extend(hotwords_tokens)
      if previous_tokens:
        prompt.extend(previous_tokens[-(self.max_length // 2 - 1) :])

    prompt.extend(tokenizer.sot_sequence)

    if without_timestamps:
      prompt.append(tokenizer.no_timestamps)

    if prefix:
      prefix_tokens = tokenizer.encode(" " + prefix.strip())
      if len(prefix_tokens) >= self.max_length // 2:
        prefix_tokens = prefix_tokens[: self.max_length // 2 - 1]
      if not without_timestamps:
        prompt.append(tokenizer.timestamp_begin)
      prompt.extend(prefix_tokens)

    return prompt

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
    if len(segments) == 0:
      return 0.0

    text_tokens_per_segment: list[list[list[int]]] = []
    for segment_group in segments:
      segment_tokens = [
        [token for token in segment["tokens"] if token < tokenizer.eot] for segment in segment_group
      ]
      text_tokens_per_segment.append(segment_tokens)

    # Flatten all tokens for alignment (same as original behavior)
    text_tokens = list(
      itertools.chain.from_iterable(itertools.chain.from_iterable(text_tokens_per_segment))
    )

    alignments = self.find_alignment(tokenizer, text_tokens, encoder_output, num_frames)
    median_max_durations: list[tuple[float, float]] = []
    for alignment in alignments:
      word_durations = np.array([word["end"] - word["start"] for word in alignment])
      word_durations = word_durations[word_durations.nonzero()]
      median_duration = np.median(word_durations) if len(word_durations) > 0 else 0.0
      median_duration = min(0.7, float(median_duration))
      max_duration = median_duration * 2

      # hack: truncate long words at sentence boundaries.
      # a better segmentation algorithm based on VAD should be able to replace this.
      if len(word_durations) > 0:
        sentence_end_marks = ".。!！?？"
        # ensure words at sentence boundaries
        # are not longer than twice the median word duration.
        for i in range(1, len(alignment)):
          if alignment[i]["end"] - alignment[i]["start"] > max_duration:
            if alignment[i]["word"] in sentence_end_marks:
              alignment[i]["end"] = alignment[i]["start"] + max_duration
            elif alignment[i - 1]["word"] in sentence_end_marks:
              alignment[i]["start"] = alignment[i]["end"] - max_duration

      merge_punctuations(alignment, prepend_punctuations, append_punctuations)
      median_max_durations.append((median_duration, max_duration))

    for segment_idx, segment in enumerate(segments):
      word_index = 0
      time_offset = segment[0]["seek"] / self.frames_per_second
      median_duration, max_duration = median_max_durations[segment_idx]
      for subsegment_idx, subsegment in enumerate(segment):
        saved_tokens = 0
        words: list[WordDict] = []

        while word_index < len(alignments[segment_idx]) and saved_tokens < len(
          text_tokens_per_segment[segment_idx][subsegment_idx]
        ):
          timing = alignments[segment_idx][word_index]

          if timing["word"]:
            words.append(
              {
                "word": timing["word"],
                "start": round(time_offset + timing["start"], 2),
                "end": round(time_offset + timing["end"], 2),
                "probability": timing["probability"],
              }
            )

          saved_tokens += len(timing["tokens"])
          word_index += 1

        # hack: truncate long words at segment boundaries.
        # a better segmentation algorithm based on VAD should be able to replace this.
        if len(words) > 0:
          # ensure the first and second word after a pause is not longer than
          # twice the median word duration.
          if words[0]["end"] - last_speech_timestamp > median_duration * 4 and (
            words[0]["end"] - words[0]["start"] > max_duration
            or (len(words) > 1 and words[1]["end"] - words[0]["start"] > max_duration * 2)
          ):
            if len(words) > 1 and words[1]["end"] - words[1]["start"] > max_duration:
              boundary = max(words[1]["end"] / 2, words[1]["end"] - max_duration)
              words[0]["end"] = words[1]["start"] = boundary
            words[0]["start"] = max(0, words[0]["end"] - max_duration)

          # prefer the segment-level start timestamp if the first word is too long.
          if (
            subsegment["start"] < words[0]["end"] and subsegment["start"] - 0.5 > words[0]["start"]
          ):
            words[0]["start"] = max(
              0,
              min(words[0]["end"] - median_duration, subsegment["start"]),
            )
          else:
            subsegment["start"] = words[0]["start"]

          # prefer the segment-level end timestamp if the last word is too long.
          if subsegment["end"] > words[-1]["start"] and subsegment["end"] + 0.5 < words[-1]["end"]:
            words[-1]["end"] = max(words[-1]["start"] + median_duration, subsegment["end"])
          else:
            subsegment["end"] = words[-1]["end"]

          last_speech_timestamp = subsegment["end"]
        segments[segment_idx][subsegment_idx]["words"] = words
    return last_speech_timestamp

  def find_alignment(
    self,
    tokenizer: Tokenizer,
    text_tokens: list[int],
    encoder_output: ctranslate2.StorageView,
    num_frames: int,
    median_filter_width: int = 7,
  ) -> list[list[WordTimingDict]]:
    if len(text_tokens) == 0:
      return []

    results = self.model.align(
      encoder_output,
      tokenizer.sot_sequence,
      text_tokens,
      num_frames,
      median_filter_width=median_filter_width,
    )
    return_list: list[list[WordTimingDict]] = []
    for result, text_token in zip(results, text_tokens):
      text_token_probs = result.text_token_probs
      alignments = result.alignments
      text_indices = np.array([pair[0] for pair in alignments])
      time_indices = np.array([pair[1] for pair in alignments])

      words, word_tokens = tokenizer.split_to_word_tokens([text_token] + [tokenizer.eot])
      if len(word_tokens) <= 1:
        # return on eot only
        # >>> np.pad([], (1, 0))
        # array([0.])
        # This results in crashes when we lookup jump_times with float, like
        # IndexError: arrays used as indices must be of integer (or boolean) type
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

  def detect_language(
    self,
    audio: np.ndarray | None = None,
    features: np.ndarray | None = None,
    vad_filter: bool = False,
    vad_parameters: VadOptions = VadOptions(),
    language_detection_segments: int = 1,
    language_detection_threshold: float = 0.5,
  ) -> tuple[str, float, list[tuple[str, float]]]:
    self.logger.debug(
      f"Detecting language: vad_filter={vad_filter}, segments={language_detection_segments}, "
      f"threshold={language_detection_threshold}"
    )
    """
        Use Whisper to detect the language of the input audio or features.

        Arguments:
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

        Returns:
            language: Detected language.
            languege_probability: Probability of the detected language.
            all_language_probs: list of tuples with all language names and probabilities.
        """
    assert audio is not None or features is not None, (
      "Either `audio` or `features` must be provided."
    )

    if audio is not None:
      if vad_filter:
        speech_chunks = get_speech_timestamps(audio, vad_parameters)
        audio_chunks, chunks_metadata = collect_chunks(audio, speech_chunks)
        audio = np.concatenate(audio_chunks, axis=0)

      audio = audio[: language_detection_segments * self.feature_extractor.n_samples]
      features = self.feature_extractor(audio)

    assert features is not None  # Should be guaranteed by logic above
    features = features[..., : language_detection_segments * self.feature_extractor.nb_max_frames]

    detected_language_info = {}
    all_language_probs: list[tuple[str, float]] = []  # Initialize to avoid unbound variable
    for i in range(0, features.shape[-1], self.feature_extractor.nb_max_frames):
      self.logger.debug(
        f"Processing language detection segment {i // self.feature_extractor.nb_max_frames + 1}"
      )
      encoder_output = self.encode(
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
    return language, language_probability, all_language_probs
