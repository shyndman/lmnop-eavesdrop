from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
  from eavesdrop.server.transcription.session import TranscriptionSessionProtocol

import ctranslate2
import numpy as np
from faster_whisper.feature_extractor import FeatureExtractor
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.utils import get_end
from structlog.stdlib import BoundLogger

from eavesdrop.server.transcription.decode_state import (
  _TranscribeContext,
  _TranscribeSegmentsResult,
)
from eavesdrop.server.transcription.generation_strategies import GenerationStrategies
from eavesdrop.server.transcription.hallucination_filter import HallucinationFilter
from eavesdrop.server.transcription.language_detection import LanguageDetector
from eavesdrop.server.transcription.models import SegmentDict, TranscriptionOptions
from eavesdrop.server.transcription.prompt_builder import PromptBuilder
from eavesdrop.server.transcription.segment_processor import SegmentProcessor
from eavesdrop.server.transcription.utils import get_ctranslate2_storage, summarize_array
from eavesdrop.server.transcription.word_alignment import WordTimestampAligner


class SegmentDecoder:
  def __init__(
    self,
    model: ctranslate2.models.Whisper,
    feature_extractor: FeatureExtractor,
    frames_per_second: float,
    logger: BoundLogger,
    language_detector: LanguageDetector,
    word_aligner: WordTimestampAligner,
    prompt_builder: PromptBuilder,
    segment_processor: SegmentProcessor,
    generation_strategies: GenerationStrategies,
    hallucination_filter: HallucinationFilter,
  ):
    self.model = model
    self.feature_extractor = feature_extractor
    self.frames_per_second = frames_per_second
    self.logger = logger
    self.language_detector = language_detector
    self.word_aligner = word_aligner
    self.prompt_builder = prompt_builder
    self.segment_processor = segment_processor
    self.generation_strategies = generation_strategies
    self.hallucination_filter = hallucination_filter

  def transcribe_segments(
    self,
    features: np.ndarray,
    tokenizer: Tokenizer,
    transcription_options: TranscriptionOptions,
    log_progress: bool,
    encoder_output: ctranslate2.StorageView | None = None,
    session: "TranscriptionSessionProtocol | None" = None,
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
      segment_index = len(ctx.all_segments)
      encoded_this_iteration = False

      if not ctx.at_beginning or encoder_output is None:
        self.logger.debug(
          "Segment encode start",
          segment_index=segment_index,
          at_beginning=ctx.at_beginning,
          seek=ctx.seek,
          time_offset=ctx.time_offset,
          **summarize_array("segment_features", segment),
        )
        encoder_output = self.encode(segment)
        encoded_this_iteration = True
        self.logger.debug(
          "Segment encode complete",
          segment_index=segment_index,
          at_beginning=ctx.at_beginning,
        )
      else:
        self.logger.debug(
          "Reusing encoder output",
          segment_index=segment_index,
          at_beginning=ctx.at_beginning,
        )

      # You may be thinking, "didn't they just detect a language earlier?", and you'd be quite
      # correct. That language was only used to initialize the tokenizer for the first pass. The
      # multilingual case is a little more interesting, so we re-detect the language here to
      # ensure we have the best possible language for the current segment.
      if transcription_options.multilingual:
        self.language_detector.update_tokenizer_language(tokenizer, encoder_output)

      # Generate the transcription
      prompt = self.prompt_builder.build_prompt(
        tokenizer,
        ctx.context_tokens,
        without_timestamps=transcription_options.without_timestamps,
        prefix=transcription_options.prefix if ctx.at_beginning else None,
        hotwords=transcription_options.hotwords,
      )
      self.logger.debug(
        "Generation start",
        segment_index=segment_index,
        encoded_this_iteration=encoded_this_iteration,
        multilingual=transcription_options.multilingual,
        beam_size=transcription_options.beam_size,
        prompt_length=len(prompt),
        context_token_count=len(ctx.context_tokens),
        word_timestamps=transcription_options.word_timestamps,
      )
      result, avg_logprob, temperature, compression_ratio, attempts = (
        self.generation_strategies.generate_with_fallback(
          model=self.model,
          encoder_output=encoder_output,
          prompt=prompt,
          tokenizer=tokenizer,
          transcription_options=transcription_options,
        )
      )
      self.logger.debug(
        "Generation complete",
        segment_index=segment_index,
        attempts=attempts,
        avg_logprob=avg_logprob,
        temperature=temperature,
        compression_ratio=compression_ratio,
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
    self.logger.debug(
      "Whisper model.encode start",
      model_device=self.model.device,
      model_device_index=self.model.device_index,
      to_cpu=to_cpu,
      **summarize_array("features", features),
    )
    features = get_ctranslate2_storage(features)
    encoded = self.model.encode(features, to_cpu=to_cpu)
    self.logger.debug("Whisper model.encode complete", to_cpu=to_cpu)
    return encoded

  def _process_word_timestamps(
    self,
    current_segments: list[SegmentDict],
    ctx: _TranscribeContext,
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
