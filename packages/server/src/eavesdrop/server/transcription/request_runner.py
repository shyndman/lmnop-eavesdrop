from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from eavesdrop.server.transcription.session import TranscriptionSessionProtocol

import ctranslate2
import numpy as np
import tokenizers
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.vad import VadOptions
from structlog.stdlib import BoundLogger

from eavesdrop.server.transcription.audio_processing import AudioProcessor
from eavesdrop.server.transcription.language_detection import LanguageDetector
from eavesdrop.server.transcription.models import (
  TranscriptionInfo,
  TranscriptionOptions,
  VadParameters,
)
from eavesdrop.server.transcription.segment_decoder import SegmentDecoder
from eavesdrop.server.transcription.utils import (
  finalize_recording_timestamps,
  get_suppressed_tokens,
  restore_speech_timestamps,
  summarize_array,
)
from eavesdrop.wire import Segment

# Whisper has a couple of modes of operation. For now, we only use transcription.
_TRANSCRIBE_TASK = "transcribe"


class RequestRunner:
  def __init__(
    self,
    logger: BoundLogger,
    model: ctranslate2.models.Whisper,
    hf_tokenizer: tokenizers.Tokenizer,
    audio_processor: AudioProcessor,
    language_detector: LanguageDetector,
    segment_decoder: SegmentDecoder,
  ):
    self.logger = logger
    self.model = model
    self.hf_tokenizer = hf_tokenizer
    self.audio_processor = audio_processor
    self.language_detector = language_detector
    self.segment_decoder = segment_decoder

  def run(
    self,
    audio: np.ndarray,
    session: "TranscriptionSessionProtocol",
    language: str | None = None,
    initial_prompt: str | None = None,
    vad_filter: bool = False,
    vad_parameters: VadParameters | None = None,
    hotwords: str | None = None,
    multilingual: bool = False,
    absolute_stream_start: float = 0.0,
    beam_size: int | None = None,
    word_timestamps: bool | None = None,
  ) -> tuple[Iterable[Segment], TranscriptionInfo]:
    default_options = TranscriptionOptions()
    resolved_vad_parameters = vad_parameters or VadParameters()
    runtime_vad_parameters = VadOptions(
      threshold=resolved_vad_parameters.threshold,
      neg_threshold=resolved_vad_parameters.neg_threshold,
      min_speech_duration_ms=resolved_vad_parameters.min_speech_duration_ms,
      max_speech_duration_s=resolved_vad_parameters.max_speech_duration_s,
      min_silence_duration_ms=resolved_vad_parameters.min_silence_duration_ms,
      speech_pad_ms=resolved_vad_parameters.speech_pad_ms,
    )
    resolved_beam_size = beam_size if beam_size is not None else default_options.beam_size
    resolved_word_timestamps = (
      word_timestamps if word_timestamps is not None else default_options.word_timestamps
    )

    # Use our new audio processing module for validation and VAD
    with session.trace_vad_stage() as tracer:
      audio, duration, duration_after_vad, speech_chunks, will_be_complete_silence = (
        self.audio_processor.validate_and_preprocess_audio(
          audio,
          vad_filter,
          runtime_vad_parameters,
        )
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
          beam_size=resolved_beam_size,
          word_timestamps=resolved_word_timestamps,
        ),
        vad_options=resolved_vad_parameters,
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
          beam_size=resolved_beam_size,
          word_timestamps=resolved_word_timestamps,
        ),
        vad_options=resolved_vad_parameters,
      )

    # Extract features using our audio processor
    with session.trace_feature_stage():
      features = self.audio_processor.extract_features(audio)
      self.logger.debug(
        "Feature extraction complete",
        requested_language=language,
        duration_after_vad=duration_after_vad,
        **summarize_array("features", features),
      )
      self.logger.debug("Language resolution start", requested_language=language)
      (language, probability), all_language_probs = self.language_detector.resolve_language(
        language, features
      )
      self.logger.debug(
        "Language resolution complete",
        language=language,
        language_probability=probability,
        candidate_count=len(all_language_probs),
      )

    # Model inference and segment processing
    with session.trace_inference_stage() as tracer:
      segments, total_attempts = self.segment_decoder.transcribe_segments(
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
            beam_size=resolved_beam_size,
            word_timestamps=resolved_word_timestamps,
            suppress_tokens=get_suppressed_tokens(tokenizer, [-1, 0]),
          )
        ),
        log_progress=False,
        encoder_output=None,
        session=session,
      )
      tracer(total_attempts, 0.0)

    # Final segment processing and results
    with session.trace_segment_stage() as tracer:
      if speech_chunks:
        segments = restore_speech_timestamps(
          segments, speech_chunks, self.audio_processor.sampling_rate
        )

      # Ensure all timestamps are recording-relative
      if absolute_stream_start > 0:
        segments = finalize_recording_timestamps(segments, absolute_stream_start)

      tracer(segments)

    info = TranscriptionInfo(
      language=language,
      language_probability=probability,
      duration=duration,
      duration_after_vad=duration_after_vad,
      transcription_options=options,
      vad_options=resolved_vad_parameters,
      all_language_probs=all_language_probs,
      speech_chunks=speech_chunks or [],
    )

    return segments, info
