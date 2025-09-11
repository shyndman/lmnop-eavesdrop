"""Generation strategies and temperature fallback for Whisper transcription.

This module handles the multi-temperature fallback strategy used by Whisper to
ensure high-quality transcriptions by trying different generation parameters
when quality thresholds are not met.
"""

from typing import NamedTuple

import ctranslate2
from faster_whisper.tokenizer import Tokenizer

from eavesdrop.server.logs import get_logger
from eavesdrop.server.transcription.models import TranscriptionOptions
from eavesdrop.server.transcription.utils import get_compression_ratio


class GenerationResult(NamedTuple):
  """Result from Whisper generation with quality metrics."""

  result: ctranslate2.models.WhisperGenerationResult
  avg_logprob: float
  temperature: float
  compression_ratio: float


def _create_max_length_error(
  prompt_length: int, max_new_tokens: int, combined_length: int, max_length: int
) -> str:
  """Create a descriptive error message for when prompt + max_new_tokens exceeds max_length."""
  return (
    f"The length of the prompt is {prompt_length}, and the `max_new_tokens` "
    f"{max_new_tokens}. Thus, the combined length of the prompt "
    f"and `max_new_tokens` is: {combined_length}. This exceeds the "
    f"`max_length` of the Whisper model: {max_length}. "
    "You should either reduce the length of your prompt, or "
    "reduce the value of `max_new_tokens`, "
    f"so that their combined length is less that {max_length}."
  )


class GenerationStrategies:
  """Handles various generation strategies and temperature fallback logic."""

  def __init__(self, max_length: int, time_precision: float):
    """Initialize the generation strategies handler.

    :param max_length: Maximum sequence length supported by the model.
    :type max_length: int
    :param time_precision: Time precision for timestamp calculations.
    :type time_precision: float
    """
    self.max_length = max_length
    self.time_precision = time_precision
    self.logger = get_logger("whisper.generation")

  def generate_with_fallback(
    self,
    model: ctranslate2.models.Whisper,
    encoder_output: ctranslate2.StorageView,
    prompt: list[int],
    tokenizer: Tokenizer,
    options: TranscriptionOptions,
  ) -> GenerationResult:
    """Generate transcription with temperature fallback strategy.

    Tries multiple temperatures to find the best balance between quality metrics
    like compression ratio and log probability. Falls back to higher temperatures
    if quality thresholds are not met.

    :param model: The CTranslate2 Whisper model.
    :type model: ctranslate2.models.Whisper
    :param encoder_output: Encoded audio features.
    :type encoder_output: ctranslate2.StorageView
    :param prompt: Token sequence to use as prompt.
    :type prompt: list[int]
    :param tokenizer: The Whisper tokenizer.
    :type tokenizer: Tokenizer
    :param options: Transcription configuration options.
    :type options: TranscriptionOptions
    :returns: GenerationResult with the best transcription attempt and quality metrics.
    :rtype: GenerationResult
    :raises ValueError: If prompt + max_new_tokens exceeds model's max_length.
    """
    decode_result = None
    all_results = []
    below_cr_threshold_results = []

    max_initial_timestamp_index = int(round(options.max_initial_timestamp / self.time_precision))

    # Calculate maximum generation length
    if options.max_new_tokens is not None:
      max_length = len(prompt) + options.max_new_tokens
    else:
      max_length = self.max_length

    if max_length > self.max_length:
      raise ValueError(
        _create_max_length_error(len(prompt), max_length - len(prompt), max_length, self.max_length)
      )

    temperature = 0.0  # Initialize to avoid unbound variable warning

    # TEMPERATURE FALLBACK STRATEGY:
    # We try each temperature in sequence until we get acceptable quality results.
    # - temperature = 0.0: Deterministic beam search for highest quality/consistency
    # - temperature > 0.0: Sampling with randomness to avoid repetition/loops
    # Quality is measured by compression ratio and average log probability thresholds.
    for temperature in options.temperatures:
      self.logger.debug(f"Trying temperature: {temperature}")

      # Configure generation parameters based on temperature
      if temperature > 0:
        # For sampling (temperature > 0): Use single beam with multiple hypotheses.
        # This allows exploring different generation paths while maintaining randomness.
        # sampling_topk=0 means we sample from the full vocabulary distribution.
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

      # Generate transcription
      result = model.generate(
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

      # Calculate quality metrics
      seq_len = len(tokens)
      cum_logprob = result.scores[0] * (seq_len**options.length_penalty)
      avg_logprob = cum_logprob / (seq_len + 1)

      text = tokenizer.decode(tokens).strip()
      compression_ratio = get_compression_ratio(text)

      self.logger.debug(
        f"Generated text (temp={temperature}): '{text[:50]}...', "
        f"compression_ratio: {compression_ratio:.3f}, avg_logprob: {avg_logprob:.3f}"
      )
      self.logger.warn(
        "Transcription results",
        scores=result.scores,
        no_speech_prob=result.no_speech_prob,
        prob_type=type(result.no_speech_prob).__name__,
      )

      decode_result = GenerationResult(
        result=result,
        avg_logprob=avg_logprob,
        temperature=temperature,
        compression_ratio=compression_ratio,
      )
      all_results.append(decode_result)

      needs_fallback = False

      # Check compression ratio threshold
      if options.compression_ratio_threshold is not None:
        if compression_ratio > options.compression_ratio_threshold:
          needs_fallback = True  # too repetitive
          self.logger.info(
            "Compression ratio threshold is not met with temperature %.1f (%f > %f)",
            temperature,
            compression_ratio,
            options.compression_ratio_threshold,
          )
        else:
          below_cr_threshold_results.append(decode_result)

      # Check log probability threshold
      if options.log_prob_threshold is not None and avg_logprob < options.log_prob_threshold:
        needs_fallback = True  # average log probability is too low
        self.logger.info(
          "Log probability threshold is not met with temperature %.1f (%f < %f)",
          temperature,
          avg_logprob,
          options.log_prob_threshold,
        )

      # Special case: silence detection overrides low log prob
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
      # All temperatures failed, select the result with highest average log probability
      decode_result = max(below_cr_threshold_results or all_results, key=lambda x: x["avg_logprob"])
      # Update temperature to pass final value for prompt_reset_on_temperature
      decode_result = GenerationResult(
        result=decode_result["result"],
        avg_logprob=decode_result["avg_logprob"],
        temperature=temperature,
        compression_ratio=decode_result["compression_ratio"],
      )

    return decode_result
