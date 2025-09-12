"""Prompt construction utilities for Whisper transcription.

This module handles the construction of prompts for Whisper model inference,
including context from previous transcriptions, hotwords, and prefixes to
improve transcription accuracy across audio boundaries.
"""

from faster_whisper.tokenizer import Tokenizer


class PromptBuilder:
  """Constructs prompts for Whisper model inference with context and constraints."""

  def __init__(self, max_length: int):
    """Initialize the prompt builder.

    :param max_length: Maximum sequence length supported by the model.
    :type max_length: int
    """
    self.max_length = max_length
    self._max_context_len = max_length // 2 - 1

  def build_prompt(
    self,
    tokenizer: Tokenizer,
    previous_tokens: list[int],
    without_timestamps: bool = False,
    prefix: str | None = None,
    hotwords: str | None = None,
  ) -> list[int]:
    """Constructs the prompt for the Whisper model.

    The prompt provides context from previous transcriptions to improve the
    accuracy of the next segment, especially across hard audio boundaries.

    :param tokenizer: The Whisper tokenizer.
    :type tokenizer: Tokenizer
    :param previous_tokens: A list of tokens from previously transcribed segments.
    :type previous_tokens: list[int]
    :param without_timestamps: Whether to exclude timestamp tokens from the prompt.
    :type without_timestamps: bool
    :param prefix: An optional prefix string to force the transcription to start with.
    :type prefix: str | None
    :param hotwords: Optional hotwords to provide as context.
    :type hotwords: str | None
    :returns: A list of tokens representing the constructed prompt.
    :rtype: list[int]
    """
    prompt: list[int] = []

    # Add previous context (tokens from prior segments and/or hotwords)

    using_hotwords = bool(hotwords and not prefix)
    if previous_tokens or using_hotwords:
      self._append_previous(tokenizer, previous_tokens, hotwords, prompt, using_hotwords)

    # Add the core start-of-transcription sequence
    prompt.extend(tokenizer.sot_sequence)

    # Add timestamp-related tokens
    if without_timestamps:
      prompt.append(tokenizer.no_timestamps)

    # Add prefix if provided
    if prefix:
      self._append_prefix(tokenizer, without_timestamps, prefix, prompt)

    return prompt

  def encode_initial_prompt(self, tokenizer: Tokenizer, initial_prompt: str | None) -> list[int]:
    """Encode the initial prompt text into tokens.

    :param tokenizer: The Whisper tokenizer for encoding the prompt text.
    :type tokenizer: Tokenizer
    :param initial_prompt: Optional initial prompt text to encode.
    :type initial_prompt: str | None
    :returns: List of encoded tokens from the initial prompt, or empty list if None.
    :rtype: list[int]
    """
    return tokenizer.encode(" " + initial_prompt.strip()) if initial_prompt else []

  def _append_previous(
    self,
    tokenizer: Tokenizer,
    previous_tokens: list[int],
    hotwords: str | None,
    prompt: list[int],
    using_hotwords: bool,
  ) -> None:
    prompt.append(tokenizer.sot_prev)

    if using_hotwords:
      assert hotwords
      hotwords_tokens = tokenizer.encode(" " + hotwords.strip())
      # Truncate if longer than max allowed context
      if len(hotwords_tokens) > self._max_context_len:
        hotwords_tokens = hotwords_tokens[: self._max_context_len]
      prompt.extend(hotwords_tokens)

    if previous_tokens:
      # Use the last N tokens as context
      prompt.extend(previous_tokens[-self._max_context_len :])

  def _append_prefix(
    self,
    tokenizer: Tokenizer,
    without_timestamps: bool,
    prefix: str,
    prompt: list[int],
  ) -> None:
    prefix_tokens = tokenizer.encode(" " + prefix.strip())
    # Truncate if longer than max allowed context
    if len(prefix_tokens) > self._max_context_len:
      prefix_tokens = prefix_tokens[: self._max_context_len]

    if not without_timestamps:
      prompt.append(tokenizer.timestamp_begin)

    prompt.extend(prefix_tokens)
