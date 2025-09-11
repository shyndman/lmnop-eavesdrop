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

    Args:
        max_length: Maximum sequence length supported by the model.
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

    Args:
        tokenizer: The Whisper tokenizer.
        previous_tokens: A list of tokens from previously transcribed segments.
        without_timestamps: Whether to exclude timestamp tokens from the prompt.
        prefix: An optional prefix string to force the transcription to start with.
        hotwords: Optional hotwords to provide as context.

    Returns:
        A list of tokens representing the constructed prompt.
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
