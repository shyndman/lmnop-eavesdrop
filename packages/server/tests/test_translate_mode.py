"""Tests for translation-mode helpers: capability gate, language and timestamp resolution."""

import pytest

from eavesdrop.server.server import resolve_session_language
from eavesdrop.server.transcription.models import (
  ModelCapabilityError,
  check_translate_supported,
)
from eavesdrop.server.transcription.request_runner import resolve_word_timestamps
from eavesdrop.wire import TranscriptionTask


class TestCheckTranslateSupported:
  def test_translate_on_monolingual_model_raises(self) -> None:
    with pytest.raises(ModelCapabilityError) as exc_info:
      check_translate_supported(TranscriptionTask.TRANSLATE, False, "distil-medium.en")

    assert "distil-medium.en" in str(exc_info.value)

  def test_translate_on_multilingual_model_passes(self) -> None:
    check_translate_supported(TranscriptionTask.TRANSLATE, True, "large-v3")

  def test_transcribe_on_monolingual_model_passes(self) -> None:
    check_translate_supported(TranscriptionTask.TRANSCRIBE, False, "distil-medium.en")


class TestResolveSessionLanguage:
  def test_translate_silent_client_autodetects(self) -> None:
    assert resolve_session_language(TranscriptionTask.TRANSLATE, None, "en") is None

  def test_translate_explicit_default_kept(self) -> None:
    assert resolve_session_language(TranscriptionTask.TRANSLATE, "en", "en") == "en"

  def test_translate_explicit_language_kept(self) -> None:
    assert resolve_session_language(TranscriptionTask.TRANSLATE, "ja", "ja") == "ja"

  def test_transcribe_silent_client_keeps_merged(self) -> None:
    assert resolve_session_language(TranscriptionTask.TRANSCRIBE, None, "en") == "en"


class TestResolveWordTimestamps:
  def test_translate_suppresses_requested(self) -> None:
    assert resolve_word_timestamps(TranscriptionTask.TRANSLATE, True) is False

  def test_translate_keeps_false(self) -> None:
    assert resolve_word_timestamps(TranscriptionTask.TRANSLATE, False) is False

  def test_transcribe_keeps_requested(self) -> None:
    assert resolve_word_timestamps(TranscriptionTask.TRANSCRIBE, True) is True
