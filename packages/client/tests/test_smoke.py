"""Smoke tests for the supported live client factory."""

from eavesdrop.client.core import EavesdropClient


def test_transcriber_factory_preserves_language_configuration() -> None:
  client = EavesdropClient.transcriber(audio_device="default", language="en")

  assert client._transcription_options.language == "en"
