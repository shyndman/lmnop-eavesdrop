"""Contract tests for wire codec behavior.

These tests lock down the wire protocol API boundary:
- every active message discriminator round-trips through JSON codecs
- invalid discriminators fail validation instead of being coerced
- decoded messages retain metadata fields clients depend on
"""

import pytest
from pydantic import ValidationError

from eavesdrop.wire import deserialize_message, serialize_message
from eavesdrop.wire.messages import (
  DisconnectMessage,
  ErrorMessage,
  HealthCheckRequest,
  LanguageDetectionMessage,
  ServerReadyMessage,
  StreamStatusMessage,
  TranscriptionMessage,
  TranscriptionSetupMessage,
)
from eavesdrop.wire.transcription import (
  Segment,
  TranscriptionSourceMode,
  UserTranscriptionOptions,
  Word,
)


def _build_contract_segment() -> Segment:
  """Build a deterministic segment fixture used to verify nested payload fidelity."""

  return Segment(
    id=4242,
    seek=160,
    start=1.25,
    end=3.75,
    text="contract fixture",
    tokens=[101, 202],
    avg_logprob=-0.35,
    compression_ratio=1.1,
    words=[Word(start=1.25, end=1.6, word="contract", probability=0.99)],
    temperature=0.0,
    time_offset=32.0,
    completed=True,
  )


@pytest.mark.parametrize(
  "message",
  [
    TranscriptionMessage(
      timestamp=1_700_000_000.1,
      stream="stream-a",
      segments=[_build_contract_segment()],
      language="en",
    ),
    StreamStatusMessage(
      timestamp=1_700_000_000.2,
      stream="stream-a",
      status="online",
      message="ready",
    ),
    ErrorMessage(
      timestamp=1_700_000_000.3,
      stream="stream-a",
      message="stream failed",
    ),
    LanguageDetectionMessage(
      timestamp=1_700_000_000.4,
      stream="stream-a",
      language="en",
      language_prob=0.87,
    ),
    ServerReadyMessage(
      timestamp=1_700_000_000.5,
      stream="stream-a",
      backend="faster_whisper",
    ),
    DisconnectMessage(
      timestamp=1_700_000_000.6,
      stream="stream-a",
      message="operator stop",
    ),
    HealthCheckRequest(timestamp=1_700_000_000.7),
    TranscriptionSetupMessage(
      timestamp=1_700_000_000.8,
      stream="stream-a",
      options=UserTranscriptionOptions(
        source_mode=TranscriptionSourceMode.FILE,
        send_last_n_segments=5,
        initial_prompt="keep names exact",
        hotwords=["Alpha", "Bravo"],
        word_timestamps=True,
        beam_size=3,
        model="distil-small.en",
      ),
    ),
  ],
)
def test_active_message_types_round_trip_without_payload_loss(message: object) -> None:
  encoded = serialize_message(message)
  decoded = deserialize_message(encoded)

  # String-level parity guarantees no field was dropped or rewritten across the wire contract.
  assert serialize_message(decoded) == encoded


def test_decode_preserves_transcription_metadata_fields() -> None:
  encoded = serialize_message(
    TranscriptionMessage(
      timestamp=1_700_000_001.0,
      stream="stream-1",
      segments=[_build_contract_segment()],
      language="fr",
    )
  )

  decoded = deserialize_message(encoded)

  assert decoded.type == "transcription"
  assert decoded.timestamp == 1_700_000_001.0
  assert decoded.stream == "stream-1"
  assert decoded.language == "fr"
  assert decoded.segments[0].id == 4242
  assert decoded.segments[0].text == "contract fixture"
  assert decoded.segments[0].time_offset == 32.0
  assert decoded.segments[0].completed is True
  assert decoded.segments[0].words[0].word == "contract"


def test_decode_preserves_control_message_metadata_fields() -> None:
  encoded = serialize_message(
    TranscriptionSetupMessage(
      timestamp=1_700_000_002.0,
      stream="stream-2",
      options=UserTranscriptionOptions(
        source_mode=TranscriptionSourceMode.LIVE,
        send_last_n_segments=2,
        initial_prompt="spell hospital names correctly",
        hotwords=["General", "Memorial"],
        word_timestamps=True,
        beam_size=2,
        model="distil-small.en",
      ),
    )
  )

  decoded = deserialize_message(encoded)

  assert decoded.type == "setup"
  assert decoded.timestamp == 1_700_000_002.0
  assert decoded.stream == "stream-2"
  assert decoded.options.send_last_n_segments == 2
  assert decoded.options.source_mode == TranscriptionSourceMode.LIVE
  assert decoded.options.initial_prompt == "spell hospital names correctly"
  assert decoded.options.hotwords == ["General", "Memorial"]
  assert decoded.options.word_timestamps is True
  assert decoded.options.beam_size == 2
  assert decoded.options.model == "distil-small.en"


def test_deserialize_rejects_unknown_discriminator_type() -> None:
  payload = '{"type":"unknown_event","timestamp":1700000003.0}'

  with pytest.raises(ValidationError) as exc_info:
    deserialize_message(payload)

  assert "type" in str(exc_info.value)


def test_deserialize_rejects_non_string_discriminator_type() -> None:
  payload = '{"type":123,"timestamp":1700000004.0}'

  with pytest.raises(ValidationError) as exc_info:
    deserialize_message(payload)

  assert "type" in str(exc_info.value)


def test_setup_options_default_source_mode_to_live_when_field_omitted() -> None:
  payload = (
    '{"type":"setup","timestamp":1700000005.0,"stream":"stream-3",'
    '"options":{"send_last_n_segments":1}}'
  )

  decoded = deserialize_message(payload)

  assert decoded.type == "setup"
  assert decoded.options.source_mode == TranscriptionSourceMode.LIVE
