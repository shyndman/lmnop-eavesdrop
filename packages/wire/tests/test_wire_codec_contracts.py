"""Contract tests for wire codec behavior.

These tests lock down the wire protocol API boundary:
- every active message discriminator round-trips through JSON codecs
- invalid discriminators fail validation instead of being coerced
- decoded messages retain metadata fields clients depend on
"""

import json
from typing import TypedDict, cast

import pytest
from pydantic import ValidationError

from eavesdrop.wire import deserialize_message, serialize_message
from eavesdrop.wire.codec import Message
from eavesdrop.wire.messages import (
  DisconnectMessage,
  ErrorMessage,
  FlushControlMessage,
  HealthCheckRequest,
  LanguageDetectionMessage,
  RecordingStartedMessage,
  ServerReadyMessage,
  StreamStatusMessage,
  TranscriptionMessage,
  TranscriptionSetupMessage,
  UtteranceCancelledMessage,
)
from eavesdrop.wire.transcription import (
  Segment,
  TranscriptionSourceMode,
  UserTranscriptionOptions,
  Word,
  compute_segment_chain_id,
)


class _WordPayload(TypedDict):
  start: float
  end: float


class _SegmentPayload(TypedDict):
  start: float
  end: float
  words: list[_WordPayload]


class _TranscriptionWirePayload(TypedDict):
  segments: list[dict[str, object]]


def _build_contract_segment() -> Segment:
  """Build a deterministic segment fixture used to verify nested payload fidelity."""

  return Segment(
    id=4242,
    start=1.25,
    end=3.75,
    text="contract fixture",
    words=[Word(start=1.25, end=1.6, word="contract")],
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
      backend="parakeet",
    ),
    DisconnectMessage(
      timestamp=1_700_000_000.6,
      stream="stream-a",
      message="operator stop",
    ),
    RecordingStartedMessage(
      timestamp=1_700_000_000.625,
      stream="stream-a",
      recording_id="rec-123",
      sample_rate_hz=16000,
    ),
    FlushControlMessage(
      timestamp=1_700_000_000.65,
      stream="stream-a",
      recording_id="rec-123",
      force_complete=False,
    ),
    UtteranceCancelledMessage(
      timestamp=1_700_000_000.66,
      stream="stream-a",
      recording_id="rec-123",
    ),
    HealthCheckRequest(timestamp=1_700_000_000.7),
    TranscriptionSetupMessage(
      timestamp=1_700_000_000.8,
      stream="stream-a",
      options=UserTranscriptionOptions(
        source_mode=TranscriptionSourceMode.FILE,
        send_last_n_segments=5,
        word_timestamps=True,
      ),
    ),
  ],
)
def test_active_message_types_round_trip_without_payload_loss(message: Message) -> None:
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
      language="en",
      recording_id="rec-456",
      flush_complete=True,
    )
  )

  payload = cast(_TranscriptionWirePayload, json.loads(encoded))
  assert "time_offset" not in payload["segments"][0]
  assert "absolute_start_time" not in payload["segments"][0]
  assert "absolute_end_time" not in payload["segments"][0]

  decoded = deserialize_message(encoded)

  assert decoded.type == "transcription"
  assert decoded.timestamp == 1_700_000_001.0
  assert decoded.stream == "stream-1"
  assert decoded.language == "en"
  assert decoded.recording_id == "rec-456"
  assert decoded.flush_complete is True
  assert decoded.segments[0].id == 4242
  assert decoded.segments[0].text == "contract fixture"
  assert decoded.segments[0].time_offset == 0.0
  assert decoded.segments[0].completed is True
  assert decoded.segments[0].words is not None
  assert decoded.segments[0].words[0].word == "contract"


def test_decode_preserves_control_message_metadata_fields() -> None:
  encoded = serialize_message(
    TranscriptionSetupMessage(
      timestamp=1_700_000_002.0,
      stream="stream-2",
      options=UserTranscriptionOptions(
        source_mode=TranscriptionSourceMode.LIVE,
        send_last_n_segments=2,
        word_timestamps=True,
        language="en",
      ),
    )
  )

  decoded = deserialize_message(encoded)

  assert decoded.type == "setup"
  assert decoded.timestamp == 1_700_000_002.0
  assert decoded.stream == "stream-2"
  assert decoded.options.send_last_n_segments == 2
  assert decoded.options.source_mode == TranscriptionSourceMode.LIVE
  assert decoded.options.word_timestamps is True
  assert decoded.options.language == "en"


def test_flush_control_round_trips_without_payload_loss() -> None:
  encoded = serialize_message(
    FlushControlMessage(
      timestamp=1_700_000_002.5,
      stream="stream-flush",
      recording_id="rec-flush",
      force_complete=False,
    )
  )

  decoded = deserialize_message(encoded)

  assert serialize_message(decoded) == encoded


def test_utterance_cancel_control_round_trips_without_payload_loss() -> None:
  encoded = serialize_message(
    UtteranceCancelledMessage(
      timestamp=1_700_000_002.6,
      stream="stream-cancel",
      recording_id="rec-cancel",
    )
  )

  decoded = deserialize_message(encoded)

  assert serialize_message(decoded) == encoded


def test_ordinary_transcription_omits_flush_complete_on_wire() -> None:
  encoded = serialize_message(
    TranscriptionMessage(
      timestamp=1_700_000_006.0,
      stream="stream-ordinary",
      segments=[_build_contract_segment()],
      language="en",
    )
  )

  assert '"flush_complete"' not in encoded
  _ = deserialize_message(encoded)


def test_live_messages_omit_none_recording_id_on_wire() -> None:
  encoded = serialize_message(
    TranscriptionMessage(
      timestamp=1_700_000_006.5,
      stream="stream-ordinary",
      segments=[_build_contract_segment()],
      language="en",
    )
  )

  assert '"recording_id"' not in encoded
  decoded = deserialize_message(encoded)
  assert isinstance(decoded, TranscriptionMessage)
  assert decoded.recording_id is None


def test_wire_payload_records_and_words_use_same_recording_timeline() -> None:
  encoded = serialize_message(
    TranscriptionMessage(
      timestamp=1_700_000_008.0,
      stream="stream-words",
      segments=[
        Segment(
          id=9001,
          start=22.0,
          end=24.0,
          text="words on timeline",
          words=[
            Word(
              start=22.1,
              end=22.4,
              word="first",
            )
          ],
          time_offset=5.0,
          completed=True,
        )
      ],
    )
  )

  payload = cast(dict[str, list[_SegmentPayload]], json.loads(encoded))["segments"][0]

  assert "time_offset" not in payload
  assert payload["start"] == pytest.approx(22.0)
  assert payload["end"] == pytest.approx(24.0)
  assert payload["words"][0]["start"] == pytest.approx(22.1)
  assert payload["words"][0]["end"] == pytest.approx(22.4)
  assert "absolute_start_time" not in payload["words"][0]
  assert "absolute_end_time" not in payload["words"][0]


def test_flush_satisfying_transcription_serializes_flush_complete_true() -> None:
  encoded = serialize_message(
    TranscriptionMessage(
      timestamp=1_700_000_007.0,
      stream="stream-flush",
      segments=[_build_contract_segment()],
      flush_complete=True,
    )
  )

  assert '"flush_complete":true' in encoded
  _ = deserialize_message(encoded)


def test_transcription_segment_words_none_round_trips_when_omitted() -> None:
  encoded = serialize_message(
    TranscriptionMessage(
      timestamp=1_700_000_007.5,
      stream="stream-no-words",
      segments=[
        Segment(
          id=5150,
          start=1.0,
          end=1.0,
          text="",
          completed=False,
        )
      ],
      flush_complete=True,
    )
  )

  assert '"words"' not in encoded
  decoded = deserialize_message(encoded)

  assert decoded.type == "transcription"
  assert decoded.segments[0].words is None


def test_deserialize_rejects_unknown_discriminator_type() -> None:
  payload = '{"type":"unknown_event","timestamp":1700000003.0}'

  with pytest.raises(ValidationError) as exc_info:
    _ = deserialize_message(payload)

  assert "type" in str(exc_info.value)


def test_deserialize_rejects_non_string_discriminator_type() -> None:
  payload = '{"type":123,"timestamp":1700000004.0}'

  with pytest.raises(ValidationError) as exc_info:
    _ = deserialize_message(payload)

  assert "type" in str(exc_info.value)


def test_setup_options_default_source_mode_to_live_when_field_omitted() -> None:
  payload = (
    '{"type":"setup","timestamp":1700000005.0,"stream":"stream-3",'
    '"options":{"send_last_n_segments":1}}'
  )

  decoded = deserialize_message(payload)

  assert decoded.type == "setup"
  assert decoded.options.source_mode == TranscriptionSourceMode.LIVE


def test_setup_defaults_enable_native_word_timestamps() -> None:
  """An omitted options payload uses English and native word timings."""
  decoded = deserialize_message('{"type":"setup","stream":"default","options":{}}')

  assert decoded.type == "setup"
  assert decoded.options.word_timestamps is True
  assert decoded.options.language is None


@pytest.mark.parametrize("language", [None, "en"])
@pytest.mark.parametrize("word_timestamps", [True, False])
def test_setup_accepts_english_compatibility_hint_and_word_timestamp_toggle(
  language: str | None, word_timestamps: bool
) -> None:
  """Supported options survive JSON validation without changing their meaning."""
  payload = json.dumps(
    {
      "type": "setup",
      "stream": "english",
      "options": {"language": language, "word_timestamps": word_timestamps},
    }
  )

  decoded = deserialize_message(payload)

  assert decoded.type == "setup"
  assert decoded.options.language == language
  assert decoded.options.word_timestamps is word_timestamps


@pytest.mark.parametrize(
  "field,value",
  [
    ("initial_prompt", "preserve names"),
    ("hotwords", ["Parakeet"]),
    ("beam_size", 3),
    ("model", "other-model"),
    ("task", "translate"),
    ("task", "transcribe"),
    ("unknown_option", True),
  ],
)
def test_setup_rejects_removed_or_unknown_options(field: str, value: object) -> None:
  """Obsolete decoder knobs fail visibly instead of being silently discarded."""
  payload = json.dumps({"type": "setup", "stream": "legacy", "options": {field: value}})

  with pytest.raises(ValidationError, match=field):
    _ = deserialize_message(payload)


@pytest.mark.parametrize("language", ["ja", "fr", "EN", "auto", "", 1, True])
def test_setup_rejects_non_english_language(language: object) -> None:
  """The compatibility language field cannot request detection or another language."""
  payload = json.dumps(
    {"type": "setup", "stream": "invalid-language", "options": {"language": language}}
  )

  with pytest.raises(ValidationError, match="language"):
    _ = deserialize_message(payload)


@pytest.mark.parametrize("word_timestamps", [None, "true", "false", 1, 0])
def test_setup_requires_boolean_word_timestamp_option(word_timestamps: object) -> None:
  """Null and coercible values are invalid for the native-word output toggle."""
  payload = json.dumps(
    {"type": "setup", "stream": "invalid-words", "options": {"word_timestamps": word_timestamps}}
  )

  with pytest.raises(ValidationError, match="word_timestamps"):
    _ = deserialize_message(payload)


def test_segment_serializes_native_timestamps_without_decoder_metadata() -> None:
  """The wire output contains transcript data without invented confidence scores."""
  segment = Segment(
    id=42,
    start=1.25,
    end=2.0,
    text="native words",
    words=[Word(start=1.25, end=1.75, word="native")],
    completed=True,
  )

  assert cast(dict[str, object], json.loads(segment.model_dump_json())) == {
    "id": 42,
    "start": 1.25,
    "end": 2.0,
    "text": "native words",
    "words": [{"start": 1.25, "end": 1.75, "word": "native"}],
    "completed": True,
    "duration": 0.75,
  }


def test_segment_completion_preserves_chain_ids_and_recording_timestamps() -> None:
  """Completion keeps the existing chained ID algorithm and recording timeline."""
  first = Segment(id=0, start=10.0, end=10.5, text="first")
  second = Segment(id=0, start=10.5, end=11.0, text="second")

  first.mark_completed(None)
  second.mark_completed(first)

  assert first.id == compute_segment_chain_id(compute_segment_chain_id(0, ""), "first")
  assert second.id == compute_segment_chain_id(first.id, "second")
  assert first.completed is True
  assert second.completed is True
  assert first.absolute_start_time == 10.0
  assert first.absolute_end_time == 10.5
  assert second.absolute_start_time == 10.5
  assert second.absolute_end_time == 11.0
