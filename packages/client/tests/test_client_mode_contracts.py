"""Contract tests for mode-specific client runtime behavior.

These tests pin the public client factories to wire-level behavior so mode-specific
configuration cannot silently drift across connection setup and streaming control flows.
"""

import asyncio
import json
from typing import cast

import pytest

from eavesdrop.client.connection import WebSocketConnection
from eavesdrop.client.core import EavesdropClient
from eavesdrop.wire import (
  ClientType,
  Segment,
  TranscriptionMessage,
  deserialize_message,
)


class FakeWebSocket:
  """Deterministic in-memory websocket stand-in used by connect-path tests."""

  def __init__(self) -> None:
    self.sent_payloads: list[str | bytes] = []
    self.closed = False

  async def send(self, payload: str | bytes) -> None:
    self.sent_payloads.append(payload)

  async def close(self) -> None:
    self.closed = True

  def __aiter__(self) -> "FakeWebSocket":
    return self

  async def __anext__(self) -> str:
    raise StopAsyncIteration


class RecordingAudioCapture:
  """Tracks start/stop transitions without touching real audio devices."""

  def __init__(self) -> None:
    self.start_calls = 0
    self.stop_calls = 0

  def start_recording(self) -> None:
    self.start_calls += 1

  def stop_recording(self) -> None:
    self.stop_calls += 1

  async def get_audio_data(self, timeout: float = 0.1) -> bytes | None:
    return None


class RecordingConnection:
  """Typed connection test double for the streaming loop entrypoint."""

  def __init__(self) -> None:
    self.audio_chunks: list[bytes] = []

  async def send_audio_data(self, audio_data: bytes) -> None:
    self.audio_chunks.append(audio_data)


def _segment(text: str) -> Segment:
  """Build a minimal valid segment payload for message routing assertions."""

  return Segment(
    id=1,
    seek=0,
    start=0.0,
    end=0.5,
    text=text,
    tokens=[1, 2],
    avg_logprob=-0.1,
    compression_ratio=1.0,
    words=None,
    temperature=None,
  )


@pytest.mark.asyncio
async def test_transcriber_factory_options_reach_setup_payload(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Factory options must survive all layers and appear in setup wire payload."""

  fake_ws = FakeWebSocket()
  captured_headers: dict[str, str] = {}

  async def fake_connect(_url: str, additional_headers: dict[str, str]) -> FakeWebSocket:
    captured_headers.update(additional_headers)
    return fake_ws

  monkeypatch.setattr("eavesdrop.client.connection.websockets.connect", fake_connect)
  monkeypatch.setattr("eavesdrop.client.core.secrets.token_hex", lambda _n: "beef")

  client = EavesdropClient.transcriber(
    audio_device="default",
    word_timestamps=True,
    initial_prompt="callsign",
    hotwords=["orion", "apollo"],
    send_last_n_segments=3,
    beam_size=5,
    model="distil-small.en",
  )

  await client.connect()

  assert captured_headers == {"X-Client-Type": "transcriber"}
  assert len(fake_ws.sent_payloads) == 1

  serialized_setup = cast(str, fake_ws.sent_payloads[0])
  assert serialized_setup

  setup_message = deserialize_message(serialized_setup)
  assert setup_message.type == "setup"
  assert setup_message.stream == "tbeef"
  assert setup_message.options.word_timestamps is True
  assert setup_message.options.initial_prompt == "callsign"
  assert setup_message.options.hotwords == ["orion", "apollo"]
  assert setup_message.options.send_last_n_segments == 3
  assert setup_message.options.beam_size == 5
  assert setup_message.options.model == "distil-small.en"

  await client.disconnect()
  assert fake_ws.closed is True


@pytest.mark.asyncio
async def test_subscriber_connect_sends_stream_subscription_header(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Subscriber mode must route stream subscriptions via websocket headers only."""

  fake_ws = FakeWebSocket()
  captured_headers: dict[str, str] = {}

  async def fake_connect(_url: str, additional_headers: dict[str, str]) -> FakeWebSocket:
    captured_headers.update(additional_headers)
    return fake_ws

  monkeypatch.setattr("eavesdrop.client.connection.websockets.connect", fake_connect)

  connection = WebSocketConnection(
    host="localhost",
    port=9090,
    stream_name="s-test",
    on_ready=lambda _backend: None,
    on_transcription=lambda _text: None,
    on_error=lambda _message: None,
    client_type=ClientType.RTSP_SUBSCRIBER,
    stream_names=["cam-a", "cam-b"],
  )

  await connection.connect()

  assert captured_headers == {
    "X-Client-Type": "rtsp_subscriber",
    "X-Stream-Names": "cam-a,cam-b",
  }
  assert fake_ws.sent_payloads == []


@pytest.mark.asyncio
async def test_subscriber_routes_only_messages_for_subscribed_streams() -> None:
  """Subscriber routing must emit callbacks only for configured stream names."""

  received_messages: list[TranscriptionMessage] = []
  received_text: list[str] = []
  received_errors: list[str] = []

  connection = WebSocketConnection(
    host="localhost",
    port=9090,
    stream_name="subscriber-session",
    on_ready=lambda _backend: None,
    on_transcription=received_text.append,
    on_error=received_errors.append,
    client_type=ClientType.RTSP_SUBSCRIBER,
    stream_names=["cam-a"],
    on_transcription_message=received_messages.append,
  )

  subscribed_payload = json.dumps(
    {
      "type": "transcription",
      "stream": "cam-a",
      "segments": [_segment(" target ").model_dump()],
      "language": "en",
    }
  )
  unsubscribed_payload = json.dumps(
    {
      "type": "transcription",
      "stream": "cam-b",
      "segments": [_segment("ignored").model_dump()],
      "language": "en",
    }
  )

  await connection._process_message(subscribed_payload)
  await connection._process_message(unsubscribed_payload)

  assert [message.stream for message in received_messages] == ["cam-a"]
  assert received_text == ["[cam-a] target"]
  assert received_errors == []


def test_transcriber_mode_requires_audio_device_configuration() -> None:
  """Mode-specific validation must fail fast for invalid transcriber input."""

  with pytest.raises(ValueError, match="audio device"):
    EavesdropClient(client_type=ClientType.TRANSCRIBER)


@pytest.mark.asyncio
async def test_start_and_stop_streaming_transition_state_with_mocks(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Streaming start/stop paths must be idempotent and update state predictably."""

  client = EavesdropClient.transcriber(audio_device="default")
  client._connected = True
  client._audio_capture = RecordingAudioCapture()
  client._connection = RecordingConnection()

  loop_started = asyncio.Event()

  async def fake_audio_streaming_loop() -> None:
    loop_started.set()

  monkeypatch.setattr(client, "_audio_streaming_loop", fake_audio_streaming_loop)

  await client.start_streaming()
  await asyncio.wait_for(loop_started.wait(), timeout=0.2)

  assert client.is_streaming() is True
  assert client._audio_capture.start_calls == 1

  await client.start_streaming()
  assert client._audio_capture.start_calls == 1

  await client.stop_streaming()
  assert client.is_streaming() is False
  assert client._audio_capture.stop_calls == 1

  await client.stop_streaming()
  assert client._audio_capture.stop_calls == 1
