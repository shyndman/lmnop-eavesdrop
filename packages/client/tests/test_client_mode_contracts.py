"""Contract tests for mode-specific client runtime behavior.

These tests pin the public client factories to wire-level behavior so mode-specific
configuration cannot silently drift across connection setup and streaming control flows.
"""

import asyncio
import json
from typing import cast

import pytest
from structlog.stdlib import BoundLogger
from websockets.asyncio.client import ClientConnection

from eavesdrop.client.audio import AudioCapture
from eavesdrop.client.connection import WebSocketConnection
from eavesdrop.client.core import EavesdropClient
from eavesdrop.client.events import (
  ConnectedEvent,
  DisconnectedEvent,
  ReconnectedEvent,
  ReconnectingEvent,
  TranscriptionEvent,
)
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


class ScriptedWebSocket(FakeWebSocket):
  """Websocket double that replays payloads then raises scripted failures."""

  def __init__(self, script: list[str | Exception]) -> None:
    super().__init__()
    self._script = script

  async def __anext__(self) -> str:
    if not self._script:
      await asyncio.sleep(0)
      raise StopAsyncIteration

    next_item = self._script.pop(0)
    if isinstance(next_item, Exception):
      raise next_item
    return next_item


class BlockingWebSocket(FakeWebSocket):
  """Websocket double that stays open until the client closes it."""

  def __init__(self) -> None:
    super().__init__()
    self._closed_event = asyncio.Event()

  async def close(self) -> None:
    await super().close()
    self._closed_event.set()

  async def __anext__(self) -> str:
    await self._closed_event.wait()
    raise StopAsyncIteration


class RecordingLogger:
  """Captures client-boundary log records for reconnect assertions."""

  def __init__(self) -> None:
    self.warning_messages: list[str] = []
    self.info_messages: list[str] = []
    self.error_messages: list[str] = []
    self.exception_messages: list[str] = []

  def debug(self, _event: str, **_kwargs: object) -> None:
    pass

  def warning(self, event: str, **_kwargs: object) -> None:
    self.warning_messages.append(event)

  def info(self, event: str, **_kwargs: object) -> None:
    self.info_messages.append(event)

  def error(self, event: str, **_kwargs: object) -> None:
    self.error_messages.append(event)

  def exception(self, event: str, **_kwargs: object) -> None:
    self.exception_messages.append(event)


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


class ScriptedAudioCapture:
  """Feeds deterministic audio chunks into the client streaming loop."""

  def __init__(self, chunks: list[bytes], client: EavesdropClient) -> None:
    self._chunks = list(chunks)
    self._client = client

  def start_recording(self) -> None:
    return None

  def stop_recording(self) -> None:
    return None

  async def get_audio_data(self, timeout: float = 0.1) -> bytes | None:
    _ = timeout
    if self._chunks:
      return self._chunks.pop(0)

    self._client._streaming = False
    return None


class RecordingConnection:
  """Typed connection test double for the streaming loop entrypoint."""

  def __init__(self) -> None:
    self.audio_chunks: list[bytes] = []
    self.command_log: list[str] = []
    self.flush_commands: list[bool] = []
    self.flush_sent = asyncio.Event()
    self.cancel_requests = 0

  async def send_audio_data(self, audio_data: bytes) -> None:
    self.audio_chunks.append(audio_data)
    self.command_log.append("audio")

  async def send_flush_control(self, *, force_complete: bool = True) -> None:
    self.flush_commands.append(force_complete)
    self.command_log.append("flush")
    self.flush_sent.set()

  async def send_utterance_cancelled(self) -> None:
    self.cancel_requests += 1
    self.command_log.append("cancel")

  def is_connected(self) -> bool:
    return True


async def _hold_open(signal: asyncio.Event) -> None:
  """Keep an async task pending until the test explicitly releases it."""

  await signal.wait()


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
async def test_connection_serializes_utterance_cancel_control_message() -> None:
  """Live cancel control must serialize the new wire message discriminator."""

  fake_ws = FakeWebSocket()
  connection = WebSocketConnection(
    host="localhost",
    port=9090,
    stream_name="stream-cancel",
    on_ready=lambda _backend: None,
    on_transcription=lambda _text: None,
    on_error=lambda _message: None,
  )
  connection.ws = cast(ClientConnection, cast(object, fake_ws))
  connection.connected = True

  await connection.send_utterance_cancelled()

  assert len(fake_ws.sent_payloads) == 1
  encoded = cast(str, fake_ws.sent_payloads[0])
  decoded = deserialize_message(encoded)
  assert decoded.type == "control_utterance_cancelled"
  assert decoded.stream == "stream-cancel"


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
async def test_live_client_event_types_expose_expected_payloads() -> None:
  """Live client events must expose stable family names and typed payloads."""

  message = TranscriptionMessage(
    stream="stream-live",
    segments=[_segment("ready")],
    language="en",
  )

  connected = ConnectedEvent(stream="stream-live")
  disconnected = DisconnectedEvent(stream="stream-live", reason="socket lost")
  reconnecting = ReconnectingEvent(stream="stream-live", attempt=2, retry_delay_s=10.0)
  reconnected = ReconnectedEvent(stream="stream-live")
  transcription = TranscriptionEvent(stream="stream-live", message=message)

  assert connected.family == "connected"
  assert disconnected.reason == "socket lost"
  assert reconnecting.retry_delay_s == 10.0
  assert reconnected.family == "reconnected"
  assert transcription.message == message


@pytest.mark.asyncio
async def test_async_iterator_yields_connected_then_transcription_event(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The live async iterator must preserve connect-before-transcription ordering."""

  fake_ws = BlockingWebSocket()

  async def fake_connect(_url: str, additional_headers: dict[str, str]) -> BlockingWebSocket:
    assert additional_headers == {"X-Client-Type": "transcriber"}
    return fake_ws

  monkeypatch.setattr("eavesdrop.client.connection.websockets.connect", fake_connect)
  monkeypatch.setattr("eavesdrop.client.core.secrets.token_hex", lambda _n: "beef")

  client = EavesdropClient.transcriber(audio_device="default")
  await client.connect()

  collected_events: list[ConnectedEvent | TranscriptionEvent] = []

  async def _collect_two_events() -> None:
    async for event in client:
      collected_events.append(cast(ConnectedEvent | TranscriptionEvent, event))
      if len(collected_events) == 2:
        break

  collector = asyncio.create_task(_collect_two_events())
  client._on_transcription_message(
    TranscriptionMessage(
      stream="tbeef",
      segments=[_segment("hello")],
      language="en",
    )
  )

  try:
    await asyncio.wait_for(collector, timeout=0.2)
  finally:
    await client.disconnect()

  assert [event.family for event in collected_events] == ["connected", "transcription"]
  assert collected_events[0].stream == "tbeef"
  transcription_event = cast(TranscriptionEvent, collected_events[1])
  assert transcription_event.message.segments[0].text == "hello"


@pytest.mark.asyncio
async def test_live_transcriber_reconnects_with_truthful_events(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Socket loss must emit reconnect events, wait 10s, and log the transition truthfully."""

  first_ws = ScriptedWebSocket([RuntimeError("socket lost")])
  second_ws = BlockingWebSocket()
  connect_results = [first_ws, second_ws]

  async def fake_connect(_url: str, additional_headers: dict[str, str]) -> FakeWebSocket:
    assert additional_headers == {"X-Client-Type": "transcriber"}
    return connect_results.pop(0)

  original_sleep = asyncio.sleep
  recorded_sleeps: list[float] = []

  async def fake_sleep(delay: float) -> None:
    recorded_sleeps.append(delay)
    await original_sleep(0)

  monkeypatch.setattr("eavesdrop.client.connection.websockets.connect", fake_connect)
  monkeypatch.setattr("eavesdrop.client.core.asyncio.sleep", fake_sleep)
  monkeypatch.setattr("eavesdrop.client.core.secrets.token_hex", lambda _n: "beef")

  client = EavesdropClient.transcriber(audio_device="default")
  logger = RecordingLogger()
  client._logger = cast(BoundLogger, cast(object, logger))

  await client.connect()

  observed_events: list[str] = []

  async def _collect_reconnect_events() -> None:
    async for event in client:
      observed_events.append(event.family)
      if len(observed_events) == 4:
        break

  collector = asyncio.create_task(_collect_reconnect_events())
  try:
    await asyncio.wait_for(collector, timeout=0.5)
  finally:
    await client.disconnect()

  assert observed_events == ["connected", "disconnected", "reconnecting", "reconnected"]
  assert recorded_sleeps == [10.0]
  assert logger.warning_messages == [
    "live transcriber disconnected",
    "live transcriber reconnect scheduled",
  ]
  assert logger.info_messages == ["live transcriber reconnected"]


@pytest.mark.asyncio
async def test_streaming_tracks_dedicated_audio_loop_task(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Streaming must expose one dedicated audio-loop task and clear it after completion."""

  client = EavesdropClient.transcriber(audio_device="default")
  audio_capture = RecordingAudioCapture()
  connection = RecordingConnection()
  client._connected = True
  client._audio_capture = cast(AudioCapture, cast(object, audio_capture))
  client._connection = cast(WebSocketConnection, cast(object, connection))

  loop_started = asyncio.Event()
  allow_exit = asyncio.Event()

  async def fake_audio_streaming_loop() -> None:
    loop_started.set()
    await allow_exit.wait()

  monkeypatch.setattr(client, "_audio_streaming_loop", fake_audio_streaming_loop)

  await client.start_streaming()
  await asyncio.wait_for(loop_started.wait(), timeout=0.2)
  audio_loop_task = client._audio_loop_task

  assert client.is_streaming() is True
  assert audio_capture.start_calls == 1
  assert audio_loop_task is not None
  assert audio_loop_task.done() is False

  stop_task = asyncio.create_task(client.stop_streaming())
  await asyncio.sleep(0)

  assert client.is_streaming() is False
  assert audio_capture.stop_calls == 1
  assert stop_task.done() is False

  allow_exit.set()
  await stop_task
  assert audio_loop_task is not None
  await audio_loop_task
  assert client._audio_loop_task is None


@pytest.mark.asyncio
async def test_start_streaming_awaits_prior_loop_and_flush_stays_valid(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Restarting streaming must await the prior loop and still allow flush after stop."""

  client = EavesdropClient.transcriber(audio_device="default")
  audio_capture = RecordingAudioCapture()
  connection = RecordingConnection()
  client._connected = True
  client._audio_capture = cast(AudioCapture, cast(object, audio_capture))
  client._connection = cast(WebSocketConnection, cast(object, connection))

  entered_loops = 0
  concurrent_loops = 0
  max_concurrent_loops = 0
  release_current_loop = asyncio.Event()
  loop_ready = asyncio.Event()

  async def fake_audio_streaming_loop() -> None:
    nonlocal entered_loops, concurrent_loops, max_concurrent_loops, release_current_loop

    entered_loops += 1
    concurrent_loops += 1
    max_concurrent_loops = max(max_concurrent_loops, concurrent_loops)
    loop_ready.set()
    try:
      await release_current_loop.wait()
    finally:
      concurrent_loops -= 1

  monkeypatch.setattr(client, "_audio_streaming_loop", fake_audio_streaming_loop)

  await client.start_streaming()
  await asyncio.wait_for(loop_ready.wait(), timeout=0.2)
  first_task = client._audio_loop_task

  first_stop = asyncio.create_task(client.stop_streaming())
  await asyncio.sleep(0)

  loop_ready = asyncio.Event()
  second_start = asyncio.create_task(client.start_streaming())
  await asyncio.sleep(0)
  assert first_stop.done() is False
  assert second_start.done() is False
  assert audio_capture.start_calls == 1

  release_current_loop.set()
  await first_stop
  assert first_task is not None
  await first_task
  release_current_loop = asyncio.Event()
  await second_start
  await asyncio.wait_for(loop_ready.wait(), timeout=0.2)

  second_task = client._audio_loop_task
  assert second_task is not None

  second_stop = asyncio.create_task(client.stop_streaming())
  await asyncio.sleep(0)
  assert second_stop.done() is False
  release_current_loop.set()
  await second_stop
  await second_task

  keepalive = asyncio.Event()
  message_task = asyncio.create_task(_hold_open(keepalive))
  client._message_task = message_task

  async def _deliver_flush_response() -> None:
    await connection.flush_sent.wait()
    client._message_queue.put_nowait(
      TranscriptionMessage(
        stream="stream-live",
        segments=[_segment("done")],
        language="en",
        flush_complete=True,
      )
    )

  deliver_task = asyncio.create_task(_deliver_flush_response())
  try:
    result = await client.flush(force_complete=False)
  finally:
    keepalive.set()
    await asyncio.gather(message_task, deliver_task)

  assert entered_loops == 2
  assert max_concurrent_loops == 1
  assert audio_capture.start_calls == 2
  assert audio_capture.stop_calls == 2
  assert client._audio_loop_task is None
  assert connection.flush_commands == [False]
  assert result.flush_complete is True


@pytest.mark.asyncio
async def test_cancel_utterance_waits_for_stream_stop_before_sending_control(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Cancel must wait for the trailing audio loop to finish before sending control."""

  client = EavesdropClient.transcriber(audio_device="default")
  audio_capture = RecordingAudioCapture()
  connection = RecordingConnection()
  client._connected = True
  client._audio_capture = cast(AudioCapture, cast(object, audio_capture))
  client._connection = cast(WebSocketConnection, cast(object, connection))
  client._message_task = asyncio.create_task(_hold_open(asyncio.Event()))

  loop_started = asyncio.Event()
  release_tail = asyncio.Event()

  async def fake_audio_streaming_loop() -> None:
    loop_started.set()
    await release_tail.wait()
    await connection.send_audio_data(b"tail")

  monkeypatch.setattr(client, "_audio_streaming_loop", fake_audio_streaming_loop)

  await client.start_streaming()
  await asyncio.wait_for(loop_started.wait(), timeout=0.2)

  cancel_task = asyncio.create_task(client.cancel_utterance())
  await asyncio.sleep(0)

  assert cancel_task.done() is False
  assert connection.cancel_requests == 0
  assert audio_capture.stop_calls == 1

  release_tail.set()
  await cancel_task

  assert client.is_streaming() is False
  assert connection.audio_chunks == [b"tail"]
  assert connection.cancel_requests == 1
  assert connection.command_log == ["audio", "cancel"]

  client._message_task.cancel()
  await asyncio.gather(client._message_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_audio_streaming_loop_delivers_capture_callback_in_send_order() -> None:
  """Capture callbacks must receive the exact sent bytes after websocket send order."""

  received_chunks: list[bytes] = []
  client = EavesdropClient.transcriber(
    audio_device="default",
    on_capture=received_chunks.append,
  )
  connection = RecordingConnection()
  client._connected = True
  client._streaming = True
  client._connection = cast(WebSocketConnection, cast(object, connection))
  client._audio_capture = cast(
    AudioCapture,
    cast(object, ScriptedAudioCapture([b"alpha", b"bravo"], client)),
  )

  await client._audio_streaming_loop()

  assert connection.audio_chunks == [b"alpha", b"bravo"]
  assert received_chunks == connection.audio_chunks


@pytest.mark.asyncio
async def test_audio_streaming_loop_skips_capture_callback_when_unset() -> None:
  """Streaming must remain a no-op for capture delivery when no callback is configured."""

  client = EavesdropClient.transcriber(audio_device="default")
  connection = RecordingConnection()
  client._connected = True
  client._streaming = True
  client._connection = cast(WebSocketConnection, cast(object, connection))
  client._audio_capture = cast(
    AudioCapture,
    cast(object, ScriptedAudioCapture([b"alpha"], client)),
  )

  await client._audio_streaming_loop()

  assert connection.audio_chunks == [b"alpha"]


@pytest.mark.asyncio
async def test_flush_waits_for_new_flush_complete_message() -> None:
  """flush() must drop stale buffered completions and return the new flush response."""

  client = EavesdropClient.transcriber(audio_device="default")
  connection = RecordingConnection()
  client._connected = True
  client._connection = cast(WebSocketConnection, cast(object, connection))

  stale_message = TranscriptionMessage(
    stream="stream-live",
    segments=[_segment("stale")],
    language="en",
    flush_complete=True,
  )
  fresh_message = TranscriptionMessage(
    stream="stream-live",
    segments=[_segment("fresh")],
    language="en",
    flush_complete=True,
  )
  client._message_queue.put_nowait(stale_message)

  keepalive = asyncio.Event()
  message_task = asyncio.create_task(_hold_open(keepalive))
  client._message_task = message_task

  async def _deliver_fresh_message() -> None:
    await connection.flush_sent.wait()
    client._message_queue.put_nowait(fresh_message)

  deliver_task = asyncio.create_task(_deliver_fresh_message())
  try:
    result = await client.flush(force_complete=True)
  finally:
    keepalive.set()
    await asyncio.gather(message_task, deliver_task)

  assert connection.flush_commands == [True]
  assert result == fresh_message


@pytest.mark.asyncio
async def test_flush_rejects_second_local_waiter_before_second_send() -> None:
  """flush() must fail fast locally when another flush call is already pending."""

  client = EavesdropClient.transcriber(audio_device="default")
  connection = RecordingConnection()
  client._connected = True
  client._connection = cast(WebSocketConnection, cast(object, connection))

  keepalive = asyncio.Event()
  message_task = asyncio.create_task(_hold_open(keepalive))
  client._message_task = message_task

  first_flush = asyncio.create_task(client.flush())
  await connection.flush_sent.wait()

  with pytest.raises(RuntimeError, match="already in progress"):
    await client.flush()

  client._message_queue.put_nowait(
    TranscriptionMessage(
      stream="stream-live",
      segments=[_segment("done")],
      language="en",
      flush_complete=True,
    )
  )

  try:
    await first_flush
  finally:
    keepalive.set()
    await message_task

  assert connection.flush_commands == [True]


@pytest.mark.asyncio
async def test_flush_surfaces_server_rejection_message() -> None:
  """flush() must propagate server-side rejection text as a runtime error."""

  client = EavesdropClient.transcriber(audio_device="default")
  connection = RecordingConnection()
  client._connected = True
  client._connection = cast(WebSocketConnection, cast(object, connection))

  keepalive = asyncio.Event()
  message_task = asyncio.create_task(_hold_open(keepalive))
  client._message_task = message_task

  flush_task = asyncio.create_task(client.flush(force_complete=True))
  await connection.flush_sent.wait()
  client._on_error("server refused flush")

  try:
    with pytest.raises(RuntimeError, match="server refused flush"):
      await flush_task
  finally:
    keepalive.set()
    await message_task

  assert connection.flush_commands == [True]


@pytest.mark.asyncio
async def test_flush_surfaces_disconnect_reason_when_socket_drops() -> None:
  """flush() must fail with the disconnect reason if the live socket drops mid-flight."""

  client = EavesdropClient.transcriber(audio_device="default")
  connection = RecordingConnection()
  client._connected = True
  client._connection = cast(WebSocketConnection, cast(object, connection))

  keepalive = asyncio.Event()
  message_task = asyncio.create_task(_hold_open(keepalive))
  client._message_task = message_task

  flush_task = asyncio.create_task(client.flush(force_complete=True))
  await connection.flush_sent.wait()
  client._on_disconnect("socket lost during flush")

  try:
    with pytest.raises(RuntimeError, match="socket lost during flush"):
      await flush_task
  finally:
    keepalive.set()
    await message_task

  assert connection.flush_commands == [True]
