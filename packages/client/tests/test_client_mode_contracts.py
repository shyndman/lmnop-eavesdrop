"""Live-client lifecycle contracts over real local WebSockets and fake microphone input."""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pytest
from websockets.asyncio.server import ServerConnection, serve

from eavesdrop.client.audio import Float32Audio, InputStreamLike
from eavesdrop.client.core import EavesdropClient
from eavesdrop.wire import (
  DisconnectMessage,
  ErrorMessage,
  FlushControlMessage,
  RecordingStartedMessage,
  Segment,
  ServerReadyMessage,
  TranscriptionMessage,
  TranscriptionSetupMessage,
  UtteranceCancelledMessage,
  Word,
  deserialize_message,
  serialize_message,
)

DEADLINE = 3.0
RECORDING_A = "recording-a"
RECORDING_B = "recording-b"


@dataclass
class CallbackTiming:
  inputBufferAdcTime: float = 0.0
  currentTime: float = 0.0
  outputBufferDacTime: float = 0.0


@dataclass
class CallbackStatus:
  input_underflow: bool = False
  input_overflow: bool = False

  def __bool__(self) -> bool:
    return self.input_underflow or self.input_overflow


class Microphone:
  """Substitute only the external input device, leaving capture and sending real."""

  def __init__(self) -> None:
    self.frames: list[Float32Audio] = []

  def InputStream(
    self,
    *,
    device: str | int | None,
    channels: int,
    samplerate: int,
    dtype: np.dtype[np.float32] | type[np.float32],
    latency: str,
    blocksize: int,
    callback: Callable[[Float32Audio, int, CallbackTiming, CallbackStatus], None],
  ) -> InputStreamLike:
    microphone = self

    class Stream:
      def start(self) -> None:
        for frame in microphone.frames:
          callback(frame, len(frame), CallbackTiming(), CallbackStatus())

      def stop(self) -> None:
        pass

      def close(self) -> None:
        pass

    return Stream()


@dataclass
class Peer:
  """Controllable remote endpoint for lifecycle races and failure responses."""

  socket: ServerConnection
  session_id: str
  received: asyncio.Queue[str | bytes] = field(default_factory=asyncio.Queue)

  async def recording_started(self) -> str:
    raw = await asyncio.wait_for(self.received.get(), DEADLINE)
    message = deserialize_message(cast(str, raw))
    match message:
      case RecordingStartedMessage(recording_id=recording_id):
        return recording_id
      case _:
        raise AssertionError(f"Expected recording boundary, received {message.type}")

  async def flush_requested(self) -> FlushControlMessage:
    raw = await asyncio.wait_for(self.received.get(), DEADLINE)
    message = deserialize_message(cast(str, raw))
    match message:
      case FlushControlMessage():
        return message
      case _:
        raise AssertionError(f"Expected flush, received {message.type}")

  async def cancelled(self) -> str | None:
    raw = await asyncio.wait_for(self.received.get(), DEADLINE)
    message = deserialize_message(cast(str, raw))
    match message:
      case UtteranceCancelledMessage(recording_id=recording_id):
        return recording_id
      case _:
        raise AssertionError(f"Expected cancellation, received {message.type}")

  async def transcript(
    self,
    recording_id: str,
    text: str,
    *,
    flushed: bool = False,
  ) -> None:
    await self.socket.send(
      serialize_message(
        TranscriptionMessage(
          stream=self.session_id,
          recording_id=recording_id,
          segments=[
            Segment(
              id=1,
              start=0.0,
              end=0.5,
              text=text,
              completed=flushed,
              words=[Word(start=0.0, end=0.5, word=text)],
            )
          ],
          flush_complete=True if flushed else None,
        )
      )
    )


@asynccontextmanager
async def connected_client(
  monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[EavesdropClient, Peer, Microphone, asyncio.Queue[Peer]]]:
  peers: asyncio.Queue[Peer] = asyncio.Queue()
  microphone = Microphone()
  monkeypatch.setattr("eavesdrop.client.audio._load_input_stream", lambda: microphone.InputStream)

  async def accept(socket: ServerConnection) -> None:
    setup = deserialize_message(await socket.recv(decode=True))
    match setup:
      case TranscriptionSetupMessage(stream=session_id):
        peer = Peer(socket, session_id)
      case _:
        raise AssertionError(f"Expected setup, received {setup.type}")
    await socket.send(serialize_message(ServerReadyMessage(stream=peer.session_id, backend="test")))
    peers.put_nowait(peer)
    async for frame in socket:
      peer.received.put_nowait(frame)

  async with serve(accept, "127.0.0.1", 0, close_timeout=DEADLINE) as server:
    address = cast(tuple[str, int], server.sockets[0].getsockname())
    client = EavesdropClient.transcriber(host="127.0.0.1", port=address[1], audio_device="test")
    await client.connect()
    peer = await asyncio.wait_for(peers.get(), DEADLINE)
    try:
      yield client, peer, microphone, peers
    finally:
      await client.disconnect()


@pytest.mark.asyncio
async def test_stop_delivers_captured_backlog_before_flush(monkeypatch: pytest.MonkeyPatch) -> None:
  async with connected_client(monkeypatch) as (client, peer, microphone, _):
    frames = [np.full((4, 1), value, dtype=np.float32) for value in (0.1, 0.2, 0.3)]
    microphone.frames = frames
    await client.start_streaming(RECORDING_A)
    await client.stop_streaming()
    pending = asyncio.create_task(client.flush(RECORDING_A))

    assert await peer.recording_started() == RECORDING_A
    delivered = [await asyncio.wait_for(peer.received.get(), DEADLINE) for _ in frames]
    assert delivered == [frame.tobytes() for frame in frames]
    assert (await peer.flush_requested()).recording_id == RECORDING_A
    await peer.transcript(RECORDING_A, "captured tail", flushed=True)
    assert (await asyncio.wait_for(pending, DEADLINE)).segments[0].text == "captured tail"


@pytest.mark.asyncio
async def test_next_recording_waits_until_final_transcript_is_pushed(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  async with connected_client(monkeypatch) as (client, peer, _, _):
    received: list[str] = []
    client.on_transcription = lambda message: received.append(message.segments[0].text)
    await client.start_streaming(RECORDING_A)
    await client.stop_streaming()
    assert await peer.recording_started() == RECORDING_A
    pending = asyncio.create_task(client.flush(RECORDING_A))
    await peer.flush_requested()
    next_recording = asyncio.create_task(client.start_streaming(RECORDING_B))

    with pytest.raises(asyncio.TimeoutError):
      await asyncio.wait_for(peer.received.get(), 0.02)
    await peer.transcript(RECORDING_A, "A finished", flushed=True)
    await asyncio.wait_for(pending, DEADLINE)
    await asyncio.wait_for(next_recording, DEADLINE)
    assert await peer.recording_started() == RECORDING_B
    assert received == ["A finished"]


@pytest.mark.asyncio
async def test_cancellation_releases_pending_flush_and_allows_next_recording(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  async with connected_client(monkeypatch) as (client, peer, _, _):
    received: list[str] = []
    client.on_transcription = lambda message: received.append(message.segments[0].text)
    await client.start_streaming(RECORDING_A)
    await client.stop_streaming()
    await peer.recording_started()
    pending = asyncio.create_task(client.flush(RECORDING_A))
    await peer.flush_requested()
    await client.cancel_utterance(RECORDING_A)
    assert await peer.cancelled() == RECORDING_A
    with pytest.raises(RuntimeError, match="cancelled"):
      await asyncio.wait_for(pending, DEADLINE)

    await asyncio.wait_for(client.start_streaming(RECORDING_B), DEADLINE)
    assert await peer.recording_started() == RECORDING_B
    await client.stop_streaming()
    next_flush = asyncio.create_task(client.flush(RECORDING_B))
    await peer.flush_requested()
    await peer.transcript(RECORDING_A, "cancelled", flushed=True)
    await peer.transcript(RECORDING_B, "B retained", flushed=True)
    assert (await asyncio.wait_for(next_flush, DEADLINE)).recording_id == RECORDING_B
    assert received == ["B retained"]


@pytest.mark.asyncio
async def test_flush_requires_fresh_matching_acknowledgement(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  async with connected_client(monkeypatch) as (client, peer, _, _):
    pushed: asyncio.Queue[TranscriptionMessage] = asyncio.Queue()
    client.on_transcription = pushed.put_nowait
    await client.start_streaming(RECORDING_A)
    await client.stop_streaming()
    await peer.recording_started()
    await peer.transcript(RECORDING_A, "earlier ack", flushed=True)
    await asyncio.wait_for(pushed.get(), DEADLINE)

    pending = asyncio.create_task(client.flush(RECORDING_A, force_complete=False))
    assert (await peer.flush_requested()).force_complete is False
    await peer.transcript(RECORDING_B, "wrong recording", flushed=True)
    await peer.transcript(RECORDING_A, "still tentative")
    assert (await asyncio.wait_for(pushed.get(), DEADLINE)).segments[0].text == "still tentative"
    assert not pending.done()
    await peer.transcript(RECORDING_A, "fresh ack", flushed=True)
    assert (await asyncio.wait_for(pending, DEADLINE)).segments[0].text == "fresh ack"


@pytest.mark.asyncio
async def test_flush_rejects_concurrency_and_server_failure_does_not_poison_next_request(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  async with connected_client(monkeypatch) as (client, peer, _, _):
    await client.start_streaming(RECORDING_A)
    await client.stop_streaming()
    await peer.recording_started()
    pending = asyncio.create_task(client.flush(RECORDING_A))
    await peer.flush_requested()
    with pytest.raises(RuntimeError, match="already in progress"):
      await client.flush(RECORDING_A)
    await peer.socket.send(
      serialize_message(ErrorMessage(stream=peer.session_id, message="server refused flush"))
    )
    with pytest.raises(RuntimeError, match="server refused flush"):
      await asyncio.wait_for(pending, DEADLINE)

    retry = asyncio.create_task(client.flush(RECORDING_A))
    await peer.flush_requested()
    await peer.transcript(RECORDING_A, "retry succeeds", flushed=True)
    assert (await asyncio.wait_for(retry, DEADLINE)).segments[0].text == "retry succeeds"


@pytest.mark.asyncio
async def test_disconnect_fails_flush_and_reconnection_supports_a_new_recording(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr("eavesdrop.client.core.RECONNECT_DELAY_SECONDS", 0.0)
  async with connected_client(monkeypatch) as (client, peer, _, peers):
    lifecycle: list[str] = []
    client.on_disconnected = lambda _session, reason: lifecycle.append(f"disconnected:{reason}")
    client.on_reconnecting = lambda _session, _attempt, _delay: lifecycle.append("reconnecting")
    reconnected = asyncio.Event()

    def on_reconnected(_session: str) -> None:
      lifecycle.append("reconnected")
      reconnected.set()

    client.on_reconnected = on_reconnected
    await client.start_streaming(RECORDING_A)
    await client.stop_streaming()
    await peer.recording_started()
    pending = asyncio.create_task(client.flush(RECORDING_A))
    await peer.flush_requested()
    await peer.socket.send(
      serialize_message(
        DisconnectMessage(stream=peer.session_id, message="socket lost during flush")
      )
    )
    await peer.socket.close()
    with pytest.raises(RuntimeError, match="socket lost during flush"):
      await asyncio.wait_for(pending, DEADLINE)
    next_peer = await asyncio.wait_for(peers.get(), DEADLINE)
    await asyncio.wait_for(reconnected.wait(), DEADLINE)
    assert lifecycle == ["disconnected:socket lost during flush", "reconnecting", "reconnected"]
    await client.start_streaming(RECORDING_B)
    await client.stop_streaming()
    assert await next_peer.recording_started() == RECORDING_B
    next_flush = asyncio.create_task(client.flush(RECORDING_B))
    await next_peer.flush_requested()
    await next_peer.transcript(RECORDING_B, "after reconnect", flushed=True)
    assert (await asyncio.wait_for(next_flush, DEADLINE)).recording_id == RECORDING_B


def test_transcriber_requires_audio_device_configuration() -> None:
  with pytest.raises(ValueError, match="audio device"):
    EavesdropClient()
