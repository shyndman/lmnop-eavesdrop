"""Real transport, shell, buffer, processor, and sink; only inference is deterministic."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from threading import Event
from typing import cast, override

import numpy as np
import pytest
from websockets.asyncio.client import ClientConnection, connect
from websockets.asyncio.server import serve

from eavesdrop.server.config import BufferConfig, TranscriptionConfig
from eavesdrop.server.connection_handler import WebSocketConnectionHandler
from eavesdrop.server.server import TranscriptionServer
from eavesdrop.server.streaming.interfaces import Float32Audio
from eavesdrop.server.streaming.processor import StreamingTranscriptionProcessor
from eavesdrop.server.transcription.models import TranscriptionInfo
from eavesdrop.server.transcription.orukeet import OrukeetTranscriber
from eavesdrop.server.transcription.session import TranscriptionSession
from eavesdrop.wire import (
  ErrorMessage,
  FlushControlMessage,
  RecordingStartedMessage,
  Segment,
  ServerReadyMessage,
  TranscriptionMessage,
  TranscriptionSetupMessage,
  UserTranscriptionOptions,
  UtteranceCancelledMessage,
  Word,
  deserialize_message,
  serialize_message,
)

DEADLINE = 3.0
STREAM = "protocol-regression"


@dataclass
class DeterministicTranscriber(OrukeetTranscriber):
  """Two revisable segments, with an optional externally released recognition pass."""

  hold: bool = False
  entered: Event = field(default_factory=Event)
  release: Event = field(default_factory=Event)
  calls: int = 0

  @override
  def transcribe(
    self,
    audio: Float32Audio,
    *,
    config: TranscriptionConfig,
    absolute_stream_start: float,
    session: TranscriptionSession,
  ) -> tuple[list[Segment], TranscriptionInfo]:
    self.calls += 1
    revision = self.calls
    self.entered.set()
    if self.hold and revision == 1:
      if not self.release.wait(10):
        raise TimeoutError("Test did not release held inference")
    duration = len(audio) / 16000
    session.update_audio_context(start_offset=absolute_stream_start, duration=duration)
    label = "A" if audio[0] > 0 else "B"
    tail_label = "B" if audio[-1] < 0 else label
    segments = [
      Segment(
        id=0,
        start=absolute_stream_start + start,
        end=absolute_stream_start + end,
        text=f"{label if index == 0 else tail_label}{index} revision {revision}",
        words=[
          Word(
            start=absolute_stream_start + start,
            end=absolute_stream_start + end,
            word=f"{label if index == 0 else tail_label}{index}",
          )
        ],
      )
      for index, (start, end) in enumerate(((0.01, 0.03), (0.05, 0.08)))
    ]
    return segments, TranscriptionInfo(
      duration=duration,
      duration_after_vad=0.08 if config.use_vad else duration,
      speech_chunks=[{"start": 0, "end": 1280}] if config.use_vad else [],
    )


@dataclass
class ProtocolServer:
  uri: str
  model: DeterministicTranscriber

  @asynccontextmanager
  async def connection(self) -> AsyncIterator[ClientConnection]:
    async with connect(
      self.uri,
      additional_headers={"X-Client-Type": "transcriber"},
      open_timeout=DEADLINE,
      close_timeout=DEADLINE,
    ) as socket:
      await socket.send(
        serialize_message(
          TranscriptionSetupMessage(
            stream=STREAM,
            options=UserTranscriptionOptions(),
          )
        )
      )
      async with asyncio.timeout(DEADLINE):
        ready = deserialize_message(await socket.recv(decode=True))
      assert isinstance(ready, ServerReadyMessage)
      assert ready.stream == STREAM
      yield socket


@asynccontextmanager
async def protocol_server(
  monkeypatch: pytest.MonkeyPatch,
  *,
  use_vad: bool = False,
  interval: float = 30.0,
  minimum: float = 1.0,
  window: int = 10,
  hold: bool = False,
  transcriber: OrukeetTranscriber | None = None,
) -> AsyncIterator[ProtocolServer]:
  model = DeterministicTranscriber(hold=hold)

  def load_model(_processor: StreamingTranscriptionProcessor) -> OrukeetTranscriber:
    return model if transcriber is None else transcriber

  monkeypatch.setattr(StreamingTranscriptionProcessor, "_load_model", load_model)
  shell = TranscriptionServer()
  shell.transcription_config = TranscriptionConfig(
    use_vad=use_vad,
    send_last_n_segments=window,
    buffer=BufferConfig(min_chunk_duration=minimum, transcription_interval=interval),
  )
  shell._connection_handler = WebSocketConnectionHandler(shell.initialize_client)
  async with serve(shell.recv_audio, "127.0.0.1", 0, close_timeout=DEADLINE) as listener:
    address = cast(tuple[str, int], listener.sockets[0].getsockname())
    port = address[1]
    try:
      yield ProtocolServer(f"ws://127.0.0.1:{port}", model)
    finally:
      model.release.set()
      listener.close()
      async with asyncio.timeout(DEADLINE):
        await listener.wait_closed()


async def start(socket: ClientConnection, recording_id: str) -> None:
  await socket.send(
    serialize_message(
      RecordingStartedMessage(
        stream=STREAM,
        recording_id=recording_id,
        sample_rate_hz=16000,
      )
    )
  )


async def audio(socket: ClientConnection, *, seconds: float = 0.1, positive: bool = True) -> None:
  await socket.send(
    np.full(int(seconds * 16000), 0.5 if positive else -0.5, dtype=np.float32).tobytes()
  )


async def flush(socket: ClientConnection, recording_id: str, *, force: bool = True) -> None:
  await socket.send(
    serialize_message(
      FlushControlMessage(
        stream=STREAM,
        recording_id=recording_id,
        force_complete=force,
      )
    )
  )


async def cancel(socket: ClientConnection, recording_id: str) -> None:
  await socket.send(
    serialize_message(
      UtteranceCancelledMessage(
        stream=STREAM,
        recording_id=recording_id,
      )
    )
  )


async def transcription(socket: ClientConnection, *, flushed: bool = False) -> TranscriptionMessage:
  async with asyncio.timeout(DEADLINE):
    while True:
      message = deserialize_message(await socket.recv(decode=True))
      assert not isinstance(message, ErrorMessage), message
      if isinstance(message, TranscriptionMessage):
        assert message.stream == STREAM
        if not flushed or message.flush_complete:
          return message


async def error(socket: ClientConnection) -> ErrorMessage:
  async with asyncio.timeout(DEADLINE):
    while True:
      message = deserialize_message(await socket.recv(decode=True))
      if isinstance(message, ErrorMessage):
        return message


async def entered(model: DeterministicTranscriber) -> None:
  assert await asyncio.to_thread(model.entered.wait, DEADLINE), "Inference never began"


async def empty_flush_responsiveness() -> None:
  """Run in a killable process: an event-loop spin cannot defeat the outer deadline."""
  with pytest.MonkeyPatch.context() as patch:
    async with protocol_server(patch) as server:
      async with server.connection() as first:
        await start(first, "empty")
        for _ in range(2):
          await flush(first, "empty")
          message = await transcription(first, flushed=True)
          assert message.recording_id == "empty"
          assert not any(segment.text.strip() for segment in message.segments)
        await audio(first)
        await flush(first, "empty")
        consumed = await transcription(first, flushed=True)
        assert any(segment.text.startswith("A") for segment in consumed.segments)
        await flush(first, "empty")
        repeated = await transcription(first, flushed=True)
        assert repeated.recording_id == "empty"
        async with server.connection() as second:
          await start(second, "responsive")
          await flush(second, "responsive")
          assert (await transcription(second, flushed=True)).recording_id == "responsive"
