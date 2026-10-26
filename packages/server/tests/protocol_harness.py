"""Real transport, shell, buffer, processor, and sink; only inference is deterministic."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from threading import Event
from typing import cast

import numpy as np
import pytest
from websockets.asyncio.client import ClientConnection, connect
from websockets.asyncio.server import serve

from eavesdrop.server.config import BufferConfig, TranscriptionConfig
from eavesdrop.server.connection_handler import WebSocketConnectionHandler
from eavesdrop.server.constants import SAMPLE_RATE
from eavesdrop.server.server import TranscriptionServer
from eavesdrop.server.streaming.interfaces import Float32Audio
from eavesdrop.server.streaming.processing.recognition import RecognitionWorker, Transcriber
from eavesdrop.server.transcription.models import TranscriptionInfo
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
INFERENCE_RELEASE_TIMEOUT = 10.0
STREAM = "protocol-regression"


@dataclass
class DeterministicTranscriber:
  """Provide deterministic recognition without initializing a production backend.

  :param hold: Whether the first recognition pass waits for explicit release.
  """

  hold: bool = False
  entered: Event = field(default_factory=Event)
  release: Event = field(default_factory=Event)
  calls: int = 0

  def transcribe(
    self,
    audio: Float32Audio,
    *,
    config: TranscriptionConfig,
    absolute_stream_start: float,
    session: TranscriptionSession,
  ) -> tuple[list[Segment], TranscriptionInfo]:
    """Recognize two timed segments and update the recording context.

    :param audio: Waveform whose sign identifies the recording.
    :param config: Controls whether deterministic VAD timing is reported.
    :param absolute_stream_start: Recording-relative start of the waveform.
    :param session: Recording context updated with the recognized duration.
    :returns: Timed segments and inference metadata.
    :raises TimeoutError: If held inference is not released before its deadline.
    """
    self.calls += 1
    revision = self.calls
    self.entered.set()
    if self.hold and revision == 1:
      if not self.release.wait(INFERENCE_RELEASE_TIMEOUT):
        raise TimeoutError("Test did not release held inference")
    duration = len(audio) / SAMPLE_RATE
    session.tracing.audio_start_offset = absolute_stream_start
    session.tracing.audio_duration = duration
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
      speech_segments=[{"start": 0, "end": 1280}] if config.use_vad else [],
    )


@dataclass
class ProtocolServer:
  """Expose a running protocol endpoint and its controllable recognizer.

  :param uri: WebSocket endpoint for the live server.
  :param model: Deterministic recognizer used by the server.
  """

  uri: str
  model: DeterministicTranscriber

  @asynccontextmanager
  async def connection(self) -> AsyncIterator[ClientConnection]:
    """Open and configure a live client connection.

    :yields: A socket after its setup acknowledgement is received.
    """
    async with connect(
      self.uri,
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
        ready = cast(ServerReadyMessage, deserialize_message(await socket.recv(decode=True)))
      assert ready.type == "server_ready"
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
  transcriber: Transcriber | None = None,
) -> AsyncIterator[ProtocolServer]:
  """Run a real server with deterministic inference.

  :param monkeypatch: Restores the model loader after the server closes.
  :param use_vad: Whether to expose deterministic VAD results.
  :param interval: Interval between recognition passes, in seconds.
  :param minimum: Minimum buffered audio duration, in seconds.
  :param window: Number of recent segments retained in outgoing updates.
  :param hold: Whether to block the first recognition pass until released.
  :param transcriber: Optional recognizer replacing the deterministic fake.
  :yields: The listening endpoint and its deterministic inference controls.
  """
  model = DeterministicTranscriber(hold=hold)

  def load_model(_worker: RecognitionWorker) -> Transcriber:
    """Return the configured recognizer at the production loader boundary.

    :param _worker: Recognition worker requesting its recognizer.
    :returns: The explicitly supplied recognizer or the standalone test fake.
    """
    return model if transcriber is None else transcriber

  monkeypatch.setattr(RecognitionWorker, "_load_model", load_model)
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
  """Start a recording.

  :param socket: Configured live connection.
  :param recording_id: Identifier assigned to the recording.
  """
  await socket.send(
    serialize_message(
      RecordingStartedMessage(
        stream=STREAM,
        recording_id=recording_id,
        sample_rate_hz=SAMPLE_RATE,
      )
    )
  )


async def audio(socket: ClientConnection, *, seconds: float = 0.1, positive: bool = True) -> None:
  """Send deterministic audio.

  :param socket: Configured live connection.
  :param seconds: Duration of audio to send.
  :param positive: Whether the waveform identifies recording A rather than B.
  """
  await socket.send(
    np.full(int(seconds * SAMPLE_RATE), 0.5 if positive else -0.5, dtype=np.float32).tobytes()
  )


async def flush(socket: ClientConnection, recording_id: str, *, force: bool = True) -> None:
  """Request a recording flush.

  :param socket: Configured live connection.
  :param recording_id: Recording whose pending work is flushed.
  :param force: Whether tentative segments must be completed.
  """
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
  """Cancel the current utterance.

  :param socket: Configured live connection.
  :param recording_id: Recording whose utterance is cancelled.
  """
  await socket.send(
    serialize_message(
      UtteranceCancelledMessage(
        stream=STREAM,
        recording_id=recording_id,
      )
    )
  )


async def transcription(socket: ClientConnection, *, flushed: bool = False) -> TranscriptionMessage:
  """Receive the next matching transcription update.

  :param socket: Configured live connection.
  :param flushed: Whether to wait specifically for a flush acknowledgement.
  :returns: The next matching transcription for the harness stream.
  """
  async with asyncio.timeout(DEADLINE):
    while True:
      message = deserialize_message(await socket.recv(decode=True))
      assert message.type != "error", message
      if message.type == "transcription":
        assert message.stream == STREAM
        if not flushed or message.flush_complete:
          return message


async def error(socket: ClientConnection) -> ErrorMessage:
  """Receive the next protocol error.

  :param socket: Configured live connection.
  :returns: The next error response.
  """
  async with asyncio.timeout(DEADLINE):
    while True:
      message = deserialize_message(await socket.recv(decode=True))
      if message.type == "error":
        return message


async def entered(model: DeterministicTranscriber) -> None:
  """Wait until recognition has entered its deterministic inference pass.

  :param model: Recognizer whose entry signal is awaited.
  """
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
