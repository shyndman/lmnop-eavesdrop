"""Retained operational contracts through real WebSocket handshakes."""

import asyncio
from typing import cast

import pytest
from protocol_harness import DEADLINE, STREAM, audio, flush, protocol_server, start, transcription
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedOK

from eavesdrop.wire import (
  ErrorMessage,
  FlushControlMessage,
  RecordingStartedMessage,
  ServerReadyMessage,
  TranscriptionSetupMessage,
  UserTranscriptionOptions,
  UtteranceCancelledMessage,
  deserialize_message,
  serialize_message,
)
from eavesdrop.wire.messages import HealthCheckRequest


@pytest.mark.asyncio
async def test_health_request_closes_normally_and_live_socket_remains_usable(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Keep the live protocol usable after a health-check connection closes.

  :param monkeypatch: Restores the deterministic inference loader.
  """
  async with protocol_server(monkeypatch) as server:
    async with connect(server.uri) as socket:
      await socket.send(serialize_message(HealthCheckRequest()))
      with pytest.raises(ConnectionClosedOK):
        async with asyncio.timeout(DEADLINE):
          await socket.recv()
    async with server.connection() as live:
      await start(live, "healthy")
      await audio(live)
      await flush(live, "healthy")
      assert (await transcription(live, flushed=True)).recording_id == "healthy"


@pytest.mark.asyncio
async def test_malformed_setup_reports_error_and_closes_connection(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Reject invalid setup options and close the rejected connection.

  :param monkeypatch: Restores the deterministic inference loader.
  """
  async with protocol_server(monkeypatch) as server:
    async with connect(server.uri) as socket:
      await socket.send('{"type":"setup","stream":"invalid","options":{"language":"fr"}}')
      async with asyncio.timeout(DEADLINE):
        rejected = deserialize_message(await socket.recv(decode=True))
        assert isinstance(rejected, ErrorMessage)
        assert rejected.message
        with pytest.raises(ConnectionClosedOK):
          await socket.recv()


@pytest.mark.asyncio
async def test_pre_setup_controls_are_rejected_without_preventing_valid_setup(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Reject early controls while permitting a subsequent valid setup.

  :param monkeypatch: Restores the deterministic inference loader.
  """
  async with protocol_server(monkeypatch) as server:
    async with connect(server.uri) as socket:
      controls = (
        FlushControlMessage(stream=STREAM, recording_id="before"),
        RecordingStartedMessage(stream=STREAM, recording_id="before", sample_rate_hz=16000),
        UtteranceCancelledMessage(stream=STREAM, recording_id="before"),
      )
      for control in controls:
        await socket.send(serialize_message(control))
        async with asyncio.timeout(DEADLINE):
          rejected = deserialize_message(await socket.recv(decode=True))
        assert isinstance(rejected, ErrorMessage)
        assert "before" in rejected.message.lower()
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
      await start(socket, "after-setup")
      await audio(socket)
      await flush(socket, "after-setup")
      assert (await transcription(socket, flushed=True)).recording_id == "after-setup"
