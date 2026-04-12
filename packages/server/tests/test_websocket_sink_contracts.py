"""Focused contract tests for ``WebSocketTranscriptionSink`` teardown behavior."""

from __future__ import annotations

import sys
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast
from unittest.mock import MagicMock

import pytest
import structlog
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

_streaming_package = ModuleType("eavesdrop.server.streaming")
_streaming_package.__path__ = [
  str(Path(__file__).resolve().parents[1] / "src" / "eavesdrop" / "server" / "streaming")
]
_ = sys.modules.setdefault("eavesdrop.server.streaming", _streaming_package)


class _LoggerMethod(Protocol):
  def assert_called_once_with(self, *args: object, **kwargs: object) -> None: ...

  def assert_not_called(self) -> None: ...


class _LoggerDouble(Protocol):
  info: _LoggerMethod
  exception: _LoggerMethod


class _SinkLike(Protocol):
  logger: structlog.stdlib.BoundLogger

  async def disconnect(self) -> None: ...

  async def send_error(self, error: str) -> None: ...


_sink_factory = cast(
  Callable[[ServerConnection, str], _SinkLike],
  import_module("eavesdrop.server.streaming.audio_flow").WebSocketTranscriptionSink,
)


class _FakeWebSocket:
  """Minimal websocket double with explicit state and controllable send failure."""

  def __init__(self, *, state: State, send_exception: Exception | None = None) -> None:
    self.state: State = state
    self.send_calls: int = 0
    self.sent_payloads: list[str] = []
    self._send_exception: Exception | None = send_exception

  async def send(self, payload: str) -> None:
    self.send_calls += 1
    if self._send_exception is not None:
      raise self._send_exception
    self.sent_payloads.append(payload)


def _as_server_connection(websocket: _FakeWebSocket) -> ServerConnection:
  """Cast a focused websocket double into the sink's runtime dependency type."""
  return cast(ServerConnection, cast(object, websocket))


def _create_sink(websocket: _FakeWebSocket) -> tuple[_SinkLike, _LoggerDouble]:
  """Create a sink with a typed logger double attached."""
  sink = _sink_factory(_as_server_connection(websocket), "stream-1")
  raw_logger = MagicMock()
  sink.logger = cast(structlog.stdlib.BoundLogger, raw_logger)
  return sink, cast(_LoggerDouble, raw_logger)


def _is_sink_closed(sink: _SinkLike) -> bool:
  """Read sink close state without relying on private-attribute typing."""
  return cast(bool, getattr(sink, "_closed"))


@pytest.mark.asyncio
async def test_disconnect_skips_message_when_socket_not_open() -> None:
  """Disconnect teardown must not attempt application data on closed sockets."""
  websocket = _FakeWebSocket(state=State.CLOSED)
  sink, logger = _create_sink(websocket)

  await sink.disconnect()

  assert websocket.send_calls == 0
  assert _is_sink_closed(sink) is True
  logger.info.assert_called_once_with("WebSocket transcription sink disconnected")


@pytest.mark.asyncio
async def test_disconnect_suppresses_disconnect_send_failure() -> None:
  """Disconnect teardown must swallow expected close-race send failures."""
  websocket = _FakeWebSocket(
    state=State.OPEN,
    send_exception=ConnectionClosed(None, None),
  )
  sink, logger = _create_sink(websocket)

  await sink.disconnect()

  assert websocket.send_calls == 1
  assert _is_sink_closed(sink) is True
  logger.exception.assert_not_called()
  logger.info.assert_called_once_with("WebSocket transcription sink disconnected")


@pytest.mark.asyncio
async def test_send_message_logs_non_disconnect_send_failures() -> None:
  """Normal outbound message failures must remain visible in logs."""
  websocket = _FakeWebSocket(
    state=State.OPEN,
    send_exception=RuntimeError("boom"),
  )
  sink, logger = _create_sink(websocket)

  await sink.send_error("fail")

  assert websocket.send_calls == 1
  logger.exception.assert_called_once_with("Error sending message to client")
