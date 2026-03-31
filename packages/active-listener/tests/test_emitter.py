"""Text-emitter boundary tests for active-listener."""

from __future__ import annotations

import pytest

from active_listener.emitter import PydotoolTextEmitter


class RecordingPydotool:
  """Captures pydotool calls without touching the workstation."""

  def __init__(self) -> None:
    self.init_calls: list[str | None] = []
    self.typed: list[str] = []

  def init(self, socket_path: str | None = None) -> None:
    self.init_calls.append(socket_path)

  def type_string(self, text: str) -> None:
    self.typed.append(text)


def test_emitter_initializes_once_and_types_text(monkeypatch: pytest.MonkeyPatch) -> None:
  recorder = RecordingPydotool()
  monkeypatch.setattr("active_listener.emitter.pydotool", recorder)

  emitter = PydotoolTextEmitter(socket_path="/tmp/ydotool.sock")
  emitter.initialize()
  emitter.initialize()
  emitter.emit_text("hello")

  assert recorder.init_calls == ["/tmp/ydotool.sock"]
  assert recorder.typed == ["hello"]


def test_emitter_requires_initialize_before_emit() -> None:
  emitter = PydotoolTextEmitter()

  with pytest.raises(RuntimeError, match="initialize"):
    emitter.emit_text("hello")
