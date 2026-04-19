"""Text-emitter boundary tests for active-listener."""

from __future__ import annotations

from pathlib import Path

import pytest

from active_listener.infra.emitter import PydotoolTextEmitter


class RecordingPydotool:
  """Captures pydotool calls without touching the workstation."""

  def __init__(self) -> None:
    self.init_calls: list[str | None] = []
    self.typed: list[str] = []
    self.failing_socket_paths: set[str] = set()

  def init(self, socket_path: str | None = None) -> None:
    self.init_calls.append(socket_path)
    if socket_path in self.failing_socket_paths:
      raise OSError(f"socket unavailable: {socket_path}")

  def type_string(
    self,
    text: str,
    *,
    hold_delay_ms: int | None = None,
    each_char_delay_ms: int | None = None,
  ) -> None:
    _ = hold_delay_ms
    _ = each_char_delay_ms
    self.typed.append(text)


def test_emitter_prefers_run_user_uid_socket(monkeypatch: pytest.MonkeyPatch) -> None:
  recorder = RecordingPydotool()

  monkeypatch.setattr("active_listener.infra.emitter.pydotool", recorder)
  monkeypatch.setattr("active_listener.infra.emitter.os.getuid", lambda: 1234)

  emitter = PydotoolTextEmitter()
  emitter.initialize()

  assert recorder.init_calls == ["/run/user/1234/.ydotool_socket"]


def test_emitter_initializes_once_and_types_text(monkeypatch: pytest.MonkeyPatch) -> None:
  recorder = RecordingPydotool()
  monkeypatch.setattr("active_listener.infra.emitter.pydotool", recorder)

  emitter = PydotoolTextEmitter(socket_path="/tmp/ydotool.sock")
  emitter.initialize()
  emitter.initialize()
  emitter.emit_text("hello")

  assert recorder.init_calls == ["/tmp/ydotool.sock"]
  assert recorder.typed == ["hello"]


def test_emitter_uses_run_user_socket_when_present(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  recorder = RecordingPydotool()
  preferred = tmp_path / "run-user.sock"
  fallback = tmp_path / "tmp.sock"
  _ = preferred.write_text("")
  _ = fallback.write_text("")

  monkeypatch.setattr("active_listener.infra.emitter.pydotool", recorder)
  monkeypatch.setattr(
    "active_listener.infra.emitter._candidate_socket_paths",
    lambda: (preferred, fallback),
  )

  emitter = PydotoolTextEmitter()
  emitter.initialize()

  assert recorder.init_calls == [str(preferred)]


def test_emitter_falls_back_to_tmp_socket_when_run_user_socket_missing(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  recorder = RecordingPydotool()
  missing = tmp_path / "missing.sock"
  fallback = tmp_path / "tmp.sock"
  recorder.failing_socket_paths.add(str(missing))

  monkeypatch.setattr("active_listener.infra.emitter.pydotool", recorder)
  monkeypatch.setattr(
    "active_listener.infra.emitter._candidate_socket_paths",
    lambda: (missing, fallback),
  )

  emitter = PydotoolTextEmitter()
  emitter.initialize()

  assert recorder.init_calls == [str(missing), str(fallback)]


def test_emitter_uses_default_init_when_no_socket_exists(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  recorder = RecordingPydotool()
  missing_one = tmp_path / "missing-one.sock"
  missing_two = tmp_path / "missing-two.sock"
  recorder.failing_socket_paths.update({str(missing_one), str(missing_two)})

  monkeypatch.setattr("active_listener.infra.emitter.pydotool", recorder)
  monkeypatch.setattr(
    "active_listener.infra.emitter._candidate_socket_paths",
    lambda: (missing_one, missing_two),
  )

  emitter = PydotoolTextEmitter()
  emitter.initialize()

  assert recorder.init_calls == [str(missing_one), str(missing_two), None]


def test_emitter_requires_initialize_before_emit() -> None:
  emitter = PydotoolTextEmitter()

  with pytest.raises(RuntimeError, match="initialize"):
    emitter.emit_text("hello")
