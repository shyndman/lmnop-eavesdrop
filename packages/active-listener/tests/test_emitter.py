"""Text-emitter boundary tests for active-listener."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import override

import pydotool
import pytest
from sdbus import SdBusUnmappedMessageError

from active_listener.infra.emitter import (
  DBUS_BUS_NAME,
  DESKTOP_OBJECT_PATH,
  MAX_EMIT_CHUNK_LENGTH,
  GnomeShellExtensionTextEmitter,
)

TEXT_MIME_TYPE = "text/plain;charset=utf-8"


@dataclass
class RecordingBus:
  close_calls: int = 0

  def close(self) -> None:
    self.close_calls += 1


@dataclass
class RecordingDesktopProxy:
  focused_windows: list[tuple[int, str]] = field(default_factory=lambda: [(42, "org.gnome.dspy")])
  current_content: tuple[str, bytes] = (TEXT_MIME_TYPE, b"seed")
  focused_window_calls: int = 0
  get_clipboard_content_calls: int = 0
  set_clipboard_text_calls: int = 0
  set_clipboard_content_calls: int = 0
  staged_texts: list[str] = field(default_factory=list)
  restored_contents: list[tuple[str, bytes]] = field(default_factory=list)
  fail_on_focused_window: bool = False
  fail_on_get_clipboard_content: bool = False
  fail_at_set_clipboard_text_call: int | None = None
  fail_on_set_clipboard_content: bool = False

  def get_focused_window(self) -> tuple[int, str]:
    self.focused_window_calls += 1
    if self.fail_on_focused_window:
      raise RuntimeError("No focused window")
    if len(self.focused_windows) == 1:
      return self.focused_windows[0]
    return self.focused_windows.pop(0)

  def get_clipboard_content(self) -> tuple[str, bytes]:
    self.get_clipboard_content_calls += 1
    if self.fail_on_get_clipboard_content:
      raise RuntimeError("clipboard unavailable")
    return self.current_content

  def set_clipboard_text(self, text: str) -> None:
    self.set_clipboard_text_calls += 1
    if self.fail_at_set_clipboard_text_call == self.set_clipboard_text_calls:
      raise RuntimeError("clipboard unavailable")
    self.current_content = (TEXT_MIME_TYPE, text.encode("utf-8"))
    self.staged_texts.append(text)

  def set_clipboard_content(self, mime_type: str, data: bytes) -> None:
    self.set_clipboard_content_calls += 1
    if self.fail_on_set_clipboard_content:
      raise RuntimeError("clipboard unavailable")
    self.current_content = (mime_type, data)
    self.restored_contents.append((mime_type, data))


@dataclass
class RecordingYdotool:
  init_calls: list[str] = field(default_factory=list)
  key_combination_calls: list[list[int]] = field(default_factory=list)
  fail_on_init: bool = False
  fail_at_key_combination_call: int | None = None

  def init(self, socket_path: str) -> None:
    self.init_calls.append(socket_path)
    if self.fail_on_init:
      raise RuntimeError("ydotool unavailable")

  def key_combination(self, keys: list[int]) -> None:
    self.key_combination_calls.append(list(keys))
    if self.fail_at_key_combination_call == len(self.key_combination_calls):
      raise RuntimeError("key injection failed")


class RecordingLogger:
  def __init__(self) -> None:
    self.exception_messages: list[str] = []

  def debug(self, event: str, **kwargs: object) -> None:
    _ = event
    _ = kwargs

  def info(self, event: str, **kwargs: object) -> None:
    _ = event
    _ = kwargs

  def warning(self, event: str, **kwargs: object) -> None:
    _ = event
    _ = kwargs

  def exception(self, event: str, **kwargs: object) -> None:
    _ = kwargs
    self.exception_messages.append(event)


def _install_proxy(
  monkeypatch: pytest.MonkeyPatch,
  *,
  bus: RecordingBus,
  desktop: RecordingDesktopProxy,
  ydotool: RecordingYdotool | None = None,
  sleep_calls: list[float] | None = None,
) -> RecordingYdotool:
  def build_desktop_proxy(
    *, service_name: str, object_path: str, bus: RecordingBus
  ) -> RecordingDesktopProxy:
    assert service_name == DBUS_BUS_NAME
    assert object_path == DESKTOP_OBJECT_PATH
    _ = bus
    return desktop

  def record_sleep(seconds: float) -> None:
    if sleep_calls is not None:
      sleep_calls.append(seconds)

  recording_ydotool = ydotool or RecordingYdotool()
  monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/test-id")
  monkeypatch.setattr("active_listener.infra.emitter.sd_bus_open_user", lambda: bus)
  monkeypatch.setattr("active_listener.infra.emitter.time.sleep", record_sleep)
  monkeypatch.setattr("active_listener.infra.emitter.pydotool.init", recording_ydotool.init)
  monkeypatch.setattr(
    "active_listener.infra.emitter.pydotool.key_combination",
    recording_ydotool.key_combination,
  )
  monkeypatch.setattr(
    "active_listener.infra.emitter.DesktopExtensionInterface",
    build_desktop_proxy,
  )
  return recording_ydotool


def test_emitter_initializes_once_without_validating_proxy(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  bus = RecordingBus()
  desktop = RecordingDesktopProxy()
  ydotool = _install_proxy(monkeypatch, bus=bus, desktop=desktop)

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()
  emitter.initialize()

  assert desktop.focused_window_calls == 0
  assert desktop.get_clipboard_content_calls == 0
  assert bus.close_calls == 0
  assert ydotool.init_calls == ["/run/user/test-id/.ydotool_socket"]


def test_emitter_closes_proxy_when_ydotool_initialization_fails(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  bus = RecordingBus()
  desktop = RecordingDesktopProxy()
  ydotool = RecordingYdotool(fail_on_init=True)
  _install_proxy(monkeypatch, bus=bus, desktop=desktop, ydotool=ydotool)
  emitter = GnomeShellExtensionTextEmitter()

  with pytest.raises(RuntimeError, match="ydotool unavailable"):
    emitter.initialize()

  assert bus.close_calls == 1
  with pytest.raises(RuntimeError, match="initialize"):
    emitter.emit_text("hello")


@pytest.mark.parametrize("wm_class", ["kitty", "so.stencil.tern"])
def test_emitter_uses_ctrl_shift_v_for_terminal_apps(
  monkeypatch: pytest.MonkeyPatch,
  wm_class: str,
) -> None:
  bus = RecordingBus()
  desktop = RecordingDesktopProxy(focused_windows=[(42, wm_class)])
  ydotool = _install_proxy(monkeypatch, bus=bus, desktop=desktop)

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()
  emitter.emit_text("hello")

  assert desktop.get_clipboard_content_calls == 1
  assert desktop.staged_texts == ["hello"]
  assert desktop.restored_contents == [(TEXT_MIME_TYPE, b"seed")]
  assert ydotool.key_combination_calls == [
    [pydotool.KEY_LEFTCTRL, pydotool.KEY_LEFTSHIFT, pydotool.KEY_V]
  ]


def test_emitter_uses_ctrl_v_for_non_terminal_apps(monkeypatch: pytest.MonkeyPatch) -> None:
  bus = RecordingBus()
  desktop = RecordingDesktopProxy(focused_windows=[(77, "org.gnome.dspy")])
  ydotool = _install_proxy(monkeypatch, bus=bus, desktop=desktop)

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()
  emitter.emit_text("hello")

  assert desktop.get_clipboard_content_calls == 1
  assert desktop.staged_texts == ["hello"]
  assert desktop.restored_contents == [(TEXT_MIME_TYPE, b"seed")]
  assert ydotool.key_combination_calls == [[pydotool.KEY_LEFTCTRL, pydotool.KEY_V]]


def test_emitter_preserves_binary_clipboard_content(monkeypatch: pytest.MonkeyPatch) -> None:
  png_bytes = b"\x89PNG\r\n\x1a\nfake-image"
  bus = RecordingBus()
  desktop = RecordingDesktopProxy(current_content=("image/png", png_bytes))
  ydotool = _install_proxy(monkeypatch, bus=bus, desktop=desktop)

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()
  emitter.emit_text("hello")

  assert desktop.staged_texts == ["hello"]
  assert desktop.restored_contents == [("image/png", png_bytes)]
  assert ydotool.key_combination_calls == [[pydotool.KEY_LEFTCTRL, pydotool.KEY_V]]


def test_emitter_skips_restore_when_clipboard_was_empty(monkeypatch: pytest.MonkeyPatch) -> None:
  bus = RecordingBus()
  desktop = RecordingDesktopProxy(current_content=("", b""))
  sleep_calls: list[float] = []
  ydotool = _install_proxy(monkeypatch, bus=bus, desktop=desktop, sleep_calls=sleep_calls)

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()
  emitter.emit_text("hello")

  assert desktop.staged_texts == ["hello"]
  assert desktop.restored_contents == []
  assert sleep_calls == []
  assert ydotool.key_combination_calls == [[pydotool.KEY_LEFTCTRL, pydotool.KEY_V]]


def test_emitter_snapshots_focused_window_once_per_emission(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  bus = RecordingBus()
  desktop = RecordingDesktopProxy(focused_windows=[(99, "kitty"), (1234, "org.gnome.dspy")])
  ydotool = _install_proxy(monkeypatch, bus=bus, desktop=desktop)

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()
  emitter.emit_text("a" * (MAX_EMIT_CHUNK_LENGTH + 1))

  assert desktop.focused_window_calls == 1
  assert desktop.get_clipboard_content_calls == 1
  assert [len(chunk) for chunk in desktop.staged_texts] == [MAX_EMIT_CHUNK_LENGTH, 1]
  assert desktop.restored_contents == [(TEXT_MIME_TYPE, b"seed")]
  assert ydotool.key_combination_calls == [
    [pydotool.KEY_LEFTCTRL, pydotool.KEY_LEFTSHIFT, pydotool.KEY_V],
    [pydotool.KEY_LEFTCTRL, pydotool.KEY_LEFTSHIFT, pydotool.KEY_V],
  ]


def test_emitter_chunks_text_at_max_length(monkeypatch: pytest.MonkeyPatch) -> None:
  bus = RecordingBus()
  desktop = RecordingDesktopProxy()
  sleep_calls: list[float] = []
  ydotool = _install_proxy(monkeypatch, bus=bus, desktop=desktop, sleep_calls=sleep_calls)

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()
  emitter.emit_text("a" * ((MAX_EMIT_CHUNK_LENGTH * 2) + 1))

  assert [len(chunk) for chunk in desktop.staged_texts] == [
    MAX_EMIT_CHUNK_LENGTH,
    MAX_EMIT_CHUNK_LENGTH,
    1,
  ]
  assert desktop.restored_contents == [(TEXT_MIME_TYPE, b"seed")]
  assert sleep_calls == [0.15, 0.15, 0.75]
  assert len(ydotool.key_combination_calls) == 3


def test_emitter_preserves_newlines_inside_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
  bus = RecordingBus()
  desktop = RecordingDesktopProxy()
  ydotool = _install_proxy(monkeypatch, bus=bus, desktop=desktop)

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()
  emitter.emit_text("hello\nworld")

  assert desktop.staged_texts == ["hello\nworld"]
  assert desktop.restored_contents == [(TEXT_MIME_TYPE, b"seed")]
  assert ydotool.key_combination_calls == [[pydotool.KEY_LEFTCTRL, pydotool.KEY_V]]


def test_emitter_restores_clipboard_after_key_injection_failure(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  bus = RecordingBus()
  desktop = RecordingDesktopProxy()
  ydotool = RecordingYdotool(fail_at_key_combination_call=2)
  _install_proxy(monkeypatch, bus=bus, desktop=desktop, ydotool=ydotool)

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()

  with pytest.raises(RuntimeError, match="key injection failed"):
    emitter.emit_text("a" * ((MAX_EMIT_CHUNK_LENGTH * 2) + 1))

  assert [len(chunk) for chunk in desktop.staged_texts] == [
    MAX_EMIT_CHUNK_LENGTH,
    MAX_EMIT_CHUNK_LENGTH,
  ]
  assert desktop.restored_contents == [(TEXT_MIME_TYPE, b"seed")]
  assert len(ydotool.key_combination_calls) == 2


def test_emitter_rebinds_after_disconnected_bus_during_emit(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  @dataclass
  class DisconnectingDesktopProxy(RecordingDesktopProxy):
    disconnect_error: Exception | None = None

    @override
    def get_focused_window(self) -> tuple[int, str]:
      if self.disconnect_error is not None:
        disconnect_error = self.disconnect_error
        self.disconnect_error = None
        raise disconnect_error
      return super().get_focused_window()

  first_bus = RecordingBus()
  second_bus = RecordingBus()
  first_desktop = DisconnectingDesktopProxy(
    disconnect_error=SdBusUnmappedMessageError(
      "System.Error.ENOTCONN",
      "Transport endpoint is not connected",
    ),
  )
  second_desktop = RecordingDesktopProxy(
    focused_windows=[(77, "org.gnome.dspy")],
    current_content=("", b""),
  )
  buses = [first_bus, second_bus]
  desktop_proxies = [first_desktop, second_desktop]

  def open_bus() -> RecordingBus:
    return buses.pop(0)

  def build_desktop_proxy(
    *, service_name: str, object_path: str, bus: RecordingBus
  ) -> RecordingDesktopProxy:
    assert service_name == DBUS_BUS_NAME
    assert object_path == DESKTOP_OBJECT_PATH
    _ = bus
    return desktop_proxies.pop(0)

  ydotool = RecordingYdotool()
  monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/test-id")
  monkeypatch.setattr("active_listener.infra.emitter.pydotool.init", ydotool.init)
  monkeypatch.setattr(
    "active_listener.infra.emitter.pydotool.key_combination",
    ydotool.key_combination,
  )
  monkeypatch.setattr("active_listener.infra.emitter.sd_bus_open_user", open_bus)
  monkeypatch.setattr(
    "active_listener.infra.emitter.DesktopExtensionInterface",
    build_desktop_proxy,
  )

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()
  emitter.emit_text("hello")

  assert first_bus.close_calls == 1
  assert second_bus.close_calls == 0
  assert second_desktop.get_clipboard_content_calls == 1
  assert second_desktop.staged_texts == ["hello"]
  assert second_desktop.restored_contents == []
  assert ydotool.key_combination_calls == [[pydotool.KEY_LEFTCTRL, pydotool.KEY_V]]


def test_emitter_restores_clipboard_after_chunk_write_failure(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  bus = RecordingBus()
  desktop = RecordingDesktopProxy(fail_at_set_clipboard_text_call=2)
  ydotool = _install_proxy(monkeypatch, bus=bus, desktop=desktop)

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()

  with pytest.raises(RuntimeError, match="clipboard unavailable"):
    emitter.emit_text("a" * ((MAX_EMIT_CHUNK_LENGTH * 2) + 1))

  assert desktop.get_clipboard_content_calls == 1
  assert desktop.staged_texts == ["a" * MAX_EMIT_CHUNK_LENGTH]
  assert desktop.restored_contents == [(TEXT_MIME_TYPE, b"seed")]
  assert ydotool.key_combination_calls == [[pydotool.KEY_LEFTCTRL, pydotool.KEY_V]]


def test_emitter_logs_restore_failure_without_failing_successful_emit(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  bus = RecordingBus()
  desktop = RecordingDesktopProxy(fail_on_set_clipboard_content=True)
  logger = RecordingLogger()
  _install_proxy(monkeypatch, bus=bus, desktop=desktop)
  monkeypatch.setattr("active_listener.infra.emitter._logger", logger)

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()
  emitter.emit_text("hello")

  assert desktop.get_clipboard_content_calls == 1
  assert desktop.staged_texts == ["hello"]
  assert desktop.restored_contents == []
  assert logger.exception_messages == ["clipboard restore failed after successful emission"]


def test_emitter_aborts_when_clipboard_snapshot_fails(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  bus = RecordingBus()
  desktop = RecordingDesktopProxy(fail_on_get_clipboard_content=True)
  ydotool = _install_proxy(monkeypatch, bus=bus, desktop=desktop)

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()

  with pytest.raises(RuntimeError, match="clipboard unavailable"):
    emitter.emit_text("hello")

  assert desktop.get_clipboard_content_calls == 1
  assert desktop.staged_texts == []
  assert desktop.restored_contents == []
  assert ydotool.key_combination_calls == []


def test_emitter_raises_when_focused_window_lookup_fails(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  bus = RecordingBus()
  desktop = RecordingDesktopProxy(fail_on_focused_window=True)
  _install_proxy(monkeypatch, bus=bus, desktop=desktop)

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()

  with pytest.raises(RuntimeError, match="No focused window"):
    emitter.emit_text("hello")


def test_emitter_requires_initialize_before_emit() -> None:
  emitter = GnomeShellExtensionTextEmitter()

  with pytest.raises(RuntimeError, match="initialize"):
    emitter.emit_text("hello")
