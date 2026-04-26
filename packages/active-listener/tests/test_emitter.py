"""Text-emitter boundary tests for active-listener."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import override

import pytest
from pydantic import ValidationError
from sdbus import SdBusUnmappedMessageError

from active_listener.infra.emitter import GnomeShellExtensionTextEmitter


def _focused_window_payload(*, window_id: int = 42, wm_class: str = "org.gnome.dspy") -> str:
  return json.dumps(
    {
      "id": window_id,
      "title": "D-Spy",
      "wm_class": wm_class,
      "wm_class_instance": wm_class,
      "pid": 333381,
      "maximized": True,
      "display": {},
      "frame_type": 0,
      "window_type": 0,
      "layer": 2,
      "monitor": 1,
      "role": None,
      "width": 1692,
      "height": 1096,
      "x": 2560,
      "y": 892,
      "in_current_workspace": True,
      "canclose": True,
      "canmaximize": True,
      "canminimize": True,
      "canshade": False,
      "moveable": True,
      "resizeable": False,
      "has_focus": True,
      "workspace": 0,
    }
  )


@dataclass
class RecordingBus:
  close_calls: int = 0

  def close(self) -> None:
    self.close_calls += 1


@dataclass
class RecordingWindowsProxy:
  focused_window_payloads: list[str]
  shortcut_results: list[bool] = field(default_factory=list)
  focused_window_calls: int = 0
  shortcut_calls: list[tuple[int, str, str]] = field(default_factory=list)

  def get_focused_window_sync(self) -> str:
    self.focused_window_calls += 1
    if len(self.focused_window_payloads) == 1:
      return self.focused_window_payloads[0]
    return self.focused_window_payloads.pop(0)

  def send_shortcut(self, winid: int, key: str, modifiers: str) -> bool:
    self.shortcut_calls.append((winid, key, modifiers))
    if self.shortcut_results:
      return self.shortcut_results.pop(0)
    return True


@dataclass
class RecordingClipboardProxy:
  current_content: str = ""
  get_current_content_calls: int = 0
  set_content_calls: int = 0
  set_contents: list[str] = field(default_factory=list)
  fail_on_get_current_content: bool = False
  fail_at_set_content_call: int | None = None

  def get_current_content(self) -> str:
    self.get_current_content_calls += 1
    if self.fail_on_get_current_content:
      raise RuntimeError("clipboard unavailable")
    return self.current_content

  def set_content(self, content: str) -> None:
    self.set_content_calls += 1
    if self.fail_at_set_content_call == self.set_content_calls:
      raise RuntimeError("clipboard unavailable")
    self.current_content = content
    self.set_contents.append(content)


@dataclass(frozen=True)
class LogRecord:
  event: str
  fields: dict[str, object]


class RecordingLogger:
  def __init__(self) -> None:
    self.debug_messages: list[str] = []
    self.info_messages: list[str] = []
    self.warning_messages: list[str] = []
    self.exception_messages: list[str] = []
    self.debug_records: list[LogRecord] = []
    self.info_records: list[LogRecord] = []
    self.warning_records: list[LogRecord] = []
    self.exception_records: list[LogRecord] = []

  def debug(self, event: str, **kwargs: object) -> None:
    self.debug_messages.append(event)
    self.debug_records.append(LogRecord(event=event, fields=kwargs))

  def info(self, event: str, **kwargs: object) -> None:
    self.info_messages.append(event)
    self.info_records.append(LogRecord(event=event, fields=kwargs))

  def warning(self, event: str, **kwargs: object) -> None:
    self.warning_messages.append(event)
    self.warning_records.append(LogRecord(event=event, fields=kwargs))

  def exception(self, event: str, **kwargs: object) -> None:
    self.exception_messages.append(event)
    self.exception_records.append(LogRecord(event=event, fields=kwargs))


def _install_proxies(
  monkeypatch: pytest.MonkeyPatch,
  *,
  bus: RecordingBus,
  windows: RecordingWindowsProxy,
  clipboard: RecordingClipboardProxy,
  sleep_calls: list[float] | None = None,
) -> None:
  def build_windows_proxy(
    *, service_name: str, object_path: str, bus: object
  ) -> RecordingWindowsProxy:
    _ = service_name
    _ = object_path
    _ = bus
    return windows

  def build_clipboard_proxy(
    *, service_name: str, object_path: str, bus: object
  ) -> RecordingClipboardProxy:
    _ = service_name
    _ = object_path
    _ = bus
    return clipboard

  def record_sleep(seconds: float) -> None:
    if sleep_calls is not None:
      sleep_calls.append(seconds)

  monkeypatch.setattr("active_listener.infra.emitter.sd_bus_open_user", lambda: bus)
  monkeypatch.setattr("active_listener.infra.emitter.time.sleep", record_sleep)
  monkeypatch.setattr(
    "active_listener.infra.emitter.WindowsExtensionInterface",
    build_windows_proxy,
  )
  monkeypatch.setattr(
    "active_listener.infra.emitter.ClipboardExtensionInterface",
    build_clipboard_proxy,
  )


def test_emitter_initializes_once_without_validating_proxies(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  bus = RecordingBus()
  windows = RecordingWindowsProxy([_focused_window_payload()])
  clipboard = RecordingClipboardProxy(current_content="seed")
  _install_proxies(monkeypatch, bus=bus, windows=windows, clipboard=clipboard)

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()
  emitter.initialize()

  assert windows.focused_window_calls == 0
  assert clipboard.get_current_content_calls == 0
  assert bus.close_calls == 0


def test_emitter_uses_ctrl_shift_v_for_kitty(monkeypatch: pytest.MonkeyPatch) -> None:
  bus = RecordingBus()
  windows = RecordingWindowsProxy([_focused_window_payload(wm_class="kitty")])
  clipboard = RecordingClipboardProxy(current_content="seed")
  _install_proxies(monkeypatch, bus=bus, windows=windows, clipboard=clipboard)

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()
  emitter.emit_text("hello")

  assert clipboard.get_current_content_calls == 1
  assert clipboard.set_contents == ["hello", "seed"]
  assert windows.shortcut_calls == [(42, "v", "CONTROL|SHIFT")]


def test_emitter_uses_ctrl_v_for_non_kitty(monkeypatch: pytest.MonkeyPatch) -> None:
  bus = RecordingBus()
  windows = RecordingWindowsProxy([_focused_window_payload(window_id=77)])
  clipboard = RecordingClipboardProxy(current_content="seed")
  _install_proxies(monkeypatch, bus=bus, windows=windows, clipboard=clipboard)

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()
  emitter.emit_text("hello")

  assert clipboard.get_current_content_calls == 1
  assert clipboard.set_contents == ["hello", "seed"]
  assert windows.shortcut_calls == [(77, "v", "CONTROL")]


def test_emitter_snapshots_focused_window_once_per_emission(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  bus = RecordingBus()
  windows = RecordingWindowsProxy(
    [
      _focused_window_payload(window_id=99, wm_class="kitty"),
      _focused_window_payload(window_id=1234),
    ]
  )
  clipboard = RecordingClipboardProxy(current_content="seed")
  _install_proxies(monkeypatch, bus=bus, windows=windows, clipboard=clipboard)

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()
  emitter.emit_text("a" * 801)

  assert windows.focused_window_calls == 1
  assert clipboard.get_current_content_calls == 1
  assert [len(chunk) for chunk in clipboard.set_contents[:-1]] == [800, 1]
  assert clipboard.set_contents[-1] == "seed"
  assert windows.shortcut_calls == [
    (99, "v", "CONTROL|SHIFT"),
    (99, "v", "CONTROL|SHIFT"),
  ]


def test_emitter_chunks_text_at_800_characters(monkeypatch: pytest.MonkeyPatch) -> None:
  bus = RecordingBus()
  windows = RecordingWindowsProxy([_focused_window_payload(), _focused_window_payload()])
  clipboard = RecordingClipboardProxy(current_content="seed")
  sleep_calls: list[float] = []
  _install_proxies(
    monkeypatch,
    bus=bus,
    windows=windows,
    clipboard=clipboard,
    sleep_calls=sleep_calls,
  )

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()
  emitter.emit_text("a" * 1601)

  assert [len(chunk) for chunk in clipboard.set_contents[:-1]] == [800, 800, 1]
  assert clipboard.set_contents[-1] == "seed"
  assert sleep_calls == [0.15, 0.15, 0.75]


def test_emitter_preserves_newlines_inside_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
  bus = RecordingBus()
  windows = RecordingWindowsProxy([_focused_window_payload(), _focused_window_payload()])
  clipboard = RecordingClipboardProxy(current_content="seed")
  _install_proxies(monkeypatch, bus=bus, windows=windows, clipboard=clipboard)

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()
  emitter.emit_text("hello\nworld")

  assert clipboard.set_contents == ["hello\nworld", "seed"]
  assert windows.shortcut_calls == [(42, "v", "CONTROL")]


def test_emitter_stops_after_send_shortcut_returns_false(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  bus = RecordingBus()
  windows = RecordingWindowsProxy(
    [_focused_window_payload(), _focused_window_payload()],
    shortcut_results=[True, False],
  )
  clipboard = RecordingClipboardProxy(current_content="seed")
  _install_proxies(monkeypatch, bus=bus, windows=windows, clipboard=clipboard)

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()

  with pytest.raises(RuntimeError, match="failed to send paste shortcut"):
    emitter.emit_text("a" * 1601)

  assert [len(chunk) for chunk in clipboard.set_contents[:-1]] == [800, 800]
  assert clipboard.set_contents[-1] == "seed"
  assert windows.shortcut_calls == [
    (42, "v", "CONTROL"),
    (42, "v", "CONTROL"),
  ]


def test_emitter_rebinds_after_disconnected_bus_during_emit(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  @dataclass
  class DisconnectingWindowsProxy(RecordingWindowsProxy):
    disconnect_error: Exception | None = None

    @override
    def get_focused_window_sync(self) -> str:
      if self.disconnect_error is not None:
        disconnect_error = self.disconnect_error
        self.disconnect_error = None
        raise disconnect_error
      return super().get_focused_window_sync()

  first_bus = RecordingBus()
  second_bus = RecordingBus()
  first_windows = DisconnectingWindowsProxy(
    [_focused_window_payload()],
    disconnect_error=SdBusUnmappedMessageError(
      "System.Error.ENOTCONN",
      "Transport endpoint is not connected",
    ),
  )
  second_windows = RecordingWindowsProxy([_focused_window_payload(window_id=77)])
  first_clipboard = RecordingClipboardProxy()
  second_clipboard = RecordingClipboardProxy()
  buses = [first_bus, second_bus]
  windows_proxies = [first_windows, second_windows]
  clipboard_proxies = [first_clipboard, second_clipboard]

  def open_bus() -> RecordingBus:
    return buses.pop(0)

  def build_windows_proxy(
    *, service_name: str, object_path: str, bus: object
  ) -> RecordingWindowsProxy:
    _ = service_name
    _ = object_path
    _ = bus
    return windows_proxies.pop(0)

  def build_clipboard_proxy(
    *, service_name: str, object_path: str, bus: object
  ) -> RecordingClipboardProxy:
    _ = service_name
    _ = object_path
    _ = bus
    return clipboard_proxies.pop(0)

  monkeypatch.setattr("active_listener.infra.emitter.sd_bus_open_user", open_bus)
  monkeypatch.setattr(
    "active_listener.infra.emitter.WindowsExtensionInterface",
    build_windows_proxy,
  )
  monkeypatch.setattr(
    "active_listener.infra.emitter.ClipboardExtensionInterface",
    build_clipboard_proxy,
  )

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()
  emitter.emit_text("hello")

  assert first_bus.close_calls == 1
  assert second_bus.close_calls == 0
  assert second_clipboard.get_current_content_calls == 1
  assert second_clipboard.set_contents == ["hello", ""]
  assert second_windows.shortcut_calls == [(77, "v", "CONTROL")]


def test_emitter_restores_clipboard_after_chunk_write_failure(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  bus = RecordingBus()
  windows = RecordingWindowsProxy([_focused_window_payload(), _focused_window_payload()])
  clipboard = RecordingClipboardProxy(current_content="seed", fail_at_set_content_call=2)
  _install_proxies(monkeypatch, bus=bus, windows=windows, clipboard=clipboard)

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()

  with pytest.raises(RuntimeError, match="clipboard unavailable"):
    emitter.emit_text("a" * 1601)

  assert clipboard.get_current_content_calls == 1
  assert clipboard.set_contents == ["a" * 800, "seed"]
  assert windows.shortcut_calls == [(42, "v", "CONTROL")]


def test_emitter_logs_restore_failure_without_failing_successful_emit(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  bus = RecordingBus()
  windows = RecordingWindowsProxy([_focused_window_payload()])
  clipboard = RecordingClipboardProxy(current_content="seed", fail_at_set_content_call=2)
  logger = RecordingLogger()
  _install_proxies(monkeypatch, bus=bus, windows=windows, clipboard=clipboard)
  monkeypatch.setattr("active_listener.infra.emitter._logger", logger)

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()
  emitter.emit_text("hello")

  assert clipboard.get_current_content_calls == 1
  assert clipboard.set_contents == ["hello"]
  assert logger.exception_messages == ["clipboard restore failed after successful emission"]


def test_emitter_aborts_when_clipboard_snapshot_fails(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  bus = RecordingBus()
  windows = RecordingWindowsProxy([_focused_window_payload()])
  clipboard = RecordingClipboardProxy(fail_on_get_current_content=True)
  _install_proxies(monkeypatch, bus=bus, windows=windows, clipboard=clipboard)

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()

  with pytest.raises(RuntimeError, match="clipboard unavailable"):
    emitter.emit_text("hello")

  assert clipboard.get_current_content_calls == 1
  assert clipboard.set_contents == []
  assert windows.shortcut_calls == []


def test_emitter_raises_for_invalid_focused_window_payload(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  bus = RecordingBus()
  windows = RecordingWindowsProxy(["not-json"])
  clipboard = RecordingClipboardProxy()
  _install_proxies(monkeypatch, bus=bus, windows=windows, clipboard=clipboard)

  emitter = GnomeShellExtensionTextEmitter()
  emitter.initialize()

  with pytest.raises(ValidationError):
    emitter.emit_text("hello")


def test_emitter_requires_initialize_before_emit() -> None:
  emitter = GnomeShellExtensionTextEmitter()

  with pytest.raises(RuntimeError, match="initialize"):
    emitter.emit_text("hello")
