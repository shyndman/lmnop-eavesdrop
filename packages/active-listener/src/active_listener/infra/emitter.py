"""Text emission boundary for finalized dictation output."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict
from sdbus import (
  DbusInterfaceCommon,
  SdBus,
  SdBusUnmappedMessageError,
  dbus_method,
  sd_bus_open_user,
)

from eavesdrop.common import get_logger

DBUS_BUS_NAME = "org.gnome.Shell"
WINDOWS_OBJECT_PATH = "/org/gnome/Shell/Extensions/Windows"
CLIPBOARD_OBJECT_PATH = "/org/gnome/Shell/Extensions/Clipboard"
WINDOWS_DBUS_INTERFACE_NAME = "org.gnome.Shell.Extensions.Windows"
CLIPBOARD_DBUS_INTERFACE_NAME = "org.gnome.Shell.Extensions.Clipboard"
PASTE_KEY = "v"
DEFAULT_PASTE_MODIFIERS = "CONTROL"
KITTY_PASTE_MODIFIERS = "CONTROL|SHIFT"
KITTY_WM_CLASS = "kitty"
MAX_EMIT_CHUNK_LENGTH = 800
# Vicinae schedules the actual paste asynchronously after a 100 ms timeout.
INTER_CHUNK_DELAY_MS = 150
DISCONNECTED_BUS_ERROR_NAME = "System.Error.ENOTCONN"


_logger = get_logger("al/emit")
ProxyCallResult = TypeVar("ProxyCallResult")


class FocusedWindowDisplay(BaseModel):
  model_config: ClassVar[ConfigDict] = ConfigDict(strict=True, extra="allow")


class FocusedWindow(BaseModel):
  model_config: ClassVar[ConfigDict] = ConfigDict(strict=True, extra="allow")

  id: int
  title: str
  wm_class: str
  wm_class_instance: str
  pid: int
  maximized: bool
  display: FocusedWindowDisplay
  frame_type: int
  window_type: int
  layer: int
  monitor: int
  role: str | None
  width: int
  height: int
  x: int
  y: int
  in_current_workspace: bool
  canclose: bool
  canmaximize: bool
  canminimize: bool
  canshade: bool
  moveable: bool
  resizeable: bool
  has_focus: bool
  workspace: int


class WindowsExtensionInterface(DbusInterfaceCommon, interface_name=WINDOWS_DBUS_INTERFACE_NAME):
  @dbus_method(method_name="GetFocusedWindowSync")
  def get_focused_window_sync(self) -> str:
    raise NotImplementedError

  @dbus_method("uss", method_name="SendShortcut")
  def send_shortcut(self, winid: int, key: str, modifiers: str) -> bool:
    _ = winid
    _ = key
    _ = modifiers
    raise NotImplementedError


class ClipboardExtensionInterface(
  DbusInterfaceCommon,
  interface_name=CLIPBOARD_DBUS_INTERFACE_NAME,
):
  @dbus_method(method_name="GetCurrentContent")
  def get_current_content(self) -> str:
    raise NotImplementedError

  @dbus_method("s", method_name="SetContent")
  def set_content(self, content: str) -> None:
    _ = content
    raise NotImplementedError


class TextEmitter(Protocol):
  """Protocol for workstation text emission backends."""

  def initialize(self) -> None:
    """Prepare the backend for later emission calls."""

  def emit_text(self, text: str) -> None:
    """Emit finalized text into the currently focused application."""


def _chunk_text(text: str) -> tuple[str, ...]:
  if text == "":
    return ()

  return tuple(
    text[offset : offset + MAX_EMIT_CHUNK_LENGTH]
    for offset in range(0, len(text), MAX_EMIT_CHUNK_LENGTH)
  )


def _parse_focused_window(payload: str) -> FocusedWindow:
  return FocusedWindow.model_validate_json(payload, strict=True)


def _select_paste_modifiers(wm_class: str) -> str:
  if wm_class == KITTY_WM_CLASS:
    return KITTY_PASTE_MODIFIERS
  return DEFAULT_PASTE_MODIFIERS


def _is_disconnected_bus_error(exc: Exception) -> bool:
  return isinstance(exc, SdBusUnmappedMessageError) and exc.args[:1] == (
    DISCONNECTED_BUS_ERROR_NAME,
  )


@dataclass
class GnomeShellExtensionTextEmitter:
  """Text emitter backed by Vicinae's GNOME Shell D-Bus extension."""

  _initialized: bool = False
  _bus: SdBus | None = None
  _windows: WindowsExtensionInterface | None = None
  _clipboard: ClipboardExtensionInterface | None = None

  def _bind_proxies(self) -> None:
    bus: SdBus | None = None
    try:
      _logger.debug("opening emitter session bus", bus_name=DBUS_BUS_NAME)
      bus = sd_bus_open_user()
      _logger.debug("binding windows proxy", object_path=WINDOWS_OBJECT_PATH)
      windows = WindowsExtensionInterface(
        service_name=DBUS_BUS_NAME,
        object_path=WINDOWS_OBJECT_PATH,
        bus=bus,
      )
      _logger.debug("binding clipboard proxy", object_path=CLIPBOARD_OBJECT_PATH)
      clipboard = ClipboardExtensionInterface(
        service_name=DBUS_BUS_NAME,
        object_path=CLIPBOARD_OBJECT_PATH,
        bus=bus,
      )
    except Exception:
      if bus is not None:
        bus.close()
      _logger.exception("emitter initialization failed")
      raise

    self._bus = bus
    self._windows = windows
    self._clipboard = clipboard
    self._initialized = True
    _logger.info("emitter initialized")

  def _close_proxies(self) -> None:
    if self._bus is not None:
      self._bus.close()

    self._bus = None
    self._windows = None
    self._clipboard = None
    self._initialized = False

  def _rebind_proxies(self, *, operation: str) -> None:
    _logger.warning("emitter bus disconnected, rebinding proxies", operation=operation)
    self._close_proxies()
    self._bind_proxies()

  def _require_windows(self) -> WindowsExtensionInterface:
    if not self._initialized or self._windows is None:
      raise RuntimeError("GnomeShellExtensionTextEmitter.initialize() must run before emit_text()")
    return self._windows

  def _require_clipboard(self) -> ClipboardExtensionInterface:
    if not self._initialized or self._clipboard is None:
      raise RuntimeError("GnomeShellExtensionTextEmitter.initialize() must run before emit_text()")
    return self._clipboard

  def _snapshot_clipboard(self) -> str:
    _logger.debug("snapshotting clipboard before emission")
    try:
      clipboard_content = self._call_with_rebind(
        operation="clipboard_get_current_content",
        call=lambda: self._require_clipboard().get_current_content(),
      )
    except Exception:
      _logger.exception("clipboard snapshot failed")
      raise
    _logger.debug("clipboard snapshot captured")
    return clipboard_content

  def _restore_clipboard(self, clipboard_content: str) -> None:
    _logger.debug("restoring clipboard after emission")
    self._call_with_rebind(
      operation="clipboard_restore_content",
      call=lambda: self._require_clipboard().set_content(clipboard_content),
    )
    _logger.debug("clipboard restore completed")

  def _emit_chunks(
    self,
    *,
    focused_window: FocusedWindow,
    paste_modifiers: str,
    chunks: tuple[str, ...],
  ) -> None:
    for chunk_index, chunk in enumerate(chunks, start=1):
      _logger.debug(
        "writing chunk to clipboard",
        chunk_index=chunk_index,
        chunk_length=len(chunk),
      )
      try:
        self._call_with_rebind(
          operation="clipboard_set_content",
          call=lambda: self._require_clipboard().set_content(chunk),
        )
      except Exception:
        _logger.exception(
          "clipboard chunk write failed",
          chunk_index=chunk_index,
          chunk_length=len(chunk),
        )
        raise
      _logger.debug(
        "clipboard chunk written",
        chunk_index=chunk_index,
        chunk_length=len(chunk),
      )

      _logger.debug(
        "sending paste shortcut",
        chunk_index=chunk_index,
        window_id=focused_window.id,
        key=PASTE_KEY,
        modifiers=paste_modifiers,
      )
      try:
        did_send = self._call_with_rebind(
          operation="send_shortcut",
          call=lambda: self._require_windows().send_shortcut(
            focused_window.id,
            PASTE_KEY,
            paste_modifiers,
          ),
        )
      except Exception:
        _logger.exception(
          "paste shortcut send failed",
          chunk_index=chunk_index,
          window_id=focused_window.id,
          key=PASTE_KEY,
          modifiers=paste_modifiers,
        )
        raise
      _logger.debug(
        "paste shortcut result",
        chunk_index=chunk_index,
        window_id=focused_window.id,
        key=PASTE_KEY,
        modifiers=paste_modifiers,
        success=did_send,
      )
      if not did_send:
        raise RuntimeError(f"failed to send paste shortcut to window {focused_window.id}")
      if chunk_index < len(chunks):
        _logger.debug(
          "waiting for pasted chunk to settle",
          chunk_index=chunk_index,
          chunk_count=len(chunks),
          delay_ms=INTER_CHUNK_DELAY_MS,
        )
        time.sleep(INTER_CHUNK_DELAY_MS / 1000)

  def _call_with_rebind(
    self,
    *,
    operation: str,
    call: Callable[[], ProxyCallResult],
  ) -> ProxyCallResult:
    try:
      return call()
    except Exception as exc:
      if not _is_disconnected_bus_error(exc):
        raise

    self._rebind_proxies(operation=operation)
    return call()

  def initialize(self) -> None:
    """Initialize the GNOME Shell extension backend exactly once.

    :returns: None
    :rtype: None
    """

    if self._initialized:
      _logger.debug("emitter already initialized")
      return

    self._bind_proxies()

  def emit_text(self, text: str) -> None:
    """Emit finalized text through the GNOME Shell extension backend.

    :param text: Finalized committed text.
    :type text: str
    :returns: None
    :rtype: None
    :raises RuntimeError: If ``initialize()`` has not run yet.
    """

    _ = self._require_windows()
    _ = self._require_clipboard()

    try:
      focused_window_payload = self._call_with_rebind(
        operation="focused_window_snapshot",
        call=lambda: self._require_windows().get_focused_window_sync(),
      )
      focused_window = _parse_focused_window(focused_window_payload)
    except Exception:
      _logger.exception("focused window snapshot invalid")
      raise

    paste_modifiers = _select_paste_modifiers(focused_window.wm_class)
    chunks = _chunk_text(text)
    _logger.debug(
      "focused window snapshot captured",
      window_id=focused_window.id,
      wm_class=focused_window.wm_class,
    )
    _logger.debug(
      "paste shortcut selected",
      window_id=focused_window.id,
      key=PASTE_KEY,
      modifiers=paste_modifiers,
    )
    _logger.debug(
      "text chunked for emission",
      text_length=len(text),
      chunk_count=len(chunks),
    )

    if not chunks:
      _logger.debug(
        "text emission completed",
        chunk_count=0,
        window_id=focused_window.id,
      )
      return

    original_clipboard_content = self._snapshot_clipboard()
    emission_succeeded = False
    try:
      self._emit_chunks(
        focused_window=focused_window,
        paste_modifiers=paste_modifiers,
        chunks=chunks,
      )
      emission_succeeded = True
    finally:
      try:
        self._restore_clipboard(original_clipboard_content)
      except Exception:
        if emission_succeeded:
          _logger.exception("clipboard restore failed after successful emission")
        else:
          _logger.exception("clipboard restore failed after emission failure")

    _logger.debug(
      "text emission completed",
      chunk_count=len(chunks),
      window_id=focused_window.id,
    )
