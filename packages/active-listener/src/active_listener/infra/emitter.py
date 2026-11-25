"""Text emission boundary for finalized dictation output."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple, Protocol, TypeVar, cast

import pydotool
from sdbus import (
  DbusInterfaceCommon,
  SdBus,
  SdBusUnmappedMessageError,
  dbus_method,
  sd_bus_open_user,
)

from eavesdrop.common import get_logger

DBUS_BUS_NAME = "org.gnome.Shell"
DESKTOP_OBJECT_PATH = "/org/gnome/Shell/Extensions/Eavesdrop"
DESKTOP_DBUS_INTERFACE_NAME = "org.gnome.Shell.Extensions.Eavesdrop"
YDOTOOL_SOCKET_NAME = ".ydotool_socket"
TERMINAL_PASTE_WM_CLASSES: frozenset[str] = frozenset({"kitty", "so.stencil.tern"})
DEFAULT_PASTE_KEYS = [pydotool.KEY_LEFTCTRL, pydotool.KEY_V]
TERMINAL_PASTE_KEYS = [
  pydotool.KEY_LEFTCTRL,
  pydotool.KEY_LEFTSHIFT,
  pydotool.KEY_V,
]
MAX_EMIT_CHUNK_LENGTH = 500
# Applications consume staged clipboard content asynchronously after key injection.
INTER_CHUNK_DELAY_MS = 150
CLIPBOARD_RESTORE_DELAY_MS = 750
DISCONNECTED_BUS_ERROR_NAME = "System.Error.ENOTCONN"


_logger = get_logger("al/emit")
ProxyCallResult = TypeVar("ProxyCallResult")


class FocusedWindow(NamedTuple):
  """Focused desktop window metadata returned by the Eavesdrop GNOME extension."""

  id: int
  wm_class: str


class ClipboardSnapshot(NamedTuple):
  """Captured clipboard payload restored after staged dictation chunks are pasted."""

  mime_type: str
  data: bytes


class DesktopExtensionInterface(
  DbusInterfaceCommon,
  interface_name=DESKTOP_DBUS_INTERFACE_NAME,
):
  """D-Bus interface exported by the Eavesdrop GNOME Shell extension."""

  @dbus_method(result_signature="ts", method_name="GetFocusedWindow")
  def get_focused_window(self) -> tuple[int, str]:
    raise NotImplementedError

  @dbus_method(result_signature="say", method_name="GetClipboardContent")
  def get_clipboard_content(self) -> tuple[str, bytes]:
    raise NotImplementedError

  @dbus_method(input_signature="say", method_name="SetClipboardContent")
  def set_clipboard_content(self, mime_type: str, data: bytes) -> None:
    _ = mime_type
    _ = data
    raise NotImplementedError

  @dbus_method(input_signature="s", method_name="SetClipboardText")
  def set_clipboard_text(self, text: str) -> None:
    _ = text
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


def _select_paste_keys(wm_class: str) -> list[int]:
  if wm_class in TERMINAL_PASTE_WM_CLASSES:
    return TERMINAL_PASTE_KEYS
  return DEFAULT_PASTE_KEYS


def _is_disconnected_bus_error(exc: Exception) -> bool:
  return isinstance(exc, SdBusUnmappedMessageError) and exc.args[:1] == (
    DISCONNECTED_BUS_ERROR_NAME,
  )


@dataclass
class GnomeShellExtensionTextEmitter:
  """Stage text through Eavesdrop's GNOME Shell extension and paste it through ydotool."""

  _initialized: bool = False
  _bus: SdBus | None = None
  _desktop: DesktopExtensionInterface | None = None

  def _bind_proxy(self) -> None:
    bus: SdBus | None = None
    try:
      _logger.debug("opening emitter session bus", bus_name=DBUS_BUS_NAME)
      bus = sd_bus_open_user()

      _logger.debug("binding desktop extension proxy", object_path=DESKTOP_OBJECT_PATH)
      desktop = DesktopExtensionInterface(
        service_name=DBUS_BUS_NAME,
        object_path=DESKTOP_OBJECT_PATH,
        bus=bus,
      )
    except Exception:
      if bus is not None:
        bus.close()
      _logger.exception("emitter initialization failed")
      raise

    self._bus = bus
    self._desktop = desktop
    self._initialized = True
    _logger.info("emitter initialized")

  def _close_proxy(self) -> None:
    if self._bus is not None:
      self._bus.close()

    self._bus = None
    self._desktop = None
    self._initialized = False

  def _rebind_proxy(self, *, operation: str) -> None:
    _logger.warning("emitter bus disconnected, rebinding proxy", operation=operation)
    self._close_proxy()
    self._bind_proxy()

  def _require_desktop(self) -> DesktopExtensionInterface:
    if not self._initialized or self._desktop is None:
      raise RuntimeError("GnomeShellExtensionTextEmitter.initialize() must run before emit_text()")
    return self._desktop

  def _snapshot_focused_window(self) -> FocusedWindow:
    try:
      window_id, wm_class = self._call_with_rebind(
        operation="focused_window_snapshot",
        call=lambda: self._require_desktop().get_focused_window(),
      )
    except Exception:
      _logger.exception("focused window snapshot failed")
      raise

    focused_window = FocusedWindow(id=window_id, wm_class=wm_class)
    _logger.debug(
      "focused window snapshot captured",
      window_id=focused_window.id,
      wm_class=focused_window.wm_class,
    )
    return focused_window

  def _snapshot_clipboard(self) -> ClipboardSnapshot:
    _logger.debug("snapshotting clipboard before emission")
    try:
      mime_type, data = self._call_with_rebind(
        operation="clipboard_get_content",
        call=lambda: self._require_desktop().get_clipboard_content(),
      )
    except Exception:
      _logger.exception("clipboard snapshot failed")
      raise

    snapshot = ClipboardSnapshot(mime_type=mime_type, data=data)
    _logger.debug(
      "clipboard snapshot captured",
      mime_type=snapshot.mime_type,
      byte_length=len(snapshot.data),
    )
    return snapshot

  def _restore_clipboard(self, snapshot: ClipboardSnapshot) -> None:
    if snapshot.mime_type == "":
      _logger.debug("skipping clipboard restore for empty snapshot")
      return

    _logger.debug(
      "waiting before clipboard restore",
      delay_ms=CLIPBOARD_RESTORE_DELAY_MS,
      mime_type=snapshot.mime_type,
    )
    time.sleep(CLIPBOARD_RESTORE_DELAY_MS / 1000)

    _logger.debug("restoring clipboard after emission", mime_type=snapshot.mime_type)
    self._call_with_rebind(
      operation="clipboard_restore_content",
      call=lambda: self._require_desktop().set_clipboard_content(
        snapshot.mime_type,
        snapshot.data,
      ),
    )
    _logger.debug("clipboard restore completed", mime_type=snapshot.mime_type)

  def _write_clipboard_chunk(self, *, chunk_index: int, chunk: str) -> None:
    _logger.debug(
      "writing chunk to clipboard",
      chunk_index=chunk_index,
      chunk_length=len(chunk),
    )
    try:
      self._call_with_rebind(
        operation="clipboard_set_text",
        call=lambda: self._require_desktop().set_clipboard_text(chunk),
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

  def _send_paste_shortcut(
    self,
    *,
    chunk_index: int,
    window_id: int,
    paste_keys: list[int],
  ) -> None:
    _logger.debug(
      "sending paste shortcut",
      chunk_index=chunk_index,
      window_id=window_id,
      key_codes=paste_keys,
    )
    try:
      pydotool.key_combination(paste_keys)
    except Exception:
      _logger.exception(
        "paste shortcut send failed",
        chunk_index=chunk_index,
        window_id=window_id,
        key_codes=paste_keys,
      )
      raise

  def _emit_chunks(
    self,
    *,
    focused_window: FocusedWindow,
    paste_keys: list[int],
    chunks: tuple[str, ...],
  ) -> None:
    for chunk_index, chunk in enumerate(chunks, start=1):
      self._write_clipboard_chunk(chunk_index=chunk_index, chunk=chunk)
      self._send_paste_shortcut(
        chunk_index=chunk_index,
        window_id=focused_window.id,
        paste_keys=paste_keys,
      )

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

    self._rebind_proxy(operation=operation)
    return call()

  def initialize(self) -> None:
    """Initialize the GNOME Shell and ydotool backends exactly once.

    :returns: None
    :rtype: None
    :raises KeyError: If ``XDG_RUNTIME_DIR`` is absent.
    """

    if self._initialized:
      _logger.debug("emitter already initialized")
      return

    socket_path = os.path.join(os.environ["XDG_RUNTIME_DIR"], YDOTOOL_SOCKET_NAME)
    self._bind_proxy()
    try:
      _logger.debug("initializing ydotool client", socket_path=socket_path)
      # HACK: python-ydotool 1.1.1 documents and implements the socket-path
      # argument, but its bundled py.typed declaration incorrectly exposes
      # init as Callable[[], None]. Keep the cast at this package boundary
      # until upstream's declaration matches the runtime API.
      initialize_ydotool = cast(Callable[[str], None], pydotool.init)
      initialize_ydotool(socket_path)
    except Exception:
      self._close_proxy()
      _logger.exception("ydotool initialization failed", socket_path=socket_path)
      raise

  def emit_text(self, text: str) -> None:
    """Emit finalized text through the GNOME Shell extension backend.

    :param text: Finalized committed text.
    :type text: str
    :returns: None
    :rtype: None
    :raises RuntimeError: If ``initialize()`` has not run yet.
    """

    focused_window = self._snapshot_focused_window()
    paste_keys = _select_paste_keys(focused_window.wm_class)
    chunks = _chunk_text(text)

    _logger.debug(
      "paste shortcut selected",
      window_id=focused_window.id,
      key_codes=paste_keys,
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

    original_clipboard = self._snapshot_clipboard()
    emission_succeeded = False
    try:
      self._emit_chunks(
        focused_window=focused_window,
        paste_keys=paste_keys,
        chunks=chunks,
      )
      emission_succeeded = True
    finally:
      try:
        self._restore_clipboard(original_clipboard)
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
