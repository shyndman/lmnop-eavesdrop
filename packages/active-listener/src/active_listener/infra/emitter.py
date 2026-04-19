"""Text emission boundary for finalized dictation output."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import pydotool


def _candidate_socket_paths() -> tuple[Path, ...]:
  uid = os.getuid()
  return (
    Path(f"/run/user/{uid}/.ydotool_socket"),
    Path("/tmp/.ydotool_socket"),
  )


def _initialize_pydotool(socket_path: str | None) -> None:
  init_with_socket = cast(Callable[[str], None], pydotool.init)

  if socket_path is not None:
    init_with_socket(socket_path)
    return

  for candidate in _candidate_socket_paths():
    try:
      init_with_socket(str(candidate))
      return
    except OSError:
      continue

  pydotool.init()


class TextEmitter(Protocol):
  """Protocol for workstation text emission backends."""

  def initialize(self) -> None:
    """Prepare the backend for later emission calls."""

  def emit_text(self, text: str) -> None:
    """Emit finalized text into the currently focused application."""


@dataclass
class PydotoolTextEmitter:
  """Text emitter backed by :mod:`pydotool`.

  :param socket_path: Optional custom ``ydotoold`` socket path.
  :type socket_path: str | None
  """

  socket_path: str | None = None
  _initialized: bool = False

  def initialize(self) -> None:
    """Initialize the ydotool backend exactly once.

    :returns: None
    :rtype: None
    """

    if self._initialized:
      return

    _initialize_pydotool(self.socket_path)
    self._initialized = True

  def emit_text(self, text: str) -> None:
    """Emit finalized text through the ydotool backend.

    :param text: Finalized committed text.
    :type text: str
    :returns: None
    :rtype: None
    :raises RuntimeError: If ``initialize()`` has not run yet.
    """

    if not self._initialized:
      raise RuntimeError("PydotoolTextEmitter.initialize() must run before emit_text()")

    pydotool.type_string(text, hold_delay_ms=1, each_char_delay_ms=1)
