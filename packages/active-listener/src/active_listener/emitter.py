"""Text emission boundary for finalized dictation output."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import pydotool


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

    if self.socket_path is None:
      pydotool.init()
    else:
      init_with_socket = cast(Callable[[str], None], pydotool.init)
      init_with_socket(self.socket_path)
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

    pydotool.type_string(text)
