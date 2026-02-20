"""
Constants for the Eavesdrop application.

These values are hardcoded and not configurable through the config file or CLI arguments.
"""

from __future__ import annotations

import os
from pathlib import Path


def _resolve_cache_path() -> str:
  """Resolve a writable cache directory respecting XDG conventions.

  :returns: Absolute path to the eavesdrop cache directory.
  :rtype: str
  """

  xdg_cache = os.environ.get("XDG_CACHE_HOME")
  base_cache = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"
  cache_dir = (base_cache / "eavesdrop").expanduser().resolve()
  cache_dir.mkdir(parents=True, exist_ok=True)
  return str(cache_dir)


CACHE_PATH = _resolve_cache_path()
"""Path for model caching resolved at runtime."""

SINGLE_MODEL = True
"""Whether to use single shared model instance - hardcoded constant."""

SAMPLE_RATE = 16_000
"""Sample rate required for the Whisper model - hardcoded constant."""
