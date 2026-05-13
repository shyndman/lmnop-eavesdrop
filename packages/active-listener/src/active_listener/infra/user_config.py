"""User configuration path helpers for active-listener."""

from __future__ import annotations

import os
from pathlib import Path

USER_CONFIG_ENV_VAR = "XDG_CONFIG_HOME"
DEFAULT_USER_CONFIG_DIRNAME = ".config"
EAVESDROP_CONFIG_DIRNAME = "eavesdrop"


def resolve_user_config_dir() -> Path:
  configured_path = os.environ.get(USER_CONFIG_ENV_VAR)
  if configured_path is not None and configured_path != "":
    return Path(configured_path)

  return Path.home() / DEFAULT_USER_CONFIG_DIRNAME
