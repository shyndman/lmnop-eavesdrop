from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import yaml

from active_listener.app import ActiveListenerConfig

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config.yaml"


class ActiveListenerConfigFileError(RuntimeError):
  pass


def load_active_listener_config(
  *,
  config_path: str | None = None,
  overrides: Mapping[str, object | None],
) -> ActiveListenerConfig:
  resolved_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
  config_data = load_active_listener_config_file(resolved_path)
  merged_config = dict(config_data)

  for field_name, value in overrides.items():
    if value is not None:
      merged_config[field_name] = value

  return ActiveListenerConfig.model_validate(merged_config, strict=True)


def load_active_listener_config_file(path: Path) -> dict[str, object]:
  try:
    with path.open(encoding="utf-8") as config_file:
      loaded_data = cast(object, yaml.safe_load(config_file))
  except FileNotFoundError as exc:
    raise ActiveListenerConfigFileError(f"Active-listener config file not found: {path}") from exc
  except yaml.YAMLError as exc:
    raise ActiveListenerConfigFileError(
      f"Active-listener config file is invalid YAML: {path}"
    ) from exc

  if loaded_data is None:
    return {}
  if not isinstance(loaded_data, dict):
    raise ActiveListenerConfigFileError(
      f"Active-listener config file must contain a YAML mapping: {path}"
    )

  return dict(cast(dict[str, object], loaded_data))
