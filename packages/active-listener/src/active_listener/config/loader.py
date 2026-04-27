from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import yaml

from active_listener.config.models import ActiveListenerConfig

USER_CONFIG_ENV_VAR = "XDG_CONFIG_HOME"
DEFAULT_USER_CONFIG_DIRNAME = ".config"
APP_CONFIG_DIRNAME = "eavesdrop"
ACTIVE_LISTENER_CONFIG_FILENAME = "active-listener.yaml"


def resolve_default_active_listener_config_path() -> Path:
  configured_path = os.environ.get(USER_CONFIG_ENV_VAR)
  base_config_dir = (
    Path(configured_path).expanduser()
    if configured_path is not None and configured_path != ""
    else Path.home() / DEFAULT_USER_CONFIG_DIRNAME
  )
  return base_config_dir / APP_CONFIG_DIRNAME / ACTIVE_LISTENER_CONFIG_FILENAME


class ActiveListenerConfigFileError(RuntimeError):
  pass


def load_active_listener_config(
  *,
  config_path: str | None = None,
  overrides: Mapping[str, object | None],
) -> ActiveListenerConfig:
  resolved_path = resolve_active_listener_config_path(config_path)
  config_data = load_active_listener_config_file(resolved_path)
  merged_config = dict(config_data)

  for field_name, value in overrides.items():
    if value is not None:
      merged_config[field_name] = value

  normalized_config = normalize_active_listener_config_paths(
    merged_config,
    config_dir=resolved_path.parent,
  )

  return ActiveListenerConfig.model_validate(normalized_config, strict=True)


def resolve_active_listener_config_path(config_path: str | None) -> Path:
  if config_path is None:
    return resolve_default_active_listener_config_path()

  return Path(config_path).expanduser().resolve()


def normalize_active_listener_config_paths(
  config_data: Mapping[str, object],
  *,
  config_dir: Path,
) -> dict[str, object]:
  normalized_config = dict(config_data)
  normalized_config["ffmpeg_path"] = normalize_config_path_value(
    normalized_config.get("ffmpeg_path"),
    config_dir=config_dir,
  )
  raw_rewrite_config = normalized_config.get("llm_rewrite")
  if not isinstance(raw_rewrite_config, Mapping):
    return normalized_config

  normalized_rewrite_config = dict(cast(Mapping[str, object], raw_rewrite_config))
  normalized_rewrite_config["prompt_path"] = normalize_config_path_value(
    normalized_rewrite_config.get("prompt_path"),
    config_dir=config_dir,
  )
  raw_provider_config = normalized_rewrite_config.get("provider")
  if isinstance(raw_provider_config, Mapping):
    normalized_provider_config = dict(cast(Mapping[str, object], raw_provider_config))
    if normalized_provider_config.get("type") == "litert":
      normalized_provider_config["model_path"] = normalize_config_path_value(
        normalized_provider_config.get("model_path"),
        config_dir=config_dir,
      )
    normalized_rewrite_config["provider"] = normalized_provider_config

  normalized_config["llm_rewrite"] = normalized_rewrite_config
  return normalized_config


def normalize_config_path_value(value: object, *, config_dir: Path) -> object:
  if not isinstance(value, str):
    return value

  configured_path = Path(value).expanduser()
  if not configured_path.is_absolute():
    configured_path = config_dir / configured_path

  return str(configured_path)


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
