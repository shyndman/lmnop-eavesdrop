from __future__ import annotations

from pathlib import Path

import pytest

from active_listener.config import (
  load_active_listener_config,
  resolve_default_active_listener_config_path,
)


def test_resolve_default_active_listener_config_path_uses_xdg_config_home(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  config_home = tmp_path / "xdg-config"
  monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

  assert resolve_default_active_listener_config_path() == (
    config_home / "eavesdrop" / "active-listener.yaml"
  )


def test_resolve_default_active_listener_config_path_falls_back_for_empty_xdg_config_home(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("XDG_CONFIG_HOME", "")
  monkeypatch.setattr(Path, "home", lambda: tmp_path)

  assert resolve_default_active_listener_config_path() == (
    tmp_path / ".config" / "eavesdrop" / "active-listener.yaml"
  )


def test_resolve_default_active_listener_config_path_falls_back_for_missing_xdg_config_home(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
  monkeypatch.setattr(Path, "home", lambda: tmp_path)

  assert resolve_default_active_listener_config_path() == (
    tmp_path / ".config" / "eavesdrop" / "active-listener.yaml"
  )


def test_load_active_listener_config_resolves_default_path_at_call_time(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  config_home = tmp_path / "xdg-config"
  expected_path = config_home / "eavesdrop" / "active-listener.yaml"
  seen_paths: list[Path] = []

  monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
  monkeypatch.setattr(
    "active_listener.config.load_active_listener_config_file",
    lambda path: seen_paths.append(path) or {},
  )
  monkeypatch.setattr(
    "active_listener.config.ActiveListenerConfig.model_validate",
    lambda _data, strict: {"strict": strict},
  )

  config = load_active_listener_config(config_path=None, overrides={})

  assert seen_paths == [expected_path]
  assert config == {"strict": True}
