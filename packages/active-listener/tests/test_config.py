from __future__ import annotations

from pathlib import Path

import pytest

from active_listener.config.loader import (
  load_active_listener_config,
  resolve_default_active_listener_config_path,
)


def _write_config(
  path: Path,
  *,
  model_path: str = "models/rewrite.litertlm",
  prompt_path: str = "prompts/rewrite_prompt.md",
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  _ = path.write_text(
    "\n".join(
      [
        'keyboard_name: "Config Keyboard"',
        'host: "config.local"',
        "port: 9090",
        'audio_device: "config-device"',
        'ydotool_socket: "/tmp/config.sock"',
        "",
        "llm_rewrite:",
        "  enabled: true",
        f'  model_path: "{model_path}"',
        f'  prompt_path: "{prompt_path}"',
        "",
      ]
    ),
    encoding="utf-8",
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


def test_load_active_listener_config_resolves_relative_rewrite_paths_against_config_dir(
  tmp_path: Path,
) -> None:
  config_path = tmp_path / "configs" / "active-listener.yaml"
  _write_config(config_path)

  config = load_active_listener_config(config_path=str(config_path), overrides={})

  assert config.llm_rewrite.model_path == str(
    config_path.parent / "models" / "rewrite.litertlm"
  )
  assert config.llm_rewrite.prompt_path == str(
    config_path.parent / "prompts" / "rewrite_prompt.md"
  )


def test_load_active_listener_config_preserves_absolute_rewrite_paths(tmp_path: Path) -> None:
  model_path = tmp_path / "models" / "rewrite.litertlm"
  prompt_path = tmp_path / "prompts" / "rewrite_prompt.md"
  config_path = tmp_path / "configs" / "active-listener.yaml"
  _write_config(
    config_path,
    model_path=str(model_path),
    prompt_path=str(prompt_path),
  )

  config = load_active_listener_config(config_path=str(config_path), overrides={})

  assert config.llm_rewrite.model_path == str(model_path)
  assert config.llm_rewrite.prompt_path == str(prompt_path)


def test_load_active_listener_config_uses_default_xdg_path_for_relative_rewrite_paths(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  config_home = tmp_path / "xdg-config"
  config_path = config_home / "eavesdrop" / "active-listener.yaml"
  _write_config(config_path)
  monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

  config = load_active_listener_config(config_path=None, overrides={})

  assert config.llm_rewrite.model_path == str(
    config_path.parent / "models" / "rewrite.litertlm"
  )
  assert config.llm_rewrite.prompt_path == str(
    config_path.parent / "prompts" / "rewrite_prompt.md"
  )


def test_load_active_listener_config_preserves_absolute_symlink_model_path(tmp_path: Path) -> None:
  model_dir = tmp_path / "models"
  model_dir.mkdir()
  real_model_path = model_dir / "bundle.bin"
  _ = real_model_path.write_text("model", encoding="utf-8")
  symlink_model_path = model_dir / "rewrite.litertlm"
  symlink_model_path.symlink_to(real_model_path)

  config_path = tmp_path / "configs" / "active-listener.yaml"
  _write_config(config_path, model_path=str(symlink_model_path))

  config = load_active_listener_config(config_path=str(config_path), overrides={})

  assert config.llm_rewrite.model_path == str(symlink_model_path)
