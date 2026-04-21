from __future__ import annotations

from pathlib import Path

import pytest

from active_listener.config.loader import (
  load_active_listener_config,
  resolve_default_active_listener_config_path,
)
from active_listener.config.models import (
  LiteRtRewriteProvider,
  LlmRewriteConfig,
  PydanticAiRewriteProvider,
)


def _write_config(
  path: Path,
  *,
  prompt_path: str = "prompts/rewrite_prompt.md",
  provider_block: str | None = None,
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  resolved_provider_block = _litert_provider_block() if provider_block is None else provider_block
  llm_rewrite_lines = [
    "llm_rewrite:",
    f'  prompt_path: "{prompt_path}"',
    *resolved_provider_block.splitlines(),
  ]
  _ = path.write_text(
    "\n".join(
      [
        'keyboard_name: "Config Keyboard"',
        'host: "config.local"',
        "port: 9090",
        'audio_device: "config-device"',
        "",
        *llm_rewrite_lines,
        "",
      ]
    ),
    encoding="utf-8",
  )


def _litert_provider_block(*, model_path: str = "models/rewrite.litertlm") -> str:
  return "\n".join(
    [
      "  provider:",
      "    type: litert",
      f'    model_path: "{model_path}"',
    ]
  )


def _pydantic_ai_provider_block(*, model: str = "openai:gpt-4.1-mini") -> str:
  return "\n".join(
    [
      "  provider:",
      "    type: pydantic_ai",
      f'    model: "{model}"',
    ]
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

  assert config.llm_rewrite == LlmRewriteConfig(
    prompt_path=str(config_path.parent / "prompts" / "rewrite_prompt.md"),
    provider=LiteRtRewriteProvider(
      type="litert",
      model_path=str(config_path.parent / "models" / "rewrite.litertlm"),
    ),
  )


def test_load_active_listener_config_preserves_absolute_rewrite_paths(tmp_path: Path) -> None:
  model_path = tmp_path / "models" / "rewrite.litertlm"
  prompt_path = tmp_path / "prompts" / "rewrite_prompt.md"
  config_path = tmp_path / "configs" / "active-listener.yaml"
  _write_config(
    config_path,
    prompt_path=str(prompt_path),
    provider_block=_litert_provider_block(model_path=str(model_path)),
  )

  config = load_active_listener_config(config_path=str(config_path), overrides={})

  assert config.llm_rewrite == LlmRewriteConfig(
    prompt_path=str(prompt_path),
    provider=LiteRtRewriteProvider(type="litert", model_path=str(model_path)),
  )


def test_load_active_listener_config_uses_default_xdg_path_for_relative_rewrite_paths(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  config_home = tmp_path / "xdg-config"
  config_path = config_home / "eavesdrop" / "active-listener.yaml"
  _write_config(config_path)
  monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

  config = load_active_listener_config(config_path=None, overrides={})

  assert config.llm_rewrite == LlmRewriteConfig(
    prompt_path=str(config_path.parent / "prompts" / "rewrite_prompt.md"),
    provider=LiteRtRewriteProvider(
      type="litert",
      model_path=str(config_path.parent / "models" / "rewrite.litertlm"),
    ),
  )


def test_load_active_listener_config_preserves_absolute_symlink_model_path(tmp_path: Path) -> None:
  model_dir = tmp_path / "models"
  model_dir.mkdir()
  real_model_path = model_dir / "bundle.bin"
  _ = real_model_path.write_text("model", encoding="utf-8")
  symlink_model_path = model_dir / "rewrite.litertlm"
  symlink_model_path.symlink_to(real_model_path)

  config_path = tmp_path / "configs" / "active-listener.yaml"
  _write_config(
    config_path, provider_block=_litert_provider_block(model_path=str(symlink_model_path))
  )

  config = load_active_listener_config(config_path=str(config_path), overrides={})

  assert config.llm_rewrite == LlmRewriteConfig(
    prompt_path=str(config_path.parent / "prompts" / "rewrite_prompt.md"),
    provider=LiteRtRewriteProvider(type="litert", model_path=str(symlink_model_path)),
  )


def test_load_active_listener_config_disables_rewrite_when_block_missing(tmp_path: Path) -> None:
  config_path = tmp_path / "configs" / "active-listener.yaml"
  config_path.parent.mkdir(parents=True, exist_ok=True)
  _ = config_path.write_text(
    "\n".join(
      [
        'keyboard_name: "Config Keyboard"',
        'host: "config.local"',
        "port: 9090",
        'audio_device: "config-device"',
        "",
      ]
    ),
    encoding="utf-8",
  )

  config = load_active_listener_config(config_path=str(config_path), overrides={})

  assert config.llm_rewrite is None


def test_load_active_listener_config_loads_pydantic_ai_provider(tmp_path: Path) -> None:
  config_path = tmp_path / "configs" / "active-listener.yaml"
  _write_config(
    config_path, provider_block=_pydantic_ai_provider_block(model="openai:gpt-4.1-mini")
  )

  config = load_active_listener_config(config_path=str(config_path), overrides={})

  assert config.llm_rewrite == LlmRewriteConfig(
    prompt_path=str(config_path.parent / "prompts" / "rewrite_prompt.md"),
    provider=PydanticAiRewriteProvider(type="pydantic_ai", model="openai:gpt-4.1-mini"),
  )


def test_load_active_listener_config_rejects_missing_provider(tmp_path: Path) -> None:
  config_path = tmp_path / "configs" / "active-listener.yaml"
  _write_config(config_path, provider_block="")

  with pytest.raises(ValueError, match="provider"):
    _ = load_active_listener_config(config_path=str(config_path), overrides={})


def test_load_active_listener_config_rejects_missing_provider_type(tmp_path: Path) -> None:
  config_path = tmp_path / "configs" / "active-listener.yaml"
  _write_config(
    config_path,
    provider_block="\n".join(
      [
        "  provider:",
        '    model_path: "models/rewrite.litertlm"',
      ]
    ),
  )

  with pytest.raises(ValueError, match="type"):
    _ = load_active_listener_config(config_path=str(config_path), overrides={})
