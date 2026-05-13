"""Persistent correction store tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from active_listener.infra.corrections import (
  CorrectionStore,
  CorrectionStoreError,
  apply_corrections,
  resolve_active_listener_corrections_path,
)


def test_resolve_active_listener_corrections_path_uses_eavesdrop_xdg_config_dir(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  config_home = tmp_path / "xdg-config"
  monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

  assert resolve_active_listener_corrections_path() == (
    config_home / "eavesdrop" / "active-listener.corrections.jsonc"
  )


def test_resolve_active_listener_corrections_path_falls_back_to_home_config_dir(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("XDG_CONFIG_HOME", "")
  monkeypatch.setattr(Path, "home", lambda: tmp_path)

  assert resolve_active_listener_corrections_path() == (
    tmp_path / ".config" / "eavesdrop" / "active-listener.corrections.jsonc"
  )


def test_missing_correction_file_loads_empty_map(tmp_path: Path) -> None:
  store = CorrectionStore(tmp_path / "missing.jsonc")

  assert store.load() == {}


def test_correction_store_rejects_non_string_values(tmp_path: Path) -> None:
  path = tmp_path / "corrections.jsonc"
  _ = path.write_text('{"alpha": 1}', encoding="utf-8")
  store = CorrectionStore(path)

  with pytest.raises(CorrectionStoreError, match="correction value must be a string"):
    _ = store.load()


def test_correction_store_preserves_comments_and_merges_values(tmp_path: Path) -> None:
  path = tmp_path / "corrections.jsonc"
  _ = path.write_text(
    '{\n  // keep this comment\n  "Kubernetties": "Kubernetes"\n}\n',
    encoding="utf-8",
  )
  store = CorrectionStore(path)

  store.merge({"Kubernetties": "Kubernetes", "pie torch": "PyTorch"})

  contents = path.read_text(encoding="utf-8")
  assert "// keep this comment" in contents
  assert store.load() == {"Kubernetties": "Kubernetes", "pie torch": "PyTorch"}


def test_correction_store_replaces_existing_value(tmp_path: Path) -> None:
  path = tmp_path / "corrections.jsonc"
  _ = path.write_text('{"pie torch":"Pie Torch"}', encoding="utf-8")
  store = CorrectionStore(path)

  store.merge({"pie torch": "PyTorch"})

  assert store.load() == {"pie torch": "PyTorch"}


def test_apply_corrections_uses_longest_key_first() -> None:
  assert apply_corrections(
    "I use pie torch and pie every day.",
    {"pie": "Pi", "pie torch": "PyTorch"},
  ) == "I use PyTorch and Pi every day."
