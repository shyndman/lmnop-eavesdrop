"""Persistent spelling correction store."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol, cast, final

from active_listener.infra.user_config import EAVESDROP_CONFIG_DIRNAME, resolve_user_config_dir

ACTIVE_LISTENER_CORRECTIONS_FILENAME = "active-listener.corrections.jsonc"
CorrectionMap = Mapping[str, str]


class _JSONCGXObject(Protocol):
  def items(self) -> Iterator[tuple[str, object]]: ...

  def lookup(self, name: str) -> Iterator[int]: ...

  def editreplacevalue(self, index: int, newvalue: object) -> None: ...

  def editinsert(self, index: int, name: str, value: object) -> None: ...


class _JSONCGXEditor(Protocol):
  root: _JSONCGXObject

  def dumps(self) -> str: ...


class _JSONCGXBuiltin(Protocol):
  value: object


class _JSONCGXModule(Protocol):
  def loadf(self, path: os.PathLike[str], allow_comments: bool = True) -> _JSONCGXEditor: ...

  def loads(
    self,
    s: str | bytes,
    name: str | None = None,
    allow_comments: bool = True,
  ) -> _JSONCGXEditor: ...


class CorrectionStoreError(RuntimeError):
  pass


class ActiveListenerCorrectionStore(Protocol):
  @property
  def path(self) -> Path: ...

  async def load_async(self) -> dict[str, str]: ...

  async def merge_async(self, corrections: CorrectionMap) -> None: ...


@final
@dataclass(frozen=True)
class CorrectionStore:
  path: Path

  @classmethod
  def default(cls) -> CorrectionStore:
    return cls(resolve_active_listener_corrections_path())

  async def load_async(self) -> dict[str, str]:
    return await asyncio.to_thread(self.load)

  async def merge_async(self, corrections: CorrectionMap) -> None:
    await asyncio.to_thread(lambda: self.merge(corrections))

  def load(self) -> dict[str, str]:
    if not self.path.exists():
      return {}

    editor = _load_jsoncgx().loadf(self.path, allow_comments=True)
    return _read_string_map(editor.root, path=self.path)

  def merge(self, corrections: CorrectionMap) -> None:
    validated_corrections = validate_corrections(corrections)
    if not validated_corrections:
      return

    self.path.parent.mkdir(parents=True, exist_ok=True)
    editor = _load_editor(self.path)
    existing = _read_string_map(editor.root, path=self.path)
    changed = False

    for key, value in validated_corrections.items():
      if existing.get(key) == value:
        continue

      indexes = list(editor.root.lookup(key))
      if indexes:
        editor.root.editreplacevalue(indexes[-1], value)
      else:
        editor.root.editinsert(len(existing), key, value)
      existing[key] = value
      changed = True

    if changed:
      _write_atomic(self.path, editor.dumps())


def resolve_active_listener_corrections_path() -> Path:
  return resolve_user_config_dir() / EAVESDROP_CONFIG_DIRNAME / ACTIVE_LISTENER_CORRECTIONS_FILENAME


def validate_corrections(corrections: CorrectionMap) -> dict[str, str]:
  validated: dict[str, str] = {}
  for key, value in corrections.items():
    if key == "" or value == "" or key == value:
      raise CorrectionStoreError("correction entries must map distinct non-empty strings")
    validated[key] = value
  return validated


def apply_corrections(transcript: str, corrections: CorrectionMap) -> str:
  corrected_transcript = transcript
  for key in sorted(corrections, key=lambda item: (-len(item), item)):
    corrected_transcript = corrected_transcript.replace(key, corrections[key])
  return corrected_transcript


def _load_editor(path: Path) -> _JSONCGXEditor:
  jsoncgx = _load_jsoncgx()
  if path.exists():
    return jsoncgx.loadf(path, allow_comments=True)
  return jsoncgx.loads("{}", allow_comments=True)


def _read_string_map(root: _JSONCGXObject, *, path: Path) -> dict[str, str]:
  corrections: dict[str, str] = {}
  for key, value in root.items():
    try:
      primitive_value = cast(_JSONCGXBuiltin, value).value
    except AttributeError as exc:
      raise CorrectionStoreError(f"correction value must be a string: {path}:{key}") from exc
    if not isinstance(primitive_value, str):
      raise CorrectionStoreError(f"correction value must be a string: {path}:{key}")
    corrections[key] = primitive_value
  return validate_corrections(corrections)


def _load_jsoncgx() -> _JSONCGXModule:
  module = import_module("jsoncgx")
  return cast(_JSONCGXModule, cast(object, module))


def _write_atomic(path: Path, contents: str) -> None:
  with NamedTemporaryFile(
    "w",
    encoding="utf-8",
    dir=path.parent,
    prefix=f".{path.name}.",
    delete=False,
  ) as temp_file:
    _ = temp_file.write(contents)
    temp_path = Path(temp_file.name)

  try:
    _ = temp_path.replace(path)
  except Exception:
    temp_path.unlink(missing_ok=True)
    raise
