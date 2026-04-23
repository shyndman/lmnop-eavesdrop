from __future__ import annotations

import importlib.util
import sqlite3
import sys
from decimal import Decimal
from pathlib import Path
from typing import Protocol, cast

import pytest

from active_listener.infra.transcript_history import (
  TRANSCRIPT_HISTORY_TABLE,
  ensure_transcript_history_schema,
)

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "transcript_history_totals.py"


class TotalsScriptModule(Protocol):
  def query_totals(self, database_path: Path) -> TranscriptHistoryTotalsLike: ...

  def main(self, argv: list[str]) -> int: ...


class TranscriptHistoryTotalsLike(Protocol):
  utterances: int
  word_count: int
  tokens_up: int
  tokens_down: int
  cost: Decimal
  duration_seconds: float


def _load_script_module() -> TotalsScriptModule:
  spec = importlib.util.spec_from_file_location("transcript_history_totals", SCRIPT_PATH)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return cast(TotalsScriptModule, cast(object, module))


def test_query_totals_returns_zeroes_for_missing_database(tmp_path: Path) -> None:
  module = _load_script_module()

  totals = module.query_totals(tmp_path / "missing.sqlite3")

  assert totals.utterances == 0
  assert totals.word_count == 0
  assert totals.tokens_up == 0
  assert totals.tokens_down == 0
  assert totals.cost == Decimal("0")
  assert totals.duration_seconds == 0.0


def test_main_prints_aggregated_totals(
  capsys: pytest.CaptureFixture[str],
  tmp_path: Path,
) -> None:
  module = _load_script_module()
  database_path = tmp_path / "history.sqlite3"
  with sqlite3.connect(str(database_path)) as connection:
    ensure_transcript_history_schema(connection)
    _ = connection.executemany(
      f"""
      INSERT INTO {TRANSCRIPT_HISTORY_TABLE} (
        pre_finalization_text,
        post_finalization_text,
        llm_model,
        tokens_in,
        tokens_out,
        cost,
        word_count,
        duration_seconds
      )
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      """,
      [
        ("alpha", "alpha ", None, None, None, None, 1, 0.5),
        (
          "bravo charlie",
          "rewritten bravo charlie ",
          "openai:gpt-4.1-mini",
          12,
          4,
          "0.00012",
          3,
          1.25,
        ),
        (
          "delta",
          "rewritten delta ",
          "openai:gpt-4.1-mini",
          8,
          2,
          "0.00003",
          2,
          0.75,
        ),
      ],
    )

  exit_code = module.main(["--db-path", str(database_path)])

  assert exit_code == 0
  assert capsys.readouterr().out == (
    f"database: {database_path}\n"
    "utterances: 3\n"
    "word count: 6\n"
    "tokens up: 20\n"
    "tokens down: 6\n"
    "cost: 0.00015\n"
    "duration: 2.500s\n"
  )
