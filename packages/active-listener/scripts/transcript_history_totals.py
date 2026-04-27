#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12.3,<3.13"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import cast

TRANSCRIPT_HISTORY_TABLE = "transcript_history"
TRANSCRIPT_AUDIO_TABLE = "transcript_audio"
TRANSCRIPT_HISTORY_OPTIONAL_COLUMNS = {
  "word_count": "INTEGER",
  "duration_seconds": "REAL",
}


@dataclass(frozen=True)
class TranscriptHistoryTotals:
  utterances: int
  word_count: int
  tokens_up: int
  tokens_down: int
  cost: Decimal
  duration_seconds: float


@dataclass(frozen=True)
class ScriptArgs:
  db_path: Path


def parse_args(argv: Sequence[str] | None = None) -> ScriptArgs:
  parser = argparse.ArgumentParser(
    description="Summarize active-listener transcript history totals."
  )
  _ = parser.add_argument(
    "--db-path",
    type=Path,
    default=resolve_transcript_history_path(),
    help="Path to the active-listener SQLite database.",
  )
  namespace = parser.parse_args(list(argv) if argv is not None else None)
  return ScriptArgs(db_path=cast(Path, namespace.db_path))


def query_totals(database_path: Path) -> TranscriptHistoryTotals:
  if not database_path.exists():
    return TranscriptHistoryTotals(
      utterances=0,
      word_count=0,
      tokens_up=0,
      tokens_down=0,
      cost=Decimal("0"),
      duration_seconds=0.0,
    )

  with sqlite3.connect(str(database_path)) as connection:
    ensure_transcript_history_schema(connection)
    aggregate_row = cast(
      tuple[int, int, int, int, float] | None,
      connection.execute(
        f"""
      SELECT
        COUNT(*),
        COALESCE(SUM(word_count), 0),
        COALESCE(SUM(tokens_in), 0),
        COALESCE(SUM(tokens_out), 0),
        COALESCE(SUM(duration_seconds), 0.0)
      FROM {TRANSCRIPT_HISTORY_TABLE}
      """
      ).fetchone(),
    )
    cost_rows = cast(
      list[tuple[str]],
      connection.execute(
        f"SELECT cost FROM {TRANSCRIPT_HISTORY_TABLE} WHERE cost IS NOT NULL"
      ).fetchall(),
    )

  assert aggregate_row is not None
  cost = sum((Decimal(cost_text) for (cost_text,) in cost_rows), start=Decimal("0"))
  return TranscriptHistoryTotals(
    utterances=int(aggregate_row[0]),
    word_count=int(aggregate_row[1]),
    tokens_up=int(aggregate_row[2]),
    tokens_down=int(aggregate_row[3]),
    cost=cost,
    duration_seconds=float(aggregate_row[4]),
  )


def format_duration(duration_seconds: float) -> str:
  total_milliseconds = max(0, round(duration_seconds * 1000))
  minutes, remaining_milliseconds = divmod(total_milliseconds, 60_000)
  seconds, milliseconds = divmod(remaining_milliseconds, 1_000)
  hours, minutes = divmod(minutes, 60)

  if hours > 0:
    return f"{hours}h {minutes}m {seconds}.{milliseconds:03d}s"
  if minutes > 0:
    return f"{minutes}m {seconds}.{milliseconds:03d}s"
  return f"{seconds}.{milliseconds:03d}s"


def resolve_transcript_history_path() -> Path:
  return Path.home() / ".local" / "share" / "eavesdrop" / "active-listener.sqlite3"


def ensure_transcript_history_schema(connection: sqlite3.Connection) -> None:
  _ = connection.execute(
    f"""
    CREATE TABLE IF NOT EXISTS {TRANSCRIPT_HISTORY_TABLE} (
      id INTEGER PRIMARY KEY,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      pre_finalization_text TEXT NOT NULL,
      post_finalization_text TEXT NOT NULL,
      llm_model TEXT,
      tokens_in INTEGER,
      tokens_out INTEGER,
      cost TEXT,
      word_count INTEGER,
      duration_seconds REAL
    )
    """
  )

  existing_columns = {
    row[1]
    for row in cast(
      list[tuple[int, str, str, int, object | None, int]],
      connection.execute(f"PRAGMA table_info({TRANSCRIPT_HISTORY_TABLE})").fetchall(),
    )
  }
  for column_name, column_definition in TRANSCRIPT_HISTORY_OPTIONAL_COLUMNS.items():
    if column_name in existing_columns:
      continue

    _ = connection.execute(
      f"ALTER TABLE {TRANSCRIPT_HISTORY_TABLE} ADD COLUMN {column_name} {column_definition}"
    )

  _ = connection.execute(
    f"""
    CREATE TABLE IF NOT EXISTS {TRANSCRIPT_AUDIO_TABLE} (
      transcript_id INTEGER PRIMARY KEY,
      audio_m4a BLOB NOT NULL,
      FOREIGN KEY (transcript_id) REFERENCES {TRANSCRIPT_HISTORY_TABLE}(id) ON DELETE CASCADE
    )
    """
  )


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)
  totals = query_totals(args.db_path)

  print(f"database: {args.db_path}")
  print(f"utterances: {totals.utterances}")
  print(f"word count: {totals.word_count}")
  print(f"tokens up: {totals.tokens_up}")
  print(f"tokens down: {totals.tokens_down}")
  print(f"cost: {totals.cost}")
  print(f"duration: {format_duration(totals.duration_seconds)}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
