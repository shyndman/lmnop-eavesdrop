"""SQLite-backed transcript history persistence for active-listener."""

from __future__ import annotations

import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import cast

from structlog.stdlib import BoundLogger
from typing_extensions import override

from active_listener.app.ports import (
  ActiveListenerTranscriptHistoryStore,
  FinalizedTranscriptRecord,
)

TRANSCRIPT_HISTORY_FILENAME = "active-listener.sqlite3"
TRANSCRIPT_HISTORY_TABLE = "transcript_history"
TRANSCRIPT_HISTORY_OPTIONAL_COLUMNS = {
  "word_count": "INTEGER",
  "duration_seconds": "REAL",
}


class SqliteTranscriptHistoryStore(ActiveListenerTranscriptHistoryStore):
  """Persist finalized transcript records into a local SQLite database.

  :param logger: Structured logger used for best-effort failure reporting.
  :type logger: BoundLogger
  """

  def __init__(self, *, logger: BoundLogger) -> None:
    self.logger: BoundLogger = logger

  @override
  def record_finalized_transcript(self, record: FinalizedTranscriptRecord) -> None:
    """Insert one finalized transcript row.

    :param record: Finalized transcript payload to persist.
    :type record: FinalizedTranscriptRecord
    :returns: None
    :rtype: None
    """

    database_path = resolve_transcript_history_path()

    try:
      database_path.parent.mkdir(parents=True, exist_ok=True)
      with sqlite3.connect(database_path) as connection:
        ensure_transcript_history_schema(connection)
        _ = connection.execute(
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
          (
            record.pre_finalization_text,
            record.post_finalization_text,
            record.llm_model,
            record.tokens_in,
            record.tokens_out,
            _serialize_cost(record.cost),
            record.word_count,
            record.duration_seconds,
          ),
        )
    except Exception:
      self.logger.exception(
        "transcript history insert failed",
        database_path=str(database_path),
      )


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

  existing_columns = _existing_columns(connection)
  for column_name, column_definition in TRANSCRIPT_HISTORY_OPTIONAL_COLUMNS.items():
    if column_name in existing_columns:
      continue

    _ = connection.execute(
      f"ALTER TABLE {TRANSCRIPT_HISTORY_TABLE} ADD COLUMN {column_name} {column_definition}"
    )


def _existing_columns(connection: sqlite3.Connection) -> set[str]:
  return {
    row[1]
    for row in cast(
      list[tuple[int, str, str, int, object | None, int]],
      connection.execute(f"PRAGMA table_info({TRANSCRIPT_HISTORY_TABLE})").fetchall(),
    )
  }


def resolve_transcript_history_path() -> Path:
  """Resolve the active-listener transcript history database path.

  :returns: Absolute path to the SQLite database file.
  :rtype: Path
  """

  return Path.home() / ".local" / "share" / "eavesdrop" / TRANSCRIPT_HISTORY_FILENAME


def _serialize_cost(cost: Decimal | None) -> str | None:
  if cost is None:
    return None

  return format(cost, "f")
