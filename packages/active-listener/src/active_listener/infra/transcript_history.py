"""SQLite-backed transcript history persistence for active-listener."""

from __future__ import annotations

import sqlite3
from decimal import Decimal
from pathlib import Path

from structlog.stdlib import BoundLogger
from typing_extensions import override

from active_listener.app.ports import (
  ActiveListenerTranscriptHistoryStore,
  FinalizedTranscriptRecord,
)

TRANSCRIPT_HISTORY_FILENAME = "active-listener.sqlite3"
TRANSCRIPT_HISTORY_TABLE = "transcript_history"


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
        self._ensure_schema(connection)
        _ = connection.execute(
          f"""
          INSERT INTO {TRANSCRIPT_HISTORY_TABLE} (
            pre_finalization_text,
            post_finalization_text,
            llm_model,
            tokens_in,
            tokens_out,
            cost
          )
          VALUES (?, ?, ?, ?, ?, ?)
          """,
          (
            record.pre_finalization_text,
            record.post_finalization_text,
            record.llm_model,
            record.tokens_in,
            record.tokens_out,
            _serialize_cost(record.cost),
          ),
        )
    except Exception:
      self.logger.exception(
        "transcript history insert failed",
        database_path=str(database_path),
      )

  def _ensure_schema(self, connection: sqlite3.Connection) -> None:
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
        cost TEXT
      )
      """
    )


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
