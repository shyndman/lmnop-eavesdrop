from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest
from structlog.stdlib import BoundLogger

from active_listener.app.ports import FinalizedTranscriptRecord
from active_listener.infra.transcript_history import (
  SqliteTranscriptHistoryStore,
  resolve_transcript_history_path,
)


@dataclass(frozen=True)
class LogRecord:
  event: str
  fields: dict[str, object]


@dataclass
class HistoryLogger:
  exception_records: list[LogRecord] = field(default_factory=list)

  def exception(self, event: str, **kwargs: object) -> None:
    self.exception_records.append(LogRecord(event=event, fields=kwargs))


def test_transcript_history_store_writes_record(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  logger = HistoryLogger()
  store = SqliteTranscriptHistoryStore(logger=cast(BoundLogger, cast(object, logger)))
  record = FinalizedTranscriptRecord(
    pre_finalization_text="alpha",
    post_finalization_text="rewritten alpha ",
    llm_model="openai:gpt-4.1-mini",
    tokens_in=12,
    tokens_out=4,
    cost=Decimal("0.00012"),
    word_count=2,
    duration_seconds=1.25,
  )

  monkeypatch.setattr(Path, "home", lambda: tmp_path)

  store.record_finalized_transcript(record)

  database_path = resolve_transcript_history_path()
  assert database_path == tmp_path / ".local" / "share" / "eavesdrop" / "active-listener.sqlite3"

  with sqlite3.connect(str(database_path)) as connection:
    row = cast(
      tuple[str, str, str, int, int, str, int, float] | None,
      connection.execute(
        """
      SELECT
        pre_finalization_text,
        post_finalization_text,
        llm_model,
        tokens_in,
        tokens_out,
        cost,
        word_count,
        duration_seconds
      FROM transcript_history
      """
      ).fetchone(),
    )

  assert row == (
    "alpha",
    "rewritten alpha ",
    "openai:gpt-4.1-mini",
    12,
    4,
    "0.00012",
    2,
    1.25,
  )


def test_transcript_history_store_logs_insert_failures(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  logger = HistoryLogger()
  store = SqliteTranscriptHistoryStore(logger=cast(BoundLogger, cast(object, logger)))

  def explode_connect(
    _path: str | Path,
    *args: object,
    **kwargs: object,
  ) -> sqlite3.Connection:
    _ = args
    _ = kwargs
    raise RuntimeError("disk full")

  monkeypatch.setattr(Path, "home", lambda: tmp_path)
  monkeypatch.setattr(sqlite3, "connect", explode_connect)

  store.record_finalized_transcript(
    FinalizedTranscriptRecord(
      pre_finalization_text="alpha",
      post_finalization_text="alpha ",
      llm_model=None,
      tokens_in=None,
      tokens_out=None,
      cost=None,
      word_count=1,
      duration_seconds=0.5,
    )
  )

  assert logger.exception_records == [
    LogRecord(
      event="transcript history insert failed",
      fields={
        "database_path": str(
          tmp_path / ".local" / "share" / "eavesdrop" / "active-listener.sqlite3"
        ),
      },
    )
  ]


def test_transcript_history_store_migrates_legacy_database(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  database_path = tmp_path / ".local" / "share" / "eavesdrop" / "active-listener.sqlite3"
  database_path.parent.mkdir(parents=True)
  with sqlite3.connect(str(database_path)) as connection:
    _ = connection.execute(
      """
      CREATE TABLE transcript_history (
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

  logger = HistoryLogger()
  store = SqliteTranscriptHistoryStore(logger=cast(BoundLogger, cast(object, logger)))
  monkeypatch.setattr(Path, "home", lambda: tmp_path)

  store.record_finalized_transcript(
    FinalizedTranscriptRecord(
      pre_finalization_text="alpha",
      post_finalization_text="alpha ",
      llm_model=None,
      tokens_in=None,
      tokens_out=None,
      cost=None,
      word_count=1,
      duration_seconds=0.5,
    )
  )

  with sqlite3.connect(str(database_path)) as connection:
    row = cast(
      tuple[int, float] | None,
      connection.execute(
        "SELECT word_count, duration_seconds FROM transcript_history ORDER BY id DESC LIMIT 1"
      ).fetchone(),
    )

  assert row == (1, 0.5)
