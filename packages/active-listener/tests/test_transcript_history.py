from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest
from structlog.stdlib import BoundLogger

from active_listener.app.ports import CapturedRecordingAudio, FinalizedTranscriptRecord
from active_listener.infra.dbus import AppStateService
from active_listener.infra.transcript_history import (
  TRANSCRIPT_AUDIO_TABLE,
  AudioArchiveError,
  SqliteTranscriptHistoryStore,
  resolve_transcript_history_path,
)


@dataclass(frozen=True)
class LogRecord:
  event: str
  fields: dict[str, object]


@dataclass
class HistoryLogger:
  info_records: list[LogRecord] = field(default_factory=list)
  exception_records: list[LogRecord] = field(default_factory=list)
  warning_records: list[LogRecord] = field(default_factory=list)

  def info(self, event: str, **kwargs: object) -> None:
    self.info_records.append(LogRecord(event=event, fields=kwargs))

  def exception(self, event: str, **kwargs: object) -> None:
    self.exception_records.append(LogRecord(event=event, fields=kwargs))

  def warning(self, event: str, **kwargs: object) -> None:
    self.warning_records.append(LogRecord(event=event, fields=kwargs))


@dataclass
class FakeDbusService:
  archive_failure_reasons: list[str] = field(default_factory=list)

  async def set_state(self, _state: object) -> None:
    return None

  async def transcription_updated(self, _runs: list[object]) -> None:
    return None

  async def spectrum_updated(self, _bars: bytes) -> None:
    return None

  async def recording_aborted(self, _reason: str) -> None:
    return None

  async def audio_archive_failed(self, reason: str) -> None:
    self.archive_failure_reasons.append(reason)

  async def pipeline_failed(self, _step: str, _reason: str) -> None:
    return None

  async def fatal_error(self, _reason: str) -> None:
    return None

  async def reconnecting(self) -> None:
    return None

  async def reconnected(self) -> None:
    return None

  async def close(self) -> None:
    return None


def _store(logger: HistoryLogger, dbus_service: FakeDbusService) -> SqliteTranscriptHistoryStore:
  return SqliteTranscriptHistoryStore(
    logger=cast(BoundLogger, cast(object, logger)),
    ffmpeg_path="/usr/bin/ffmpeg",
    dbus_service=cast(AppStateService, cast(object, dbus_service)),
  )


def _record() -> FinalizedTranscriptRecord:
  return FinalizedTranscriptRecord(
    pre_finalization_text="alpha",
    post_finalization_text="rewritten alpha ",
    llm_model="openai:gpt-4.1-mini",
    tokens_in=12,
    tokens_out=4,
    cost=Decimal("0.00012"),
    word_count=2,
    duration_seconds=1.25,
  )


def _captured_audio(*, pcm_f32le: bytes = b"audio") -> CapturedRecordingAudio:
  return CapturedRecordingAudio(
    pcm_f32le=pcm_f32le,
    sample_rate_hz=16000,
    channels=1,
  )


def test_transcript_history_store_writes_record_and_audio_row(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  logger = HistoryLogger()
  dbus_service = FakeDbusService()
  store = _store(logger, dbus_service)

  monkeypatch.setattr(Path, "home", lambda: tmp_path)
  monkeypatch.setattr(
    "active_listener.infra.transcript_history.encode_m4a",
    lambda *_args, **_kwargs: b"m4a-bytes",
  )

  store.record_finalized_recording(_record(), _captured_audio())

  database_path = resolve_transcript_history_path()
  assert database_path == tmp_path / ".local" / "share" / "eavesdrop" / "active-listener.sqlite3"

  with sqlite3.connect(str(database_path)) as connection:
    transcript_row = cast(
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
    audio_row = cast(
      tuple[bytes] | None,
      connection.execute(f"SELECT audio_m4a FROM {TRANSCRIPT_AUDIO_TABLE}").fetchone(),
    )

  assert transcript_row == (
    "alpha",
    "rewritten alpha ",
    "openai:gpt-4.1-mini",
    12,
    4,
    "0.00012",
    2,
    1.25,
  )
  assert audio_row == (b"m4a-bytes",)
  assert dbus_service.archive_failure_reasons == []
  assert logger.info_records == [
    LogRecord(
      event="archiving transcript audio",
      fields={
        "database_path": str(database_path),
        "transcript_id": 1,
        "pcm_bytes": 5,
        "sample_rate_hz": 16000,
        "channels": 1,
      },
    ),
    LogRecord(
      event="transcript audio archived",
      fields={
        "database_path": str(database_path),
        "transcript_id": 1,
        "audio_bytes": 9,
      },
    ),
  ]


def test_transcript_history_store_logs_insert_failures(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  logger = HistoryLogger()
  store = _store(logger, FakeDbusService())

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

  store.record_finalized_recording(_record(), _captured_audio())

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


def test_transcript_history_store_migrates_legacy_database_and_adds_audio_table(
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
  store = _store(logger, FakeDbusService())
  monkeypatch.setattr(Path, "home", lambda: tmp_path)
  monkeypatch.setattr(
    "active_listener.infra.transcript_history.encode_m4a",
    lambda *_args, **_kwargs: b"m4a-bytes",
  )

  store.record_finalized_recording(_record(), _captured_audio())

  with sqlite3.connect(str(database_path)) as connection:
    transcript_row = cast(
      tuple[int, float] | None,
      connection.execute(
        "SELECT word_count, duration_seconds FROM transcript_history ORDER BY id DESC LIMIT 1"
      ).fetchone(),
    )
    audio_table_row = cast(
      tuple[str] | None,
      connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (TRANSCRIPT_AUDIO_TABLE,),
      ).fetchone(),
    )

  assert transcript_row == (2, 1.25)
  assert audio_table_row == (TRANSCRIPT_AUDIO_TABLE,)


def test_transcript_history_store_keeps_transcript_only_rows_when_audio_is_empty(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  logger = HistoryLogger()
  store = _store(logger, FakeDbusService())
  monkeypatch.setattr(Path, "home", lambda: tmp_path)

  store.record_finalized_recording(_record(), _captured_audio(pcm_f32le=b""))

  database_path = resolve_transcript_history_path()
  with sqlite3.connect(str(database_path)) as connection:
    transcript_count = cast(
      tuple[int],
      connection.execute("SELECT COUNT(*) FROM transcript_history").fetchone(),
    )
    audio_count = cast(
      tuple[int],
      connection.execute(f"SELECT COUNT(*) FROM {TRANSCRIPT_AUDIO_TABLE}").fetchone(),
    )

  assert transcript_count == (1,)
  assert audio_count == (0,)
  assert logger.info_records == [
    LogRecord(
      event="transcript audio archive skipped",
      fields={
        "database_path": str(database_path),
        "transcript_id": 1,
        "reason": "empty capture",
      },
    )
  ]


@pytest.mark.asyncio
async def test_transcript_history_store_preserves_transcript_row_on_encode_failure(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  logger = HistoryLogger()
  dbus_service = FakeDbusService()
  store = _store(logger, dbus_service)
  monkeypatch.setattr(Path, "home", lambda: tmp_path)

  def explode_encode(*_args: object, **_kwargs: object) -> bytes:
    raise AudioArchiveError("encoder exploded")

  monkeypatch.setattr("active_listener.infra.transcript_history.encode_m4a", explode_encode)

  store.record_finalized_recording(_record(), _captured_audio())
  await asyncio.sleep(0)

  database_path = resolve_transcript_history_path()
  with sqlite3.connect(str(database_path)) as connection:
    transcript_count = cast(
      tuple[int],
      connection.execute("SELECT COUNT(*) FROM transcript_history").fetchone(),
    )
    audio_count = cast(
      tuple[int],
      connection.execute(f"SELECT COUNT(*) FROM {TRANSCRIPT_AUDIO_TABLE}").fetchone(),
    )

  assert transcript_count == (1,)
  assert audio_count == (0,)
  assert dbus_service.archive_failure_reasons == ["encoder exploded"]
  assert logger.exception_records[-1] == LogRecord(
    event="transcript audio archive failed",
    fields={
      "database_path": str(database_path),
      "transcript_id": 1,
    },
  )


@pytest.mark.asyncio
async def test_transcript_history_store_preserves_transcript_row_on_audio_insert_failure(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  logger = HistoryLogger()
  dbus_service = FakeDbusService()
  store = _store(logger, dbus_service)
  monkeypatch.setattr(Path, "home", lambda: tmp_path)
  monkeypatch.setattr(
    "active_listener.infra.transcript_history.encode_m4a",
    lambda *_args, **_kwargs: b"m4a-bytes",
  )

  def explode_insert(
    _connection: sqlite3.Connection,
    _transcript_id: int,
    _audio_m4a: bytes,
  ) -> None:
    raise sqlite3.IntegrityError("audio insert exploded")

  monkeypatch.setattr(
    "active_listener.infra.transcript_history.insert_transcript_audio",
    explode_insert,
  )

  store.record_finalized_recording(_record(), _captured_audio())
  await asyncio.sleep(0)

  database_path = resolve_transcript_history_path()
  with sqlite3.connect(str(database_path)) as connection:
    transcript_count = cast(
      tuple[int],
      connection.execute("SELECT COUNT(*) FROM transcript_history").fetchone(),
    )
    audio_count = cast(
      tuple[int],
      connection.execute(f"SELECT COUNT(*) FROM {TRANSCRIPT_AUDIO_TABLE}").fetchone(),
    )

  assert transcript_count == (1,)
  assert audio_count == (0,)
  assert dbus_service.archive_failure_reasons == ["audio insert exploded"]
  assert logger.exception_records[-1] == LogRecord(
    event="transcript audio archive failed",
    fields={
      "database_path": str(database_path),
      "transcript_id": 1,
    },
  )
