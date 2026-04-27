"""SQLite-backed transcript history persistence for active-listener."""

from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import cast

from structlog.stdlib import BoundLogger
from typing_extensions import override

from active_listener.app.ports import (
  ActiveListenerTranscriptHistoryStore,
  CapturedRecordingAudio,
  FinalizedTranscriptRecord,
)
from active_listener.infra.dbus import AppStateService

TRANSCRIPT_HISTORY_FILENAME = "active-listener.sqlite3"
TRANSCRIPT_HISTORY_TABLE = "transcript_history"
TRANSCRIPT_AUDIO_TABLE = "transcript_audio"
TRANSCRIPT_HISTORY_OPTIONAL_COLUMNS = {
  "word_count": "INTEGER",
  "duration_seconds": "REAL",
}


class AudioArchiveError(RuntimeError):
  """Raised when recorded audio cannot be encoded or inserted."""


class SqliteTranscriptHistoryStore(ActiveListenerTranscriptHistoryStore):
  """Persist finalized transcript records into a local SQLite database.

  :param logger: Structured logger used for best-effort failure reporting.
  :type logger: BoundLogger
  :param ffmpeg_path: Resolved FFmpeg binary path validated at startup.
  :type ffmpeg_path: str
  :param dbus_service: DBus publisher used for archive-failure notifications.
  :type dbus_service: AppStateService
  """

  def __init__(
    self,
    *,
    logger: BoundLogger,
    ffmpeg_path: str,
    dbus_service: AppStateService,
  ) -> None:
    self.logger: BoundLogger = logger
    self.ffmpeg_path: str = ffmpeg_path
    self.dbus_service: AppStateService = dbus_service

  @override
  def record_finalized_recording(
    self,
    record: FinalizedTranscriptRecord,
    captured_audio: CapturedRecordingAudio,
  ) -> None:
    """Insert one finalized transcript row and best-effort archived audio.

    :param record: Finalized transcript payload to persist.
    :type record: FinalizedTranscriptRecord
    :param captured_audio: Captured recording audio snapshot.
    :type captured_audio: CapturedRecordingAudio
    :returns: None
    :rtype: None
    """

    database_path = resolve_transcript_history_path()

    try:
      database_path.parent.mkdir(parents=True, exist_ok=True)
      with sqlite3.connect(database_path) as connection:
        configure_transcript_history_connection(connection)
        ensure_transcript_history_schema(connection)
        transcript_id = insert_transcript_history(connection, record)
        if captured_audio.pcm_f32le == b"":
          self.logger.info(
            "transcript audio archive skipped",
            database_path=str(database_path),
            transcript_id=transcript_id,
            reason="empty capture",
          )
          return

        try:
          self.logger.info(
            "archiving transcript audio",
            database_path=str(database_path),
            transcript_id=transcript_id,
            pcm_bytes=len(captured_audio.pcm_f32le),
            sample_rate_hz=captured_audio.sample_rate_hz,
            channels=captured_audio.channels,
          )
          audio_m4a = encode_m4a(
            self.ffmpeg_path,
            captured_audio.pcm_f32le,
            sample_rate_hz=captured_audio.sample_rate_hz,
            channels=captured_audio.channels,
          )
          insert_transcript_audio(connection, transcript_id, audio_m4a)
          self.logger.info(
            "transcript audio archived",
            database_path=str(database_path),
            transcript_id=transcript_id,
            audio_bytes=len(audio_m4a),
          )
        except Exception as exc:
          self.logger.exception(
            "transcript audio archive failed",
            database_path=str(database_path),
            transcript_id=transcript_id,
          )
          self._schedule_archive_failure_notification(str(exc))
    except Exception:
      self.logger.exception(
        "transcript history insert failed",
        database_path=str(database_path),
      )

  def _schedule_archive_failure_notification(self, reason: str) -> None:
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      self.logger.warning("audio archive notification skipped", reason=reason)
      return

    task = loop.create_task(self.dbus_service.audio_archive_failed(reason))
    task.add_done_callback(self._log_archive_notification_result)

  def _log_archive_notification_result(self, task: asyncio.Task[None]) -> None:
    try:
      _ = task.result()
    except asyncio.CancelledError:
      return
    except Exception:
      self.logger.exception("audio archive notification failed")


def configure_transcript_history_connection(connection: sqlite3.Connection) -> None:
  _ = connection.execute("PRAGMA foreign_keys = ON")


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

  _ = connection.execute(
    f"""
    CREATE TABLE IF NOT EXISTS {TRANSCRIPT_AUDIO_TABLE} (
      transcript_id INTEGER PRIMARY KEY,
      audio_m4a BLOB NOT NULL,
      FOREIGN KEY (transcript_id) REFERENCES {TRANSCRIPT_HISTORY_TABLE}(id) ON DELETE CASCADE
    )
    """
  )


def insert_transcript_history(
  connection: sqlite3.Connection,
  record: FinalizedTranscriptRecord,
) -> int:
  cursor = connection.execute(
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
  row_id = cursor.lastrowid
  if row_id is None:
    raise AudioArchiveError("transcript history insert did not return a row id")
  return int(row_id)


def insert_transcript_audio(
  connection: sqlite3.Connection,
  transcript_id: int,
  audio_m4a: bytes,
) -> None:
  _ = connection.execute(
    f"""
    INSERT INTO {TRANSCRIPT_AUDIO_TABLE} (
      transcript_id,
      audio_m4a
    )
    VALUES (?, ?)
    """,
    (transcript_id, audio_m4a),
  )


def encode_m4a(
  ffmpeg_path: str,
  pcm_f32le: bytes,
  *,
  sample_rate_hz: int = 16_000,
  channels: int = 1,
) -> bytes:
  with tempfile.TemporaryDirectory(prefix="active-listener-audio-") as temp_dir:
    output_path = Path(temp_dir) / "recording.m4a"
    result = subprocess.run(
      [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate_hz),
        "-ac",
        str(channels),
        "-i",
        "pipe:0",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-profile:a",
        "aac_low",
        "-f",
        "ipod",
        str(output_path),
      ],
      input=pcm_f32le,
      capture_output=True,
      check=False,
    )
    if result.returncode != 0:
      stderr_text = result.stderr.decode("utf-8", errors="replace").strip()
      raise AudioArchiveError(stderr_text or "ffmpeg audio encode failed")

    if not output_path.exists():
      raise AudioArchiveError("ffmpeg completed without creating an output file")

    audio_m4a = output_path.read_bytes()
    if audio_m4a == b"":
      raise AudioArchiveError("ffmpeg created an empty m4a file")

  return audio_m4a


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
