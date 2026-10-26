"""Finalize owned recording snapshots independently of foreground application state."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from langfuse import LangfuseSpan
from structlog.stdlib import BoundLogger

from active_listener.app.ports import (
  ActiveListenerClient,
  ActiveListenerRewriteClient,
  ActiveListenerTranscriptHistoryStore,
  FinalizedTranscriptRecord,
  FinishedRecording,
  RewriteResult,
)
from active_listener.config.models import ActiveListenerConfig
from active_listener.infra.audio import encode_recording_audio
from active_listener.infra.corrections import ActiveListenerCorrectionStore, CorrectionMap
from active_listener.infra.dbus import AppStateService
from active_listener.infra.emitter import TextEmitter
from active_listener.infra.langfuse import (
  RecordingOutput,
  build_langfuse_audio_attachment,
  end_recording_observation,
  update_recording_observation,
)
from active_listener.recording.reducer import (
  TextRun,
  apply_segment_reduction,
  build_completed_text_runs,
  reduce_new_segments,
  render_text,
  serialize_runs_for_rewrite,
  serialize_runs_without_commands,
)
from active_listener.recording.rewrite import TranscriptRewriter
from active_listener.recording.text_shaping import shape_runs
from eavesdrop.wire import TranscriptionMessage


@dataclass(frozen=True)
class PreparedTranscript:
  """Recording-local shaped text awaiting optional rewriting and emission."""

  raw_text: str
  runs: list[TextRun]
  rewrite_input: str


class RecordingFinalizer:
  """Serialize finalization without reading mutable foreground application state.

  :param config: Validated rewrite and audio archive configuration.
  :param client: Server flush dependency.
  :param emitter: Workstation text destination.
  :param logger: Failure and result diagnostics.
  :param rewrite_client: Optional rewrite provider implementation.
  :param history_store: Successfully emitted recording destination.
  :param dbus_service: Pipeline failure notification destination.
  :param correction_store: Persistent user spelling corrections.
  """

  def __init__(
    self,
    *,
    config: ActiveListenerConfig,
    client: ActiveListenerClient,
    emitter: TextEmitter,
    logger: BoundLogger,
    rewrite_client: ActiveListenerRewriteClient,
    history_store: ActiveListenerTranscriptHistoryStore,
    dbus_service: AppStateService,
    correction_store: ActiveListenerCorrectionStore,
  ) -> None:
    self.config: ActiveListenerConfig = config
    self.client: ActiveListenerClient = client
    self.emitter: TextEmitter = emitter
    self.logger: BoundLogger = logger
    self._rewriter: TranscriptRewriter = TranscriptRewriter(
      config.llm_rewrite, rewrite_client, correction_store, logger
    )
    self.history_store: ActiveListenerTranscriptHistoryStore = history_store
    self.dbus_service: AppStateService = dbus_service
    self._lock: asyncio.Lock = asyncio.Lock()
    self._flush_lock: asyncio.Lock = asyncio.Lock()
    self._pending_flushes: int = 0
    self._pending_flushes_drained: asyncio.Event = asyncio.Event()
    self._pending_flushes_drained.set()

  def reserve_flush(self) -> None:
    """Reserve a flush before scheduling finalization, blocking the next start."""
    self._pending_flushes += 1
    self._pending_flushes_drained.clear()

  async def wait_for_pending_flushes(self) -> None:
    """Wait until every accepted prior recording has received its flush result."""
    _ = await self._pending_flushes_drained.wait()

  def cancel_reserved_flush(self) -> None:
    """Release a reservation whose finalization task could not be scheduled."""
    self._release_reserved_flush()

  def _release_reserved_flush(self) -> None:
    self._pending_flushes -= 1
    if self._pending_flushes == 0:
      self._pending_flushes_drained.set()

  async def finalize_recording(
    self,
    *,
    finished_recording: FinishedRecording,
    recording_observation: LangfuseSpan | None,
    llm_available: bool,
    llm_active: bool,
  ) -> None:
    """Flush and finalize one transferred recording using finish-time decisions.

    :param finished_recording: Owned audio, transcript, and correction-load snapshot.
    :param recording_observation: Observation transferred from foreground ownership.
    :param llm_available: Whether rewriting was configured when recording finished.
    :param llm_active: Whether rewriting was enabled when recording finished.
    :return: None after emission or a logged pipeline failure.
    """
    try:
      try:
        async with self._flush_lock:
          message = await self.client.flush(finished_recording.recording_id, force_complete=True)
      except Exception:
        self.logger.exception("recording finalization failed")
        update_recording_observation(
          recording_observation,
          logger=self.logger,
          level="ERROR",
          status_message="recording finalization failed",
        )
        return
      finally:
        self._release_reserved_flush()
      async with self._lock:
        prepared = await self._prepare_transcript(
          finished_recording, message, recording_observation
        )
        if prepared is not None:
          await self._emit_recording(
            finished_recording,
            prepared,
            message.stream,
            recording_observation,
            llm_available=llm_available,
            llm_active=llm_active,
          )
    except Exception:
      self.logger.exception("recording finalization pipeline failed")
      update_recording_observation(
        recording_observation,
        logger=self.logger,
        level="ERROR",
        status_message="recording finalization pipeline failed",
      )
    finally:
      end_recording_observation(recording_observation, logger=self.logger)

  async def _prepare_transcript(
    self,
    recording: FinishedRecording,
    message: TranscriptionMessage,
    observation: LangfuseSpan | None,
  ) -> PreparedTranscript | None:
    corrections = await self._resolve_corrections(recording, session_id=message.stream)
    state = recording.reducer_state
    reduction = reduce_new_segments(message.segments, state.last_id)
    apply_segment_reduction(state, reduction)
    state.last_id = reduction.last_id
    raw_text = render_text(state.completed_words)
    runs = shape_runs(build_completed_text_runs(state), corrections)
    rewrite_input = serialize_runs_for_rewrite(runs)
    if not raw_text or not rewrite_input.strip():
      self.logger.info("recording finalized without committed text", session_id=message.stream)
      return None
    self.logger.info("finalized raw transcript", session_id=message.stream, raw_text=raw_text)
    if observation is not None:
      _ = observation.update(
        input={"raw_transcript": raw_text, "rewrite_input": rewrite_input},
        metadata={
          "component": "active-listener",
          "session_id": message.stream,
          "recording_id": recording.recording_id,
          "duration_seconds": state.duration_seconds,
          "word_count": len(state.completed_words),
        },
      )
    return PreparedTranscript(raw_text, runs, rewrite_input)

  async def _emit_recording(
    self,
    recording: FinishedRecording,
    prepared: PreparedTranscript,
    session_id: str,
    observation: LangfuseSpan | None,
    *,
    llm_available: bool,
    llm_active: bool,
  ) -> None:
    rewrite = await self._optional_rewrite(prepared, recording, session_id, observation, llm_active)
    text = rewrite.text if rewrite is not None else serialize_runs_without_commands(prepared.runs)
    if not text.strip():
      self.logger.info("recording finalized without committed text", session_id=session_id)
      return
    text = f"{text} "
    try:
      self.emitter.emit_text(text)
    except Exception:
      self.logger.exception("text emission failed", session_id=session_id)
      update_recording_observation(
        observation, logger=self.logger, level="ERROR", status_message="text emission failed"
      )
      return
    mode = _llm_mode(llm_available=llm_available, rewrite_ran=rewrite is not None)
    source = _emitted_text_source(mode)
    self.logger.info("recording finalization mode", session_id=session_id, llm_mode=mode)
    self.logger.info(
      "text emitted", session_id=session_id, emitted_text=text, text_length=len(text), source=source
    )
    record = _history_record(recording, prepared.raw_text, text, rewrite)
    audio = await self._archive_recording(recording, record, session_id)
    self._observe_output(recording, record, audio, observation, session_id, mode, source)

  async def _optional_rewrite(
    self,
    prepared: PreparedTranscript,
    recording: FinishedRecording,
    session_id: str,
    observation: LangfuseSpan | None,
    active: bool,
  ) -> RewriteResult | None:
    if not active:
      return None
    try:
      return await self._rewriter.rewrite(
        transcript=prepared.rewrite_input,
        session_id=session_id,
        recording_id=recording.recording_id,
        recording_observation=observation,
      )
    except Exception as exc:
      self.logger.exception(
        "dictation pipeline step failed",
        session_id=session_id,
        step="rewrite_with_llm",
        reason=str(exc),
      )
      await self.dbus_service.pipeline_failed("rewrite_with_llm", str(exc))
      return None

  async def _archive_recording(
    self,
    recording: FinishedRecording,
    record: FinalizedTranscriptRecord,
    session_id: str,
  ) -> bytes | None:
    audio: bytes | None = None
    try:
      audio = encode_recording_audio(recording.captured_audio, ffmpeg_path=self.config.ffmpeg_path)
    except Exception as exc:
      self.logger.exception("recording audio encode failed", session_id=session_id)
      await self.dbus_service.audio_archive_failed(str(exc))
    self.history_store.record_finalized_recording(record, audio)
    return audio

  def _observe_output(
    self,
    recording: FinishedRecording,
    record: FinalizedTranscriptRecord,
    audio: bytes | None,
    observation: LangfuseSpan | None,
    session_id: str,
    mode: str,
    source: str,
  ) -> None:
    if observation is None:
      return
    attachment = build_langfuse_audio_attachment(audio_bytes=audio)
    output: RecordingOutput = {"emitted_text": record.post_finalization_text}
    if attachment is not None:
      output["captured_audio"] = attachment
    _ = observation.update(
      output=output,
      metadata={
        "component": "active-listener",
        "session_id": session_id,
        "recording_id": recording.recording_id,
        "llm_mode": mode,
        "source": source,
        "word_count": record.word_count,
        "duration_seconds": record.duration_seconds,
        "llm_model": record.llm_model,
        "tokens_in": record.tokens_in,
        "tokens_out": record.tokens_out,
        "cost": format(record.cost, "f") if record.cost is not None else None,
      },
    )

  async def _resolve_corrections(
    self,
    recording: FinishedRecording,
    *,
    session_id: str,
  ) -> CorrectionMap:
    task = recording.correction_load_task
    if task is None:
      return {}
    try:
      return await task
    except Exception:
      self.logger.exception("stored correction load failed", session_id=session_id)
      return {}


def _llm_mode(*, llm_available: bool, rewrite_ran: bool) -> str:
  if not llm_available:
    return "unavailable"
  return "active" if rewrite_ran else "bypassed"


def _emitted_text_source(llm_mode: str) -> str:
  match llm_mode:
    case "unavailable":
      return "raw"
    case "active":
      return "pipeline_llm"
    case _:
      return "pipeline_no_llm"


def _history_record(
  recording: FinishedRecording,
  raw_text: str,
  text: str,
  rewrite: RewriteResult | None,
) -> FinalizedTranscriptRecord:
  return FinalizedTranscriptRecord(
    pre_finalization_text=raw_text,
    post_finalization_text=text,
    llm_model=rewrite.model if rewrite is not None else None,
    tokens_in=rewrite.input_tokens if rewrite is not None else None,
    tokens_out=rewrite.output_tokens if rewrite is not None else None,
    cost=rewrite.cost if rewrite is not None else None,
    word_count=len(text.split()),
    duration_seconds=recording.reducer_state.duration_seconds,
  )
