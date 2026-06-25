"""Recording finalization pipeline for active-listener."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

from structlog.stdlib import BoundLogger

import active_listener.infra.rewrite as rewrite_module
from active_listener.app.ports import (
  ActiveListenerClient,
  ActiveListenerRewriteClient,
  ActiveListenerTranscriptHistoryStore,
  FinalizedTranscriptRecord,
  FinishedRecording,
  RewriteResult,
)
from active_listener.config.models import (
  ActiveListenerConfig,
  LiteRtRewriteProvider,
  LlmRewriteConfig,
)
from active_listener.infra.audio import encode_recording_audio
from active_listener.infra.corrections import (
  ActiveListenerCorrectionStore,
  CorrectionMap,
  CorrectionStore,
)
from active_listener.infra.dbus import AppStateService
from active_listener.infra.emitter import TextEmitter
from active_listener.recording.reducer import (
  apply_segment_reduction,
  build_completed_text_runs,
  reduce_new_segments,
  render_text,
  serialize_runs_for_rewrite,
  serialize_runs_without_commands,
)
from active_listener.recording.text_shaping import shape_runs

from ..infra.langfuse import (
  RecordingObservation,
  build_langfuse_audio_attachment,
  end_recording_observation,
  start_rewrite_observation,
  update_recording_observation,
)

LlmAvailabilityReader = Callable[[], bool]
LlmActiveReader = Callable[[], bool]
DisconnectGenerationReader = Callable[[], int]


@dataclass
class RecordingFinalizer:
  """Finalize a finished recording into emitted workstation text."""

  config: ActiveListenerConfig
  client: ActiveListenerClient
  emitter: TextEmitter
  logger: BoundLogger
  rewrite_client: ActiveListenerRewriteClient
  history_store: ActiveListenerTranscriptHistoryStore
  dbus_service: AppStateService
  current_llm_available: LlmAvailabilityReader
  current_llm_active: LlmActiveReader
  current_disconnect_generation: DisconnectGenerationReader
  correction_store: ActiveListenerCorrectionStore = field(default_factory=CorrectionStore.default)
  _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
  _flush_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
  _pending_flushes: int = 0
  _pending_flushes_drained: asyncio.Event = field(default_factory=asyncio.Event)

  def __post_init__(self) -> None:
    _ = self._pending_flushes_drained.set()

  def reserve_flush(self) -> None:
    self._pending_flushes += 1
    _ = self._pending_flushes_drained.clear()

  async def wait_for_pending_flushes(self) -> None:
    _ = await self._pending_flushes_drained.wait()

  def cancel_reserved_flush(self) -> None:
    self._release_reserved_flush()

  def _release_reserved_flush(self) -> None:
    self._pending_flushes -= 1
    if self._pending_flushes == 0:
      _ = self._pending_flushes_drained.set()

  async def finalize_recording(
    self,
    *,
    disconnect_generation: int,
    finished_recording: FinishedRecording,
    recording_observation: RecordingObservation | None,
  ) -> None:
    try:
      reducer_state = finished_recording.reducer_state
      try:
        async with self._flush_lock:
          try:
            message = await self.client.flush(finished_recording.recording_id, force_complete=True)
          finally:
            self._release_reserved_flush()
      except Exception:
        update_recording_observation(
          recording_observation,
          logger=self.logger,
          level="ERROR",
          status_message="recording finalization failed",
        )
        self.logger.exception("recording finalization failed")
        return

      if self.current_disconnect_generation() != disconnect_generation:
        update_recording_observation(
          recording_observation,
          logger=self.logger,
          status_message="recording emission skipped after disconnect",
        )
        self.logger.warning("skipping emission after disconnect", stream=message.stream)
        return

      async with self._lock:
        if self.current_disconnect_generation() != disconnect_generation:
          update_recording_observation(
            recording_observation,
            logger=self.logger,
            status_message="recording emission skipped after disconnect",
          )
          self.logger.warning("skipping emission after disconnect", stream=message.stream)
          return
        # Resolve stored corrections locally. The session clears its own load task during
        # teardown and a new recording may already own the live session, so the finalizer
        # never reads or writes live-session state: it shapes this recording's committed text
        # from the reducer state and correction task carried on the finished recording.
        corrections = await self._resolve_corrections(finished_recording, stream=message.stream)
        # Reduce the final flushed window into this recording's private reducer state.
        reduction = reduce_new_segments(message.segments, reducer_state.last_id)
        apply_segment_reduction(reducer_state, reduction)
        reducer_state.last_id = reduction.last_id
        # raw_text is the unshaped "before" snapshot, retained only for history/observation.
        raw_text = render_text(reducer_state.completed_words)
        if not raw_text:
          self.logger.info("recording finalized without committed text", stream=message.stream)
          return

        self.logger.info("finalized raw transcript", stream=message.stream, raw_text=raw_text)
        completed_runs = shape_runs(build_completed_text_runs(reducer_state), corrections)
        # LLM-input view: command runs are wrapped in instruction markers. Marker-wrapping is
        # an LLM-boundary concern, so the shaping pipeline never needs to know the LLM exists.
        rewrite_input = serialize_runs_for_rewrite(completed_runs)
        # Shaping can empty the committed text (e.g. a dictation that was only a hallucinated
        # "thank you"); skip the rewrite rather than calling the LLM on empty input.
        if not rewrite_input.strip():
          self.logger.info("recording finalized without committed text", stream=message.stream)
          return
        if recording_observation is not None:
          _ = recording_observation.update(
            input={
              "raw_transcript": raw_text,
              "rewrite_input": rewrite_input,
            },
            metadata={
              "component": "active-listener",
              "stream": message.stream,
              "recording_id": finished_recording.recording_id,
              "duration_seconds": reducer_state.duration_seconds,
              "word_count": len(reducer_state.completed_words),
            },
          )

        rewrite_result: RewriteResult | None = None
        if self.current_llm_active():
          try:
            rewrite_result = await self._rewrite_with_llm(
              transcript=rewrite_input,
              stream=message.stream,
              recording_id=finished_recording.recording_id,
              recording_observation=recording_observation,
            )
          except Exception as exc:
            self.logger.exception(
              "dictation pipeline step failed",
              stream=message.stream,
              step="rewrite_with_llm",
              reason=str(exc),
            )
            await self.dbus_service.pipeline_failed("rewrite_with_llm", str(exc))
            return
          final_text = rewrite_result.text
        else:
          # No LLM to consume the dictated command instructions, so each command run is
          # dropped entirely rather than pasted literally into the editor. This path is pure
          # and cannot fail, so it carries no failure signal.
          final_text = serialize_runs_without_commands(completed_runs)
        if not final_text.strip():
          self.logger.info("recording finalized without committed text", stream=message.stream)
          return
        # Trailing space (formerly the append_trailing_space pipeline step).
        final_text = f"{final_text} "

        llm_mode = _llm_mode(
          llm_available=self.current_llm_available(),
          rewrite_ran=rewrite_result is not None,
        )
        emitted_text_source = _emitted_text_source(llm_mode)

        try:
          self.emitter.emit_text(final_text)
        except Exception:
          update_recording_observation(
            recording_observation,
            logger=self.logger,
            level="ERROR",
            status_message="text emission failed",
          )
          self.logger.exception("text emission failed", stream=message.stream)
          return

        self.logger.info("recording finalization mode", stream=message.stream, llm_mode=llm_mode)
        self.logger.info(
          "text emitted",
          stream=message.stream,
          emitted_text=final_text,
          text_length=len(final_text),
          source=emitted_text_source,
        )
        finalized_record = FinalizedTranscriptRecord(
          pre_finalization_text=raw_text,
          post_finalization_text=final_text,
          llm_model=(rewrite_result.model if rewrite_result is not None else None),
          tokens_in=(rewrite_result.input_tokens if rewrite_result is not None else None),
          tokens_out=(rewrite_result.output_tokens if rewrite_result is not None else None),
          cost=(rewrite_result.cost if rewrite_result is not None else None),
          word_count=_count_words(final_text),
          duration_seconds=reducer_state.duration_seconds,
        )
        archived_audio: bytes | None = None
        try:
          archived_audio = encode_recording_audio(
            finished_recording.captured_audio,
            ffmpeg_path=self.config.ffmpeg_path,
          )
        except Exception as exc:
          self.logger.exception("recording audio encode failed", stream=message.stream)
          await self.dbus_service.audio_archive_failed(str(exc))
        self.history_store.record_finalized_recording(finalized_record, archived_audio)
        audio_attachment = build_langfuse_audio_attachment(audio_bytes=archived_audio)
        if recording_observation is not None:
          output_payload: dict[str, object] = {"emitted_text": final_text}
          if audio_attachment is not None:
            output_payload["captured_audio"] = audio_attachment
          _ = recording_observation.update(
            output=output_payload,
            metadata={
              "component": "active-listener",
              "stream": message.stream,
              "recording_id": finished_recording.recording_id,
              "llm_mode": llm_mode,
              "source": emitted_text_source,
              "word_count": finalized_record.word_count,
              "duration_seconds": finalized_record.duration_seconds,
              "llm_model": finalized_record.llm_model,
              "tokens_in": finalized_record.tokens_in,
              "tokens_out": finalized_record.tokens_out,
              "cost": (
                format(finalized_record.cost, "f") if finalized_record.cost is not None else None
              ),
            },
          )
    finally:
      end_recording_observation(recording_observation, logger=self.logger)

  async def _resolve_corrections(
    self, finished_recording: FinishedRecording, *, stream: str
  ) -> CorrectionMap:
    """Await this recording's stored-correction load and return the map.

    The session drops its own correction load task during teardown, so the surviving task
    travels on the finished recording. Resolving it here keeps the finalizer independent of
    live-session state, which a subsequent recording may already own.

    :param finished_recording: The recording being finalized.
    :type finished_recording: FinishedRecording
    :param stream: Stream identifier for logging.
    :type stream: str
    :returns: Loaded spelling corrections, or an empty map on failure or absence.
    :rtype: CorrectionMap
    """
    task = finished_recording.correction_load_task
    if task is None:
      return {}
    try:
      return await task
    except Exception:
      self.logger.exception("stored correction load failed", stream=stream)
      return {}

  async def _rewrite_with_llm(
    self,
    *,
    transcript: str,
    stream: str,
    recording_id: str,
    recording_observation: RecordingObservation | None,
  ) -> RewriteResult:
    rewrite_config = self.config.llm_rewrite
    # Defensive: unreachable while current_llm_active() gates the call site.
    if rewrite_config is None:
      raise rewrite_module.RewriteClientError("rewrite is disabled")

    #! This very deliberately happens on each recording run. DO NOT ALTER THIS. Do not ask
    # about loading up front. Just leave it.
    loaded_prompt = rewrite_module.load_active_listener_rewrite_prompt(rewrite_config.prompt_path)
    prompt = loaded_prompt.prompt
    prompt_path = str(loaded_prompt.prompt_path)
    self.logger.info(
      "rewrite prompt loaded",
      stream=stream,
      prompt_path=prompt_path,
    )
    self.logger.info(
      "rewrite started",
      stream=stream,
      prompt_path=prompt_path,
      rewrite_input=transcript,
    )
    provider = _rewrite_provider_type(rewrite_config)
    model = _rewrite_model(rewrite_config)

    with start_rewrite_observation(
      session_id=stream,
      recording_id=recording_id,
      provider=provider,
      model=model,
      prompt_path=prompt_path,
      stream=stream,
      transcript=transcript,
      parent_observation=recording_observation,
    ) as observation:
      try:
        rewrite_result = await self.rewrite_client.rewrite_text(
          instructions=prompt.instructions,
          transcript=transcript,
        )
      except Exception as exc:
        if observation is not None:
          _ = observation.update(level="ERROR", status_message=str(exc))
        raise

      if observation is not None:
        _ = observation.update(
          output=rewrite_result.text,
          usage_details=_langfuse_usage_details(rewrite_result),
          cost_details=_langfuse_cost_details(rewrite_result),
        )

    self.logger.info(
      "rewrite succeeded",
      stream=stream,
      prompt_path=prompt_path,
      rewrite_input=transcript,
      rewritten_text=rewrite_result.text,
    )
    if rewrite_result.corrections:
      try:
        await self.correction_store.merge_async(rewrite_result.corrections)
        self.logger.info(
          "rewrite corrections persisted",
          stream=stream,
          correction_count=len(rewrite_result.corrections),
          path=str(self.correction_store.path),
        )
      except Exception:
        self.logger.exception(
          "rewrite correction persistence failed",
          stream=stream,
          correction_count=len(rewrite_result.corrections),
          path=str(self.correction_store.path),
        )
    return rewrite_result


def _count_words(text: str) -> int:
  return len(text.split())


def _llm_mode(*, llm_available: bool, rewrite_ran: bool) -> str:
  if not llm_available:
    return "unavailable"

  if rewrite_ran:
    return "active"

  return "bypassed"


def _emitted_text_source(llm_mode: str) -> str:
  if llm_mode == "unavailable":
    return "raw"

  if llm_mode == "active":
    return "pipeline_llm"

  return "pipeline_no_llm"


def _rewrite_provider_type(config: LlmRewriteConfig) -> str:
  return config.provider.type


def _rewrite_model(config: LlmRewriteConfig) -> str:
  provider = config.provider
  if isinstance(provider, LiteRtRewriteProvider):
    return provider.model_path

  return provider.model


def _langfuse_usage_details(rewrite_result: RewriteResult) -> dict[str, int] | None:
  usage_details: dict[str, int] = {}

  if rewrite_result.input_tokens is not None:
    usage_details["input"] = rewrite_result.input_tokens

  if rewrite_result.output_tokens is not None:
    usage_details["output"] = rewrite_result.output_tokens

  if not usage_details:
    return None

  return usage_details


def _langfuse_cost_details(rewrite_result: RewriteResult) -> dict[str, float] | None:
  if rewrite_result.cost is None:
    return None

  return {"total": float(rewrite_result.cost)}
