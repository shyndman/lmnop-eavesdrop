"""Recording finalization pipeline for active-listener."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
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
from active_listener.infra.dbus import AppStateService
from active_listener.infra.emitter import TextEmitter
from active_listener.recording.reducer import (
  RecordingReducerState,
  TranscriptionUpdate,
  build_completed_text_runs,
  render_text,
  serialize_text_runs,
)
from eavesdrop.wire import TranscriptionMessage

from ..infra.langfuse import (
  build_langfuse_audio_attachment,
  start_recording_observation,
  start_rewrite_observation,
)


def _empty_metadata() -> dict[str, object]:
  return {}


@dataclass(frozen=True)
class PipelineContext:
  transcript: str
  llm_enabled: bool
  metadata: Mapping[str, object] = field(default_factory=_empty_metadata)


PipelineStep = Callable[[PipelineContext], Awaitable[PipelineContext]]

# Spoken symbol words recognized by `_replace_symbols`. DO NOT expand this list
# casually — it is deliberately tiny so that normal prose words never collide.
_SYMBOL_WORDS = {
  "backslash": "\\",
  "dot": ".",
  "slash": "/",
  "tild": "~",
  "tilde": "~",
}
_SYMBOL_PATTERN = re.compile(
  r"\s*\b(" + "|".join(_SYMBOL_WORDS) + r")\b\s*",
  re.IGNORECASE,
)

# Case-insensitive whole-phrase replacements applied BEFORE symbol fusion.
# Keys are matched case-insensitively; values are substituted verbatim with
# their canonical casing. Add freely — keep keys lowercase for readability.
_REPLACEMENTS: dict[str, str] = {
  "debass": "D-Bus",
  "hillary": "hilary",
  "tild": "tilde",
  "yamel": "yaml",
}
_REPLACEMENT_PATTERN = (
  re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _REPLACEMENTS) + r")\b",
    re.IGNORECASE,
  )
  if _REPLACEMENTS
  else None
)
_INSTRUCTION_TAG_PATTERN = re.compile(r"</?\s*instruction\s*>", re.IGNORECASE)
_THANK_YOU_PATTERN = re.compile(r"\b(?:(escape)\s+)?(thank you)\b([,.!?;:]*)", re.IGNORECASE)
_HORIZONTAL_WHITESPACE_PATTERN = re.compile(r"[ \t]{2,}")
_REMOVE_PRECEDING_THANK_YOU_INSTRUCTION = (
  "<instruction>remove the preceding thank you</instruction>"
)
_REWRITE_RESULT_METADATA_KEY = "rewrite_result"

RecordingMessageIngestor = Callable[
  [RecordingReducerState, TranscriptionMessage],
  TranscriptionUpdate | None,
]
LlmAvailabilityReader = Callable[[], bool]
LlmActiveReader = Callable[[], bool]
DisconnectGenerationReader = Callable[[], int]


@dataclass
class RecordingFinalizer:
  """Finalize a finished recording into emitted workstation text."""

  config: ActiveListenerConfig
  client: ActiveListenerClient
  emitter: TextEmitter
  ffmpeg_path: str | None
  logger: BoundLogger
  rewrite_client: ActiveListenerRewriteClient
  history_store: ActiveListenerTranscriptHistoryStore
  dbus_service: AppStateService
  ingest_transcription_message: RecordingMessageIngestor
  current_llm_available: LlmAvailabilityReader
  current_llm_active: LlmActiveReader
  current_disconnect_generation: DisconnectGenerationReader
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
  ) -> None:
    reducer_state = finished_recording.reducer_state
    try:
      async with self._flush_lock:
        try:
          message = await self.client.flush(finished_recording.recording_id, force_complete=True)
        finally:
          self._release_reserved_flush()
    except Exception:
      self.logger.exception("recording finalization failed")
      return

    if self.current_disconnect_generation() != disconnect_generation:
      self.logger.warning("skipping emission after disconnect", stream=message.stream)
      return

    async with self._lock:
      if self.current_disconnect_generation() != disconnect_generation:
        self.logger.warning("skipping emission after disconnect", stream=message.stream)
        return

      _ = self.ingest_transcription_message(reducer_state, message)
      completed_runs = build_completed_text_runs(reducer_state)
      raw_text = render_text(reducer_state.completed_words)
      if not raw_text:
        self.logger.info("recording finalized without committed text", stream=message.stream)
        return

      self.logger.info("finalized raw transcript", stream=message.stream, raw_text=raw_text)
      rewrite_input = serialize_text_runs(completed_runs)
      audio_attachment = build_langfuse_audio_attachment(
        captured_audio=finished_recording.captured_audio,
        ffmpeg_path=self.ffmpeg_path,
        logger=self.logger,
      )

      with start_recording_observation(
        session_id=message.stream,
        stream=message.stream,
        recording_id=finished_recording.recording_id,
        raw_text=raw_text,
        rewrite_input=rewrite_input,
        audio_attachment=audio_attachment,
        duration_seconds=reducer_state.duration_seconds,
        word_count=len(reducer_state.completed_words),
      ) as recording_observation:
        pipeline_steps = self._pipeline_steps(stream=message.stream)
        pipeline_context = await self._run_pipeline(
          context=PipelineContext(
            transcript=rewrite_input,
            llm_enabled=self.current_llm_active(),
          ),
          steps=pipeline_steps,
          stream=message.stream,
        )
        if pipeline_context is None:
          if recording_observation is not None:
            _ = recording_observation.update(
              level="ERROR",
              status_message="finalization pipeline failed",
            )
          return

        final_text = pipeline_context.transcript
        rewrite_result = _rewrite_result_from_metadata(pipeline_context.metadata)

        llm_mode = _llm_mode(
          llm_available=self.current_llm_available(),
          rewrite_ran=rewrite_result is not None,
        )
        emitted_text_source = _emitted_text_source(llm_mode)

        try:
          self.emitter.emit_text(final_text)
        except Exception:
          if recording_observation is not None:
            _ = recording_observation.update(level="ERROR", status_message="text emission failed")
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
        self.history_store.record_finalized_recording(
          finalized_record,
          finished_recording.captured_audio,
        )
        if recording_observation is not None:
          _ = recording_observation.update(
            output={"emitted_text": final_text},
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

  def _pipeline_steps(self, *, stream: str) -> list[PipelineStep]:
    async def rewrite_with_llm(context: PipelineContext) -> PipelineContext:
      return await self._rewrite_with_llm(context=context, stream=stream)

    async def append_trailing_space(context: PipelineContext) -> PipelineContext:
      return _update_pipeline_context(
        context,
        transcript=f"{context.transcript} ",
      )

    steps: list[PipelineStep] = [
      self._strip_instruction_tags_when_llm_disabled,
      self._apply_replacements,
      self._apply_thank_you_escape,
      self._replace_symbols,
      rewrite_with_llm,
      append_trailing_space,
    ]

    return steps

  async def _strip_instruction_tags_when_llm_disabled(
    self,
    context: PipelineContext,
  ) -> PipelineContext:
    if context.llm_enabled:
      return context

    return _update_pipeline_context(
      context,
      transcript=_strip_instruction_tags(context.transcript),
    )

  async def _apply_replacements(self, context: PipelineContext) -> PipelineContext:
    # Case-insensitive whole-word/phrase substitution driven by `_REPLACEMENTS`.
    # Extend the map at module level; this function should stay dumb.
    if _REPLACEMENT_PATTERN is None:
      return context
    replacement_pattern = _REPLACEMENT_PATTERN
    return _update_pipeline_context(
      context,
      transcript=replacement_pattern.sub(
        lambda m: _REPLACEMENTS[m.group(1).lower()],
        context.transcript,
      ),
    )

  async def _replace_symbols(self, context: PipelineContext) -> PipelineContext:
    #! This satisfies the design intent. Do not touch.
    #
    # Fuse spoken symbol words into their glyphs, swallowing adjacent whitespace:
    # e.g. "tild slash dot omp slash agent slash skills" -> "~/.omp/agent/skills".
    symbol_pattern = _SYMBOL_PATTERN
    return _update_pipeline_context(
      context,
      transcript=symbol_pattern.sub(
        lambda m: _SYMBOL_WORDS[m.group(1).lower()],
        context.transcript,
      ),
    )

  async def _apply_thank_you_escape(self, context: PipelineContext) -> PipelineContext:
    return _update_pipeline_context(
      context,
      transcript=_apply_thank_you_policy(
        context.transcript,
        llm_enabled=context.llm_enabled,
      ),
    )

  async def _run_pipeline(
    self,
    *,
    context: PipelineContext,
    steps: Iterable[PipelineStep],
    stream: str,
  ) -> PipelineContext | None:
    for step in steps:
      try:
        context = await step(context)
      except Exception as exc:
        self.logger.exception(
          "dictation pipeline step failed",
          stream=stream,
          step=step.__name__,
          reason=str(exc),
        )
        await self.dbus_service.pipeline_failed(step.__name__, str(exc))
        return None

    return context

  async def _rewrite_with_llm(
    self,
    *,
    context: PipelineContext,
    stream: str,
  ) -> PipelineContext:
    if not context.llm_enabled:
      return context

    rewrite_config = self.config.llm_rewrite
    if rewrite_config is None:
      raise rewrite_module.RewriteClientError("rewrite is disabled")

    prompt_path: str | None = None

    #! This very deliberately happens on each recording run. DO NOT ALTER THIS. Do not ask
    # about loading up front. Just leave it.
    loaded_prompt = rewrite_module.load_active_listener_rewrite_prompt(rewrite_config.prompt_path)
    prompt = loaded_prompt.prompt
    prompt_path = str(loaded_prompt.prompt_path)
    rewrite_input = context.transcript
    self.logger.info(
      "rewrite prompt loaded",
      stream=stream,
      prompt_path=prompt_path,
    )
    self.logger.info(
      "rewrite started",
      stream=stream,
      prompt_path=prompt_path,
      rewrite_input=rewrite_input,
    )
    provider = _rewrite_provider_type(rewrite_config)
    model = _rewrite_model(rewrite_config)

    with start_rewrite_observation(
      session_id=stream,
      provider=provider,
      model=model,
      prompt_path=prompt_path,
      stream=stream,
      transcript=rewrite_input,
    ) as observation:
      try:
        rewrite_result = await self.rewrite_client.rewrite_text(
          instructions=prompt.instructions,
          transcript=rewrite_input,
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
      rewrite_input=rewrite_input,
      rewritten_text=rewrite_result.text,
    )
    return _update_pipeline_context(
      context,
      transcript=rewrite_result.text,
      metadata_updates={_REWRITE_RESULT_METADATA_KEY: rewrite_result},
    )


def _count_words(text: str) -> int:
  return len(text.split())


def _update_pipeline_context(
  context: PipelineContext,
  *,
  transcript: str,
  metadata_updates: Mapping[str, object] | None = None,
) -> PipelineContext:
  metadata = (
    context.metadata if metadata_updates is None else {**context.metadata, **metadata_updates}
  )
  return PipelineContext(
    transcript=transcript,
    llm_enabled=context.llm_enabled,
    metadata=metadata,
  )


def _rewrite_result_from_metadata(metadata: Mapping[str, object]) -> RewriteResult | None:
  rewrite_result = metadata.get(_REWRITE_RESULT_METADATA_KEY)
  if rewrite_result is None:
    return None

  if not isinstance(rewrite_result, RewriteResult):
    raise TypeError("rewrite_result metadata must be a RewriteResult")

  return rewrite_result


def _strip_instruction_tags(text: str) -> str:
  return _INSTRUCTION_TAG_PATTERN.sub("", text)


def _apply_thank_you_policy(text: str, *, llm_enabled: bool) -> str:
  def replace_match(match: re.Match[str]) -> str:
    thank_you_text = f"{match.group(2)}{match.group(3)}"
    if match.group(1) is not None:
      return thank_you_text

    if not llm_enabled:
      return ""

    return f"{thank_you_text} {_REMOVE_PRECEDING_THANK_YOU_INSTRUCTION}"

  return _HORIZONTAL_WHITESPACE_PATTERN.sub(
    " ",
    _THANK_YOU_PATTERN.sub(replace_match, text),
  ).strip()


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
