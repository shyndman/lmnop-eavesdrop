"""Recording finalization pipeline for active-listener."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field

from structlog.stdlib import BoundLogger

import active_listener.infra.rewrite as rewrite_module
from active_listener.app.ports import (
  ActiveListenerClient,
  ActiveListenerRewriteClient,
  ActiveListenerTranscriptHistoryStore,
  FinalizedTranscriptRecord,
  RewriteResult,
)
from active_listener.config.models import ActiveListenerConfig
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


@dataclass(frozen=True)
class FinalizationState:
  text: str
  rewrite_input: str | None = None
  rewrite_result: RewriteResult | None = None


PipelineStep = Callable[[FinalizationState], Awaitable[FinalizationState]]

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
_REPLACEMENTS: dict[str, str] = {"debass": "D-Bus", "tild": "tilde", "yamel": "yaml"}
_REPLACEMENT_PATTERN = (
  re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _REPLACEMENTS) + r")\b",
    re.IGNORECASE,
  )
  if _REPLACEMENTS
  else None
)

RecordingMessageIngestor = Callable[
  [RecordingReducerState, TranscriptionMessage],
  TranscriptionUpdate | None,
]
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
  ingest_transcription_message: RecordingMessageIngestor
  current_disconnect_generation: DisconnectGenerationReader
  _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

  async def finalize_recording(
    self,
    *,
    disconnect_generation: int,
    reducer_state: RecordingReducerState,
  ) -> None:
    async with self._lock:
      try:
        message = await self.client.flush(force_complete=True)
      except Exception:
        self.logger.exception("recording finalization failed")
        return

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

      pipeline_steps = self._pipeline_steps(stream=message.stream)
      finalization_state = await self._run_pipeline(
        state=FinalizationState(
          text=raw_text,
          rewrite_input=serialize_text_runs(completed_runs),
        ),
        steps=pipeline_steps,
        stream=message.stream,
      )
      if finalization_state is None:
        return

      final_text = finalization_state.text

      emitted_text_source = "pipeline" if self.config.llm_rewrite is not None else "raw"

      try:
        self.emitter.emit_text(final_text)
      except Exception:
        self.logger.exception("text emission failed", stream=message.stream)
        return

      self.logger.info(
        "text emitted",
        stream=message.stream,
        emitted_text=final_text,
        text_length=len(final_text),
        source=emitted_text_source,
      )
      self.history_store.record_finalized_transcript(
        FinalizedTranscriptRecord(
          pre_finalization_text=raw_text,
          post_finalization_text=final_text,
          llm_model=(
            finalization_state.rewrite_result.model
            if finalization_state.rewrite_result is not None
            else None
          ),
          tokens_in=(
            finalization_state.rewrite_result.input_tokens
            if finalization_state.rewrite_result is not None
            else None
          ),
          tokens_out=(
            finalization_state.rewrite_result.output_tokens
            if finalization_state.rewrite_result is not None
            else None
          ),
          cost=(
            finalization_state.rewrite_result.cost
            if finalization_state.rewrite_result is not None
            else None
          ),
          word_count=_count_words(final_text),
          duration_seconds=reducer_state.duration_seconds,
        )
      )

  def _pipeline_steps(self, *, stream: str) -> list[PipelineStep]:
    async def rewrite_with_llm(state: FinalizationState) -> FinalizationState:
      return await self._rewrite_with_llm(state=state, stream=stream)

    async def append_trailing_space(state: FinalizationState) -> FinalizationState:
      return FinalizationState(
        text=f"{state.text} ",
        rewrite_input=state.rewrite_input,
        rewrite_result=state.rewrite_result,
      )

    steps: list[PipelineStep] = [
      self._apply_replacements,
      self._replace_symbols,
    ]

    if self.config.llm_rewrite is not None:
      steps.append(rewrite_with_llm)

    steps.append(append_trailing_space)

    return steps

  async def _apply_replacements(self, state: FinalizationState) -> FinalizationState:
    # Case-insensitive whole-word/phrase substitution driven by `_REPLACEMENTS`.
    # Extend the map at module level; this function should stay dumb.
    if _REPLACEMENT_PATTERN is None:
      return state
    replacement_pattern = _REPLACEMENT_PATTERN
    return FinalizationState(
      text=replacement_pattern.sub(
        lambda m: _REPLACEMENTS[m.group(1).lower()],
        state.text,
      ),
      rewrite_input=self._rewrite_pipeline_text(
        state.rewrite_input,
        lambda text: replacement_pattern.sub(
          lambda m: _REPLACEMENTS[m.group(1).lower()],
          text,
        ),
      ),
      rewrite_result=state.rewrite_result,
    )

  async def _replace_symbols(self, state: FinalizationState) -> FinalizationState:
    #! This satisfies the design intent. Do not touch.
    #
    # Fuse spoken symbol words into their glyphs, swallowing adjacent whitespace:
    # e.g. "tild slash dot omp slash agent slash skills" -> "~/.omp/agent/skills".
    symbol_pattern = _SYMBOL_PATTERN
    return FinalizationState(
      text=symbol_pattern.sub(
        lambda m: _SYMBOL_WORDS[m.group(1).lower()],
        state.text,
      ),
      rewrite_input=self._rewrite_pipeline_text(
        state.rewrite_input,
        lambda text: symbol_pattern.sub(
          lambda m: _SYMBOL_WORDS[m.group(1).lower()],
          text,
        ),
      ),
      rewrite_result=state.rewrite_result,
    )

  async def _run_pipeline(
    self,
    *,
    state: FinalizationState,
    steps: Iterable[PipelineStep],
    stream: str,
  ) -> FinalizationState | None:
    for step in steps:
      try:
        state = await step(state)
      except Exception as exc:
        self.logger.exception(
          "dictation pipeline step failed",
          stream=stream,
          step=step.__name__,
          reason=str(exc),
        )
        await self.dbus_service.pipeline_failed(step.__name__, str(exc))
        return None

    return state

  async def _rewrite_with_llm(
    self,
    *,
    state: FinalizationState,
    stream: str,
  ) -> FinalizationState:
    rewrite_config = self.config.llm_rewrite
    if rewrite_config is None:
      raise rewrite_module.RewriteClientError("rewrite is disabled")

    prompt_path: str | None = None

    #! This very deliberately happens on each recording run. DO NOT ALTER THIS. Do not ask
    # about loading up front. Just leave it.
    loaded_prompt = rewrite_module.load_active_listener_rewrite_prompt(rewrite_config.prompt_path)
    prompt = loaded_prompt.prompt
    prompt_path = str(loaded_prompt.prompt_path)
    rewrite_input = state.rewrite_input or state.text
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
    rewrite_result = await self.rewrite_client.rewrite_text(
      instructions=prompt.instructions,
      transcript=rewrite_input,
    )
    self.logger.info(
      "rewrite succeeded",
      stream=stream,
      prompt_path=prompt_path,
      rewrite_input=rewrite_input,
      rewritten_text=rewrite_result.text,
    )
    return FinalizationState(
      text=rewrite_result.text,
      rewrite_input=state.rewrite_input,
      rewrite_result=rewrite_result,
    )

  def _rewrite_pipeline_text(
    self,
    text: str | None,
    transform: Callable[[str], str],
  ) -> str | None:
    if text is None:
      return None
    return transform(text)


def _count_words(text: str) -> int:
  return len(text.split())
