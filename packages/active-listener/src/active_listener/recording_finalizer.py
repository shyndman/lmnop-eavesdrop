"""Recording finalization pipeline for active-listener."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import active_listener.rewrite as rewrite_module
from active_listener.dbus_service import AppStateService
from active_listener.emitter import TextEmitter
from active_listener.reducer import RecordingReducerState, render_text
from active_listener.service_ports import (
  ActiveListenerClient,
  ActiveListenerLogger,
  ActiveListenerRewriteClient,
)
from active_listener.settings import ActiveListenerConfig
from eavesdrop.wire import TranscriptionMessage

PipelineStep = Callable[[str], Awaitable[str]]
RecordingMessageIngestor = Callable[[RecordingReducerState, TranscriptionMessage], None]
DisconnectGenerationReader = Callable[[], int]


@dataclass
class RecordingFinalizer:
  """Finalize a finished recording into emitted workstation text."""

  config: ActiveListenerConfig
  client: ActiveListenerClient
  emitter: TextEmitter
  logger: ActiveListenerLogger
  rewrite_client: ActiveListenerRewriteClient
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

      self.ingest_transcription_message(reducer_state, message)
      raw_text = render_text(reducer_state.parts)
      if not raw_text:
        self.logger.info("recording finalized without committed text", stream=message.stream)
        return

      self.logger.info("finalized raw transcript", stream=message.stream, raw_text=raw_text)

      pipeline_steps = self._pipeline_steps(stream=message.stream)
      final_text = await self._run_pipeline(
        text=raw_text,
        steps=pipeline_steps,
        stream=message.stream,
      )
      if final_text is None:
        return

      emitted_text_source = "raw" if not pipeline_steps else "pipeline"

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

  def _pipeline_steps(self, *, stream: str) -> tuple[PipelineStep, ...]:
    if not self.config.llm_rewrite.enabled:
      return ()

    async def rewrite_with_llm(text: str) -> str:
      return await self._rewrite_with_llm(text=text, stream=stream)

    #! Yes, I'm aware there's only a single step right now. More are coming soon. Do not touch this
    # functionality.
    return (rewrite_with_llm,)

  async def _run_pipeline(
    self,
    *,
    text: str,
    steps: tuple[PipelineStep, ...],
    stream: str,
  ) -> str | None:
    for step in steps:
      try:
        text = await step(text)
      except Exception as exc:
        self.logger.exception(
          "dictation pipeline step failed",
          stream=stream,
          step=step.__name__,
          reason=str(exc),
        )
        await self.dbus_service.pipeline_failed(step.__name__, str(exc))
        return None

    return text

  async def _rewrite_with_llm(self, *, text: str, stream: str) -> str:
    prompt_path: str | None = None

    #! This very deliberately happens on each recording run. DO NOT ALTER THIS. Do not ask
    # about loading up front. Just leave it.
    loaded_prompt = rewrite_module.load_active_listener_rewrite_prompt(
      self.config.llm_rewrite.prompt_path
    )
    prompt = loaded_prompt.prompt
    prompt_path = str(loaded_prompt.prompt_path)
    self.logger.info(
      "rewrite prompt loaded",
      stream=stream,
      prompt_path=prompt_path,
      model_name=prompt.model_name,
      prompt_metadata=prompt.metadata,
    )
    self.logger.info(
      "rewrite prompt rendered",
      stream=stream,
      prompt_path=prompt_path,
      model_name=prompt.model_name,
      instructions=prompt.instructions,
    )
    self.logger.info(
      "rewrite started",
      stream=stream,
      base_url=self.config.llm_rewrite.base_url,
      prompt_path=prompt_path,
      model_name=prompt.model_name,
      raw_text=text,
    )
    rewritten_text = await self.rewrite_client.rewrite_text(
      model_name=prompt.model_name,
      instructions=prompt.instructions,
      transcript=text,
    )
    self.logger.info(
      "rewrite succeeded",
      stream=stream,
      prompt_path=prompt_path,
      model_name=prompt.model_name,
      raw_text=text,
      rewritten_text=rewritten_text,
    )
    return rewritten_text
