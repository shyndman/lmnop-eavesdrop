"""Load the current rewrite prompt, trace the provider call, and persist corrections."""

from __future__ import annotations

from langfuse import LangfuseSpan
from structlog.stdlib import BoundLogger

import active_listener.infra.rewrite as rewrite_module
from active_listener.app.ports import ActiveListenerRewriteClient, RewriteResult
from active_listener.config.models import LiteRtRewriteProvider, LlmRewriteConfig
from active_listener.infra.corrections import ActiveListenerCorrectionStore
from active_listener.infra.langfuse import start_rewrite_observation


class TranscriptRewriter:
  """Apply the configured rewrite independently of foreground application state.

  :param config: Configured provider and prompt, or None when rewriting is disabled.
  :param client: Provider request implementation.
  :param corrections: Persistent spelling-correction destination.
  :param logger: Prompt, provider, and correction-persistence diagnostics.
  """

  def __init__(
    self,
    config: LlmRewriteConfig | None,
    client: ActiveListenerRewriteClient,
    corrections: ActiveListenerCorrectionStore,
    logger: BoundLogger,
  ) -> None:
    self.config: LlmRewriteConfig | None = config
    self.client: ActiveListenerRewriteClient = client
    self.corrections: ActiveListenerCorrectionStore = corrections
    self.logger: BoundLogger = logger

  async def rewrite(
    self,
    *,
    transcript: str,
    session_id: str,
    recording_id: str,
    recording_observation: LangfuseSpan | None,
  ) -> RewriteResult:
    """Rewrite with the prompt reloaded for this recording, never cached at startup.

    :param transcript: Shaped transcript with command instructions marked.
    :param session_id: Server session identifier for tracing.
    :param recording_id: Recording identifier for trace ownership.
    :param recording_observation: Transferred parent observation, if tracing is enabled.
    :return: Provider text, usage, cost, and learned spelling corrections.
    :raises rewrite_module.RewriteClientError: If rewriting is unconfigured or fails.
    """
    config = self.config
    if config is None:
      self.logger.error("rewrite requested without configuration")
      raise rewrite_module.RewriteClientError("rewrite is disabled")
    # Reload on each recording so edits apply without restarting.
    loaded = rewrite_module.load_active_listener_rewrite_prompt(config.prompt_path)
    path = str(loaded.prompt_path)
    match config.provider:
      case LiteRtRewriteProvider(model_path=model):
        pass
      case provider:
        model = provider.model
    self.logger.info("rewrite prompt loaded", session_id=session_id, prompt_path=path)
    self.logger.info(
      "rewrite started", session_id=session_id, prompt_path=path, rewrite_input=transcript
    )
    with start_rewrite_observation(
      session_id=session_id,
      recording_id=recording_id,
      provider=config.provider.type,
      model=model,
      prompt_path=path,
      transcript=transcript,
      parent_observation=recording_observation,
    ) as observation:
      result = await self._request_rewrite(loaded.prompt.instructions, transcript, observation)
    self.logger.info(
      "rewrite succeeded",
      session_id=session_id,
      prompt_path=path,
      rewrite_input=transcript,
      rewritten_text=result.text,
    )
    await self._persist_corrections(result, session_id)
    return result

  async def _request_rewrite(
    self,
    instructions: str,
    transcript: str,
    observation: LangfuseSpan | None,
  ) -> RewriteResult:
    try:
      result = await self.client.rewrite_text(instructions=instructions, transcript=transcript)
    except Exception as exc:
      if observation is not None:
        _ = observation.update(level="ERROR", status_message=str(exc))
      raise
    if observation is not None:
      _ = observation.update(
        output=result.text, usage_details=_usage_details(result), cost_details=_cost_details(result)
      )
    return result

  async def _persist_corrections(self, result: RewriteResult, session_id: str) -> None:
    if not result.corrections:
      return
    try:
      await self.corrections.merge_async(result.corrections)
      self.logger.info(
        "rewrite corrections persisted",
        session_id=session_id,
        correction_count=len(result.corrections),
        path=str(self.corrections.path),
      )
    except Exception:
      self.logger.exception(
        "rewrite correction persistence failed",
        session_id=session_id,
        correction_count=len(result.corrections),
        path=str(self.corrections.path),
      )


def _usage_details(result: RewriteResult) -> dict[str, int] | None:
  usage: dict[str, int] = {}
  if result.input_tokens is not None:
    usage["input"] = result.input_tokens
  if result.output_tokens is not None:
    usage["output"] = result.output_tokens
  return usage or None


def _cost_details(result: RewriteResult) -> dict[str, float] | None:
  return {"total": float(result.cost)} if result.cost is not None else None
