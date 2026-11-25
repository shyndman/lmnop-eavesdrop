"""Cloud LLM rewrite client and prompt loading for active-listener."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import ClassVar, Self, final

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai import Agent
from pydantic_ai.run import AgentRunResult

from active_listener.app.ports import RewriteResult
from active_listener.infra.corrections import validate_corrections
from active_listener.infra.langfuse import initialize_langfuse
from active_listener.infra.user_config import EAVESDROP_CONFIG_DIRNAME, resolve_user_config_dir

ACTIVE_LISTENER_PROMPT_FILENAME = "active-listener.rewrite.system.md"


@dataclass(frozen=True)
class LoadedRewritePrompt:
  instructions: str


@dataclass(frozen=True)
class LoadedRewritePromptFile:
  prompt_path: Path
  prompt: LoadedRewritePrompt


class RewritePromptError(RuntimeError):
  def __init__(self, message: str, *, prompt_path: Path | None = None) -> None:
    super().__init__(message)
    self.prompt_path: Path | None = prompt_path


class RewriteClientError(RuntimeError):
  pass


class RewriteClientTimeoutError(RewriteClientError):
  pass


#! DO NOT REMOVE/ALTER COMMENTS BELOW. THEY ARE CRITICAL INSTRUCTIONS FOR THE MODEL.
class StructuredRewriteOutput(BaseModel):
  model_config: ClassVar[ConfigDict] = ConfigDict(use_attribute_docstrings=True)

  text: str
  """The transcript rewritten according to the rules and user-provided instructions"""
  corrections: dict[str, str] = Field(default_factory=dict)
  """Persistent recurring spelling or word substitutions explicitly marked by the speaker
  in instruction text (excluding one-off corrections)"""

  @model_validator(mode="after")
  def validate_output(self) -> Self:
    self.text = self.text.strip()
    if self.text == "":
      raise ValueError("rewrite model returned empty output")
    self.corrections = validate_corrections(self.corrections)
    return self


@final
class PydanticAiRewriteClient:
  def __init__(self, *, model: str) -> None:
    self.model: str = model
    _ = initialize_langfuse()
    self._agent: Agent[None, StructuredRewriteOutput] = Agent(
      output_type=StructuredRewriteOutput,
      instrument=True,
      name="active-listener-rewrite",
    )

  async def rewrite_text(
    self,
    *,
    instructions: str,
    transcript: str,
  ) -> RewriteResult:
    try:
      response: AgentRunResult[StructuredRewriteOutput] = await self._agent.run(
        transcript,
        instructions=instructions,
        model=self.model,
      )
    except Exception as exc:
      raise RewriteClientError(f"rewrite request failed: {exc}") from exc

    rewrite_output = response.output
    if rewrite_output.text == "":
      raise RewriteClientError("rewrite model returned empty output")

    usage = response.usage()
    return RewriteResult(
      text=rewrite_output.text,
      model=self.model,
      input_tokens=usage.input_tokens,
      output_tokens=usage.output_tokens,
      cost=_extract_total_cost(response),
      corrections=rewrite_output.corrections,
    )

  async def close(self) -> None:
    return None


@dataclass(frozen=True)
class DisabledRewriteClient:
  async def rewrite_text(
    self,
    *,
    instructions: str,
    transcript: str,
  ) -> RewriteResult:
    _ = instructions
    _ = transcript
    raise RewriteClientError("rewrite is disabled")

  async def close(self) -> None:
    return None


def load_active_listener_rewrite_prompt(configured_prompt_path: str) -> LoadedRewritePromptFile:
  prompt_path = resolve_active_listener_prompt_path(configured_prompt_path)
  prompt = load_rewrite_prompt(prompt_path)
  return LoadedRewritePromptFile(prompt_path=prompt_path, prompt=prompt)


def resolve_active_listener_prompt_path(configured_prompt_path: str) -> Path:
  override_prompt_path = resolve_active_listener_override_prompt_path()
  if override_prompt_path.exists():
    return override_prompt_path

  return resolve_prompt_path(configured_prompt_path)


def resolve_active_listener_override_prompt_path() -> Path:
  return resolve_user_config_dir() / EAVESDROP_CONFIG_DIRNAME / ACTIVE_LISTENER_PROMPT_FILENAME


def load_rewrite_prompt(prompt_path: str | Path) -> LoadedRewritePrompt:
  resolved_prompt_path = resolve_prompt_path(prompt_path)

  try:
    instructions = resolved_prompt_path.read_text(encoding="utf-8")
  except Exception as exc:
    raise RewritePromptError(
      f"failed to load rewrite prompt: {resolved_prompt_path}",
      prompt_path=resolved_prompt_path,
    ) from exc

  if instructions.strip() == "":
    raise RewritePromptError("rewrite prompt is empty", prompt_path=resolved_prompt_path)

  return LoadedRewritePrompt(instructions=instructions)


def _extract_total_cost(response: AgentRunResult[StructuredRewriteOutput]) -> Decimal | None:
  try:
    return response.response.cost().total_price
  except Exception:
    return None


def resolve_prompt_path(raw_path: str | Path) -> Path:
  configured_path = Path(raw_path).expanduser()
  return configured_path.resolve()
