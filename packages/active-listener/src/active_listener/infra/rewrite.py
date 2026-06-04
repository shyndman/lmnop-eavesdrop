from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from importlib import import_module
from pathlib import Path
from types import TracebackType
from typing import Literal, NotRequired, Protocol, Self, TypedDict, cast, final

from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_ai import Agent
from pydantic_ai.run import AgentRunResult

from active_listener.app.ports import RewriteResult
from active_listener.infra.corrections import validate_corrections
from active_listener.infra.user_config import EAVESDROP_CONFIG_DIRNAME, resolve_user_config_dir

from .langfuse import initialize_langfuse

ACTIVE_LISTENER_PROMPT_FILENAME = "active-listener.rewrite.system.md"


class LiteRtTextContent(TypedDict):
  type: Literal["text"]
  text: str


class LiteRtMessage(TypedDict):
  role: Literal["system"]
  content: list[LiteRtTextContent]


class LiteRtResponseContent(TypedDict):
  type: NotRequired[str]
  text: NotRequired[str]


class LiteRtResponse(TypedDict):
  content: NotRequired[list[LiteRtResponseContent]]


class LiteRtConversation(Protocol):
  def __enter__(self) -> LiteRtConversation: ...

  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc: BaseException | None,
    traceback: TracebackType | None,
  ) -> bool | None: ...

  def send_message(self, prompt: str) -> LiteRtResponse: ...


class LiteRtEngine(Protocol):
  def __enter__(self) -> LiteRtEngine: ...

  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc: BaseException | None,
    traceback: TracebackType | None,
  ) -> bool | None: ...

  def create_conversation(
    self,
    *,
    messages: Sequence[LiteRtMessage] | None = None,
  ) -> LiteRtConversation: ...


class LiteRtBackend(Protocol):
  CPU: object


class LiteRtEngineFactory(Protocol):
  def __call__(self, model_path: str, *, backend: object) -> LiteRtEngine: ...


class LiteRtModule(Protocol):
  Backend: LiteRtBackend
  Engine: LiteRtEngineFactory


litert_lm: LiteRtModule | None = None


def _load_litert_module() -> LiteRtModule:
  global litert_lm
  if litert_lm is None:
    litert_lm = cast(LiteRtModule, cast(object, import_module("litert_lm")))
  return litert_lm


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
  text: str
  """The transcript rewritten according to the rules and user-provided instructions"""
  corrections: dict[str, str] = Field(default_factory=dict)
  """A mapping of corrections the user made by explicitly spelling a corrections letter by letter
  in instruction text"""

  @model_validator(mode="after")
  def validate_output(self) -> Self:
    self.text = self.text.strip()
    if self.text == "":
      raise ValueError("rewrite model returned empty output")
    self.corrections = validate_corrections(self.corrections)
    return self


@final
class LiteRtRewriteClient:
  def __init__(self, *, model_path: str) -> None:
    self.model_path: str = model_path
    self._engine: LiteRtEngine = self._open_engine(model_path)
    try:
      _ = self._engine.__enter__()
    except Exception as exc:
      raise RewriteClientError(f"failed to initialize LiteRT rewrite model: {model_path}") from exc
    self._closed: bool = False

  async def rewrite_text(
    self,
    *,
    instructions: str,
    transcript: str,
  ) -> RewriteResult:
    try:
      with self._engine.create_conversation(
        messages=[build_system_message(instructions)]
      ) as conversation:
        response = conversation.send_message(transcript)
    except Exception as exc:
      raise RewriteClientError("rewrite request failed") from exc

    rewrite_output = parse_structured_rewrite_output(extract_rewrite_output(response))
    return RewriteResult(
      text=rewrite_output.text,
      model=self.model_path,
      input_tokens=None,
      output_tokens=None,
      cost=None,
      corrections=rewrite_output.corrections,
    )

  async def close(self) -> None:
    if self._closed:
      return

    self._closed = True
    try:
      _ = self._engine.__exit__(None, None, None)
    except Exception as exc:
      raise RewriteClientError("failed to close rewrite client") from exc

  def _open_engine(self, model_path: str) -> LiteRtEngine:
    try:
      litert_module = _load_litert_module()
      return litert_module.Engine(model_path, backend=litert_module.Backend.CPU)
    except Exception as exc:
      raise RewriteClientError(f"failed to initialize LiteRT rewrite model: {model_path}") from exc


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
      raise RewriteClientError("rewrite request failed") from exc

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


def build_system_message(instructions: str) -> LiteRtMessage:
  return {
    "role": "system",
    "content": [{"type": "text", "text": instructions}],
  }


def extract_rewrite_output(response: LiteRtResponse) -> str:
  rewritten_parts: list[str] = []

  for item in response.get("content", []):
    if item.get("type") != "text":
      continue

    text = item.get("text")
    if text is None or text == "":
      continue

    rewritten_parts.append(text)

  rewritten_text = "".join(rewritten_parts).strip()
  if rewritten_text == "":
    raise RewriteClientError("rewrite model returned empty output")

  return rewritten_text


def parse_structured_rewrite_output(raw_output: str) -> StructuredRewriteOutput:
  try:
    decoded = cast(object, json.loads(raw_output))
    return StructuredRewriteOutput.model_validate(decoded)
  except (json.JSONDecodeError, ValidationError, ValueError) as exc:
    raise RewriteClientError("rewrite model returned invalid structured output") from exc


def _extract_total_cost(response: AgentRunResult[StructuredRewriteOutput]) -> Decimal | None:
  try:
    return response.response.cost().total_price
  except Exception:
    return None


def resolve_prompt_path(raw_path: str | Path) -> Path:
  configured_path = Path(raw_path).expanduser()
  return configured_path.resolve()
