from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Literal, NotRequired, Protocol, TypedDict, cast, final

import litert_lm
from pydantic_ai import Agent
from pydantic_ai.run import AgentRunResult

USER_CONFIG_ENV_VAR = "XDG_CONFIG_HOME"
DEFAULT_USER_CONFIG_DIRNAME = ".config"
EAVESDROP_CONFIG_DIRNAME = "eavesdrop"
ACTIVE_LISTENER_PROMPT_FILENAME = "active-listener.system.md"


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
  ) -> str:
    try:
      with self._engine.create_conversation(
        messages=[build_system_message(instructions)]
      ) as conversation:
        response = conversation.send_message(transcript)
    except Exception as exc:
      raise RewriteClientError("rewrite request failed") from exc

    return extract_rewrite_output(response)

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
      return cast(LiteRtEngine, litert_lm.Engine(model_path, backend=litert_lm.Backend.CPU))
    except Exception as exc:
      raise RewriteClientError(f"failed to initialize LiteRT rewrite model: {model_path}") from exc


@final
class PydanticAiRewriteClient:
  def __init__(self, *, model: str) -> None:
    self.model: str = model
    self._agent: Agent[None, str] = Agent(output_type=str)

  async def rewrite_text(
    self,
    *,
    instructions: str,
    transcript: str,
  ) -> str:
    try:
      response: AgentRunResult[str] = await self._agent.run(
        transcript,
        instructions=instructions,
        model=self.model,
      )
    except Exception as exc:
      raise RewriteClientError("rewrite request failed") from exc

    model_response = response.response
    print("OpenRouter usage:", asdict(model_response.usage))
    print("OpenRouter cost:", model_response.cost())

    rewritten_text = response.output.strip()
    if rewritten_text == "":
      raise RewriteClientError("rewrite model returned empty output")

    return rewritten_text

  async def close(self) -> None:
    return None


@dataclass(frozen=True)
class DisabledRewriteClient:
  async def rewrite_text(
    self,
    *,
    instructions: str,
    transcript: str,
  ) -> str:
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


def resolve_user_config_dir() -> Path:
  configured_path = os.environ.get(USER_CONFIG_ENV_VAR)
  if configured_path is not None and configured_path != "":
    return Path(configured_path)

  return Path.home() / DEFAULT_USER_CONFIG_DIRNAME


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


def resolve_prompt_path(raw_path: str | Path) -> Path:
  configured_path = Path(raw_path).expanduser()
  return configured_path.resolve()
