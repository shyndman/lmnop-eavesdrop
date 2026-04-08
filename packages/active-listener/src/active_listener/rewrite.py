from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

import frontmatter
from jinja2 import Environment, StrictUndefined
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

REPO_ROOT = Path(__file__).resolve().parents[4]
USER_CONFIG_ENV_VAR = "XDG_CONFIG_HOME"
DEFAULT_USER_CONFIG_DIRNAME = ".config"
ACTIVE_LISTENER_CONFIG_DIRNAME = "active-listener"
ACTIVE_LISTENER_PROMPT_FILENAME = "system.md"


@dataclass(frozen=True)
class LoadedRewritePrompt:
  model_name: str
  metadata: dict[str, object]
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


@dataclass(frozen=True)
class LlmRewriteClient:
  base_url: str
  timeout_s: int

  async def rewrite_text(
    self,
    *,
    model_name: str,
    instructions: str,
    transcript: str,
  ) -> str:
    provider = OpenAIProvider(base_url=self.base_url, api_key="ollama")
    model = OpenAIChatModel(model_name, provider=provider)
    agent = Agent(model, instructions=instructions)

    try:
      result = await asyncio.wait_for(agent.run(transcript), timeout=self.timeout_s)
    except asyncio.TimeoutError as exc:
      raise RewriteClientTimeoutError("rewrite request timed out") from exc
    except Exception as exc:
      raise RewriteClientError("rewrite request failed") from exc

    rewritten_text = result.output.strip()
    if rewritten_text == "":
      raise RewriteClientError("rewrite model returned empty output")

    return rewritten_text


def load_active_listener_rewrite_prompt(configured_prompt_path: str) -> LoadedRewritePromptFile:
  prompt_path = resolve_active_listener_prompt_path(configured_prompt_path)
  prompt = load_rewrite_prompt(prompt_path)
  return LoadedRewritePromptFile(prompt_path=prompt_path, prompt=prompt)


def resolve_active_listener_prompt_path(configured_prompt_path: str) -> Path:
  override_prompt_path = resolve_active_listener_override_prompt_path()
  if override_prompt_path.exists():
    return override_prompt_path

  return resolve_repo_path(configured_prompt_path)


def resolve_active_listener_override_prompt_path() -> Path:
  return (
    resolve_user_config_dir()
    / ACTIVE_LISTENER_CONFIG_DIRNAME
    / ACTIVE_LISTENER_PROMPT_FILENAME
  )


def resolve_user_config_dir() -> Path:
  configured_path = os.environ.get(USER_CONFIG_ENV_VAR)
  if configured_path is not None and configured_path != "":
    return Path(configured_path)

  return Path.home() / DEFAULT_USER_CONFIG_DIRNAME


def load_rewrite_prompt(prompt_path: str | Path) -> LoadedRewritePrompt:
  resolved_prompt_path = resolve_prompt_path(prompt_path)

  try:
    post = frontmatter.load(str(resolved_prompt_path), encoding="utf-8")
  except Exception as exc:
    raise RewritePromptError(
      f"failed to load rewrite prompt: {resolved_prompt_path}",
      prompt_path=resolved_prompt_path,
    ) from exc

  metadata = dict(post.metadata)
  model_name = metadata.pop("model", None)
  if not isinstance(model_name, str) or model_name == "":
    raise RewritePromptError(
      "rewrite prompt front matter must define 'model'",
      prompt_path=resolved_prompt_path,
    )

  template_environment = Environment(undefined=StrictUndefined, autoescape=False)
  template = template_environment.from_string(post.content)

  try:
    instructions = str(template.render(**metadata))
  except Exception as exc:
    raise RewritePromptError(
      "failed to render rewrite prompt",
      prompt_path=resolved_prompt_path,
    ) from exc

  return LoadedRewritePrompt(
    model_name=model_name,
    metadata=metadata,
    instructions=instructions,
  )


def resolve_repo_path(raw_path: str) -> Path:
  path = Path(raw_path)
  if path.is_absolute():
    return path
  return REPO_ROOT / path


def resolve_prompt_path(raw_path: str | Path) -> Path:
  if isinstance(raw_path, Path):
    return raw_path

  return resolve_repo_path(raw_path)
