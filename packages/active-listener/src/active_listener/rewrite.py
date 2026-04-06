from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import frontmatter
from jinja2 import Environment, StrictUndefined
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class LoadedRewritePrompt:
  model_name: str
  metadata: dict[str, object]
  instructions: str


class RewritePromptError(RuntimeError):
  pass


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
    provider = OpenAIProvider(base_url=self.base_url)
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


def load_rewrite_prompt(prompt_path: str) -> LoadedRewritePrompt:
  resolved_prompt_path = resolve_repo_path(prompt_path)

  try:
    post = frontmatter.load(str(resolved_prompt_path), encoding="utf-8")
  except Exception as exc:
    raise RewritePromptError(f"failed to load rewrite prompt: {resolved_prompt_path}") from exc

  metadata = dict(post.metadata)
  model_name = metadata.pop("model", None)
  if not isinstance(model_name, str) or model_name == "":
    raise RewritePromptError("rewrite prompt front matter must define 'model'")

  template_environment = Environment(undefined=StrictUndefined, autoescape=False)
  template = template_environment.from_string(post.content)

  try:
    instructions = str(template.render(**metadata))
  except Exception as exc:
    raise RewritePromptError("failed to render rewrite prompt") from exc

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
