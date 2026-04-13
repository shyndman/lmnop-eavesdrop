"""Prompt loading and rewrite client tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from typing_extensions import override

from active_listener.rewrite import (
  LlmRewriteClient,
  LoadedRewritePromptFile,
  RewriteClientError,
  RewriteClientTimeoutError,
  RewritePromptError,
  load_active_listener_rewrite_prompt,
  load_rewrite_prompt,
  resolve_active_listener_override_prompt_path,
)


class StubRunResult:
  def __init__(self, output: str) -> None:
    self.output: str = output


class StubAgent:
  created: list[dict[str, object]] = []
  next_result: str = "rewritten text"
  next_error: Exception | None = None

  def __init__(self, model: object, *, instructions: str) -> None:
    self.model: object = model
    self.instructions: str = instructions
    self.__class__.created.append({"model": model, "instructions": instructions})

  async def run(self, transcript: str) -> StubRunResult:
    if self.__class__.next_error is not None:
      raise self.__class__.next_error
    if self.__class__.next_result == "":
      return StubRunResult(output="")
    return StubRunResult(output=f"{self.__class__.next_result}:{transcript}")


class StubOpenAIProvider:
  created: list[dict[str, str]] = []

  def __init__(self, *, base_url: str, api_key: str) -> None:
    self.base_url: str = base_url
    self.api_key: str = api_key
    self.__class__.created.append({"base_url": base_url, "api_key": api_key})


class StubOpenAIChatModel:
  created: list[dict[str, object]] = []

  def __init__(self, model_name: str, *, provider: object) -> None:
    self.model_name: str = model_name
    self.provider: object = provider
    self.__class__.created.append({"model_name": model_name, "provider": provider})


@pytest.fixture(autouse=True)
def reset_stubs() -> None:
  StubAgent.created.clear()
  StubAgent.next_result = "rewritten text"
  StubAgent.next_error = None
  StubOpenAIProvider.created.clear()
  StubOpenAIChatModel.created.clear()


def test_load_rewrite_prompt_parses_front_matter_and_metadata(tmp_path: Path) -> None:
  prompt_path = tmp_path / "rewrite_prompt.md"
  _ = prompt_path.write_text(
    """---
model: llama3
voice: concise
related_words:
  - alpha
  - bravo
---
Voice: {{ voice }}
Words: {{ related_words | join(', ') }}
""",
    encoding="utf-8",
  )

  prompt = load_rewrite_prompt(str(prompt_path))

  assert prompt.model_name == "llama3"
  assert prompt.metadata == {"voice": "concise", "related_words": ["alpha", "bravo"]}
  assert prompt.instructions == "Voice: concise\nWords: alpha, bravo"


def test_load_rewrite_prompt_requires_model(tmp_path: Path) -> None:
  prompt_path = tmp_path / "rewrite_prompt.md"
  _ = prompt_path.write_text("---\nvoice: concise\n---\nHello\n", encoding="utf-8")

  with pytest.raises(RewritePromptError, match="must define 'model'"):
    _ = load_rewrite_prompt(str(prompt_path))


def test_load_rewrite_prompt_raises_for_malformed_front_matter(tmp_path: Path) -> None:
  prompt_path = tmp_path / "rewrite_prompt.md"
  _ = prompt_path.write_text("---\nmodel: [unterminated\n---\nHello\n", encoding="utf-8")

  with pytest.raises(RewritePromptError, match="failed to load rewrite prompt"):
    _ = load_rewrite_prompt(str(prompt_path))


def test_load_rewrite_prompt_raises_for_missing_template_variable(tmp_path: Path) -> None:
  prompt_path = tmp_path / "rewrite_prompt.md"
  _ = prompt_path.write_text(
    "---\nmodel: llama3\n---\nHello {{ missing_value }}\n",
    encoding="utf-8",
  )

  with pytest.raises(RewritePromptError, match="failed to render rewrite prompt"):
    _ = load_rewrite_prompt(str(prompt_path))


def test_load_rewrite_prompt_renders_list_metadata(tmp_path: Path) -> None:
  prompt_path = tmp_path / "rewrite_prompt.md"
  _ = prompt_path.write_text(
    """---
model: llama3
related_words:
  - alpha
  - bravo
---
{{ related_words | join(' / ') }}
""",
    encoding="utf-8",
  )

  prompt = load_rewrite_prompt(str(prompt_path))

  assert prompt.instructions == "alpha / bravo"


def test_load_active_listener_rewrite_prompt_prefers_user_override(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  config_dir = tmp_path / "config"

  def fake_override_prompt_path() -> Path:
    return config_dir / "system.md"

  override_path = config_dir / "system.md"
  _ = override_path.parent.mkdir(parents=True)
  _ = override_path.write_text("---\nmodel: override\n---\nOverride prompt\n", encoding="utf-8")

  configured_prompt_path = tmp_path / "configured.md"
  _ = configured_prompt_path.write_text(
    "---\nmodel: fallback\n---\nFallback prompt\n", encoding="utf-8"
  )
  monkeypatch.setattr(
    "active_listener.rewrite.resolve_active_listener_override_prompt_path",
    fake_override_prompt_path,
  )

  loaded_prompt = load_active_listener_rewrite_prompt(str(configured_prompt_path))

  assert loaded_prompt == LoadedRewritePromptFile(
    prompt_path=override_path,
    prompt=load_rewrite_prompt(override_path),
  )


def test_load_active_listener_rewrite_prompt_falls_back_without_override(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  config_dir = tmp_path / "config"

  def fake_override_prompt_path() -> Path:
    return config_dir / "system.md"

  configured_prompt_path = tmp_path / "configured.md"
  _ = configured_prompt_path.write_text(
    "---\nmodel: fallback\n---\nFallback prompt\n", encoding="utf-8"
  )
  monkeypatch.setattr(
    "active_listener.rewrite.resolve_active_listener_override_prompt_path",
    fake_override_prompt_path,
  )

  loaded_prompt = load_active_listener_rewrite_prompt(str(configured_prompt_path))

  assert loaded_prompt == LoadedRewritePromptFile(
    prompt_path=configured_prompt_path,
    prompt=load_rewrite_prompt(configured_prompt_path),
  )


def test_load_active_listener_rewrite_prompt_reloads_override_each_time(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  config_dir = tmp_path / "config"

  def fake_override_prompt_path() -> Path:
    return config_dir / "system.md"

  override_path = config_dir / "system.md"
  _ = override_path.parent.mkdir(parents=True)
  monkeypatch.setattr(
    "active_listener.rewrite.resolve_active_listener_override_prompt_path",
    fake_override_prompt_path,
  )

  _ = override_path.write_text("---\nmodel: override\n---\nfirst prompt\n", encoding="utf-8")
  first_loaded_prompt = load_active_listener_rewrite_prompt(
    "packages/active-listener/src/active_listener/rewrite_prompt.md"
  )

  _ = override_path.write_text("---\nmodel: override\n---\nsecond prompt\n", encoding="utf-8")
  second_loaded_prompt = load_active_listener_rewrite_prompt(
    "packages/active-listener/src/active_listener/rewrite_prompt.md"
  )

  assert first_loaded_prompt.prompt.instructions == "first prompt"
  assert second_loaded_prompt.prompt.instructions == "second prompt"
  assert first_loaded_prompt.prompt_path == second_loaded_prompt.prompt_path == override_path


def test_resolve_active_listener_override_prompt_path_uses_eavesdrop_xdg_config_dir(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  config_home = tmp_path / "xdg-config"
  monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

  assert resolve_active_listener_override_prompt_path() == (
    config_home / "eavesdrop" / "active-listener.system.md"
  )


def test_resolve_active_listener_override_prompt_path_falls_back_to_home_config_dir(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("XDG_CONFIG_HOME", "")
  monkeypatch.setattr(Path, "home", lambda: tmp_path)

  assert resolve_active_listener_override_prompt_path() == (
    tmp_path / ".config" / "eavesdrop" / "active-listener.system.md"
  )


def test_resolve_active_listener_override_prompt_path_uses_home_config_dir_when_unset(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
  monkeypatch.setattr(Path, "home", lambda: tmp_path)

  assert resolve_active_listener_override_prompt_path() == (
    tmp_path / ".config" / "eavesdrop" / "active-listener.system.md"
  )


@pytest.mark.asyncio
async def test_rewrite_client_returns_plain_text_output(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr("active_listener.rewrite.OpenAIProvider", StubOpenAIProvider)
  monkeypatch.setattr("active_listener.rewrite.OpenAIChatModel", StubOpenAIChatModel)
  monkeypatch.setattr("active_listener.rewrite.Agent", StubAgent)

  client = LlmRewriteClient(base_url="http://localhost:11434/v1", timeout_s=30)
  rewritten = await client.rewrite_text(
    model_name="llama3",
    instructions="Rewrite this transcript.",
    transcript="alpha",
  )

  assert rewritten == "rewritten text:alpha"
  assert StubOpenAIProvider.created == [
    {"base_url": "http://localhost:11434/v1", "api_key": "ollama"}
  ]
  assert StubOpenAIChatModel.created[0]["model_name"] == "llama3"
  assert StubAgent.created[0]["instructions"] == "Rewrite this transcript."


@pytest.mark.asyncio
async def test_rewrite_client_propagates_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
  class HangingAgent(StubAgent):
    @override
    async def run(self, transcript: str) -> StubRunResult:
      _ = transcript
      await asyncio.sleep(1.1)
      return StubRunResult(output="too late")

  monkeypatch.setattr("active_listener.rewrite.OpenAIProvider", StubOpenAIProvider)
  monkeypatch.setattr("active_listener.rewrite.OpenAIChatModel", StubOpenAIChatModel)
  monkeypatch.setattr("active_listener.rewrite.Agent", HangingAgent)

  client = LlmRewriteClient(base_url="http://localhost:11434/v1", timeout_s=1)

  with pytest.raises(RewriteClientTimeoutError, match="timed out"):
    _ = await client.rewrite_text(
      model_name="llama3",
      instructions="Rewrite this transcript.",
      transcript="alpha",
    )


@pytest.mark.asyncio
async def test_rewrite_client_rejects_empty_output(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr("active_listener.rewrite.OpenAIProvider", StubOpenAIProvider)
  monkeypatch.setattr("active_listener.rewrite.OpenAIChatModel", StubOpenAIChatModel)
  monkeypatch.setattr("active_listener.rewrite.Agent", StubAgent)
  StubAgent.next_result = ""

  client = LlmRewriteClient(base_url="http://localhost:11434/v1", timeout_s=30)

  with pytest.raises(RewriteClientError, match="empty output"):
    _ = await client.rewrite_text(
      model_name="llama3",
      instructions="Rewrite this transcript.",
      transcript="alpha",
    )


@pytest.mark.asyncio
async def test_rewrite_client_propagates_model_errors(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr("active_listener.rewrite.OpenAIProvider", StubOpenAIProvider)
  monkeypatch.setattr("active_listener.rewrite.OpenAIChatModel", StubOpenAIChatModel)
  monkeypatch.setattr("active_listener.rewrite.Agent", StubAgent)
  StubAgent.next_error = RuntimeError("boom")

  client = LlmRewriteClient(base_url="http://localhost:11434/v1", timeout_s=30)

  with pytest.raises(RewriteClientError, match="rewrite request failed"):
    _ = await client.rewrite_text(
      model_name="llama3",
      instructions="Rewrite this transcript.",
      transcript="alpha",
    )
