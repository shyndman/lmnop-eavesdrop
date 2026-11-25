"""Prompt loading and rewrite client tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import cast, final

import pytest
from pydantic_ai.usage import RunUsage

import active_listener.infra.rewrite as rewrite_module
from active_listener.app.ports import RewriteResult
from active_listener.infra.rewrite import (
  LoadedRewritePromptFile,
  PydanticAiRewriteClient,
  RewriteClientError,
  RewritePromptError,
  StructuredRewriteOutput,
  load_active_listener_rewrite_prompt,
  load_rewrite_prompt,
  resolve_active_listener_override_prompt_path,
)


@dataclass(frozen=True)
class StubPriceCalculation:
  total_price: Decimal


@dataclass(frozen=True)
class StubPydanticAiModelResponse:
  usage: RunUsage = field(default_factory=RunUsage)
  cost_result: StubPriceCalculation | None = None

  def cost(self) -> StubPriceCalculation:
    assert self.cost_result is not None
    return self.cost_result


@dataclass(frozen=True)
class StubPydanticAiRunResult:
  output: StructuredRewriteOutput
  run_usage: RunUsage = field(default_factory=RunUsage)
  cost_result: StubPriceCalculation | None = None

  @property
  def response(self) -> StubPydanticAiModelResponse:
    return StubPydanticAiModelResponse(
      usage=self.run_usage,
      cost_result=self.cost_result,
    )

  def usage(self) -> RunUsage:
    return self.run_usage


@final
class StubAgent:
  init_calls: list[dict[str, object]] = []
  responses: list[object] = []
  run_calls: list[dict[str, object]] = []

  def __init__(
    self,
    *,
    output_type: object,
    instrument: object | None = None,
    name: object | None = None,
  ) -> None:
    self.output_type: object = output_type
    self.__class__.init_calls.append(
      {
        "output_type": output_type,
        "instrument": instrument,
        "name": name,
      }
    )

  async def run(
    self,
    user_prompt: str,
    *,
    instructions: str,
    model: str,
  ) -> StubPydanticAiRunResult:
    self.__class__.run_calls.append(
      {
        "user_prompt": user_prompt,
        "instructions": instructions,
        "model": model,
      }
    )
    response = self.__class__.responses.pop(0)
    if isinstance(response, Exception):
      raise response
    return cast(StubPydanticAiRunResult, response)


@pytest.fixture(autouse=True)
def reset_stubs() -> None:
  StubAgent.init_calls.clear()
  StubAgent.responses.clear()
  StubAgent.run_calls.clear()


def test_load_rewrite_prompt_reads_markdown_contents(tmp_path: Path) -> None:
  prompt_path = tmp_path / "rewrite_prompt.md"
  expected_contents = "# Rewrite this transcript\n\nKeep the commands intact.\n"
  _ = prompt_path.write_text(expected_contents, encoding="utf-8")

  prompt = load_rewrite_prompt(str(prompt_path))

  assert prompt.instructions == expected_contents


def test_load_rewrite_prompt_rejects_empty_contents(tmp_path: Path) -> None:
  prompt_path = tmp_path / "rewrite_prompt.md"
  _ = prompt_path.write_text("  \n\n", encoding="utf-8")

  with pytest.raises(RewritePromptError, match="rewrite prompt is empty"):
    _ = load_rewrite_prompt(str(prompt_path))


def test_load_active_listener_rewrite_prompt_prefers_user_override(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  config_dir = tmp_path / "config"

  def fake_override_prompt_path() -> Path:
    return config_dir / "system.md"

  override_path = config_dir / "system.md"
  _ = override_path.parent.mkdir(parents=True)
  _ = override_path.write_text("Override prompt\n", encoding="utf-8")

  configured_prompt_path = tmp_path / "configured.md"
  _ = configured_prompt_path.write_text("Fallback prompt\n", encoding="utf-8")
  monkeypatch.setattr(
    "active_listener.infra.rewrite.resolve_active_listener_override_prompt_path",
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
  _ = configured_prompt_path.write_text("Fallback prompt\n", encoding="utf-8")
  monkeypatch.setattr(
    "active_listener.infra.rewrite.resolve_active_listener_override_prompt_path",
    fake_override_prompt_path,
  )

  loaded_prompt = load_active_listener_rewrite_prompt(str(configured_prompt_path))

  assert loaded_prompt == LoadedRewritePromptFile(
    prompt_path=configured_prompt_path.resolve(),
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
    "active_listener.infra.rewrite.resolve_active_listener_override_prompt_path",
    fake_override_prompt_path,
  )

  _ = override_path.write_text("first prompt\n", encoding="utf-8")
  first_loaded_prompt = load_active_listener_rewrite_prompt(str(tmp_path / "configured.md"))

  _ = override_path.write_text("second prompt\n", encoding="utf-8")
  second_loaded_prompt = load_active_listener_rewrite_prompt(str(tmp_path / "configured.md"))

  assert first_loaded_prompt.prompt.instructions == "first prompt\n"
  assert second_loaded_prompt.prompt.instructions == "second prompt\n"
  assert first_loaded_prompt.prompt_path == second_loaded_prompt.prompt_path == override_path


def test_resolve_active_listener_override_prompt_path_uses_eavesdrop_xdg_config_dir(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  config_home = tmp_path / "xdg-config"
  monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

  assert resolve_active_listener_override_prompt_path() == (
    config_home / "eavesdrop" / "active-listener.rewrite.system.md"
  )


def test_resolve_active_listener_override_prompt_path_falls_back_to_home_config_dir(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("XDG_CONFIG_HOME", "")
  monkeypatch.setattr(Path, "home", lambda: tmp_path)

  assert resolve_active_listener_override_prompt_path() == (
    tmp_path / ".config" / "eavesdrop" / "active-listener.rewrite.system.md"
  )


def test_resolve_active_listener_override_prompt_path_uses_home_config_dir_when_unset(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
  monkeypatch.setattr(Path, "home", lambda: tmp_path)

  assert resolve_active_listener_override_prompt_path() == (
    tmp_path / ".config" / "eavesdrop" / "active-listener.rewrite.system.md"
  )


def test_structured_rewrite_output_schema_includes_attribute_docstrings() -> None:
  schema = StructuredRewriteOutput.model_json_schema()
  properties = cast(dict[str, dict[str, object]], schema["properties"])

  assert "rewritten" in str(properties["text"].get("description", ""))
  assert "recurring" in str(properties["corrections"].get("description", ""))


@pytest.mark.asyncio
async def test_pydantic_ai_rewrite_client_uses_model_and_instructions(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(rewrite_module, "Agent", StubAgent)
  StubAgent.responses.append(
    StubPydanticAiRunResult(
      output=StructuredRewriteOutput(text=" rewritten alpha ", corrections={"alfa": "alpha"})
    )
  )

  client = PydanticAiRewriteClient(model="openai:gpt-4.1-mini")
  rewritten = await client.rewrite_text(instructions="Prompt A", transcript="alpha")

  assert rewritten == RewriteResult(
    text="rewritten alpha",
    model="openai:gpt-4.1-mini",
    input_tokens=0,
    output_tokens=0,
    cost=None,
    corrections={"alfa": "alpha"},
  )
  assert StubAgent.init_calls == [
    {
      "output_type": StructuredRewriteOutput,
      "instrument": True,
      "name": "active-listener-rewrite",
    }
  ]
  assert StubAgent.run_calls == [
    {
      "user_prompt": "alpha",
      "instructions": "Prompt A",
      "model": "openai:gpt-4.1-mini",
    }
  ]


@pytest.mark.asyncio
async def test_pydantic_ai_rewrite_client_returns_usage_and_cost_metadata(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(rewrite_module, "Agent", StubAgent)
  usage = RunUsage(
    requests=1,
    input_tokens=12,
    output_tokens=4,
    details={"cache_discount": 1},
  )
  cost_result = StubPriceCalculation(total_price=Decimal("0.00012"))
  StubAgent.responses.append(
    StubPydanticAiRunResult(
      output=StructuredRewriteOutput(text=" rewritten alpha "),
      run_usage=usage,
      cost_result=cost_result,
    )
  )

  client = PydanticAiRewriteClient(model="openai:gpt-4.1-mini")

  rewritten = await client.rewrite_text(instructions="Prompt A", transcript="alpha")

  assert rewritten == RewriteResult(
    text="rewritten alpha",
    model="openai:gpt-4.1-mini",
    input_tokens=12,
    output_tokens=4,
    cost=Decimal("0.00012"),
    corrections={},
  )


@pytest.mark.asyncio
async def test_pydantic_ai_rewrite_client_rejects_empty_output(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(rewrite_module, "Agent", StubAgent)
  StubAgent.responses.append(
    StubPydanticAiRunResult(output=StructuredRewriteOutput.model_construct(text="", corrections={}))
  )

  client = PydanticAiRewriteClient(model="openai:gpt-4.1-mini")

  with pytest.raises(RewriteClientError, match="empty output"):
    _ = await client.rewrite_text(instructions="Prompt A", transcript="alpha")


@pytest.mark.asyncio
async def test_pydantic_ai_rewrite_client_propagates_model_errors(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(rewrite_module, "Agent", StubAgent)
  StubAgent.responses.append(RuntimeError("boom"))

  client = PydanticAiRewriteClient(model="openai:gpt-4.1-mini")

  with pytest.raises(RewriteClientError, match="rewrite request failed: boom"):
    _ = await client.rewrite_text(instructions="Prompt A", transcript="alpha")
