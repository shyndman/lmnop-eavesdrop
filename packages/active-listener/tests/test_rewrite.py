"""Prompt loading and rewrite client tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import cast, final

import pytest
from pydantic_ai.usage import RunUsage

import active_listener.infra.rewrite as rewrite_module
from active_listener.infra.rewrite import (
  LiteRtRewriteClient,
  LoadedRewritePromptFile,
  PydanticAiRewriteClient,
  RewriteClientError,
  RewritePromptError,
  load_active_listener_rewrite_prompt,
  load_rewrite_prompt,
  resolve_active_listener_override_prompt_path,
)


@final
class StubConversation:
  def __init__(self, *, messages: list[rewrite_module.LiteRtMessage], response: object) -> None:
    self.messages: list[rewrite_module.LiteRtMessage] = messages
    self.response: object = response
    self.sent_prompts: list[str] = []
    self.entered: bool = False
    self.exited: bool = False

  def __enter__(self) -> StubConversation:
    self.entered = True
    return self

  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc: BaseException | None,
    traceback: object | None,
  ) -> bool | None:
    _ = exc_type
    _ = exc
    _ = traceback
    self.exited = True
    return None

  def send_message(self, prompt: str) -> rewrite_module.LiteRtResponse:
    self.sent_prompts.append(prompt)
    if isinstance(self.response, Exception):
      raise self.response
    return cast(rewrite_module.LiteRtResponse, self.response)


@final
class StubBackend:
  CPU: str = "cpu"


@final
class StubEngine:
  created: list[StubEngine] = []
  conversations: list[StubConversation] = []
  responses: list[object] = []
  init_error: Exception | None = None
  close_calls: int = 0

  def __init__(self, model_path: str, *, backend: object) -> None:
    if self.__class__.init_error is not None:
      raise self.__class__.init_error

    self.model_path: str = model_path
    self.backend: object = backend
    self.__class__.created.append(self)

  def __enter__(self) -> StubEngine:
    return self

  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc: BaseException | None,
    traceback: object | None,
  ) -> bool | None:
    _ = exc_type
    _ = exc
    _ = traceback
    self.__class__.close_calls += 1
    return None

  def create_conversation(
    self,
    *,
    messages: list[rewrite_module.LiteRtMessage],
  ) -> StubConversation:
    response = self.__class__.responses.pop(0)
    conversation = StubConversation(messages=messages, response=response)
    self.__class__.conversations.append(conversation)
    return conversation


@dataclass(frozen=True)
class StubPydanticAiModelResponse:
  usage: RunUsage = field(default_factory=RunUsage)
  cost_result: object | None = None

  def cost(self) -> object:
    return self.cost_result


@dataclass(frozen=True)
class StubPydanticAiRunResult:
  output: str
  run_usage: RunUsage = field(default_factory=RunUsage)
  cost_result: object | None = None

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
  responses: list[object] = []
  run_calls: list[dict[str, object]] = []

  def __init__(self, *, output_type: object) -> None:
    self.output_type: object = output_type

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
  StubEngine.created.clear()
  StubEngine.conversations.clear()
  StubEngine.responses.clear()
  StubEngine.init_error = None
  StubEngine.close_calls = 0
  StubAgent.responses.clear()
  StubAgent.run_calls.clear()


def test_dependency_import_resolves() -> None:
  assert "litert_lm" in rewrite_module.__dict__


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
async def test_rewrite_client_uses_fresh_conversation_per_request(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    rewrite_module,
    "litert_lm",
    SimpleNamespace(Engine=StubEngine, Backend=StubBackend),
  )
  StubEngine.responses.extend(
    [
      {"content": [{"type": "text", "text": " rewritten alpha "}]},
      {"content": [{"type": "text", "text": "rewritten beta"}]},
    ]
  )

  client = LiteRtRewriteClient(model_path="/tmp/rewrite/model.litertlm")
  first = await client.rewrite_text(instructions="Prompt A", transcript="alpha")
  second = await client.rewrite_text(instructions="Prompt B", transcript="beta")
  await client.close()

  assert first == "rewritten alpha"
  assert second == "rewritten beta"
  assert len(StubEngine.created) == 1
  assert StubEngine.created[0].model_path == "/tmp/rewrite/model.litertlm"
  assert len(StubEngine.conversations) == 2
  assert StubEngine.conversations[0].messages == [
    {"role": "system", "content": [{"type": "text", "text": "Prompt A"}]}
  ]
  assert StubEngine.conversations[1].messages == [
    {"role": "system", "content": [{"type": "text", "text": "Prompt B"}]}
  ]
  assert StubEngine.conversations[0].sent_prompts == ["alpha"]
  assert StubEngine.conversations[1].sent_prompts == ["beta"]
  assert StubEngine.close_calls == 1


@pytest.mark.asyncio
async def test_rewrite_client_extracts_documented_response_shape(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    rewrite_module,
    "litert_lm",
    SimpleNamespace(Engine=StubEngine, Backend=StubBackend),
  )
  StubEngine.responses.append(
    {
      "content": [
        {"type": "text", "text": "Hello"},
        {"type": "image"},
        {"type": "text", "text": ", world"},
      ]
    }
  )

  client = LiteRtRewriteClient(model_path="/tmp/rewrite/model.litertlm")
  rewritten = await client.rewrite_text(instructions="Rewrite this transcript.", transcript="alpha")

  assert rewritten == "Hello, world"


@pytest.mark.asyncio
async def test_rewrite_client_rejects_empty_output(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(
    rewrite_module,
    "litert_lm",
    SimpleNamespace(Engine=StubEngine, Backend=StubBackend),
  )
  StubEngine.responses.append({"content": [{"type": "text", "text": "   "}]})

  client = LiteRtRewriteClient(model_path="/tmp/rewrite/model.litertlm")

  with pytest.raises(RewriteClientError, match="empty output"):
    _ = await client.rewrite_text(instructions="Rewrite this transcript.", transcript="alpha")


@pytest.mark.asyncio
async def test_rewrite_client_propagates_model_errors(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(
    rewrite_module,
    "litert_lm",
    SimpleNamespace(Engine=StubEngine, Backend=StubBackend),
  )
  StubEngine.responses.append(RuntimeError("boom"))

  client = LiteRtRewriteClient(model_path="/tmp/rewrite/model.litertlm")

  with pytest.raises(RewriteClientError, match="rewrite request failed"):
    _ = await client.rewrite_text(instructions="Rewrite this transcript.", transcript="alpha")


def test_rewrite_client_fails_fast_when_engine_initialization_fails(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    rewrite_module,
    "litert_lm",
    SimpleNamespace(Engine=StubEngine, Backend=StubBackend),
  )
  StubEngine.init_error = RuntimeError("bad bundle")

  with pytest.raises(
    RewriteClientError,
    match="failed to initialize LiteRT rewrite model: /tmp/rewrite/model.litertlm",
  ):
    _ = LiteRtRewriteClient(model_path="/tmp/rewrite/model.litertlm")


@pytest.mark.asyncio
async def test_pydantic_ai_rewrite_client_uses_model_and_instructions(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(rewrite_module, "Agent", StubAgent)
  StubAgent.responses.append(StubPydanticAiRunResult(output=" rewritten alpha "))

  client = PydanticAiRewriteClient(model="openai:gpt-4.1-mini")
  rewritten = await client.rewrite_text(instructions="Prompt A", transcript="alpha")

  assert rewritten == "rewritten alpha"
  assert StubAgent.run_calls == [
    {
      "user_prompt": "alpha",
      "instructions": "Prompt A",
      "model": "openai:gpt-4.1-mini",
    }
  ]


@pytest.mark.asyncio
async def test_pydantic_ai_rewrite_client_prints_openrouter_usage(
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> None:
  monkeypatch.setattr(rewrite_module, "Agent", StubAgent)
  usage = RunUsage(
    requests=1,
    input_tokens=12,
    output_tokens=4,
    details={"cache_discount": 1},
  )
  cost_result = {"total_price": "0.00012"}
  StubAgent.responses.append(
    StubPydanticAiRunResult(
      output=" rewritten alpha ",
      run_usage=usage,
      cost_result=cost_result,
    )
  )

  client = PydanticAiRewriteClient(model="openai:gpt-4.1-mini")

  _ = await client.rewrite_text(instructions="Prompt A", transcript="alpha")

  assert capsys.readouterr().out == (
    f"OpenRouter usage: {asdict(usage)}\n"
    f"OpenRouter cost: {cost_result}\n"
  )


@pytest.mark.asyncio
async def test_pydantic_ai_rewrite_client_rejects_empty_output(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(rewrite_module, "Agent", StubAgent)
  StubAgent.responses.append(StubPydanticAiRunResult(output="   "))

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

  with pytest.raises(RewriteClientError, match="rewrite request failed"):
    _ = await client.rewrite_text(instructions="Prompt A", transcript="alpha")
