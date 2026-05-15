"""Langfuse tracing tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import pytest

from active_listener.infra import langfuse as langfuse_module


class FakeObservation:
  pass


@dataclass
class FakeLangfuseClient:
  trace_seeds: list[str] = field(default_factory=list)
  start_calls: list[dict[str, object]] = field(default_factory=list)

  def create_trace_id(self, *, seed: str) -> str:
    self.trace_seeds.append(seed)
    return f"trace::{seed}"

  @contextmanager
  def start_as_current_observation(self, **kwargs: object) -> Iterator[FakeObservation]:
    self.start_calls.append(kwargs)
    yield FakeObservation()


def test_start_rewrite_observation_pins_trace_context(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  propagated_calls: list[dict[str, object]] = []
  client = FakeLangfuseClient()

  @contextmanager
  def fake_propagate_attributes(**kwargs: object) -> Iterator[None]:
    propagated_calls.append(kwargs)
    yield None

  monkeypatch.setattr(langfuse_module, "initialize_langfuse", lambda: True)
  monkeypatch.setattr(langfuse_module, "get_client", lambda: client)
  monkeypatch.setattr(langfuse_module, "propagate_attributes", fake_propagate_attributes)

  with langfuse_module.start_rewrite_observation(
    session_id="session-1",
    recording_id="recording-1",
    provider="litert",
    model="model-1",
    prompt_path="/tmp/rewrite/system.md",
    stream="stream-1",
    transcript="alpha",
  ) as observation:
    assert observation is not None

  assert client.trace_seeds == ["recording-1"]
  assert propagated_calls == [
    {
      "metadata": {
        "component": "active-listener",
        "provider": "litert",
        "stream": "stream-1",
      },
      "tags": ["active-listener", "rewrite", "litert"],
      "session_id": "session-1",
    }
  ]
  assert client.start_calls == [
    {
      "name": "active-listener-rewrite",
      "trace_context": {"trace_id": "trace::recording-1"},
      "input": "alpha",
      "metadata": {
        "component": "active-listener",
        "provider": "litert",
        "model": "model-1",
        "prompt_path": "/tmp/rewrite/system.md",
        "stream": "stream-1",
      },
    }
  ]
