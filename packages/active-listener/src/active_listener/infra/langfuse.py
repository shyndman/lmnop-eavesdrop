from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol, Self, cast

from langfuse import get_client, propagate_attributes
from pydantic_ai import Agent

LANGFUSE_PUBLIC_KEY_ENV_VAR = "LANGFUSE_PUBLIC_KEY"
LANGFUSE_SECRET_KEY_ENV_VAR = "LANGFUSE_SECRET_KEY"
_langfuse_initialized = False


class RewriteObservation(Protocol):
  def update(
    self,
    *,
    input: object | None = None,
    output: object | None = None,
    metadata: object | None = None,
    level: str | None = None,
    status_message: str | None = None,
    usage_details: dict[str, int] | None = None,
    cost_details: dict[str, float] | None = None,
  ) -> Self: ...


def has_langfuse_credentials() -> bool:
  public_key = os.getenv(LANGFUSE_PUBLIC_KEY_ENV_VAR)
  secret_key = os.getenv(LANGFUSE_SECRET_KEY_ENV_VAR)
  return bool(public_key) and bool(secret_key)


def initialize_langfuse() -> bool:
  global _langfuse_initialized

  if _langfuse_initialized:
    return True

  if not has_langfuse_credentials():
    return False

  _ = get_client()
  Agent.instrument_all()
  _langfuse_initialized = True
  return True


def flush_langfuse() -> None:
  if not _langfuse_initialized:
    return

  get_client().flush()


@contextmanager
def start_rewrite_observation(
  *,
  provider: str,
  model: str,
  prompt_path: str,
  stream: str,
  transcript: str,
) -> Iterator[RewriteObservation | None]:
  if not initialize_langfuse():
    yield None
    return

  with propagate_attributes(
    metadata={
      "component": "active-listener",
      "provider": provider,
      "stream": stream,
    },
    tags=["active-listener", "rewrite", provider],
    trace_name="active-listener-rewrite",
  ):
    with get_client().start_as_current_observation(
      name="active-listener-rewrite",
      input=transcript,
      metadata={
        "component": "active-listener",
        "provider": provider,
        "model": model,
        "prompt_path": prompt_path,
        "stream": stream,
      },
    ) as observation:
      yield cast(RewriteObservation, cast(object, observation))
