from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from importlib import import_module
from typing import Protocol, Self, cast

LANGFUSE_PUBLIC_KEY_ENV_VAR = "LANGFUSE_PUBLIC_KEY"
LANGFUSE_SECRET_KEY_ENV_VAR = "LANGFUSE_SECRET_KEY"
_langfuse_initialized = False


class ServerObservation(Protocol):
  def update(
    self,
    *,
    input: object | None = None,
    output: object | None = None,
    metadata: object | None = None,
    level: str | None = None,
    status_message: str | None = None,
  ) -> Self: ...


class LangfuseClient(Protocol):
  def flush(self) -> None: ...

  def create_trace_id(self, *, seed: str) -> str: ...

  def start_as_current_observation(
    self,
    *,
    as_type: str,
    name: str,
    trace_context: dict[str, str] | None = None,
    input: object | None = None,
    metadata: object | None = None,
  ) -> AbstractContextManager[object]: ...


class LangfuseModule(Protocol):
  def get_client(self) -> LangfuseClient: ...

  def propagate_attributes(
    self,
    *,
    session_id: str,
    metadata: dict[str, object] | None = None,
    tags: list[str] | None = None,
    trace_name: str | None = None,
  ) -> AbstractContextManager[object]: ...


def langfuse_module() -> LangfuseModule:
  return cast(LangfuseModule, cast(object, import_module("langfuse")))


def langfuse_client() -> LangfuseClient:
  return langfuse_module().get_client()


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

  _ = langfuse_client()
  _langfuse_initialized = True
  return True


def flush_langfuse() -> None:
  if not _langfuse_initialized:
    return

  langfuse_client().flush()


def recording_trace_id(recording_id: str) -> str:
  return langfuse_client().create_trace_id(seed=recording_id)


@contextmanager
def start_recording_observation(
  *,
  stream: str,
  recording_id: str | None,
  name: str,
  input: object | None = None,
  metadata: dict[str, object] | None = None,
) -> Iterator[ServerObservation | None]:
  if recording_id is None or not initialize_langfuse():
    yield None
    return

  observation_metadata: dict[str, object] = {
    "component": "server",
    "stream": stream,
    "recording_id": recording_id,
  }
  if metadata is not None:
    observation_metadata.update(metadata)

  with langfuse_module().propagate_attributes(
    session_id=stream,
    metadata={
      "component": "server",
      "stream": stream,
      "recording_id": recording_id,
    },
    tags=["server", "transcription", "recording"],
    trace_name="eavesdrop-recording",
  ):
    with langfuse_client().start_as_current_observation(
      as_type="span",
      name=name,
      trace_context={"trace_id": recording_trace_id(recording_id)},
      input=input,
      metadata=observation_metadata,
    ) as observation:
      yield cast(ServerObservation, observation)


@contextmanager
def start_stage_observation(
  *,
  stream: str,
  recording_id: str | None,
  name: str,
  input: object | None = None,
  metadata: dict[str, object] | None = None,
) -> Iterator[ServerObservation | None]:
  if recording_id is None or not initialize_langfuse():
    yield None
    return

  observation_metadata: dict[str, object] = {
    "component": "server",
    "stream": stream,
    "recording_id": recording_id,
  }
  if metadata is not None:
    observation_metadata.update(metadata)

  with langfuse_client().start_as_current_observation(
    as_type="span",
    name=name,
    input=input,
    metadata=observation_metadata,
  ) as observation:
    yield cast(ServerObservation, observation)
