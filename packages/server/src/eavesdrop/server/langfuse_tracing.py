from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from pydantic import JsonValue

if TYPE_CHECKING:
  from langfuse import Langfuse, LangfuseSpan

LANGFUSE_PUBLIC_KEY_ENV_VAR = "LANGFUSE_PUBLIC_KEY"
LANGFUSE_SECRET_KEY_ENV_VAR = "LANGFUSE_SECRET_KEY"
_langfuse_initialized = False


def langfuse_client() -> Langfuse:
  """Return the process-wide Langfuse client.

  :returns: The client managing observations and delivery.
  """
  from langfuse import get_client

  return get_client()


def has_langfuse_credentials() -> bool:
  """Check whether both required Langfuse credential variables are populated.

  :returns: Whether tracing can be initialized.
  """
  public_key = os.getenv(LANGFUSE_PUBLIC_KEY_ENV_VAR)
  secret_key = os.getenv(LANGFUSE_SECRET_KEY_ENV_VAR)
  return bool(public_key) and bool(secret_key)


def initialize_langfuse() -> bool:
  """Initialize tracing once when credentials are available.

  :returns: Whether tracing is available for this process.
  """
  global _langfuse_initialized

  if _langfuse_initialized:
    return True

  if not has_langfuse_credentials():
    return False

  _ = langfuse_client()
  _langfuse_initialized = True
  return True


def flush_langfuse() -> None:
  """Flush observations if tracing was initialized for this process."""
  if not _langfuse_initialized:
    return

  langfuse_client().flush()


def recording_trace_id(recording_id: str) -> str:
  """Derive the trace identity shared by a recording's observations.

  :param recording_id: Recording identifier supplied by the client.
  :returns: Deterministic Langfuse trace identifier.
  """
  return langfuse_client().create_trace_id(seed=recording_id)


@contextmanager
def start_recording_observation(
  *,
  stream: str,
  recording_id: str | None,
  name: str,
  input: JsonValue = None,
  metadata: dict[str, JsonValue] | None = None,
) -> Iterator[LangfuseSpan | None]:
  """Open a recording observation, disabled when identity or credentials are absent.

  :param stream: Connection session identifier.
  :param recording_id: Recording identifier, or None before recording start.
  :param name: Observation name.
  :param input: JSON-compatible recognition input metadata.
  :param metadata: Additional structured observation metadata.
  :yields: The active observation, or None when tracing is disabled.
  """
  if recording_id is None or not initialize_langfuse():
    yield None
    return
  from langfuse import propagate_attributes

  observation_metadata: dict[str, JsonValue] = {
    "component": "server",
    "stream": stream,
    "recording_id": recording_id,
  }
  if metadata is not None:
    observation_metadata.update(metadata)

  with propagate_attributes(
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
      yield observation


@contextmanager
def start_stage_observation(
  *,
  stream: str,
  recording_id: str | None,
  name: str,
  input: JsonValue = None,
  metadata: dict[str, JsonValue] | None = None,
) -> Iterator[LangfuseSpan | None]:
  """Open a recognition stage observation in its recording trace.

  :param stream: Connection session identifier.
  :param recording_id: Recording identifier, or None before recording start.
  :param name: Stage observation name.
  :param input: JSON-compatible stage input.
  :param metadata: Additional structured stage metadata.
  :yields: The active observation, or None when tracing is disabled.
  """
  if recording_id is None or not initialize_langfuse():
    yield None
    return
  from langfuse import propagate_attributes

  observation_metadata: dict[str, JsonValue] = {
    "component": "server",
    "stream": stream,
    "recording_id": recording_id,
  }
  if metadata is not None:
    observation_metadata.update(metadata)

  with propagate_attributes(
    session_id=stream,
    metadata={
      "component": "server",
      "stream": stream,
      "recording_id": recording_id,
    },
    tags=["server", "transcription", "stage"],
    trace_name="eavesdrop-recording",
  ):
    with langfuse_client().start_as_current_observation(
      as_type="span",
      name=name,
      input=input,
      metadata=observation_metadata,
    ) as observation:
      yield observation
