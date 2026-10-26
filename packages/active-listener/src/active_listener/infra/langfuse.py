from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import NotRequired, TypedDict

from langfuse import LangfuseSpan, get_client, propagate_attributes
from langfuse.api.media.types.media_content_type import MediaContentType
from langfuse.media import LangfuseMedia
from langfuse.types import SpanLevel
from pydantic import JsonValue
from pydantic_ai import Agent
from structlog.stdlib import BoundLogger

LANGFUSE_PUBLIC_KEY_ENV_VAR = "LANGFUSE_PUBLIC_KEY"
LANGFUSE_SECRET_KEY_ENV_VAR = "LANGFUSE_SECRET_KEY"
_langfuse_initialized = False


class RecordingInput(TypedDict):
  """Shaped and unshaped transcript views attached to recording observations."""

  raw_transcript: str
  rewrite_input: str


class RecordingOutput(TypedDict):
  """Emitted text and optional archived audio attached to an observation."""

  emitted_text: str
  captured_audio: NotRequired[LangfuseMedia]


def record_recording_event(
  *,
  parent_observation: LangfuseSpan | None,
  name: str,
  metadata: dict[str, JsonValue] | None = None,
) -> None:
  if parent_observation is None:
    return

  with parent_observation.start_as_current_observation(
    as_type="span",
    name=name,
    metadata=metadata,
  ):
    return


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


def recording_trace_id(recording_id: str) -> str:
  return get_client().create_trace_id(seed=recording_id)


def build_langfuse_audio_attachment(
  *,
  audio_bytes: bytes | None,
) -> LangfuseMedia | None:
  if audio_bytes is None:
    return None
  if not initialize_langfuse():
    return None
  return LangfuseMedia(
    content_bytes=audio_bytes,
    content_type=MediaContentType.AUDIO_MP4,
  )


def record_session_event(
  *,
  session_id: str,
  name: str,
  metadata: dict[str, JsonValue] | None = None,
) -> None:
  if not initialize_langfuse():
    return

  with propagate_attributes(
    session_id=session_id,
    tags=["active-listener", "session"],
    trace_name="active-listener-session",
  ):
    with get_client().start_as_current_observation(
      as_type="span",
      name=name,
      metadata=metadata,
    ):
      return


def start_recording_observation(
  *,
  session_id: str,
  recording_id: str,
) -> LangfuseSpan | None:
  if not initialize_langfuse():
    return None

  with propagate_attributes(
    session_id=session_id,
    metadata={
      "component": "active-listener",
      "session_id": session_id,
      "recording_id": recording_id,
    },
    tags=["active-listener", "recording"],
    trace_name="eavesdrop-recording",
  ):
    observation = get_client().start_observation(
      as_type="span",
      name="active-listener-recording",
      trace_context={"trace_id": recording_trace_id(recording_id)},
      metadata={
        "component": "active-listener",
        "session_id": session_id,
        "recording_id": recording_id,
      },
    )

  return observation


def end_recording_observation(
  observation: LangfuseSpan | None,
  *,
  logger: BoundLogger,
  metadata: dict[str, JsonValue] | None = None,
  level: SpanLevel | None = None,
  status_message: str | None = None,
) -> None:
  if observation is None:
    return

  update_recording_observation(
    observation,
    logger=logger,
    metadata=metadata,
    level=level,
    status_message=status_message,
  )

  try:
    _ = observation.end()
  except Exception:
    logger.exception("langfuse recording observation close failed")


def update_recording_observation(
  observation: LangfuseSpan | None,
  *,
  logger: BoundLogger,
  metadata: dict[str, JsonValue] | None = None,
  level: SpanLevel | None = None,
  status_message: str | None = None,
) -> None:
  if observation is None:
    return

  try:
    if metadata is not None or level is not None or status_message is not None:
      _ = observation.update(
        metadata=metadata,
        level=level,
        status_message=status_message,
      )
  except Exception:
    logger.exception("langfuse recording observation update failed")


@contextmanager
def start_rewrite_observation(
  *,
  session_id: str,
  recording_id: str,
  provider: str,
  model: str,
  prompt_path: str,
  transcript: str,
  parent_observation: LangfuseSpan | None = None,
) -> Iterator[LangfuseSpan | None]:
  if not initialize_langfuse():
    yield None
    return

  with propagate_attributes(
    metadata={
      "component": "active-listener",
      "provider": provider,
      "session_id": session_id,
    },
    tags=["active-listener", "rewrite", provider],
    session_id=session_id,
  ):
    observation_metadata: dict[str, JsonValue] = {
      "component": "active-listener",
      "provider": provider,
      "model": model,
      "prompt_path": prompt_path,
      "session_id": session_id,
    }

    if parent_observation is None:
      with get_client().start_as_current_observation(
        as_type="span",
        name="active-listener-rewrite",
        trace_context={"trace_id": recording_trace_id(recording_id)},
        input=transcript,
        metadata=observation_metadata,
      ) as observation:
        yield observation
      return

    with parent_observation.start_as_current_observation(
      as_type="span",
      name="active-listener-rewrite",
      input=transcript,
      metadata=observation_metadata,
    ) as observation:
      yield observation


class RecordingTrace:
  """Own the foreground observation until it is handed to finalization.

  :param logger: Sink for tracing failures, which never interrupt recording.
  """

  def __init__(self, logger: BoundLogger) -> None:
    self.logger: BoundLogger = logger
    self.session_id: str | None = None
    self._observation: LangfuseSpan | None = None

  def start(self, *, recording_id: str) -> None:
    """Start the foreground recording observation for the current server session.

    :param recording_id: Identifier of the recording beginning now.
    :return: None; failures are logged without interrupting capture.
    """
    if self.session_id is None:
      return
    try:
      self._observation = start_recording_observation(
        session_id=self.session_id, recording_id=recording_id
      )
    except Exception:
      self.logger.exception(
        "langfuse recording observation start failed",
        session_id=self.session_id,
        recording_id=recording_id,
      )

  def transfer_and_clear_observation(self) -> LangfuseSpan | None:
    """Transfer the foreground observation and clear the owned reference.

    :return: Previously owned observation, or None after transfer/no tracing.
    """
    observation = self._observation
    self._observation = None
    return observation

  def session_event(
    self,
    *,
    name: str,
    metadata: dict[str, JsonValue],
    session_id: str | None = None,
  ) -> None:
    """Publish connection metadata without affecting recording policy."""
    resolved_session_id = session_id or self.session_id
    if resolved_session_id is None:
      return
    try:
      record_session_event(session_id=resolved_session_id, name=name, metadata=metadata)
    except Exception:
      self.logger.exception("langfuse session event failed", name=name)

  def recording_event(
    self,
    *,
    name: str,
    metadata: dict[str, JsonValue],
    recording_observation: LangfuseSpan | None = None,
  ) -> None:
    """Publish against the foreground or explicitly transferred observation."""
    observation = recording_observation if recording_observation is not None else self._observation
    try:
      record_recording_event(parent_observation=observation, name=name, metadata=metadata)
    except Exception:
      self.logger.exception("langfuse recording event failed", name=name)
