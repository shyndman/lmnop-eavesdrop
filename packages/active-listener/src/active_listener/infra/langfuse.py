from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Protocol, cast

from langfuse import get_client, propagate_attributes
from langfuse.api.media.types.media_content_type import MediaContentType
from langfuse.media import LangfuseMedia
from pydantic_ai import Agent
from structlog.stdlib import BoundLogger

from active_listener.app.ports import CapturedRecordingAudio
from active_listener.infra.audio import encode_mp3

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
  ) -> object: ...


class RecordingObservation(Protocol):
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
  ) -> object: ...

  def end(self, *, end_time: int | None = None) -> object: ...

  def start_as_current_observation(
    self,
    *,
    name: str,
    input: object | None = None,
    output: object | None = None,
    metadata: object | None = None,
    level: str | None = None,
    status_message: str | None = None,
    usage_details: dict[str, int] | None = None,
    cost_details: dict[str, float] | None = None,
  ) -> AbstractContextManager[object]: ...


def _recording_observation(observation: object) -> RecordingObservation:
  return cast(RecordingObservation, observation)


def _rewrite_observation(observation: object) -> RewriteObservation:
  return cast(RewriteObservation, observation)


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
  captured_audio: CapturedRecordingAudio,
  ffmpeg_path: str | None,
  logger: BoundLogger,
) -> LangfuseMedia | None:
  if ffmpeg_path is None or captured_audio.pcm_f32le == b"":
    return None

  if not initialize_langfuse():
    return None

  try:
    audio_mp3 = encode_mp3(
      ffmpeg_path,
      captured_audio.pcm_f32le,
      sample_rate_hz=captured_audio.sample_rate_hz,
      channels=captured_audio.channels,
    )
  except Exception:
    logger.exception(
      "langfuse audio attachment failed",
      pcm_bytes=len(captured_audio.pcm_f32le),
      sample_rate_hz=captured_audio.sample_rate_hz,
      channels=captured_audio.channels,
    )
    return None

  return LangfuseMedia(content_bytes=audio_mp3, content_type=MediaContentType.AUDIO_MPEG)


def record_session_event(
  *,
  session_id: str,
  name: str,
  metadata: dict[str, object] | None = None,
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
  stream: str,
  recording_id: str,
) -> RecordingObservation | None:
  if not initialize_langfuse():
    return None

  with propagate_attributes(
    session_id=stream,
    metadata={
      "component": "active-listener",
      "stream": stream,
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
        "stream": stream,
        "recording_id": recording_id,
      },
    )

  return _recording_observation(observation)


def end_recording_observation(
  observation: RecordingObservation | None,
  *,
  logger: BoundLogger,
  metadata: object | None = None,
  level: str | None = None,
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
  observation: RecordingObservation | None,
  *,
  logger: BoundLogger,
  metadata: object | None = None,
  level: str | None = None,
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
  stream: str,
  transcript: str,
  parent_observation: RecordingObservation | None = None,
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
    session_id=session_id,
  ):
    observation_metadata = {
      "component": "active-listener",
      "provider": provider,
      "model": model,
      "prompt_path": prompt_path,
      "stream": stream,
    }

    if parent_observation is None:
      with get_client().start_as_current_observation(
        name="active-listener-rewrite",
        trace_context={"trace_id": recording_trace_id(recording_id)},
        input=transcript,
        metadata=observation_metadata,
      ) as observation:
        yield _rewrite_observation(observation)
      return

    with parent_observation.start_as_current_observation(
      name="active-listener-rewrite",
      input=transcript,
      metadata=observation_metadata,
    ) as observation:
      yield _rewrite_observation(observation)
