from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol, Self, cast

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
  ) -> Self: ...


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


@contextmanager
def start_recording_observation(
  *,
  stream: str,
  recording_id: str,
  raw_text: str,
  rewrite_input: str | None,
  audio_attachment: LangfuseMedia | None,
  duration_seconds: float | None,
  word_count: int,
) -> Iterator[RecordingObservation | None]:
  if not initialize_langfuse():
    yield None
    return

  observation_input: dict[str, object] = {
    "raw_transcript": raw_text,
  }
  if rewrite_input is not None:
    observation_input["rewrite_input"] = rewrite_input
  if audio_attachment is not None:
    observation_input["captured_audio_mp3"] = audio_attachment

  observation_metadata: dict[str, object] = {
    "component": "active-listener",
    "stream": stream,
    "recording_id": recording_id,
    "duration_seconds": duration_seconds,
    "word_count": word_count,
  }

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
    with get_client().start_as_current_observation(
      as_type="span",
      name="active-listener-recording",
      trace_context={"trace_id": recording_trace_id(recording_id)},
      input=observation_input,
      metadata=observation_metadata,
    ) as observation:
      yield cast(RecordingObservation, cast(object, observation))


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
    with get_client().start_as_current_observation(
      name="active-listener-rewrite",
      trace_context={"trace_id": recording_trace_id(recording_id)},
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
