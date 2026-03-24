"""Contract tests for VAD tracer logging semantics."""

from dataclasses import dataclass, field
from typing import cast

from structlog.stdlib import BoundLogger

from eavesdrop.server.transcription.models import SpeechChunk
from eavesdrop.server.transcription.session import TranscriptionSession, create_session


@dataclass
class LogEvent:
  """Single structured log emission captured for assertions."""

  message: str
  fields: dict[str, float | int | str]


@dataclass
class RecordingLogger:
  """Minimal logger stub that records info-level structured events."""

  events: list[LogEvent] = field(default_factory=list)

  def info(self, message: str, **kwargs: float | int | str) -> None:
    self.events.append(LogEvent(message=message, fields=kwargs))


def _create_recording_session() -> tuple[TranscriptionSession, RecordingLogger]:
  """Create a transcription session with a recording logger attached."""
  session = create_session("stream-1")
  logger = RecordingLogger()
  session.logger = cast(BoundLogger, cast(object, logger))
  session.update_audio_context(0.0, 1.0)
  return session, logger


def test_vad_tracer_logs_no_voice_activity_for_empty_speech_chunks() -> None:
  """Empty VAD output must not be reported as detected voice."""
  session, logger = _create_recording_session()
  tracer = session.trace_vad_stage()

  tracer([], sample_rate=16000, total_samples=16000)

  assert logger.events == [
    LogEvent(
      message="No voice activity",
      fields={"t_start": 0.0, "t_end": 1.0, "total": 1.0},
    )
  ]


def test_vad_tracer_logs_voice_detected_for_non_empty_speech_chunks() -> None:
  """Non-empty VAD output must continue to be reported as detected voice."""
  session, logger = _create_recording_session()
  tracer = session.trace_vad_stage()
  speech_chunks: list[SpeechChunk] = [{"start": 0, "end": 8000}]

  tracer(speech_chunks, sample_rate=16000, total_samples=16000)

  assert len(logger.events) == 1
  event = logger.events[0]
  assert event.message == "Voice detected!"
  assert event.fields["chunk_count"] == 1
  assert event.fields["speech"] == 0.5
  assert event.fields["silence"] == 0.5


def test_vad_tracer_logs_vad_not_applied_when_chunks_are_missing() -> None:
  """Missing speech chunk metadata should be logged distinctly from silence."""
  session, logger = _create_recording_session()
  tracer = session.trace_vad_stage()

  tracer(None, sample_rate=16000, total_samples=16000)

  assert logger.events == [
    LogEvent(
      message="VAD not applied",
      fields={"t_start": 0.0, "t_end": 1.0, "total": 1.0},
    )
  ]
