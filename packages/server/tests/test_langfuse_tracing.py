"""Contract tests for backend Langfuse tracing behavior."""

from pytest import MonkeyPatch

from eavesdrop.server.transcription.models import SpeechChunk
from eavesdrop.server.transcription.session import create_session


def test_transcription_tracers_remain_operational_without_langfuse_credentials(
  monkeypatch: MonkeyPatch,
) -> None:
  """Backend pipeline tracing must keep recording timings when Langfuse is disabled."""
  monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
  monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

  session = create_session("stream-1")
  session.update_recording_context("rec-1")
  session.update_audio_context(start_offset=2.0, duration=1.5)
  speech_chunks: list[SpeechChunk] = [{"start": 0, "end": 8000}]

  with session.trace_pipeline():
    with session.trace_vad_stage() as tracer:
      tracer(speech_chunks, sample_rate=16000, total_samples=24000)
    with session.trace_feature_stage() as tracer:
      tracer()
    with session.trace_inference_stage() as tracer:
      tracer(attempts=2, final_temperature=0.0)
    with session.trace_segment_stage() as tracer:
      tracer([])

  assert session.recording_id == "rec-1"
  assert set(session.stage_timings) == {
    "audio_preprocessing",
    "feature_extraction",
    "model_inference",
    "segment_processing",
    "total_pipeline",
  }
  assert session.generation_attempts == 2
  assert session.get_recording_time_range() == (2.0, 3.5)
