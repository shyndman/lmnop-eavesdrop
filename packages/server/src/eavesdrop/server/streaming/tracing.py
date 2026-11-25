"""Structured tracing helpers for streaming transcription processor."""

from typing import SupportsFloat, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from eavesdrop.common import Range, Seconds, get_logger
from eavesdrop.server.transcription.models import SpeechChunk
from eavesdrop.wire import Segment

tracing_logger = get_logger("tracing")
Float32Audio = NDArray[np.float32]


class TracingClientSegmentDict(TypedDict):
  id: int
  start: str
  end: str
  text: str
  completed: bool
  synthetic: bool


def log_vad_analysis(
  audio: Float32Audio,
  duration: float,
  speech_chunks: list[SpeechChunk] | None,
  sample_rate: int = 16000,
) -> None:
  """Emit structured VAD timeline and audio energy tracing for one chunk."""
  rms = float(cast(SupportsFloat, np.sqrt(np.mean(audio**2)))) if len(audio) > 0 else 0.0
  peak = float(cast(SupportsFloat, np.max(np.abs(audio)))) if len(audio) > 0 else 0.0
  zero_crossings = int(np.count_nonzero(np.diff(np.signbit(audio)))) if len(audio) > 1 else 0
  zcr = zero_crossings / len(audio) * sample_rate if len(audio) > 0 else 0.0

  if not speech_chunks:
    tracing_logger.info(
      "VAD analysis",
      speech_ranges=[],
      speech_duration="0.00s",
      silence_duration=f"{duration:.2f}s",
      audio_duration=f"{duration:.2f}s",
      speech_ratio="0.0%",
      timeline="~" * int(duration * 10),
      audio_rms=f"{rms:.6f}",
      audio_peak=f"{peak:.6f}",
      audio_zcr=f"{zcr:.1f}",
    )
    return

  speech_ranges = [
    f"{chunk['start'] / sample_rate:.2f}-{chunk['end'] / sample_rate:.2f}s"
    for chunk in speech_chunks
  ]
  total_speech_duration = sum(
    (chunk["end"] - chunk["start"]) / sample_rate for chunk in speech_chunks
  )
  timeline = ["~"] * int(duration * 10)
  for speech_chunk in speech_chunks:
    start_idx = int((speech_chunk["start"] / sample_rate) * 10)
    end_idx = int((speech_chunk["end"] / sample_rate) * 10)
    for idx in range(start_idx, min(end_idx, len(timeline))):
      timeline[idx] = "S"

  tracing_logger.info(
    "VAD analysis",
    speech_ranges=speech_ranges,
    speech_duration=f"{total_speech_duration:.2f}s",
    silence_duration=f"{duration - total_speech_duration:.2f}s",
    audio_duration=f"{duration:.2f}s",
    speech_ratio=f"{total_speech_duration / duration:.1%}",
    timeline="".join(timeline),
    audio_rms=f"{rms:.6f}",
    audio_peak=f"{peak:.6f}",
    audio_zcr=f"{zcr:.1f}",
  )


def log_generated_segments(segments: list[Segment], duration: float) -> None:
  """Emit structured tracing for raw recognizer segments."""
  raw_segments = [
    {
      "id": seg.id,
      "timespan": Range(Seconds(seg.start), Seconds(seg.end)),
      "text": seg.text[:50] + ("..." if len(seg.text) > 50 else ""),
      "completed": seg.completed,
    }
    for seg in segments
  ]
  tracing_logger.info(
    "speech recognition generated segments",
    segment_count=len(segments),
    segments=raw_segments,
    audio_duration=Seconds(duration),
  )


def log_client_segments(segments: list[Segment]) -> None:
  """Emit structured tracing for client-bound segment envelopes."""
  client_segments: list[TracingClientSegmentDict] = [
    {
      "id": seg.id,
      "start": f"{seg.start:.2f}s",
      "end": f"{seg.end:.2f}s",
      "text": seg.text[:30] + ("..." if len(seg.text) > 30 else ""),
      "completed": seg.completed,
      "synthetic": seg.text == "" and seg.start == seg.end,
    }
    for seg in segments
  ]
  completed_count = sum(1 for seg in segments if seg.completed)
  tracing_logger.info(
    "Sending segments to client",
    total_segments=len(segments),
    completed_segments=completed_count,
    incomplete_segments=len(segments) - completed_count,
    segments=client_segments,
  )
