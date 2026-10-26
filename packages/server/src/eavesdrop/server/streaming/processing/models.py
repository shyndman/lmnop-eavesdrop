"""Typed audio chunks and identity-bearing recognition outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from eavesdrop.server.streaming.interfaces import Float32Audio
from eavesdrop.server.transcription.models import SpeechChunk, TranscriptionInfo
from eavesdrop.wire import Segment


@dataclass
class AudioChunk:
  """Audio data with associated metadata for processing.

  ``start_time`` is a buffer offset. In live WebSocket mode recording boundaries
  reset the buffer, so the value is recording-relative for the active recording.
  """

  data: Float32Audio
  duration: float
  start_time: float


@dataclass
class ChunkTranscriptionResult:
  """Result of transcribing an audio chunk."""

  status: "TranscriptionPassStatus"
  # Absolute sample indexes are in the buffer timeline. In live WebSocket mode,
  # recording boundaries reset the buffer, so these are recording-relative samples.
  chunk_start_sample: int
  chunk_sample_count: int
  segments: list[Segment] | None
  info: TranscriptionInfo | None
  processing_time: float
  audio_duration: float
  speech_chunks: list[SpeechChunk] | None = None
  utterance_generation: int = 0
  recording_id: str | None = None


class TranscriptionPassStatus(str, Enum):
  """Outcome for one transcription pass over a buffered audio chunk."""

  TRANSCRIBED = "transcribed"
  INTERRUPTED_BEFORE_COMMIT = "interrupted_before_commit"
