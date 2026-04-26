"""Regression tests for absolute timestamp behavior in transcription output."""

import json
import sys
from types import ModuleType
from typing import cast

import pytest

from eavesdrop.server.transcription.models import SpeechChunk
from eavesdrop.server.transcription.session import create_session
from eavesdrop.server.transcription.utils import (
  finalize_recording_timestamps,
  restore_speech_timestamps,
)
from eavesdrop.wire import Segment, TranscriptionMessage, Word, serialize_message


class MockSpeechTimestampsMap:
  """Fake faster-whisper timestamp map that returns chunk-local times."""

  def __init__(self, chunks: list[SpeechChunk], rate: int) -> None:
    _ = chunks
    _ = rate

  def get_chunk_index(self, t: float) -> int:
    _ = t
    return 0

  def get_original_time(self, t: float, chunk_index: int = 0) -> float:
    _ = chunk_index
    return t % 100.0


class MockVadModule(ModuleType):
  SpeechTimestampsMap: type[MockSpeechTimestampsMap] = MockSpeechTimestampsMap


def _install_mock_vad(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setitem(sys.modules, "faster_whisper", ModuleType("faster_whisper"))
  monkeypatch.setitem(sys.modules, "faster_whisper.vad", MockVadModule("faster_whisper.vad"))


def _segment(*, start: float, end: float, time_offset: float) -> Segment:
  return Segment(
    id=1,
    seek=0,
    start=start,
    end=end,
    text="hello",
    tokens=[1, 2, 3],
    avg_logprob=-0.25,
    compression_ratio=1.0,
    words=None,
    temperature=0.0,
    time_offset=time_offset,
    completed=False,
  )


def test_vad_restoration_with_late_flattening_preserves_offset(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _install_mock_vad(monkeypatch)
  absolute_offset = 100.0
  local_start = 0.5
  local_end = 1.5
  word = Word(
    start=local_start + 0.1,
    end=local_start + 0.5,
    word="hello",
    probability=0.99,
  )
  segment = Segment(
    id=1,
    seek=0,
    start=local_start,
    end=local_end,
    text="hello",
    tokens=[1, 2, 3],
    avg_logprob=-0.25,
    compression_ratio=1.0,
    words=[word],
    temperature=0.0,
    time_offset=0.0,
    completed=False,
  )

  speech_chunks: list[SpeechChunk] = [SpeechChunk(start=0, end=160000)]
  sampling_rate = 16000

  _ = restore_speech_timestamps([segment], speech_chunks, sampling_rate)
  _ = finalize_recording_timestamps([segment], absolute_offset)

  assert abs(segment.start - (absolute_offset + local_start + 0.1)) < 1e-9
  assert abs(word.start - (absolute_offset + local_start + 0.1)) < 1e-9
  assert segment.time_offset == absolute_offset


def test_finalize_recording_timestamps_direct() -> None:
  word = Word(start=0.1, end=0.5, word="a", probability=1.0)
  segment = _segment(start=0.0, end=1.0, time_offset=0.0)
  segment.words = [word]

  _ = finalize_recording_timestamps([segment], 50.0)

  assert segment.start == 50.0
  assert segment.end == 51.0
  assert segment.time_offset == 50.0
  assert word.start == 50.1
  assert word.end == 50.5


def test_segment_absolute_times_apply_offset_beyond_30_seconds() -> None:
  segment = _segment(start=2.5, end=4.0, time_offset=31.25)

  assert abs(segment.absolute_start_time - 33.75) < 1e-9
  assert abs(segment.absolute_end_time - 35.25) < 1e-9


def test_absolute_times_remain_monotonic_across_window_boundaries() -> None:
  first = _segment(start=2.0, end=4.0, time_offset=28.0)
  second = _segment(start=0.1, end=1.1, time_offset=32.0)

  assert abs(first.absolute_start_time - 30.0) < 1e-9
  assert abs(first.absolute_end_time - 32.0) < 1e-9
  assert abs(second.absolute_start_time - 32.1) < 1e-9
  assert abs(second.absolute_end_time - 33.1) < 1e-9
  assert second.absolute_start_time > first.absolute_end_time


def test_transcription_message_serializes_flattened_timestamps() -> None:
  segment = Segment(
    id=1,
    seek=0,
    start=43.0,
    end=45.0,
    text="hello",
    tokens=[1, 2, 3],
    avg_logprob=-0.25,
    compression_ratio=1.0,
    words=None,
    temperature=0.0,
    time_offset=40.0,
    completed=False,
  )
  message = TranscriptionMessage(stream="s1", segments=[segment], language="en")

  payload = cast(dict[str, object], json.loads(serialize_message(message)))
  serialized_segments = cast(list[dict[str, object]], payload["segments"])
  serialized_segment = serialized_segments[0]

  assert "time_offset" not in serialized_segment
  assert "absolute_start_time" not in serialized_segment
  assert "absolute_end_time" not in serialized_segment
  assert serialized_segment["start"] == 43.0
  assert serialized_segment["end"] == 45.0


def test_transcription_message_word_timestamps_share_recording_timeline() -> None:
  segment = Segment(
    id=1,
    seek=0,
    start=40.0,
    end=42.5,
    text="worded hello",
    tokens=[1, 2, 3],
    avg_logprob=-0.25,
    compression_ratio=1.0,
    words=[
      Word(
        start=40.25,
        end=40.6,
        word="word",
        probability=0.99,
      )
    ],
    temperature=0.0,
    time_offset=40.0,
    completed=False,
  )
  message = TranscriptionMessage(stream="s1", segments=[segment], language="en")

  payload = cast(dict[str, object], json.loads(serialize_message(message)))
  serialized_segments = cast(list[dict[str, object]], payload["segments"])
  serialized_segment = serialized_segments[0]
  serialized_words = cast(list[dict[str, object]], serialized_segment["words"])
  serialized_word = serialized_words[0]

  assert "time_offset" not in serialized_segment
  assert "absolute_start_time" not in serialized_word
  assert "absolute_end_time" not in serialized_word
  assert serialized_word["start"] == 40.25
  assert serialized_word["end"] == 40.6


def test_session_reports_absolute_chunk_time_range() -> None:
  session = create_session("stream-1")

  session.update_audio_context(start_offset=61.0, duration=1.5)

  assert session.get_absolute_time_range() == (61.0, 62.5)
