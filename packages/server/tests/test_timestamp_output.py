"""Regression tests for recording-relative timestamp behavior in transcription output."""

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from protocol_harness import flush, protocol_server, start, transcription

from eavesdrop.server.transcription.models import SpeechChunk, VadParameters
from eavesdrop.server.transcription.orukeet import OrukeetTranscriber
from eavesdrop.server.transcription.session import create_session
from eavesdrop.server.transcription.utils import FloatAudio, finalize_recording_timestamps
from eavesdrop.wire import Segment, Word


def _segment(*, start: float, end: float, time_offset: float) -> Segment:
  return Segment(
    id=1,
    start=start,
    end=end,
    text="hello",
    words=[],
    time_offset=time_offset,
    completed=False,
  )


def test_finalize_recording_timestamps_direct() -> None:
  word = Word(start=0.1, end=0.5, word="a")
  segment = _segment(start=0.0, end=1.0, time_offset=0.0)
  segment.words = [word]

  _ = finalize_recording_timestamps([segment], 50.0)

  assert segment.start == 50.0
  assert segment.end == 51.0
  assert segment.time_offset == 50.0
  assert word.start == 50.1
  assert word.end == 50.5


def test_segment_recording_relative_times_apply_offset_beyond_30_seconds() -> None:
  segment = _segment(start=2.5, end=4.0, time_offset=31.25)

  assert abs(segment.absolute_start_time - 33.75) < 1e-9
  assert abs(segment.absolute_end_time - 35.25) < 1e-9


def test_recording_relative_times_remain_monotonic_across_window_boundaries() -> None:
  first = _segment(start=2.0, end=4.0, time_offset=28.0)
  second = _segment(start=0.1, end=1.1, time_offset=32.0)

  assert abs(first.absolute_start_time - 30.0) < 1e-9
  assert abs(first.absolute_end_time - 32.0) < 1e-9
  assert abs(second.absolute_start_time - 32.1) < 1e-9
  assert abs(second.absolute_end_time - 33.1) < 1e-9
  assert second.absolute_start_time > first.absolute_end_time


def test_session_reports_recording_relative_chunk_time_range() -> None:
  session = create_session("stream-1")

  session.update_audio_context(start_offset=61.0, duration=1.5)

  assert session.get_recording_time_range() == (61.0, 62.5)


@dataclass
class _NativeHypothesis:
  text: str = "Hello."
  tokens: list[str] = field(default_factory=lambda: ["▁Hel", "lo", "."])
  timestamps: list[float] = field(default_factory=lambda: [0.08, 0.16, 0.24])
  durations: list[float] = field(default_factory=lambda: [0.08, 0.08, 0.0])


class _NativeRecognizer:
  def recognize(self, audio: FloatAudio) -> _NativeHypothesis:
    return _NativeHypothesis()


class _SpeechSpans:
  def detect(self, audio: FloatAudio, parameters: VadParameters) -> list[SpeechChunk]:
    # Deterministic VAD substitute: zero is silence, nonzero is speech.
    active = cast(NDArray[np.bool_], np.concatenate(([False], audio != 0, [False])))
    changes = cast(NDArray[np.bool_], active[1:] != active[:-1])
    boundaries = np.flatnonzero(changes)
    return [
      {
        "start": int(cast(np.intp, boundaries[index])),
        "end": int(cast(np.intp, boundaries[index + 1])),
      }
      for index in range(0, boundaries.size, 2)
    ]


@pytest.mark.asyncio
async def test_native_words_keep_leading_internal_silence_and_segment_window_offsets(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  adapter = OrukeetTranscriber(_NativeRecognizer(), _SpeechSpans())
  async with protocol_server(
    monkeypatch,
    use_vad=True,
    window=2,
    minimum=0.1,
    interval=30,
    transcriber=adapter,
  ) as server:
    async with server.connection() as socket:
      await start(socket, "native")
      waveform = np.zeros(64000, dtype=np.float32)
      waveform[8000:16000] = 0.5
      waveform[32000:40000] = 0.5
      await socket.send(waveform.tobytes())
      interim = await transcription(socket)
      spoken = [segment for segment in interim.segments if segment.text.strip()]
      assert [segment.start for segment in spoken] == pytest.approx([0.58, 2.08])
      assert [segment.words[0].start for segment in spoken] == pytest.approx([0.58, 2.08])
      assert [segment.end for segment in spoken] == pytest.approx([0.74, 2.24])
      assert [segment.words[-1].end for segment in spoken] == pytest.approx([0.74, 2.24])
      await flush(socket, "native")
      first = await transcription(socket, flushed=True)
      assert first.recording_id == "native"
      assert all(segment.completed for segment in first.segments if segment.text.strip())
      # Completion must consume the full VAD speech span, not re-recognize the
      # tail after the native word's earlier end during the flush.
      first_spoken = [segment for segment in first.segments if segment.text.strip()]
      assert [segment.start for segment in first_spoken] == pytest.approx([0.58, 2.08])
      assert [segment.end for segment in first_spoken] == pytest.approx([0.74, 2.24])
      assert [segment.id for segment in first_spoken] == [segment.id for segment in spoken]

      following = np.zeros(32000, dtype=np.float32)
      following[8000:16000] = 0.5
      await socket.send(following.tobytes())
      await flush(socket, "native")
      second = await transcription(socket, flushed=True)
      spoken = [segment for segment in second.segments if segment.text.strip()]
      assert len(spoken) == 2
      assert [segment.start for segment in spoken] == pytest.approx([2.08, 4.58])
      assert [segment.words[0].start for segment in spoken] == pytest.approx([2.08, 4.58])
      assert [segment.end for segment in spoken] == pytest.approx([2.24, 4.74])
      assert len({segment.id for segment in spoken}) == 2
