"""Completion and revision behavior observed through the live protocol."""

import numpy as np
import pytest
from protocol_harness import audio, cancel, flush, protocol_server, start, transcription


@pytest.mark.asyncio
@pytest.mark.parametrize("silence_seconds", [1, 46])
async def test_no_vad_all_segments_remain_revisable_until_explicit_completion(
  monkeypatch: pytest.MonkeyPatch,
  silence_seconds: int,
) -> None:
  """Combine all no-VAD portions into one revisable timed segment until forced completion.

  :param monkeypatch: Restores the deterministic inference loader.
  :param silence_seconds: Silence duration, including a case beyond buffer cleanup limits.
  """
  async with protocol_server(monkeypatch, minimum=0.1, interval=0.05, window=1) as server:
    async with server.connection() as socket:
      await start(socket, "revisions")
      # Speech followed by silence beyond the completion threshold must not authorize completion.
      waveform = np.concatenate(
        (
          np.full(3200, 0.5, dtype=np.float32),
          np.zeros(16000 * silence_seconds, dtype=np.float32),
        )
      )
      for offset in range(0, len(waveform), 160000):
        await socket.send(waveform[offset : offset + 160000].tobytes())
      first = await transcription(socket)
      pending = [segment for segment in first.segments if segment.text.strip()]
      assert len(pending) == 1
      assert all(not segment.completed for segment in first.segments)
      first_text = [segment.text for segment in pending]
      assert pending[0].text == "A0 revision 1 A1 revision 1"
      assert [word.word for word in pending[0].words] == ["A0", "A1"]
      assert [(word.start, word.end) for word in pending[0].words] == [(0.01, 0.03), (0.05, 0.08)]

      await audio(socket, seconds=0.2)
      while True:
        revised = await transcription(socket)
        assert all(not segment.completed for segment in revised.segments)
        revised_text = [segment.text for segment in revised.segments if segment.text.strip()]
        if revised_text != first_text:
          break
      assert len(revised_text) == 1

      await flush(socket, "revisions", force=False)
      provisional = await transcription(socket, flushed=True)
      assert provisional.recording_id == "revisions"
      assert len([segment for segment in provisional.segments if segment.text.strip()]) == 1
      assert all(not segment.completed for segment in provisional.segments)

      await flush(socket, "revisions", force=True)
      completed = await transcription(socket, flushed=True)
      spoken = [segment for segment in completed.segments if segment.text.strip()]
      assert len(spoken) == 1
      assert all(segment.completed for segment in spoken)
      assert [word.word for word in spoken[0].words] == ["A0", "A1"]
      assert "A0" in spoken[0].text and "A1" in spoken[0].text
      assert [(word.start, word.end) for word in spoken[0].words] == [(0.01, 0.03), (0.05, 0.08)]

      await audio(socket, seconds=0.2, positive=False)
      while True:
        following = await transcription(socket)
        new_segments = [segment for segment in following.segments if segment.text.startswith("B")]
        if new_segments:
          break
      assert len(new_segments) == 1
      assert all(not segment.completed for segment in new_segments)
      assert all(
        segment.completed for segment in following.segments if segment.text.startswith("A")
      )
      await flush(socket, "revisions", force=False)
      still_pending = await transcription(socket, flushed=True)
      assert all(
        not segment.completed for segment in still_pending.segments if segment.text.startswith("B")
      )
      await flush(socket, "revisions")
      final = await transcription(socket, flushed=True)
      assert all(segment.completed for segment in final.segments if segment.text.strip())


@pytest.mark.asyncio
async def test_enabled_vad_retains_silence_completion_and_incomplete_tail(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Retain VAD silence completion and its tentative trailing segment.

  :param monkeypatch: Restores the deterministic inference loader.
  """
  async with protocol_server(monkeypatch, use_vad=True, minimum=0.1, interval=0.05) as server:
    async with server.connection() as socket:
      await start(socket, "silence")
      await audio(socket, seconds=2)
      result = await transcription(socket)
      spoken = [segment for segment in result.segments if segment.text.strip()]
      assert len(spoken) == 2
      assert all(segment.completed for segment in spoken)
      assert len([segment for segment in result.segments if not segment.completed]) == 1
      assert result.segments[-1].completed is False
      assert result.segments[-1].words == []
      await flush(socket, "silence")
      assert (await transcription(socket, flushed=True)).recording_id == "silence"


@pytest.mark.asyncio
async def test_cancellation_clears_previously_completed_utterance_history(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Remove completed utterance history when cancellation starts a new utterance.

  :param monkeypatch: Restores the deterministic inference loader.
  """
  async with protocol_server(monkeypatch) as server:
    async with server.connection() as socket:
      await start(socket, "same-recording")
      await audio(socket)
      await flush(socket, "same-recording")
      initial = await transcription(socket, flushed=True)
      assert any(segment.text.startswith("A") and segment.completed for segment in initial.segments)
      await cancel(socket, "same-recording")
      await audio(socket, positive=False)
      await flush(socket, "same-recording")
      reset = await transcription(socket, flushed=True)
      assert reset.recording_id == "same-recording"
      assert not any(segment.text.startswith("A") for segment in reset.segments)
      assert len([segment for segment in reset.segments if segment.text.startswith("B")]) == 1
