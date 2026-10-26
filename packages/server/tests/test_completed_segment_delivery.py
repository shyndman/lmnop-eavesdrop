"""Newly completed speech reaches the wire before completed history is windowed."""

import pytest
from protocol_harness import audio, entered, error, flush, protocol_server, start, transcription


@pytest.mark.asyncio
@pytest.mark.parametrize("use_vad", [False, True])
async def test_force_completion_delivers_all_new_segments_before_windowing_history(
  monkeypatch: pytest.MonkeyPatch,
  use_vad: bool,
) -> None:
  """Deliver every new completion, then retain only the configured history.

  :param monkeypatch: Fixture substituting deterministic recognition.
  :param use_vad: Whether silence-based completion is enabled.
  :returns: None.
  """
  async with protocol_server(monkeypatch, use_vad=use_vad, window=1) as server:
    async with server.connection() as socket:
      await start(socket, "complete")
      await audio(socket)
      if not use_vad:
        await flush(socket, "complete", force=False)
        provisional = await transcription(socket, flushed=True)
        spoken = [segment for segment in provisional.segments if segment.text.strip()]
        assert " ".join(segment.text for segment in spoken) == "A0 revision 1 A1 revision 1"
        assert all(not segment.completed for segment in spoken)

      await flush(socket, "complete")
      completed = await transcription(socket, flushed=True)
      spoken = [segment for segment in completed.segments if segment.text.strip()]
      assert completed.recording_id == "complete"
      expected_revision = 1 if use_vad else 2
      assert " ".join(segment.text for segment in spoken) == (
        f"A0 revision {expected_revision} A1 revision {expected_revision}"
      )
      assert all(segment.completed for segment in spoken)
      assert [word.word for segment in spoken for word in segment.words] == ["A0", "A1"]
      assert spoken[0].start == pytest.approx(0.01)
      assert spoken[-1].end == pytest.approx(0.08)
      assert [word.start for segment in spoken for word in segment.words] == pytest.approx(
        [0.01, 0.05]
      )
      assert [word.end for segment in spoken for word in segment.words] == pytest.approx(
        [0.03, 0.08]
      )

      await flush(socket, "complete")
      history = await transcription(socket, flushed=True)
      retained = [segment for segment in history.segments if segment.text.strip()]
      assert [segment.id for segment in retained] == [spoken[-1].id]
      assert [segment.text for segment in retained] == [spoken[-1].text]
      assert [word.word for segment in retained for word in segment.words] == (
        ["A1"] if use_vad else ["A0", "A1"]
      )


@pytest.mark.asyncio
async def test_completions_suppressed_before_flush_boundary_are_delivered_with_flush(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Preserve undelivered VAD completions across a held recognition pass.

  :param monkeypatch: Fixture substituting held deterministic recognition.
  :returns: None.
  """
  async with protocol_server(monkeypatch, use_vad=True, window=1, minimum=0.1, hold=True) as server:
    async with server.connection() as socket:
      await start(socket, "held")
      await audio(socket, seconds=0.2)
      await entered(server.model)
      await audio(socket, seconds=0.2)
      await flush(socket, "held")
      # A rejected stale control provides a wire-visible barrier: the accepted
      # flush boundary includes the second frame before inference is released.
      await flush(socket, "stale")
      await error(socket)
      server.model.release.set()
      completed = await transcription(socket, flushed=True)
      spoken = [segment for segment in completed.segments if segment.text.strip()]
      assert completed.recording_id == "held"
      assert [segment.text for segment in spoken] == [
        "A0 revision 1",
        "A0 revision 2",
        "A1 revision 2",
      ]
      assert all(segment.completed for segment in spoken)
      assert [word.word for segment in spoken for word in segment.words] == [
        "A0",
        "A0",
        "A1",
      ]
      assert [segment.start for segment in spoken] == pytest.approx([0.01, 0.04, 0.08])
      assert [word.start for segment in spoken for word in segment.words] == pytest.approx(
        [0.01, 0.04, 0.08]
      )
