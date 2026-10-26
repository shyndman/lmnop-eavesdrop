"""Recording lifecycle contracts exercised over real local WebSockets."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
from protocol_harness import (
  audio,
  cancel,
  entered,
  error,
  flush,
  protocol_server,
  start,
  transcription,
)

from eavesdrop.wire import ErrorMessage, deserialize_message


def test_empty_consumed_and_repeated_flush_leave_second_socket_responsive() -> None:
  """Keep a second socket responsive after empty, consumed, and repeated flushes."""
  directory = str(Path(__file__).parent)
  environment = os.environ.copy()
  environment["PYTHONPATH"] = os.pathsep.join((directory, environment.get("PYTHONPATH", "")))
  result = subprocess.run(
    [
      sys.executable,
      "-c",
      (
        "import asyncio; from protocol_harness import empty_flush_responsiveness; "
        "asyncio.run(empty_flush_responsiveness())"
      ),
    ],
    env=environment,
    capture_output=True,
    text=True,
    timeout=20,
    check=False,
  )
  assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.asyncio
async def test_short_audio_flush_interrupts_long_normal_interval(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Flush short audio without waiting for the normal transcription interval.

  :param monkeypatch: Fixture configuring the real WebSocket server's model.
  """
  async with protocol_server(monkeypatch, interval=30, minimum=1) as server:
    async with server.connection() as socket:
      await start(socket, "short")
      await audio(socket, seconds=0.1)
      await flush(socket, "short")
      result = await transcription(socket, flushed=True)
      assert result.recording_id == "short"
      assert len([segment for segment in result.segments if segment.text.strip()]) == 1
      assert all(segment.completed for segment in result.segments if segment.text.strip())


@pytest.mark.asyncio
async def test_normal_two_recordings_reset_transcript_and_recording_zero(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Reset transcript content and recording-relative timestamps between recordings.

  :param monkeypatch: Fixture configuring the real WebSocket server's model.
  """
  async with protocol_server(monkeypatch) as server:
    async with server.connection() as socket:
      for identifier, positive, prefix in (("A", True, "A"), ("B", False, "B")):
        await start(socket, identifier)
        await audio(socket, positive=positive)
        await flush(socket, identifier)
        result = await transcription(socket, flushed=True)
        spoken = [segment for segment in result.segments if segment.text.strip()]
        assert result.recording_id == identifier
        assert len(spoken) == 1
        assert all(segment.text.startswith(prefix) for segment in spoken)
        assert spoken[0].start == pytest.approx(0.01)
        assert spoken[0].words[0].start == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_boundary_queues_b_while_a_recognition_and_flush_are_outstanding(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Queue B while A is outstanding and deliver each recording's own timed results.

  :param monkeypatch: Fixture configuring the real WebSocket server's model.
  """
  async with protocol_server(monkeypatch, minimum=0.1, interval=0.02, hold=True) as server:
    async with server.connection() as socket:
      await start(socket, "A")
      await audio(socket, seconds=0.2)
      await entered(server.model)
      await flush(socket, "A")
      await start(socket, "B")
      await audio(socket, positive=False)
      await flush(socket, "B")
      server.model.release.set()
      first = await transcription(socket, flushed=True)
      assert first.recording_id == "A"
      assert all(segment.text.startswith("A") for segment in first.segments if segment.text.strip())
      assert len([segment for segment in first.segments if segment.text.strip()]) == 1
      second = await transcription(socket, flushed=True)
      assert second.recording_id == "B"
      spoken = [segment for segment in second.segments if segment.text.strip()]
      assert len(spoken) == 1
      assert all(segment.text.startswith("B") for segment in spoken)
      assert spoken[0].start == pytest.approx(0.01)
      assert all(segment.completed for segment in spoken)


@pytest.mark.asyncio
async def test_cancel_outstanding_work_and_restart_suppresses_cancelled_results(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Suppress cancelled recognition results when a replacement recording starts.

  :param monkeypatch: Fixture configuring the real WebSocket server's model.
  """
  async with protocol_server(monkeypatch, minimum=0.1, interval=0.02, hold=True) as server:
    async with server.connection() as socket:
      await start(socket, "cancelled")
      await audio(socket, seconds=0.2)
      await entered(server.model)
      await flush(socket, "cancelled")
      await cancel(socket, "cancelled")
      # A rejected control is a transport-visible barrier after the preceding cancellation.
      await flush(socket, "not-the-active-recording")
      await error(socket)
      await start(socket, "replacement")
      await audio(socket, positive=False)
      await flush(socket, "replacement")
      server.model.release.set()
      while True:
        result = await transcription(socket)
        assert result.recording_id == "replacement"
        assert all(not segment.text.startswith("A") for segment in result.segments)
        if result.flush_complete:
          break
      spoken = [segment for segment in result.segments if segment.text.strip()]
      assert len(spoken) == 1
      assert spoken[0].start == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_stale_controls_cannot_cancel_or_flush_current_recording(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Reject stale controls without cancelling or flushing the current recording.

  :param monkeypatch: Fixture configuring the real WebSocket server's model.
  """
  async with protocol_server(monkeypatch) as server:
    async with server.connection() as socket:
      await start(socket, "old")
      await audio(socket)
      await flush(socket, "old")
      await transcription(socket, flushed=True)
      await start(socket, "current")
      await audio(socket, positive=False)
      await cancel(socket, "old")
      await error(socket)
      await flush(socket, "old")
      await error(socket)
      await flush(socket, "current")
      result = await transcription(socket, flushed=True)
      assert result.recording_id == "current"
      assert len([segment for segment in result.segments if segment.text.startswith("B")]) == 1


@pytest.mark.asyncio
async def test_disconnect_releases_connection_and_server_remains_usable(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Release a disconnected socket while keeping the server usable by another.

  :param monkeypatch: Fixture configuring the real WebSocket server's model.
  """
  async with protocol_server(monkeypatch, minimum=0.1, interval=0.02, hold=True) as server:
    async with server.connection() as first:
      await start(first, "disconnected")
      await audio(first, seconds=0.2)
      await entered(server.model)
      await first.close()
      async with asyncio.timeout(3):
        await first.wait_closed()
    server.model.release.set()
    async with server.connection() as second:
      await start(second, "survivor")
      await audio(second, positive=False)
      await flush(second, "survivor")
      result = await transcription(second, flushed=True)
      assert result.recording_id == "survivor"
      assert len([segment for segment in result.segments if segment.text.startswith("B")]) == 1


@pytest.mark.asyncio
async def test_flush_interrupts_interval_and_includes_newly_accepted_audio(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Interrupt the normal interval and include newly accepted audio in the flush.

  :param monkeypatch: Fixture configuring the real WebSocket server's model.
  """
  async with protocol_server(monkeypatch, interval=30, minimum=0.1) as server:
    async with server.connection() as socket:
      await start(socket, "waiting")
      await audio(socket, seconds=0.2)
      ordinary = await transcription(socket)
      assert all(not segment.completed for segment in ordinary.segments)
      await audio(socket, positive=False)
      await flush(socket, "waiting")
      result = await transcription(socket, flushed=True)
      assert result.recording_id == "waiting"
      assert any(word.word == "B1" for segment in result.segments for word in segment.words)
      assert all(segment.completed for segment in result.segments if segment.text.strip())


@pytest.mark.asyncio
@pytest.mark.parametrize("discard_a", [False, True])
async def test_deferred_recording_requires_explicit_completion_or_cancellation(
  monkeypatch: pytest.MonkeyPatch,
  discard_a: bool,
) -> None:
  """Defer B until A is explicitly completed or cancelled after a tentative flush.

  :param monkeypatch: Fixture configuring the real WebSocket server's model.
  :param discard_a: Whether to cancel A instead of explicitly completing it.
  """
  async with protocol_server(monkeypatch) as server:
    async with server.connection() as socket:
      await start(socket, "A")
      await audio(socket)
      await flush(socket, "A", force=False)
      provisional = await transcription(socket, flushed=True)
      assert all(not segment.completed for segment in provisional.segments)
      await start(socket, "B")
      await audio(socket, positive=False)
      await flush(socket, "B")
      # A stale control is an observable barrier even while B is deferred.
      await flush(socket, "stale")
      barrier = cast(ErrorMessage, deserialize_message(await socket.recv(decode=True)))
      assert barrier.type == "error"
      assert barrier.message == (
        "Live control rejected: recording_id does not match the active recording identifier"
      )
      with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
          await socket.recv()
      if discard_a:
        await cancel(socket, "A")
      else:
        await flush(socket, "A")
        completed_a = await transcription(socket, flushed=True)
        assert completed_a.recording_id == "A"
        assert [word.word for segment in completed_a.segments for word in segment.words] == [
          "A0",
          "A1",
        ]
        assert all(segment.completed for segment in completed_a.segments if segment.text.strip())
      completed_b = await transcription(socket, flushed=True)
      assert completed_b.recording_id == "B"
      assert [word.word for segment in completed_b.segments for word in segment.words] == [
        "B0",
        "B1",
      ]
      assert completed_b.segments[0].start == pytest.approx(0.01)
      assert all(segment.completed for segment in completed_b.segments if segment.text.strip())
