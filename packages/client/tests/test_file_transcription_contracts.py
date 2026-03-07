"""Contract tests for one-shot ``EavesdropClient.transcribe_file`` behavior."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from eavesdrop.client.core import EavesdropClient, FileTranscriptionResult
from eavesdrop.wire import Segment, TranscriptionMessage


def _segment(*, segment_id: int, text: str, completed: bool) -> Segment:
  """Create deterministic segment fixtures used by reducer contract tests."""
  return Segment(
    id=segment_id,
    seek=0,
    start=0.0,
    end=0.5,
    text=text,
    tokens=[1, 2],
    avg_logprob=-0.1,
    compression_ratio=1.0,
    words=None,
    temperature=0.0,
    completed=completed,
  )


@pytest.mark.asyncio
async def test_transcribe_file_rejects_reentrant_operation(tmp_path) -> None:
  """transcribe_file must fail fast if a previous operation is still active."""
  audio_path = tmp_path / "clip.wav"
  audio_path.write_bytes(b"bytes")

  client = EavesdropClient.transcriber(audio_device="default")
  await client._operation_lock.acquire()

  with pytest.raises(RuntimeError, match="already in progress"):
    await client.transcribe_file(str(audio_path))

  client._operation_lock.release()


@pytest.mark.asyncio
async def test_transcribe_file_timeout_cleans_up_connection(tmp_path) -> None:
  """Timeout must trigger cleanup and raise TimeoutError to the caller."""
  audio_path = tmp_path / "clip.wav"
  audio_path.write_bytes(b"bytes")

  client = EavesdropClient.transcriber(audio_device="default")
  client.disconnect = AsyncMock()

  async def _slow_operation(*, file_bytes: bytes) -> FileTranscriptionResult:
    await asyncio.sleep(1.0)
    raise AssertionError("operation should have timed out")

  client._transcribe_file_operation = _slow_operation

  with pytest.raises(TimeoutError, match="timed out"):
    await client.transcribe_file(str(audio_path), timeout_s=0.01)

  client.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_transcribe_file_cancellation_cleans_up_connection(tmp_path) -> None:
  """Cancellation must cleanup and re-raise cancellation."""
  audio_path = tmp_path / "clip.wav"
  audio_path.write_bytes(b"bytes")

  client = EavesdropClient.transcriber(audio_device="default")
  client.disconnect = AsyncMock()

  gate = asyncio.Event()

  async def _blocking_operation(*, file_bytes: bytes) -> FileTranscriptionResult:
    await gate.wait()
    raise AssertionError("operation should have been cancelled")

  client._transcribe_file_operation = _blocking_operation

  operation_task = asyncio.create_task(client.transcribe_file(str(audio_path), timeout_s=None))
  await asyncio.sleep(0)
  operation_task.cancel()

  with pytest.raises(asyncio.CancelledError):
    await operation_task

  client.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_transcribe_file_operation_uploads_file_bytes_and_eof() -> None:
  """One-shot operation must upload bytes, send EOF, then reduce terminal output."""
  client = EavesdropClient.transcriber(audio_device="default")

  fake_connection = MagicMock()
  fake_connection.send_file_bytes = AsyncMock()
  fake_connection.send_end_of_audio = AsyncMock()

  async def _connect_with_fake_connection(
    setup_options,
  ) -> None:
    client._connection = fake_connection
    client._connected = True

  expected_result = FileTranscriptionResult(segments=[], text="", language=None, warnings=[])
  client.connect = AsyncMock(side_effect=_connect_with_fake_connection)
  client._collect_file_result = AsyncMock(return_value=expected_result)

  result = await client._transcribe_file_operation(file_bytes=b"abc")

  assert result == expected_result
  client.connect.assert_awaited_once()
  fake_connection.send_file_bytes.assert_awaited_once_with(b"abc")
  fake_connection.send_end_of_audio.assert_awaited_once()


@pytest.mark.asyncio
async def test_collect_file_result_treats_disconnect_as_terminal_signal() -> None:
  """Explicit disconnect event should terminate collection after reducing queued messages."""
  client = EavesdropClient.transcriber(audio_device="default")
  client._message_task = None

  message = TranscriptionMessage(
    stream="stream-1",
    segments=[
      _segment(segment_id=10, text="committed", completed=True),
      _segment(segment_id=11, text="tail", completed=False),
    ],
    language="en",
  )
  client._message_queue.put_nowait(message)

  async def _trigger_disconnect() -> None:
    await asyncio.sleep(0.05)
    client._on_disconnect("done")

  disconnect_task = asyncio.create_task(_trigger_disconnect())
  result = await client._collect_file_result()
  await disconnect_task

  assert [segment.id for segment in result.segments] == [10]
  assert result.text == "committed"
  assert result.language == "en"


@pytest.mark.asyncio
async def test_collect_file_result_treats_message_loop_exit_as_terminal_fallback() -> None:
  """Socket/message-loop completion must terminate collection when queue is drained."""
  client = EavesdropClient.transcriber(audio_device="default")

  completed_message = TranscriptionMessage(
    stream="stream-3",
    segments=[
      _segment(segment_id=21, text="alpha", completed=True),
      _segment(segment_id=22, text="tail", completed=False),
    ],
    language="en",
  )
  client._message_queue.put_nowait(completed_message)

  message_task = asyncio.create_task(asyncio.sleep(0))
  await message_task
  client._message_task = message_task

  result = await client._collect_file_result()

  assert [segment.id for segment in result.segments] == [21]
  assert result.text == "alpha"


def test_reducer_warns_and_commits_window_when_sentinel_missing() -> None:
  """Reducer must warn and continue progress when previous sentinel id is absent."""
  client = EavesdropClient.transcriber(audio_device="default")

  warnings: list[str] = []
  reduced_segments: list[Segment] = []
  message = TranscriptionMessage(
    stream="stream-2",
    segments=[
      _segment(segment_id=100, text="alpha", completed=True),
      _segment(segment_id=101, text="beta", completed=True),
      _segment(segment_id=999, text="tail", completed=False),
    ],
    language="en",
  )

  last_committed = client._reduce_windowed_segments(
    message=message,
    reduced_segments=reduced_segments,
    last_committed_id=42,
    warnings=warnings,
  )

  assert warnings
  assert warnings[0].startswith("Reducer sentinel missing")
  assert [segment.id for segment in reduced_segments] == [100, 101]
  assert last_committed == 101
