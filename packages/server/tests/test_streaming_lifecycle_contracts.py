"""Deterministic lifecycle contract tests for ``WebSocketStreamingClient``.

These tests pin orchestration guarantees around completion handling:
- EOF ingestion drives completion and teardown.
- Pending task cancellation happens before stop orchestration.
- Stop processing/disconnect occurs before source shutdown.
- Active-task teardown remains bounded.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from eavesdrop.server.streaming.client import WebSocketStreamingClient


def _build_client(*, processor: MagicMock, audio_source: MagicMock) -> WebSocketStreamingClient:
  """Create a minimally wired client instance for lifecycle orchestration tests."""
  client = WebSocketStreamingClient.__new__(WebSocketStreamingClient)
  client.websocket = MagicMock()
  client.stream_name = "stream-1"
  client.logger = MagicMock()
  client.session = MagicMock()
  client.buffer = MagicMock()
  client.transcription_sink = MagicMock()
  client.processor = processor
  client.audio_source = audio_source
  client._processing_task = None
  client._audio_task = None
  client._completion_task = None
  client._exit = False
  return client


async def _run_until_cancelled(cancelled_event: asyncio.Event) -> None:
  """Block forever until cancellation, then signal deterministic cancellation observation."""
  try:
    await asyncio.Future()
  except asyncio.CancelledError:
    cancelled_event.set()
    raise


async def test_ingestion_eof_completes_and_tears_down_cleanly() -> None:
  """EOF from audio ingestion must complete orchestration and stop the client exactly once."""
  processing_cancelled = asyncio.Event()

  processor = MagicMock()
  processor.initialize = AsyncMock()

  async def _start_processing() -> None:
    await _run_until_cancelled(processing_cancelled)

  processor.start_processing = _start_processing
  processor.stop_processing = AsyncMock()

  audio_source = MagicMock()
  audio_source.read_audio = AsyncMock(return_value=None)
  audio_source.close = MagicMock()

  client = _build_client(processor=processor, audio_source=audio_source)

  completion_task = await client.start()
  await asyncio.wait_for(completion_task, timeout=0.5)

  assert processing_cancelled.is_set()
  processor.initialize.assert_awaited_once()
  processor.stop_processing.assert_awaited_once()
  audio_source.close.assert_called_once()
  assert client._exit is True


async def test_completion_cancels_pending_task_before_stop() -> None:
  """Completion wait must cancel the still-running task before invoking stop orchestration."""
  pending_cancelled = asyncio.Event()

  processor = MagicMock()
  processor.stop_processing = AsyncMock()

  audio_source = MagicMock()
  audio_source.close = MagicMock()

  client = _build_client(processor=processor, audio_source=audio_source)
  client._audio_task = asyncio.create_task(asyncio.sleep(0))
  client._processing_task = asyncio.create_task(_run_until_cancelled(pending_cancelled))

  async def _stop_guard() -> None:
    assert pending_cancelled.is_set()

  client.stop = AsyncMock(side_effect=_stop_guard)

  await asyncio.wait_for(client._wait_for_completion(), timeout=0.5)

  assert client._processing_task.cancelled()
  client.stop.assert_awaited_once()


async def test_completion_path_runs_disconnect_before_source_close() -> None:
  """Completion-triggered stop must disconnect processor output before closing audio source."""
  events: list[str] = []

  sink = MagicMock()

  async def _disconnect() -> None:
    events.append("disconnect")

  sink.disconnect = AsyncMock(side_effect=_disconnect)

  processor = MagicMock()

  async def _stop_processing() -> None:
    events.append("stop_processing")
    await sink.disconnect()

  processor.stop_processing = AsyncMock(side_effect=_stop_processing)

  audio_source = MagicMock()
  audio_source.close = MagicMock(side_effect=lambda: events.append("audio_close"))

  client = _build_client(processor=processor, audio_source=audio_source)
  client._audio_task = asyncio.create_task(asyncio.sleep(0))
  client._processing_task = asyncio.create_task(asyncio.sleep(3600))

  await asyncio.wait_for(client._wait_for_completion(), timeout=0.5)

  assert events == ["stop_processing", "disconnect", "audio_close"]
  processor.stop_processing.assert_awaited_once()
  audio_source.close.assert_called_once()


async def test_stop_during_active_processing_is_bounded() -> None:
  """Stopping during active tasks must complete quickly while cancelling both running loops."""
  processing_cancelled = asyncio.Event()
  audio_cancelled = asyncio.Event()

  processor = MagicMock()
  processor.stop_processing = AsyncMock()

  audio_source = MagicMock()
  audio_source.close = MagicMock()

  client = _build_client(processor=processor, audio_source=audio_source)
  client._processing_task = asyncio.create_task(_run_until_cancelled(processing_cancelled))
  client._audio_task = asyncio.create_task(_run_until_cancelled(audio_cancelled))

  # Yield once so both tasks begin running before stop() attempts cancellation.
  await asyncio.sleep(0)

  await asyncio.wait_for(client.stop(), timeout=0.5)

  assert processing_cancelled.is_set()
  assert audio_cancelled.is_set()
  assert client._processing_task.cancelled()
  assert client._audio_task.cancelled()
  processor.stop_processing.assert_awaited_once()
  audio_source.close.assert_called_once()
