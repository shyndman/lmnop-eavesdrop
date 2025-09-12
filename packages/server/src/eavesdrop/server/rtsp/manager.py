import asyncio
from typing import TYPE_CHECKING, TypedDict

from eavesdrop.server.config import TranscriptionConfig
from eavesdrop.server.logs import get_logger
from eavesdrop.server.rtsp.client import RTSPTranscriptionClient

if TYPE_CHECKING:
  from eavesdrop.server.rtsp.cache import RTSPTranscriptionCache
  from eavesdrop.server.rtsp.subscriber import RTSPSubscriberManager


class StreamStatusDict(TypedDict):
  """Status information for a single RTSP stream."""

  url: str
  reconnect_count: int
  chunks_read: int
  total_bytes: int
  transcriptions_completed: int
  transcription_errors: int
  task_running: bool
  stopped: bool
  buffer_duration: float
  processed_duration: float
  available_duration: float
  processor_active: bool
  segments_processed: int


class StreamSummaryDict(TypedDict):
  """Summary statistics for all RTSP streams."""

  total_created: int
  active_streams: int
  failed_streams: int


class RTSPManagerStatusDict(TypedDict):
  """Complete status information for the RTSP manager."""

  summary: StreamSummaryDict
  streams: dict[str, StreamStatusDict]


class RTSPClientManager:
  """
  Manager for RTSP transcription clients.

  Provides centralized management of multiple RTSP streams, handling their
  lifecycle, health monitoring, and graceful shutdown. Mirrors the pattern
  used by WebSocketClientManager for WebSocket clients.
  """

  def __init__(
    self,
    transcription_config: TranscriptionConfig,
    subscriber_manager: "RTSPSubscriberManager",
    transcription_cache: "RTSPTranscriptionCache",
  ):
    """
    Initialize the RTSP client manager.

    :param
        transcription_config: Global transcription configuration
        subscriber_manager: Manager for WebSocket subscribers
        transcription_cache: Cache for storing transcription history
    """
    self.transcription_config = transcription_config
    self.subscriber_manager = subscriber_manager
    self.transcription_cache = transcription_cache
    self.clients: dict[str, RTSPTranscriptionClient] = {}
    self.tasks: dict[str, asyncio.Task] = {}
    self.logger = get_logger("rtsp/mgr")

    # Statistics
    self.total_streams_created = 0
    self.active_streams = 0
    self.failed_streams = 0

  async def add_stream(self, stream_name: str, rtsp_url: str) -> bool:
    """
    Add a new RTSP stream for transcription.

    :param
        stream_name: Unique name for the stream
        rtsp_url: RTSP URL to connect to

    :returns:
        True if stream was added successfully, False otherwise
    """
    if stream_name in self.clients:
      self.logger.warning("Stream already exists", stream=stream_name)
      return False

    try:
      self.logger.info("Adding RTSP stream", stream=stream_name, url=rtsp_url)

      # Create RTSP client with subscriber manager and cache
      client = RTSPTranscriptionClient(
        stream_name,
        rtsp_url,
        self.transcription_config,
        self.subscriber_manager,
        self.transcription_cache,
      )
      self.clients[stream_name] = client

      # Create and start task for this client
      task = asyncio.create_task(client.run())
      task.set_name(f"rtsp_stream_{stream_name}")
      self.tasks[stream_name] = task

      self.total_streams_created += 1
      self.active_streams += 1

      self.logger.info(
        "RTSP stream added successfully", stream=stream_name, active_streams=self.active_streams
      )

      return True

    except Exception:
      self.failed_streams += 1
      self.logger.exception("Failed to add RTSP stream", stream=stream_name)

      # Clean up partial state
      self.clients.pop(stream_name, None)
      task = self.tasks.pop(stream_name, None)
      if task and not task.done():
        task.cancel()

      return False

  async def remove_stream(self, stream_name: str) -> bool:
    """
    Remove an RTSP stream and stop its transcription.

    :param
        stream_name: Name of the stream to remove

    :returns:
        True if stream was removed successfully, False if not found
    """
    if stream_name not in self.clients:
      self.logger.warning("Stream not found for removal", stream=stream_name)
      return False

    try:
      self.logger.info("Removing RTSP stream", stream=stream_name)

      # Stop the client
      client = self.clients[stream_name]
      await client.stop()

      # Cancel and clean up task
      task = self.tasks[stream_name]
      if not task.done():
        task.cancel()
        try:
          await task
        except asyncio.CancelledError:
          pass

      # Remove from tracking
      del self.clients[stream_name]
      del self.tasks[stream_name]

      self.active_streams -= 1

      self.logger.info(
        "RTSP stream removed successfully", stream=stream_name, active_streams=self.active_streams
      )

      return True

    except Exception:
      self.logger.exception("Failed to remove RTSP stream", stream=stream_name)
      return False

  async def start_all_streams(self, stream_config: dict[str, str]) -> None:
    """
    Start all RTSP streams from configuration.

    :param
        stream_config: Dictionary mapping stream names to RTSP URLs
    """
    self.logger.info("Starting all RTSP streams", stream_count=len(stream_config))

    successful_streams = []
    failed_streams = []

    for stream_name, rtsp_url in stream_config.items():
      try:
        success = await self.add_stream(stream_name, rtsp_url)
        if success:
          successful_streams.append(stream_name)
        else:
          failed_streams.append(stream_name)
      except Exception:
        self.logger.exception("Exception adding stream", stream=stream_name)
        failed_streams.append(stream_name)

    self.logger.info(
      "RTSP stream startup completed",
      successful=len(successful_streams),
      failed=len(failed_streams),
      successful_streams=successful_streams,
      failed_streams=failed_streams if failed_streams else None,
    )

  async def stop_all_streams(self) -> None:
    """
    Stop all active RTSP streams gracefully.

    Used during server shutdown to ensure clean cleanup of all resources.
    """
    if not self.clients:
      self.logger.debug("No RTSP streams to stop")
      return

    self.logger.info("Stopping all RTSP streams", active_streams=self.active_streams)

    # Stop all clients concurrently
    stop_tasks = []
    for stream_name, client in self.clients.items():
      self.logger.debug("Stopping RTSP stream", stream=stream_name)
      stop_task = asyncio.create_task(client.stop())
      stop_task.set_name(f"stop_rtsp_{stream_name}")
      stop_tasks.append(stop_task)

    # Wait for all stops to complete
    if stop_tasks:
      await asyncio.gather(*stop_tasks, return_exceptions=True)

    # Cancel any remaining tasks
    cancel_tasks = []
    for stream_name, task in self.tasks.items():
      if not task.done():
        self.logger.debug("Cancelling RTSP task", stream=stream_name)
        task.cancel()
        cancel_tasks.append(task)

    # Wait for cancellations
    if cancel_tasks:
      await asyncio.gather(*cancel_tasks, return_exceptions=True)

    # Clean up tracking
    self.clients.clear()
    self.tasks.clear()
    self.active_streams = 0

    self.logger.info("All RTSP streams stopped successfully")

  def get_stream_status(self) -> RTSPManagerStatusDict:
    """
    Get status information for all RTSP streams.

    :returns:
        Dictionary with stream status information
    """
    status: RTSPManagerStatusDict = {
      "summary": {
        "total_created": self.total_streams_created,
        "active_streams": self.active_streams,
        "failed_streams": self.failed_streams,
      },
      "streams": {},
    }

    for stream_name, client in self.clients.items():
      task = self.tasks.get(stream_name)
      status["streams"][stream_name] = {
        "url": client.rtsp_url,
        "reconnect_count": client.reconnect_count,
        "chunks_read": client.chunks_read,
        "total_bytes": client.total_bytes,
        "transcriptions_completed": getattr(client, "transcriptions_completed", 0),
        "transcription_errors": getattr(client, "transcription_errors", 0),
        "task_running": task is not None and not task.done(),
        "stopped": client.stopped,
        "buffer_duration": client.stream_buffer.total_duration,
        "processed_duration": client.stream_buffer.processed_duration,
        "available_duration": client.stream_buffer.available_duration,
        "processor_active": client.processor is not None and not client.processor.exit,
        "segments_processed": getattr(client.processor, "segments_processed", 0),
      }

    return status

  def get_active_stream_names(self) -> list[str]:
    """Get list of active stream names."""
    return list(self.clients.keys())

  def get_stream_count(self) -> int:
    """Get count of active streams."""
    return len(self.clients)

  async def restart_failed_streams(self) -> None:
    """
    Restart any streams that have failed tasks.

    This can be called periodically to recover from unexpected failures.
    """
    failed_streams = []

    for stream_name, task in list(self.tasks.items()):
      if task.done() and not self.clients[stream_name].stopped:
        # Task completed but client wasn't explicitly stopped - likely a failure
        failed_streams.append(stream_name)

    if failed_streams:
      self.logger.warning("Detected failed streams", streams=failed_streams)

      for stream_name in failed_streams:
        client = self.clients[stream_name]
        rtsp_url = client.rtsp_url

        # Remove the failed stream
        await self.remove_stream(stream_name)

        # Re-add it to restart
        await self.add_stream(stream_name, rtsp_url)

        self.logger.info("Restarted failed stream", stream=stream_name)
