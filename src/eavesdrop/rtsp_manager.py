import asyncio

from .logs import get_logger
from .rtsp import RTSPTranscriptionClient
from .rtsp_models import RTSPModelManager


class RTSPClientManager:
  """
  Manager for RTSP transcription clients.

  Provides centralized management of multiple RTSP streams, handling their
  lifecycle, health monitoring, and graceful shutdown. Mirrors the pattern
  used by ClientManager for WebSocket clients.
  """

  def __init__(self, model_manager: RTSPModelManager):
    """
    Initialize the RTSP client manager.

    Args:
        model_manager: Shared model manager for all RTSP clients
    """
    self.model_manager = model_manager
    self.clients: dict[str, RTSPTranscriptionClient] = {}
    self.tasks: dict[str, asyncio.Task] = {}
    self.logger = get_logger("rtsp_client_manager")

    # Statistics
    self.total_streams_created = 0
    self.active_streams = 0
    self.failed_streams = 0

  async def add_stream(self, stream_name: str, rtsp_url: str) -> bool:
    """
    Add a new RTSP stream for transcription.

    Args:
        stream_name: Unique name for the stream
        rtsp_url: RTSP URL to connect to

    Returns:
        True if stream was added successfully, False otherwise
    """
    if stream_name in self.clients:
      self.logger.warning("Stream already exists", stream=stream_name)
      return False

    try:
      self.logger.info("Adding RTSP stream", stream=stream_name, url=rtsp_url)

      # Get shared transcriber from model manager
      transcriber = await self.model_manager.get_transcriber()

      # Create RTSP client
      client = RTSPTranscriptionClient(stream_name, rtsp_url, transcriber)
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

    Args:
        stream_name: Name of the stream to remove

    Returns:
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

    Args:
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

    # Clean up model manager
    await self.model_manager.cleanup()

    self.logger.info("All RTSP streams stopped successfully")

  def get_stream_status(self) -> dict[str, dict]:
    """
    Get status information for all RTSP streams.

    Returns:
        Dictionary with stream status information
    """
    status = {
      "summary": {
        "total_created": self.total_streams_created,
        "active_streams": self.active_streams,
        "failed_streams": self.failed_streams,
        "model_info": self.model_manager.get_model_info(),
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
        "processor_active": not client.processor.stopped,
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
