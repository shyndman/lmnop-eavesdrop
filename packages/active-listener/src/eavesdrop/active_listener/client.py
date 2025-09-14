"""Eavesdrop client integration wrapper.

Provides integration with eavesdrop client library including connection management,
transcription message handling, and error recovery for real-time audio streaming.
"""

import time

import structlog

from eavesdrop.active_listener.text_manager import ConnectionState
from eavesdrop.client import EavesdropClient


class EavesdropClientWrapper:
  """Wrapper for EavesdropClient with connection management and health monitoring."""

  def __init__(self, host: str, port: int, audio_device: str):
    self._host = host
    self._port = port
    self._audio_device = audio_device
    self._client: EavesdropClient | None = None
    self._connection_state = ConnectionState()
    self.logger = structlog.get_logger("🤸👂")

  async def initialize(self) -> None:
    """Initialize the eavesdrop client connection."""
    try:
      self.logger.info("Initializing eavesdrop client", host=self._host, port=self._port)

      self._client = self._create_client()
      await self._client.connect()

      self._connection_state.is_connected = True
      self._connection_state.last_message_time = time.time()
      self._connection_state.error_message = None

      self.logger.info("Client initialized successfully")

    except Exception as e:
      self._connection_state.is_connected = False
      self._connection_state.error_message = str(e)
      self.logger.exception("Failed to initialize client")
      raise

  def _create_client(self) -> EavesdropClient:
    """Create and configure the eavesdrop client."""
    return EavesdropClient.transcriber(
      host=self._host,
      port=self._port,
      audio_device=self._audio_device,
      hotwords=["com"],
    )

  def __aiter__(self):
    """Make wrapper itself async iterable."""
    if not self._client:
      raise Exception("Client not initialized")
    return self._client.__aiter__()

  async def __anext__(self):
    """Forward async iteration to the underlying client."""
    if not self._client:
      raise StopAsyncIteration

    try:
      message = await self._client.__anext__()
      self.update_last_message_time()
      return message
    except Exception as e:
      self._connection_state.is_connected = False
      self._connection_state.error_message = str(e)
      self.logger.exception("Error receiving message")
      raise

  async def start_streaming(self) -> None:
    """Start audio streaming from the configured device."""
    if not self._client:
      raise Exception("Client not initialized")

    try:
      await self._client.start_streaming()
      self._connection_state.is_streaming = True
      self.logger.info("Audio streaming started", device=self._audio_device)

    except Exception:
      self._connection_state.is_streaming = False
      self.logger.exception("Failed to start audio streaming")
      raise

  async def stop_streaming(self) -> None:
    """Stop audio streaming."""
    if not self._client:
      return

    try:
      await self._client.stop_streaming()
      self._connection_state.is_streaming = False
      self.logger.info("Audio streaming stopped")

    except Exception:
      self.logger.exception("Error stopping audio stream")

  def check_connection_health(self) -> bool:
    """Check if connection is healthy based on recent message activity."""
    if not self._connection_state.is_connected:
      return False

    # Check if we've received messages recently (within 30 seconds)
    current_time = time.time()
    time_since_last_message = current_time - self._connection_state.last_message_time

    return time_since_last_message < 30.0

  def update_last_message_time(self) -> None:
    """Update the timestamp of the last received message."""
    self._connection_state.last_message_time = time.time()

  async def attempt_reconnection(self) -> bool:
    """Attempt to reconnect to the server."""
    if not self._client:
      return False

    try:
      self._connection_state.reconnection_attempts += 1
      self.logger.info(
        "Attempting reconnection", attempt=self._connection_state.reconnection_attempts
      )

      await self._client.connect()

      self._connection_state.is_connected = True
      self._connection_state.error_message = None
      self.update_last_message_time()

      self.logger.info("Reconnection successful")
      return True

    except Exception as e:
      self._connection_state.is_connected = False
      self._connection_state.error_message = str(e)
      self.logger.exception("Reconnection failed")
      return False

  async def shutdown(self) -> None:
    """Gracefully shutdown the client connection."""
    try:
      if self._connection_state.is_streaming:
        await self.stop_streaming()

      if self._client and self._connection_state.is_connected:
        await self._client.disconnect()

      self._connection_state.is_connected = False
      self._connection_state.is_streaming = False

      self.logger.info("Client shutdown complete")

    except Exception:
      self.logger.exception("Error during shutdown")
