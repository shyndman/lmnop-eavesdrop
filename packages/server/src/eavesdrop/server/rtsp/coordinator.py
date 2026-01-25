"""
RTSP stream coordination for initialization, lifecycle, and shutdown.

Consolidates RTSP component management that was previously scattered across
TranscriptionServer methods.
"""

from eavesdrop.common import get_logger
from eavesdrop.server.config import RTSPConfig, TranscriptionConfig
from eavesdrop.server.rtsp.cache import RTSPTranscriptionCache
from eavesdrop.server.rtsp.manager import RTSPClientManager, RTSPManagerStatusDict
from eavesdrop.server.rtsp.subscriber import RTSPSubscriberManager


class RTSPStreamCoordinator:
  """
  Coordinates RTSP stream initialization, lifecycle, and shutdown.

  Owns and manages the RTSP component hierarchy:

  - RTSPTranscriptionCache: Caches transcription results with listener-aware retention
  - RTSPSubscriberManager: Manages WebSocket subscribers to RTSP streams
  - RTSPClientManager: Manages RTSP stream clients and their transcription pipelines

  This class consolidates the RTSP-related code that was previously in
  TranscriptionServer._initialize_rtsp_streams() and _run_with_rtsp_streams().

  :param rtsp_config: RTSP stream configuration including stream URLs and cache settings.
  :type rtsp_config: RTSPConfig
  :param transcription_config: Transcription processing configuration.
  :type transcription_config: TranscriptionConfig
  """

  def __init__(
    self,
    rtsp_config: RTSPConfig,
    transcription_config: TranscriptionConfig,
  ) -> None:
    self._rtsp_config = rtsp_config
    self._transcription_config = transcription_config
    self._cache: RTSPTranscriptionCache | None = None
    self._subscriber_manager: RTSPSubscriberManager | None = None
    self._client_manager: RTSPClientManager | None = None
    self._initialized = False
    self._logger = get_logger("rtsp/coord")

  @property
  def subscriber_manager(self) -> RTSPSubscriberManager | None:
    """
    Get the subscriber manager for handling WebSocket subscriptions.

    :returns: The subscriber manager, or None if not initialized.
    :rtype: RTSPSubscriberManager | None
    """
    return self._subscriber_manager

  @property
  def client_manager(self) -> RTSPClientManager | None:
    """
    Get the client manager for RTSP stream management.

    :returns: The client manager, or None if not initialized.
    :rtype: RTSPClientManager | None
    """
    return self._client_manager

  @property
  def is_initialized(self) -> bool:
    """
    Check if the coordinator has been initialized.

    :returns: True if initialized, False otherwise.
    :rtype: bool
    """
    return self._initialized

  @property
  def stream_count(self) -> int:
    """
    Get the number of active RTSP streams.

    :returns: Number of active streams, or 0 if not initialized.
    :rtype: int
    """
    if self._client_manager:
      return self._client_manager.get_stream_count()
    return 0

  async def initialize(self) -> None:
    """
    Initialize RTSP components and start all configured streams.

    Creates the cache, subscriber manager, and client manager in the correct
    dependency order, then starts all streams defined in the configuration.

    :raises RuntimeError: If initialization fails.
    """
    if self._initialized:
      self._logger.warning("RTSP coordinator already initialized, skipping")
      return

    try:
      self._logger.info("Initializing RTSP transcription system")

      # Create transcription cache
      self._cache = RTSPTranscriptionCache(self._rtsp_config.cache)

      # Create RTSP subscriber manager with cache
      available_streams = set(self._rtsp_config.streams.keys())
      self._subscriber_manager = RTSPSubscriberManager(available_streams, self._cache)

      # Create RTSP client manager with subscriber manager and cache
      self._client_manager = RTSPClientManager(
        self._transcription_config, self._subscriber_manager, self._cache
      )

      self._logger.info(
        "RTSP subscriber manager created",
        available_streams=list(available_streams),
      )

      # Start all configured streams
      await self._client_manager.start_all_streams(self._rtsp_config.streams)

      self._initialized = True
      self._logger.info(
        "RTSP transcription system initialized",
        active_streams=self._client_manager.get_stream_count(),
      )

    except Exception:
      self._logger.exception("Failed to initialize RTSP system")
      # Clean up on failure
      self._cleanup_components()
      raise RuntimeError("RTSP initialization failed")

  async def shutdown(self) -> None:
    """
    Gracefully shutdown all RTSP streams and components.

    Stops all streams and cleans up resources. Safe to call multiple times.
    """
    if not self._initialized:
      return

    self._logger.info("Shutting down RTSP streams")

    if self._client_manager:
      await self._client_manager.stop_all_streams()

    self._cleanup_components()
    self._logger.info("RTSP coordinator shutdown complete")

  def get_status(self) -> RTSPManagerStatusDict:
    """
    Get status information for all RTSP streams.

    :returns: Dictionary with stream status information including summary
              and per-stream details.
    :rtype: RTSPManagerStatusDict
    """
    if self._client_manager:
      return self._client_manager.get_stream_status()
    return {
      "summary": {
        "total_created": 0,
        "active_streams": 0,
        "failed_streams": 0,
      },
      "streams": {},
    }

  def _cleanup_components(self) -> None:
    """Reset all component references to None."""
    self._client_manager = None
    self._subscriber_manager = None
    self._cache = None
    self._initialized = False
