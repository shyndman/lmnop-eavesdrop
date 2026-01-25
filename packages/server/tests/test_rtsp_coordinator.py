"""Tests for the RTSPStreamCoordinator class."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from eavesdrop.server.config import RTSPCacheConfig, RTSPConfig, TranscriptionConfig
from eavesdrop.server.rtsp.coordinator import RTSPStreamCoordinator


class TestRTSPStreamCoordinator:
  """Tests for RTSPStreamCoordinator initialization and lifecycle."""

  @pytest.fixture
  def rtsp_config(self) -> RTSPConfig:
    """Create test RTSP configuration with one stream."""
    return RTSPConfig(
      streams={"office": "rtsp://camera1:554/stream"},
      cache=RTSPCacheConfig(),
    )

  @pytest.fixture
  def multi_stream_config(self) -> RTSPConfig:
    """Create test RTSP configuration with multiple streams."""
    return RTSPConfig(
      streams={
        "office": "rtsp://camera1:554/stream",
        "lobby": "rtsp://camera2:554/stream",
        "entrance": "rtsp://camera3:554/stream",
      },
      cache=RTSPCacheConfig(),
    )

  @pytest.fixture
  def transcription_config(self) -> TranscriptionConfig:
    """Create test transcription configuration."""
    return TranscriptionConfig()

  @pytest.fixture
  def coordinator(
    self, rtsp_config: RTSPConfig, transcription_config: TranscriptionConfig
  ) -> RTSPStreamCoordinator:
    """Create coordinator instance for testing."""
    return RTSPStreamCoordinator(rtsp_config, transcription_config)

  def test_initial_state(self, coordinator: RTSPStreamCoordinator) -> None:
    """Test coordinator initial state before initialization."""
    assert not coordinator.is_initialized
    assert coordinator.subscriber_manager is None
    assert coordinator.client_manager is None
    assert coordinator.stream_count == 0

  def test_get_status_before_init(self, coordinator: RTSPStreamCoordinator) -> None:
    """Test get_status returns empty status before initialization."""
    status = coordinator.get_status()

    assert status["summary"]["total_created"] == 0
    assert status["summary"]["active_streams"] == 0
    assert status["summary"]["failed_streams"] == 0
    assert status["streams"] == {}


class TestRTSPStreamCoordinatorInitialization:
  """Tests for RTSPStreamCoordinator.initialize() method."""

  @pytest.fixture
  def rtsp_config(self) -> RTSPConfig:
    """Create test RTSP configuration."""
    return RTSPConfig(
      streams={"office": "rtsp://camera1:554/stream"},
      cache=RTSPCacheConfig(),
    )

  @pytest.fixture
  def transcription_config(self) -> TranscriptionConfig:
    """Create test transcription configuration."""
    return TranscriptionConfig()

  @pytest.fixture
  def coordinator(
    self, rtsp_config: RTSPConfig, transcription_config: TranscriptionConfig
  ) -> RTSPStreamCoordinator:
    """Create coordinator instance for testing."""
    return RTSPStreamCoordinator(rtsp_config, transcription_config)

  @pytest.fixture
  def mock_components(self):
    """Create mock RTSP components for testing."""
    with (
      patch("eavesdrop.server.rtsp.coordinator.RTSPTranscriptionCache") as mock_cache_cls,
      patch("eavesdrop.server.rtsp.coordinator.RTSPSubscriberManager") as mock_sub_mgr_cls,
      patch("eavesdrop.server.rtsp.coordinator.RTSPClientManager") as mock_client_mgr_cls,
    ):
      mock_cache = MagicMock()
      mock_cache_cls.return_value = mock_cache

      mock_sub_mgr = MagicMock()
      mock_sub_mgr_cls.return_value = mock_sub_mgr

      mock_client_mgr = MagicMock()
      mock_client_mgr.start_all_streams = AsyncMock()
      mock_client_mgr.stop_all_streams = AsyncMock()
      mock_client_mgr.get_stream_count.return_value = 1
      mock_client_mgr.get_stream_status.return_value = {
        "summary": {"total_created": 1, "active_streams": 1, "failed_streams": 0},
        "streams": {},
      }
      mock_client_mgr_cls.return_value = mock_client_mgr

      yield {
        "cache_cls": mock_cache_cls,
        "cache": mock_cache,
        "sub_mgr_cls": mock_sub_mgr_cls,
        "sub_mgr": mock_sub_mgr,
        "client_mgr_cls": mock_client_mgr_cls,
        "client_mgr": mock_client_mgr,
      }

  @pytest.mark.asyncio
  async def test_initialize_creates_components_in_order(
    self, coordinator: RTSPStreamCoordinator, mock_components: dict
  ) -> None:
    """Test that initialize creates cache, subscriber manager, then client manager."""
    await coordinator.initialize()

    # Verify all components were created
    mock_components["cache_cls"].assert_called_once()
    mock_components["sub_mgr_cls"].assert_called_once()
    mock_components["client_mgr_cls"].assert_called_once()

    # Verify subscriber manager was created with cache
    call_args = mock_components["sub_mgr_cls"].call_args
    assert call_args[0][1] == mock_components["cache"]

    # Verify client manager was created with subscriber manager and cache
    call_args = mock_components["client_mgr_cls"].call_args
    assert call_args[0][1] == mock_components["sub_mgr"]
    assert call_args[0][2] == mock_components["cache"]

  @pytest.mark.asyncio
  async def test_initialize_starts_all_streams(
    self,
    coordinator: RTSPStreamCoordinator,
    mock_components: dict,
    rtsp_config: RTSPConfig,
  ) -> None:
    """Test that initialize starts all configured streams."""
    await coordinator.initialize()

    mock_components["client_mgr"].start_all_streams.assert_called_once_with(rtsp_config.streams)

  @pytest.mark.asyncio
  async def test_initialize_sets_initialized_flag(
    self, coordinator: RTSPStreamCoordinator, mock_components: dict
  ) -> None:
    """Test that initialize sets is_initialized to True on success."""
    assert not coordinator.is_initialized

    await coordinator.initialize()

    assert coordinator.is_initialized

  @pytest.mark.asyncio
  async def test_initialize_exposes_subscriber_manager(
    self, coordinator: RTSPStreamCoordinator, mock_components: dict
  ) -> None:
    """Test that subscriber_manager is accessible after initialization."""
    assert coordinator.subscriber_manager is None

    await coordinator.initialize()

    assert coordinator.subscriber_manager is mock_components["sub_mgr"]

  @pytest.mark.asyncio
  async def test_initialize_exposes_client_manager(
    self, coordinator: RTSPStreamCoordinator, mock_components: dict
  ) -> None:
    """Test that client_manager is accessible after initialization."""
    assert coordinator.client_manager is None

    await coordinator.initialize()

    assert coordinator.client_manager is mock_components["client_mgr"]

  @pytest.mark.asyncio
  async def test_double_initialize_is_idempotent(
    self, coordinator: RTSPStreamCoordinator, mock_components: dict
  ) -> None:
    """Test that calling initialize twice only initializes once."""
    await coordinator.initialize()
    await coordinator.initialize()

    # Should only create components once
    mock_components["cache_cls"].assert_called_once()
    mock_components["sub_mgr_cls"].assert_called_once()
    mock_components["client_mgr_cls"].assert_called_once()

  @pytest.mark.asyncio
  async def test_initialize_failure_cleans_up(
    self, coordinator: RTSPStreamCoordinator, mock_components: dict
  ) -> None:
    """Test that initialization failure cleans up partial state."""
    mock_components["client_mgr"].start_all_streams.side_effect = Exception("Connection failed")

    with pytest.raises(RuntimeError, match="RTSP initialization failed"):
      await coordinator.initialize()

    assert not coordinator.is_initialized
    assert coordinator.subscriber_manager is None
    assert coordinator.client_manager is None


class TestRTSPStreamCoordinatorShutdown:
  """Tests for RTSPStreamCoordinator.shutdown() method."""

  @pytest.fixture
  def rtsp_config(self) -> RTSPConfig:
    """Create test RTSP configuration."""
    return RTSPConfig(
      streams={"office": "rtsp://camera1:554/stream"},
      cache=RTSPCacheConfig(),
    )

  @pytest.fixture
  def transcription_config(self) -> TranscriptionConfig:
    """Create test transcription configuration."""
    return TranscriptionConfig()

  @pytest.fixture
  def coordinator(
    self, rtsp_config: RTSPConfig, transcription_config: TranscriptionConfig
  ) -> RTSPStreamCoordinator:
    """Create coordinator instance for testing."""
    return RTSPStreamCoordinator(rtsp_config, transcription_config)

  @pytest.fixture
  def mock_components(self):
    """Create mock RTSP components for testing."""
    with (
      patch("eavesdrop.server.rtsp.coordinator.RTSPTranscriptionCache") as mock_cache_cls,
      patch("eavesdrop.server.rtsp.coordinator.RTSPSubscriberManager") as mock_sub_mgr_cls,
      patch("eavesdrop.server.rtsp.coordinator.RTSPClientManager") as mock_client_mgr_cls,
    ):
      mock_cache = MagicMock()
      mock_cache_cls.return_value = mock_cache

      mock_sub_mgr = MagicMock()
      mock_sub_mgr_cls.return_value = mock_sub_mgr

      mock_client_mgr = MagicMock()
      mock_client_mgr.start_all_streams = AsyncMock()
      mock_client_mgr.stop_all_streams = AsyncMock()
      mock_client_mgr.get_stream_count.return_value = 1
      mock_client_mgr_cls.return_value = mock_client_mgr

      yield {
        "cache_cls": mock_cache_cls,
        "cache": mock_cache,
        "sub_mgr_cls": mock_sub_mgr_cls,
        "sub_mgr": mock_sub_mgr,
        "client_mgr_cls": mock_client_mgr_cls,
        "client_mgr": mock_client_mgr,
      }

  @pytest.mark.asyncio
  async def test_shutdown_stops_all_streams(
    self, coordinator: RTSPStreamCoordinator, mock_components: dict
  ) -> None:
    """Test that shutdown stops all streams via client manager."""
    await coordinator.initialize()
    await coordinator.shutdown()

    mock_components["client_mgr"].stop_all_streams.assert_called_once()

  @pytest.mark.asyncio
  async def test_shutdown_clears_components(
    self, coordinator: RTSPStreamCoordinator, mock_components: dict
  ) -> None:
    """Test that shutdown clears all component references."""
    await coordinator.initialize()
    assert coordinator.subscriber_manager is not None
    assert coordinator.client_manager is not None

    await coordinator.shutdown()

    assert coordinator.subscriber_manager is None
    assert coordinator.client_manager is None
    assert not coordinator.is_initialized

  @pytest.mark.asyncio
  async def test_shutdown_before_init_is_safe(self, coordinator: RTSPStreamCoordinator) -> None:
    """Test that shutdown before initialization is a no-op."""
    # Should not raise
    await coordinator.shutdown()

    assert not coordinator.is_initialized

  @pytest.mark.asyncio
  async def test_double_shutdown_is_safe(
    self, coordinator: RTSPStreamCoordinator, mock_components: dict
  ) -> None:
    """Test that calling shutdown twice is safe."""
    await coordinator.initialize()
    await coordinator.shutdown()
    await coordinator.shutdown()

    # stop_all_streams should only be called once
    mock_components["client_mgr"].stop_all_streams.assert_called_once()


class TestRTSPStreamCoordinatorStatus:
  """Tests for RTSPStreamCoordinator.get_status() method."""

  @pytest.fixture
  def rtsp_config(self) -> RTSPConfig:
    """Create test RTSP configuration."""
    return RTSPConfig(
      streams={"office": "rtsp://camera1:554/stream"},
      cache=RTSPCacheConfig(),
    )

  @pytest.fixture
  def transcription_config(self) -> TranscriptionConfig:
    """Create test transcription configuration."""
    return TranscriptionConfig()

  @pytest.fixture
  def coordinator(
    self, rtsp_config: RTSPConfig, transcription_config: TranscriptionConfig
  ) -> RTSPStreamCoordinator:
    """Create coordinator instance for testing."""
    return RTSPStreamCoordinator(rtsp_config, transcription_config)

  @pytest.fixture
  def mock_components(self):
    """Create mock RTSP components for testing."""
    with (
      patch("eavesdrop.server.rtsp.coordinator.RTSPTranscriptionCache") as mock_cache_cls,
      patch("eavesdrop.server.rtsp.coordinator.RTSPSubscriberManager") as mock_sub_mgr_cls,
      patch("eavesdrop.server.rtsp.coordinator.RTSPClientManager") as mock_client_mgr_cls,
    ):
      mock_cache = MagicMock()
      mock_cache_cls.return_value = mock_cache

      mock_sub_mgr = MagicMock()
      mock_sub_mgr_cls.return_value = mock_sub_mgr

      mock_client_mgr = MagicMock()
      mock_client_mgr.start_all_streams = AsyncMock()
      mock_client_mgr.stop_all_streams = AsyncMock()
      mock_client_mgr.get_stream_count.return_value = 1
      mock_client_mgr.get_stream_status.return_value = {
        "summary": {"total_created": 1, "active_streams": 1, "failed_streams": 0},
        "streams": {"office": {"url": "rtsp://camera1:554/stream"}},
      }
      mock_client_mgr_cls.return_value = mock_client_mgr

      yield {
        "cache_cls": mock_cache_cls,
        "cache": mock_cache,
        "sub_mgr_cls": mock_sub_mgr_cls,
        "sub_mgr": mock_sub_mgr,
        "client_mgr_cls": mock_client_mgr_cls,
        "client_mgr": mock_client_mgr,
      }

  @pytest.mark.asyncio
  async def test_get_status_after_init(
    self, coordinator: RTSPStreamCoordinator, mock_components: dict
  ) -> None:
    """Test get_status delegates to client manager after initialization."""
    await coordinator.initialize()

    status = coordinator.get_status()

    mock_components["client_mgr"].get_stream_status.assert_called_once()
    assert status["summary"]["active_streams"] == 1
    assert "office" in status["streams"]

  @pytest.mark.asyncio
  async def test_stream_count_after_init(
    self, coordinator: RTSPStreamCoordinator, mock_components: dict
  ) -> None:
    """Test stream_count reflects client manager state."""
    await coordinator.initialize()

    assert coordinator.stream_count == 1
    mock_components["client_mgr"].get_stream_count.assert_called()
