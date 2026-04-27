"""Tests for the RTSPStreamCoordinator class."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import final

import pytest
from pytest import MonkeyPatch

from eavesdrop.server.config import RTSPCacheConfig, RTSPConfig, TranscriptionConfig
from eavesdrop.server.rtsp.coordinator import RTSPStreamCoordinator
from eavesdrop.server.rtsp.manager import RTSPManagerStatusDict


@final
class FakeCache:
  pass


@final
class FakeSubscriberManager:
  available_streams: set[str]
  cache: FakeCache

  def __init__(self, available_streams: set[str], cache: FakeCache) -> None:
    self.available_streams = available_streams
    self.cache = cache


@final
class FakeClientManager:
  transcription_config: TranscriptionConfig
  subscriber_manager: FakeSubscriberManager
  cache: FakeCache
  stream_count: int
  status: RTSPManagerStatusDict
  start_calls: list[dict[str, str]]
  stop_calls: int
  get_stream_count_calls: int
  get_stream_status_calls: int
  start_error: Exception | None

  def __init__(
    self,
    transcription_config: TranscriptionConfig,
    subscriber_manager: FakeSubscriberManager,
    cache: FakeCache,
    *,
    stream_count: int = 1,
    status: RTSPManagerStatusDict | None = None,
  ) -> None:
    self.transcription_config = transcription_config
    self.subscriber_manager = subscriber_manager
    self.cache = cache
    self.stream_count = stream_count
    self.status = status or {
      "summary": {"total_created": 1, "active_streams": 1, "failed_streams": 0},
      "streams": {},
    }
    self.start_calls = []
    self.stop_calls = 0
    self.get_stream_count_calls = 0
    self.get_stream_status_calls = 0
    self.start_error = None

  async def start_all_streams(self, streams: dict[str, str]) -> None:
    self.start_calls.append(streams)
    if self.start_error is not None:
      raise self.start_error

  async def stop_all_streams(self) -> None:
    self.stop_calls += 1

  def get_stream_count(self) -> int:
    self.get_stream_count_calls += 1
    return self.stream_count

  def get_stream_status(self) -> RTSPManagerStatusDict:
    self.get_stream_status_calls += 1
    return self.status


@dataclass
class CacheFactory:
  instance: FakeCache = field(default_factory=FakeCache)
  call_count: int = 0

  def __call__(self, config: RTSPCacheConfig) -> FakeCache:
    del config
    self.call_count += 1
    return self.instance


@dataclass
class SubscriberManagerFactory:
  instance: FakeSubscriberManager | None = None
  call_count: int = 0
  call_args: tuple[set[str], FakeCache] | None = None

  def __call__(self, available_streams: set[str], cache: FakeCache) -> FakeSubscriberManager:
    self.call_count += 1
    self.call_args = (available_streams, cache)
    self.instance = FakeSubscriberManager(available_streams, cache)
    return self.instance


@dataclass
class ClientManagerFactory:
  instance: FakeClientManager
  call_count: int = 0
  call_args: tuple[TranscriptionConfig, FakeSubscriberManager, FakeCache] | None = None

  def __call__(
    self,
    transcription_config: TranscriptionConfig,
    subscriber_manager: FakeSubscriberManager,
    cache: FakeCache,
  ) -> FakeClientManager:
    self.call_count += 1
    self.call_args = (transcription_config, subscriber_manager, cache)
    self.instance.transcription_config = transcription_config
    self.instance.subscriber_manager = subscriber_manager
    self.instance.cache = cache
    return self.instance


@dataclass
class ComponentBundle:
  cache_factory: CacheFactory
  subscriber_factory: SubscriberManagerFactory
  client_factory: ClientManagerFactory


@pytest.fixture
def rtsp_config() -> RTSPConfig:
  return RTSPConfig(
    streams={"office": "rtsp://camera1:554/stream"},
    cache=RTSPCacheConfig(),
  )


@pytest.fixture
def transcription_config() -> TranscriptionConfig:
  return TranscriptionConfig()


@pytest.fixture
def coordinator(
  rtsp_config: RTSPConfig, transcription_config: TranscriptionConfig
) -> RTSPStreamCoordinator:
  return RTSPStreamCoordinator(rtsp_config, transcription_config)


@pytest.fixture
def mock_components(monkeypatch: MonkeyPatch) -> Iterator[ComponentBundle]:
  client_manager = FakeClientManager(
    transcription_config=TranscriptionConfig(),
    subscriber_manager=FakeSubscriberManager(set(), FakeCache()),
    cache=FakeCache(),
  )
  components = ComponentBundle(
    cache_factory=CacheFactory(),
    subscriber_factory=SubscriberManagerFactory(),
    client_factory=ClientManagerFactory(instance=client_manager),
  )
  monkeypatch.setattr(
    "eavesdrop.server.rtsp.coordinator.RTSPTranscriptionCache",
    components.cache_factory,
  )
  monkeypatch.setattr(
    "eavesdrop.server.rtsp.coordinator.RTSPSubscriberManager",
    components.subscriber_factory,
  )
  monkeypatch.setattr(
    "eavesdrop.server.rtsp.coordinator.RTSPClientManager",
    components.client_factory,
  )
  yield components


class TestRTSPStreamCoordinator:
  @pytest.fixture
  def multi_stream_config(self) -> RTSPConfig:
    return RTSPConfig(
      streams={
        "office": "rtsp://camera1:554/stream",
        "lobby": "rtsp://camera2:554/stream",
        "entrance": "rtsp://camera3:554/stream",
      },
      cache=RTSPCacheConfig(),
    )

  def test_initial_state(self, coordinator: RTSPStreamCoordinator) -> None:
    assert not coordinator.is_initialized
    assert coordinator.subscriber_manager is None
    assert coordinator.client_manager is None
    assert coordinator.stream_count == 0

  def test_get_status_before_init(self, coordinator: RTSPStreamCoordinator) -> None:
    status = coordinator.get_status()

    assert status["summary"]["total_created"] == 0
    assert status["summary"]["active_streams"] == 0
    assert status["summary"]["failed_streams"] == 0
    assert status["streams"] == {}


class TestRTSPStreamCoordinatorInitialization:
  @pytest.mark.asyncio
  async def test_initialize_creates_components_in_order(
    self,
    coordinator: RTSPStreamCoordinator,
    mock_components: ComponentBundle,
    transcription_config: TranscriptionConfig,
  ) -> None:
    await coordinator.initialize()

    assert mock_components.cache_factory.call_count == 1
    assert mock_components.subscriber_factory.call_count == 1
    assert mock_components.client_factory.call_count == 1
    assert mock_components.subscriber_factory.call_args == (
      {"office"},
      mock_components.cache_factory.instance,
    )
    assert mock_components.client_factory.call_args == (
      transcription_config,
      mock_components.subscriber_factory.instance,
      mock_components.cache_factory.instance,
    )

  @pytest.mark.asyncio
  async def test_initialize_starts_all_streams(
    self,
    coordinator: RTSPStreamCoordinator,
    mock_components: ComponentBundle,
    rtsp_config: RTSPConfig,
  ) -> None:
    await coordinator.initialize()

    assert mock_components.client_factory.instance.start_calls == [rtsp_config.streams]

  @pytest.mark.asyncio
  @pytest.mark.usefixtures("mock_components")
  async def test_initialize_sets_initialized_flag(self, coordinator: RTSPStreamCoordinator) -> None:
    await coordinator.initialize()

    assert coordinator.is_initialized

  @pytest.mark.asyncio
  async def test_initialize_exposes_subscriber_manager(
    self, coordinator: RTSPStreamCoordinator, mock_components: ComponentBundle
  ) -> None:
    await coordinator.initialize()

    assert coordinator.subscriber_manager is mock_components.subscriber_factory.instance

  @pytest.mark.asyncio
  async def test_initialize_exposes_client_manager(
    self, coordinator: RTSPStreamCoordinator, mock_components: ComponentBundle
  ) -> None:
    await coordinator.initialize()

    assert coordinator.client_manager is mock_components.client_factory.instance

  @pytest.mark.asyncio
  async def test_double_initialize_is_idempotent(
    self, coordinator: RTSPStreamCoordinator, mock_components: ComponentBundle
  ) -> None:
    await coordinator.initialize()
    await coordinator.initialize()

    assert mock_components.cache_factory.call_count == 1
    assert mock_components.subscriber_factory.call_count == 1
    assert mock_components.client_factory.call_count == 1

  @pytest.mark.asyncio
  async def test_initialize_failure_cleans_up(
    self, coordinator: RTSPStreamCoordinator, mock_components: ComponentBundle
  ) -> None:
    mock_components.client_factory.instance.start_error = Exception("Connection failed")

    with pytest.raises(RuntimeError, match="RTSP initialization failed"):
      await coordinator.initialize()

    assert not coordinator.is_initialized
    assert coordinator.subscriber_manager is None
    assert coordinator.client_manager is None


class TestRTSPStreamCoordinatorShutdown:
  @pytest.mark.asyncio
  async def test_shutdown_stops_all_streams(
    self, coordinator: RTSPStreamCoordinator, mock_components: ComponentBundle
  ) -> None:
    await coordinator.initialize()
    await coordinator.shutdown()

    assert mock_components.client_factory.instance.stop_calls == 1

  @pytest.mark.asyncio
  async def test_shutdown_clears_components(self, coordinator: RTSPStreamCoordinator) -> None:
    await coordinator.initialize()
    assert coordinator.subscriber_manager is not None
    assert coordinator.client_manager is not None

    await coordinator.shutdown()

    assert coordinator.subscriber_manager is None
    assert coordinator.client_manager is None
    assert not coordinator.is_initialized

  @pytest.mark.asyncio
  async def test_shutdown_before_init_is_safe(self, coordinator: RTSPStreamCoordinator) -> None:
    await coordinator.shutdown()

    assert not coordinator.is_initialized

  @pytest.mark.asyncio
  async def test_double_shutdown_is_safe(
    self, coordinator: RTSPStreamCoordinator, mock_components: ComponentBundle
  ) -> None:
    await coordinator.initialize()
    await coordinator.shutdown()
    await coordinator.shutdown()

    assert mock_components.client_factory.instance.stop_calls == 1


class TestRTSPStreamCoordinatorStatus:
  @pytest.fixture
  def mock_components(self, monkeypatch: MonkeyPatch) -> Iterator[ComponentBundle]:
    status: RTSPManagerStatusDict = {
      "summary": {"total_created": 1, "active_streams": 1, "failed_streams": 0},
      "streams": {
        "office": {
          "url": "rtsp://camera1:554/stream",
          "reconnect_count": 0,
          "chunks_read": 0,
          "total_bytes": 0,
          "transcriptions_completed": 0,
          "transcription_errors": 0,
          "task_running": True,
          "stopped": False,
          "buffer_duration": 0.0,
          "processed_duration": 0.0,
          "available_duration": 0.0,
          "processor_active": True,
          "segments_processed": 0,
        }
      },
    }
    client_manager = FakeClientManager(
      transcription_config=TranscriptionConfig(),
      subscriber_manager=FakeSubscriberManager(set(), FakeCache()),
      cache=FakeCache(),
      status=status,
    )
    components = ComponentBundle(
      cache_factory=CacheFactory(),
      subscriber_factory=SubscriberManagerFactory(),
      client_factory=ClientManagerFactory(instance=client_manager),
    )
    monkeypatch.setattr(
      "eavesdrop.server.rtsp.coordinator.RTSPTranscriptionCache",
      components.cache_factory,
    )
    monkeypatch.setattr(
      "eavesdrop.server.rtsp.coordinator.RTSPSubscriberManager",
      components.subscriber_factory,
    )
    monkeypatch.setattr(
      "eavesdrop.server.rtsp.coordinator.RTSPClientManager",
      components.client_factory,
    )
    yield components

  @pytest.mark.asyncio
  async def test_get_status_after_init(
    self, coordinator: RTSPStreamCoordinator, mock_components: ComponentBundle
  ) -> None:
    await coordinator.initialize()

    status = coordinator.get_status()

    assert mock_components.client_factory.instance.get_stream_status_calls == 1
    assert status["summary"]["active_streams"] == 1
    assert "office" in status["streams"]

  @pytest.mark.asyncio
  async def test_stream_count_after_init(
    self, coordinator: RTSPStreamCoordinator, mock_components: ComponentBundle
  ) -> None:
    await coordinator.initialize()

    assert coordinator.stream_count == 1
    assert mock_components.client_factory.instance.get_stream_count_calls >= 1
