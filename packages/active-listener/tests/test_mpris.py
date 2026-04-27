"""MPRIS boundary tests for active-listener."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import pytest
from _pytest.monkeypatch import MonkeyPatch
from structlog.stdlib import BoundLogger

from active_listener.infra.mpris import (
  PLAYBACK_STATUS_PLAYING,
  NoopMediaPlaybackController,
  PlayerctldMediaPlaybackController,
)


@dataclass
class RecordingLogger:
  debug_messages: list[str] = field(default_factory=list)
  exception_messages: list[str] = field(default_factory=list)

  def debug(self, event: str, **kwargs: object) -> None:
    _ = kwargs
    self.debug_messages.append(event)

  def exception(self, event: str, **kwargs: object) -> None:
    _ = kwargs
    self.exception_messages.append(event)


@dataclass
class FakeBus:
  close_calls: int = 0

  def close(self) -> None:
    self.close_calls += 1


@dataclass
class FakePlayer:
  playback_status: AwaitableValue[str]
  pause_calls: int = 0
  play_calls: int = 0
  pause_error: Exception | None = None

  async def pause(self) -> None:
    self.pause_calls += 1
    if self.pause_error is not None:
      raise self.pause_error

  async def play(self) -> None:
    self.play_calls += 1


@dataclass(frozen=True)
class AwaitableValue[T]:
  value: T

  def __await__(self):
    async def _resolve() -> T:
      return self.value

    return _resolve().__await__()


@pytest.mark.asyncio
async def test_noop_controller_never_schedules_resume() -> None:
  controller = NoopMediaPlaybackController()

  assert await controller.pause_if_playing() is False
  assert await controller.resume() is None


@pytest.mark.asyncio
async def test_pause_if_playing_pauses_active_player_and_closes_bus(
  monkeypatch: MonkeyPatch,
) -> None:
  logger = RecordingLogger()
  bus = FakeBus()
  player = FakePlayer(playback_status=AwaitableValue(PLAYBACK_STATUS_PLAYING))

  controller = PlayerctldMediaPlaybackController(logger=cast(BoundLogger, cast(object, logger)))
  monkeypatch.setattr(controller, "_open_player", lambda: (bus, player))

  assert await controller.pause_if_playing() is True
  assert player.pause_calls == 1
  assert bus.close_calls == 1
  assert logger.debug_messages == ["playerctld paused active player"]


@pytest.mark.asyncio
async def test_pause_if_playing_skips_pause_for_non_playing_player(
  monkeypatch: MonkeyPatch,
) -> None:
  logger = RecordingLogger()
  bus = FakeBus()
  player = FakePlayer(playback_status=AwaitableValue("Paused"))

  controller = PlayerctldMediaPlaybackController(logger=cast(BoundLogger, cast(object, logger)))
  monkeypatch.setattr(controller, "_open_player", lambda: (bus, player))

  assert await controller.pause_if_playing() is False
  assert player.pause_calls == 0
  assert bus.close_calls == 1
  assert logger.debug_messages == []


@pytest.mark.asyncio
async def test_pause_if_playing_still_returns_true_when_pause_fails(
  monkeypatch: MonkeyPatch,
) -> None:
  logger = RecordingLogger()
  bus = FakeBus()
  player = FakePlayer(
    playback_status=AwaitableValue(PLAYBACK_STATUS_PLAYING),
    pause_error=RuntimeError("pause failed"),
  )

  controller = PlayerctldMediaPlaybackController(logger=cast(BoundLogger, cast(object, logger)))
  monkeypatch.setattr(controller, "_open_player", lambda: (bus, player))

  assert await controller.pause_if_playing() is True
  assert player.pause_calls == 1
  assert bus.close_calls == 1
  assert logger.exception_messages == ["playerctld pause failed"]


@pytest.mark.asyncio
async def test_resume_plays_player_and_closes_bus(monkeypatch: MonkeyPatch) -> None:
  logger = RecordingLogger()
  bus = FakeBus()
  player = FakePlayer(playback_status=AwaitableValue("Paused"))

  controller = PlayerctldMediaPlaybackController(logger=cast(BoundLogger, cast(object, logger)))
  monkeypatch.setattr(controller, "_open_player", lambda: (bus, player))

  await controller.resume()

  assert player.play_calls == 1
  assert bus.close_calls == 1
  assert logger.debug_messages == ["playerctld resumed prior player"]
