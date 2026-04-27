"""MPRIS/playerctld media playback boundary for active-listener."""

from __future__ import annotations

from dataclasses import dataclass

from sdbus import (
  DbusInterfaceCommonAsync,
  SdBus,
  dbus_method_async,
  dbus_property_async,
  sd_bus_open_user,
)
from structlog.stdlib import BoundLogger

PLAYERCTLD_BUS_NAME = "org.mpris.MediaPlayer2.playerctld"
PLAYERCTLD_OBJECT_PATH = "/org/mpris/MediaPlayer2"
PLAYER_INTERFACE_NAME = "org.mpris.MediaPlayer2.Player"
PLAYBACK_STATUS_PLAYING = "Playing"


class PlayerctldPlayerInterface(DbusInterfaceCommonAsync, interface_name=PLAYER_INTERFACE_NAME):
  @dbus_property_async(property_signature="s", property_name="PlaybackStatus")
  def playback_status(self) -> str:
    raise NotImplementedError

  @dbus_method_async(method_name="Pause")
  async def pause(self) -> None:
    raise NotImplementedError

  @dbus_method_async(method_name="Play")
  async def play(self) -> None:
    raise NotImplementedError


@dataclass
class NoopMediaPlaybackController:
  async def pause_if_playing(self) -> bool:
    return False

  async def resume(self) -> None:
    return None


@dataclass
class PlayerctldMediaPlaybackController:
  logger: BoundLogger

  async def pause_if_playing(self) -> bool:
    bus, player = self._open_player()
    try:
      was_playing_before_recording = await player.playback_status == PLAYBACK_STATUS_PLAYING
      if not was_playing_before_recording:
        return False

      try:
        await player.pause()
      except Exception:
        self.logger.exception("playerctld pause failed")
      else:
        self.logger.debug("playerctld paused active player")
      return True
    finally:
      bus.close()

  async def resume(self) -> None:
    bus, player = self._open_player()
    try:
      await player.play()
      self.logger.debug("playerctld resumed prior player")
    finally:
      bus.close()

  def _open_player(self) -> tuple[SdBus, PlayerctldPlayerInterface]:
    bus = sd_bus_open_user()
    player = PlayerctldPlayerInterface.new_proxy(
      PLAYERCTLD_BUS_NAME,
      PLAYERCTLD_OBJECT_PATH,
      bus=bus,
    )
    return bus, player
