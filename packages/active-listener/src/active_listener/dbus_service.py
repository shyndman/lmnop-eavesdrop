"""DBus app-state publication boundary for active-listener."""

from __future__ import annotations

from dataclasses import dataclass
from types import new_class
from typing import TYPE_CHECKING, Protocol, cast

from sdbus import (
  DbusInterfaceCommonAsync,
  DbusPropertyEmitsChangeFlag,
  SdBus,
  dbus_property_async,
  dbus_signal_async,
  request_default_bus_name_async,
  sd_bus_open_user,
  set_default_bus,
)
from sdbus.exceptions import SdBusRequestNameExistsError

from active_listener.state import ForegroundPhase

DBUS_BUS_NAME = "ca.lmnop.Eavesdrop.ActiveListener"
DBUS_OBJECT_PATH = "/ca/lmnop/Eavesdrop/ActiveListener"
DBUS_INTERFACE_NAME = "ca.lmnop.Eavesdrop.ActiveListener1"


class DbusExportHandle(Protocol):
  def stop(self) -> None: ...


class DbusSignalEmitter(Protocol):
  def emit(self, payload: str | None) -> None: ...


class AppStateService(Protocol):
  async def set_state(self, state: ForegroundPhase) -> None: ...

  async def recording_aborted(self, reason: str) -> None: ...

  async def fatal_error(self, reason: str) -> None: ...

  async def reconnecting(self) -> None: ...

  async def reconnected(self) -> None: ...

  async def close(self) -> None: ...


class DbusServiceError(RuntimeError):
  """Raised when DBus setup fails."""


class DbusDuplicateInstanceError(DbusServiceError):
  """Raised when another process already owns the DBus name."""


if TYPE_CHECKING:

  class ActiveListenerDbusInterface:
    _state: str = ""
    state: object = object()
    recording_aborted: object = object()
    fatal_error: object = object()
    reconnecting: object = object()
    reconnected: object = object()

    def __init__(self, initial_state: ForegroundPhase) -> None:
      _ = initial_state
      raise NotImplementedError

    async def set_state(self, _state: ForegroundPhase) -> None:
      raise NotImplementedError

    async def current_state(self) -> str:
      raise NotImplementedError

    def export_to_dbus(self, object_path: str, bus: SdBus | None = None) -> DbusExportHandle:
      _ = object_path
      _ = bus
      raise NotImplementedError

    @classmethod
    def new_proxy(
      cls,
      service_name: str,
      object_path: str,
      bus: SdBus | None = None,
    ) -> ActiveListenerDbusInterface:
      _ = service_name
      _ = object_path
      _ = bus
      raise NotImplementedError
else:

  def _build_active_listener_dbus_interface() -> type[DbusInterfaceCommonAsync]:
    def define_namespace(namespace: dict[str, object]) -> None:
      def __init__(self, initial_state: ForegroundPhase) -> None:
        DbusInterfaceCommonAsync.__init__(self)
        self._state = initial_state.value

      @dbus_property_async(
        property_signature="s",
        flags=DbusPropertyEmitsChangeFlag,
        property_name="State",
      )
      def state(self) -> str:
        return self._state

      def set_local_state(self, value: str) -> None:
        self._state = value

      state.setter_private(set_local_state)

      @dbus_signal_async(
        signal_signature="s",
        signal_args_names=("reason",),
        signal_name="RecordingAborted",
      )
      def recording_aborted(self) -> str:
        raise NotImplementedError

      @dbus_signal_async(
        signal_signature="s",
        signal_args_names=("reason",),
        signal_name="FatalError",
      )
      def fatal_error(self) -> str:
        raise NotImplementedError

      @dbus_signal_async(signal_signature="", signal_name="Reconnecting")
      def reconnecting(self) -> None:
        raise NotImplementedError

      @dbus_signal_async(signal_signature="", signal_name="Reconnected")
      def reconnected(self) -> None:
        raise NotImplementedError

      async def set_state(self, state: ForegroundPhase) -> None:
        if state.value == self._state:
          return
        set_local_state(self, state.value)
        self.properties_changed.emit(
          (
            DBUS_INTERFACE_NAME,
            {
              "State": (
                "s",
                state.value,
              )
            },
            [],
          )
        )

      async def current_state(self) -> str:
        return self._state

      namespace["__init__"] = __init__
      namespace["state"] = state
      namespace["recording_aborted"] = recording_aborted
      namespace["fatal_error"] = fatal_error
      namespace["reconnecting"] = reconnecting
      namespace["reconnected"] = reconnected
      namespace["set_state"] = set_state
      namespace["current_state"] = current_state

    return new_class(
      "ActiveListenerDbusInterface",
      (DbusInterfaceCommonAsync,),
      {"interface_name": DBUS_INTERFACE_NAME},
      define_namespace,
    )

  ActiveListenerDbusInterface = _build_active_listener_dbus_interface()


@dataclass
class NoopDbusService:
  async def set_state(self, state: ForegroundPhase) -> None:
    _ = state

  async def recording_aborted(self, reason: str) -> None:
    _ = reason

  async def fatal_error(self, reason: str) -> None:
    _ = reason

  async def reconnecting(self) -> None:
    return None

  async def reconnected(self) -> None:
    return None

  async def close(self) -> None:
    return None


@dataclass
class SdbusDbusService:
  bus: SdBus
  interface: ActiveListenerDbusInterface
  export_handle: DbusExportHandle

  @classmethod
  async def connect(cls) -> SdbusDbusService:
    try:
      bus = sd_bus_open_user()
      set_default_bus(bus)
      await request_default_bus_name_async(DBUS_BUS_NAME)
    except SdBusRequestNameExistsError as exc:
      raise DbusDuplicateInstanceError(
        "another active-listener instance is already running"
      ) from exc
    except Exception as exc:
      raise DbusServiceError(str(exc)) from exc

    interface = ActiveListenerDbusInterface(initial_state=ForegroundPhase.STARTING)

    try:
      export_handle = interface.export_to_dbus(DBUS_OBJECT_PATH, bus)
    except Exception:
      bus.close()
      raise

    return cls(bus=bus, interface=interface, export_handle=export_handle)

  async def set_state(self, state: ForegroundPhase) -> None:
    await self.interface.set_state(state)

  async def recording_aborted(self, reason: str) -> None:
    cast(DbusSignalEmitter, self.interface.recording_aborted).emit(reason)

  async def fatal_error(self, reason: str) -> None:
    cast(DbusSignalEmitter, self.interface.fatal_error).emit(reason)

  async def reconnecting(self) -> None:
    cast(DbusSignalEmitter, self.interface.reconnecting).emit(None)

  async def reconnected(self) -> None:
    cast(DbusSignalEmitter, self.interface.reconnected).emit(None)

  async def close(self) -> None:
    self.export_handle.stop()
    self.bus.close()
