"""Focused DBus boundary tests for active-listener."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from typing import Protocol, cast

import pytest
from sdbus import DbusPropertyReadOnlyError

from active_listener.dbus_service import (
  DBUS_BUS_NAME,
  DBUS_OBJECT_PATH,
  ActiveListenerDbusInterface,
  DbusDuplicateInstanceError,
  SdbusDbusService,
)
from active_listener.state import ForegroundPhase

requires_user_bus = pytest.mark.skipif(
  os.getenv("DBUS_SESSION_BUS_ADDRESS") in {None, ""},
  reason="requires user session bus",
)


class WritablePropertyProxy(Protocol):
  async def set_async(self, value: str) -> None: ...


class EmptySignalProxy(Protocol):
  def __aiter__(self) -> AsyncIterator[None]: ...


class PropertyDescriptor(Protocol):
  property_name: str
  property_setter_is_public: bool


class SignalDescriptor(Protocol):
  signal_name: str


class IntrospectableProxy(Protocol):
  async def dbus_introspect(self) -> str: ...


async def receive_next_signal(iterator: AsyncIterator[None]) -> None:
  await anext(iterator)


@requires_user_bus
@pytest.mark.asyncio
async def test_interface_contract_names_match_spec() -> None:
  state_descriptor = cast(PropertyDescriptor, ActiveListenerDbusInterface.__dict__["state"])
  recording_aborted_signal = cast(
    SignalDescriptor,
    ActiveListenerDbusInterface.__dict__["recording_aborted"],
  )
  reconnecting_signal = cast(
    SignalDescriptor,
    ActiveListenerDbusInterface.__dict__["reconnecting"],
  )
  reconnected_signal = cast(
    SignalDescriptor,
    ActiveListenerDbusInterface.__dict__["reconnected"],
  )

  assert state_descriptor.property_name == "State"
  assert state_descriptor.property_setter_is_public is False
  assert recording_aborted_signal.signal_name == "RecordingAborted"
  assert reconnecting_signal.signal_name == "Reconnecting"
  assert reconnected_signal.signal_name == "Reconnected"


@pytest.mark.asyncio
async def test_interface_state_is_locally_mutable() -> None:
  interface = ActiveListenerDbusInterface(initial_state=ForegroundPhase.STARTING)

  assert await interface.current_state() == "starting"

  await interface.set_state(ForegroundPhase.IDLE)

  assert await interface.current_state() == "idle"


@requires_user_bus
@pytest.mark.asyncio
async def test_property_is_read_only_over_dbus_and_empty_signals_emit() -> None:
  service = await SdbusDbusService.connect()
  proxy = ActiveListenerDbusInterface.new_proxy(DBUS_BUS_NAME, DBUS_OBJECT_PATH, bus=service.bus)

  try:
    with pytest.raises(DbusPropertyReadOnlyError, match="State"):
      await cast(WritablePropertyProxy, proxy.state).set_async("idle")

    reconnecting_iter = cast(EmptySignalProxy, proxy.reconnecting).__aiter__()
    reconnected_iter = cast(EmptySignalProxy, proxy.reconnected).__aiter__()
    reconnecting_task = asyncio.create_task(receive_next_signal(reconnecting_iter))
    reconnected_task = asyncio.create_task(receive_next_signal(reconnected_iter))
    await asyncio.sleep(0.05)

    await service.reconnecting()
    await service.reconnected()

    assert await asyncio.wait_for(reconnecting_task, timeout=2) is None
    assert await asyncio.wait_for(reconnected_task, timeout=2) is None
  finally:
    await service.close()


@requires_user_bus
@pytest.mark.asyncio
async def test_dbus_introspection_matches_locked_contract() -> None:
  service = await SdbusDbusService.connect()
  proxy = ActiveListenerDbusInterface.new_proxy(DBUS_BUS_NAME, DBUS_OBJECT_PATH, bus=service.bus)

  try:
    introspection_xml = await cast(IntrospectableProxy, cast(object, proxy)).dbus_introspect()
    interface_block = introspection_xml.split(
      '<interface name="ca.lmnop.Eavesdrop.ActiveListener1">',
      maxsplit=1,
    )[1].split("</interface>", maxsplit=1)[0]

    assert DBUS_BUS_NAME == "ca.lmnop.Eavesdrop.ActiveListener"
    assert DBUS_OBJECT_PATH == "/ca/lmnop/Eavesdrop/ActiveListener"
    assert 'interface name="ca.lmnop.Eavesdrop.ActiveListener1"' in introspection_xml
    assert '<property name="State" type="s" access="read">' in introspection_xml
    assert '<signal name="RecordingAborted">' in introspection_xml
    assert '<signal name="Reconnecting">' in introspection_xml
    assert '<signal name="Reconnected">' in introspection_xml
    assert interface_block.count("<signal name=") == 3
  finally:
    await service.close()


@requires_user_bus
@pytest.mark.asyncio
async def test_duplicate_name_acquisition_maps_to_duplicate_instance_error() -> None:
  first_service = await SdbusDbusService.connect()

  try:
    with pytest.raises(DbusDuplicateInstanceError, match="already running"):
      _ = await SdbusDbusService.connect()
  finally:
    await first_service.close()
