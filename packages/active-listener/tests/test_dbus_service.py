"""Focused DBus boundary tests for active-listener."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from typing import Protocol, cast

import pytest
from sdbus import DbusPropertyReadOnlyError

from active_listener.app.state import ForegroundPhase
from active_listener.infra.dbus import (
  DBUS_BUS_NAME,
  DBUS_OBJECT_PATH,
  ActiveListenerDbusInterface,
  DbusDuplicateInstanceError,
  SdbusDbusService,
)

requires_user_bus = pytest.mark.skipif(
  os.getenv("DBUS_SESSION_BUS_ADDRESS") in {None, ""},
  reason="requires user session bus",
)


class WritablePropertyProxy(Protocol):
  async def get_async(self) -> str: ...

  async def set_async(self, value: str) -> None: ...


class EmptySignalProxy(Protocol):
  def __aiter__(self) -> AsyncIterator[None]: ...


class StringSignalProxy(Protocol):
  def __aiter__(self) -> AsyncIterator[str]: ...


class PairSignalProxy(Protocol):
  def __aiter__(self) -> AsyncIterator[tuple[str, str]]: ...


class PropertyDescriptor(Protocol):
  property_name: str
  property_setter_is_public: bool


class SignalDescriptor(Protocol):
  signal_name: str


class IntrospectableProxy(Protocol):
  async def dbus_introspect(self) -> str: ...


async def receive_next_signal(iterator: AsyncIterator[None]) -> None:
  await anext(iterator)


async def receive_next_string_signal(iterator: AsyncIterator[str]) -> str:
  return await anext(iterator)


async def receive_next_pair_signal(iterator: AsyncIterator[tuple[str, str]]) -> tuple[str, str]:
  return await anext(iterator)


async def connect_test_service_or_skip() -> SdbusDbusService:
  try:
    return await SdbusDbusService.connect()
  except DbusDuplicateInstanceError:
    pytest.skip("DBus name already owned by another active-listener process")


@requires_user_bus
@pytest.mark.asyncio
async def test_interface_contract_names_match_spec() -> None:
  state_descriptor = cast(PropertyDescriptor, ActiveListenerDbusInterface.__dict__["state"])
  recording_aborted_signal = cast(
    SignalDescriptor,
    ActiveListenerDbusInterface.__dict__["recording_aborted"],
  )
  pipeline_failed_signal = cast(
    SignalDescriptor,
    ActiveListenerDbusInterface.__dict__["pipeline_failed"],
  )
  fatal_error_signal = cast(
    SignalDescriptor,
    ActiveListenerDbusInterface.__dict__["fatal_error"],
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
  assert pipeline_failed_signal.signal_name == "PipelineFailed"
  assert fatal_error_signal.signal_name == "FatalError"
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
  service = await connect_test_service_or_skip()
  proxy = ActiveListenerDbusInterface.new_proxy(DBUS_BUS_NAME, DBUS_OBJECT_PATH, bus=service.bus)

  try:
    state_proxy = cast(WritablePropertyProxy, proxy.state)

    assert await state_proxy.get_async() == ForegroundPhase.STARTING.value

    with pytest.raises(DbusPropertyReadOnlyError, match="State"):
      await state_proxy.set_async("idle")

    for state in (
      ForegroundPhase.IDLE,
      ForegroundPhase.RECORDING,
      ForegroundPhase.RECONNECTING,
      ForegroundPhase.IDLE,
    ):
      await service.set_state(state)
      assert await state_proxy.get_async() == state.value

    fatal_error_iter = cast(StringSignalProxy, proxy.fatal_error).__aiter__()
    pipeline_failed_iter = cast(PairSignalProxy, proxy.pipeline_failed).__aiter__()
    reconnecting_iter = cast(EmptySignalProxy, proxy.reconnecting).__aiter__()
    reconnected_iter = cast(EmptySignalProxy, proxy.reconnected).__aiter__()
    fatal_error_task = asyncio.create_task(receive_next_string_signal(fatal_error_iter))
    pipeline_failed_task = asyncio.create_task(receive_next_pair_signal(pipeline_failed_iter))
    reconnecting_task = asyncio.create_task(receive_next_signal(reconnecting_iter))
    reconnected_task = asyncio.create_task(receive_next_signal(reconnected_iter))
    await asyncio.sleep(0.05)

    await service.fatal_error("boom")
    await service.pipeline_failed("rewrite_with_llm", "timed out")
    await service.reconnecting()
    await service.reconnected()

    assert await asyncio.wait_for(fatal_error_task, timeout=2) == "boom"
    assert await asyncio.wait_for(pipeline_failed_task, timeout=2) == (
      "rewrite_with_llm",
      "timed out",
    )
    assert await asyncio.wait_for(reconnecting_task, timeout=2) is None
    assert await asyncio.wait_for(reconnected_task, timeout=2) is None
    assert await state_proxy.get_async() == ForegroundPhase.IDLE.value
  finally:
    await service.close()


@requires_user_bus
@pytest.mark.asyncio
async def test_dbus_introspection_matches_locked_contract() -> None:
  service = await connect_test_service_or_skip()
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
    assert '<signal name="PipelineFailed">' in introspection_xml
    assert '<signal name="FatalError">' in introspection_xml
    assert '<signal name="Reconnecting">' in introspection_xml
    assert '<signal name="Reconnected">' in introspection_xml
    assert interface_block.count("<signal name=") == 5
  finally:
    await service.close()


@requires_user_bus
@pytest.mark.asyncio
async def test_duplicate_name_acquisition_maps_to_duplicate_instance_error() -> None:
  first_service = await connect_test_service_or_skip()

  try:
    with pytest.raises(DbusDuplicateInstanceError, match="already running"):
      _ = await SdbusDbusService.connect()
  finally:
    await first_service.close()
