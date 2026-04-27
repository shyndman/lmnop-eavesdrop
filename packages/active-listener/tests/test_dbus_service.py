"""Focused DBus boundary tests for active-listener."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol, cast

import pytest
from sdbus import DbusPropertyReadOnlyError

from active_listener.app.state import (
  AppAction,
  AppActionDecision,
  ForegroundPhase,
  StartOrFinishResult,
)
from active_listener.infra.dbus import (
  DBUS_BUS_NAME,
  DBUS_OBJECT_PATH,
  ActiveListenerDbusInterface,
  DbusDuplicateInstanceError,
  NoopDbusService,
  SdbusDbusService,
)
from active_listener.recording.reducer import TextRun

requires_user_bus = pytest.mark.skipif(
  os.getenv("DBUS_SESSION_BUS_ADDRESS") in {None, ""},
  reason="requires user session bus",
)


class WritablePropertyProxy(Protocol):
  async def get_async(self) -> str: ...

  async def set_async(self, value: str) -> None: ...


class WritableBoolPropertyProxy(Protocol):
  async def get_async(self) -> bool: ...

  async def set_async(self, value: bool) -> None: ...


class EmptySignalProxy(Protocol):
  def __aiter__(self) -> AsyncIterator[None]: ...


class StringSignalProxy(Protocol):
  def __aiter__(self) -> AsyncIterator[str]: ...


class BytesSignalProxy(Protocol):
  def __aiter__(self) -> AsyncIterator[bytes]: ...


class PairSignalProxy(Protocol):
  def __aiter__(self) -> AsyncIterator[tuple[str, str]]: ...


class TranscriptionUpdatedSignalProxy(Protocol):
  def __aiter__(self) -> AsyncIterator[list[tuple[str, bool, bool]]]: ...


class PropertyDescriptor(Protocol):
  property_name: str
  property_setter_is_public: bool


class SignalDescriptor(Protocol):
  signal_name: str


class MethodDescriptor(Protocol):
  method_name: str


class IntrospectableProxy(Protocol):
  async def dbus_introspect(self) -> str: ...


class StartOrFinishProxy(Protocol):
  async def start_or_finish_recording(self) -> str: ...


class SetLlmActiveProxy(Protocol):
  async def set_llm_active(self, active: bool) -> bool: ...


@dataclass
class FakeRecordingControl:
  decisions: list[AppActionDecision]
  actions: list[AppAction] = field(default_factory=list)

  async def handle_action(self, action: AppAction) -> AppActionDecision:
    self.actions.append(action)
    return self.decisions.pop(0)


@dataclass
class FakeLlmRuntimeControl:
  llm_available: bool
  llm_active: bool
  calls: list[bool] = field(default_factory=list)

  def current_llm_available(self) -> bool:
    return self.llm_available

  def current_llm_active(self) -> bool:
    return self.llm_active

  async def set_llm_active(self, active: bool) -> bool:
    self.calls.append(active)
    if not self.llm_available:
      raise RuntimeError("llm unavailable")
    self.llm_active = active
    return self.llm_active


async def receive_next_signal(iterator: AsyncIterator[None]) -> None:
  await anext(iterator)


async def receive_next_string_signal(iterator: AsyncIterator[str]) -> str:
  return await anext(iterator)


async def receive_next_bytes_signal(iterator: AsyncIterator[bytes]) -> bytes:
  return await anext(iterator)


async def receive_next_pair_signal(iterator: AsyncIterator[tuple[str, str]]) -> tuple[str, str]:
  return await anext(iterator)


async def receive_next_transcription_update(
  iterator: AsyncIterator[list[tuple[str, bool, bool]]],
) -> list[tuple[str, bool, bool]]:
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
  llm_available_descriptor = cast(
    PropertyDescriptor,
    ActiveListenerDbusInterface.__dict__["llm_available"],
  )
  llm_active_descriptor = cast(
    PropertyDescriptor,
    ActiveListenerDbusInterface.__dict__["llm_active"],
  )
  start_or_finish_method = cast(
    MethodDescriptor,
    ActiveListenerDbusInterface.__dict__["start_or_finish_recording"],
  )
  set_llm_active_method = cast(
    MethodDescriptor,
    ActiveListenerDbusInterface.__dict__["set_llm_active"],
  )
  recording_aborted_signal = cast(
    SignalDescriptor,
    ActiveListenerDbusInterface.__dict__["recording_aborted"],
  )
  transcription_updated_signal = cast(
    SignalDescriptor,
    ActiveListenerDbusInterface.__dict__["transcription_updated"],
  )
  spectrum_updated_signal = cast(
    SignalDescriptor,
    ActiveListenerDbusInterface.__dict__["spectrum_updated"],
  )
  pipeline_failed_signal = cast(
    SignalDescriptor,
    ActiveListenerDbusInterface.__dict__["pipeline_failed"],
  )
  audio_archive_failed_signal = cast(
    SignalDescriptor,
    ActiveListenerDbusInterface.__dict__["audio_archive_failed"],
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
  assert llm_available_descriptor.property_name == "LlmAvailable"
  assert llm_available_descriptor.property_setter_is_public is False
  assert llm_active_descriptor.property_name == "LlmActive"
  assert llm_active_descriptor.property_setter_is_public is False
  assert start_or_finish_method.method_name == "StartOrFinishRecording"
  assert set_llm_active_method.method_name == "SetLlmActive"
  assert transcription_updated_signal.signal_name == "TranscriptionUpdated"
  assert spectrum_updated_signal.signal_name == "SpectrumUpdated"
  assert recording_aborted_signal.signal_name == "RecordingAborted"
  assert audio_archive_failed_signal.signal_name == "AudioArchiveFailed"
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


@pytest.mark.asyncio
async def test_noop_service_accepts_spectrum_updates() -> None:
  service = NoopDbusService()

  await service.spectrum_updated(bytes(range(50)))
  await service.audio_archive_failed("disk full")


@requires_user_bus
@pytest.mark.asyncio
async def test_property_is_read_only_over_dbus_and_empty_signals_emit() -> None:
  service = await connect_test_service_or_skip()
  proxy = ActiveListenerDbusInterface.new_proxy(DBUS_BUS_NAME, DBUS_OBJECT_PATH, bus=service.bus)
  llm_runtime_control = FakeLlmRuntimeControl(llm_available=True, llm_active=True)
  service.attach_llm_runtime_control(llm_runtime_control)

  try:
    state_proxy = cast(WritablePropertyProxy, proxy.state)
    llm_available_proxy = cast(WritableBoolPropertyProxy, proxy.llm_available)
    llm_active_proxy = cast(WritableBoolPropertyProxy, proxy.llm_active)

    assert await state_proxy.get_async() == ForegroundPhase.STARTING.value
    assert await llm_available_proxy.get_async() is True
    assert await llm_active_proxy.get_async() is True

    with pytest.raises(DbusPropertyReadOnlyError, match="State"):
      await state_proxy.set_async("idle")
    with pytest.raises(DbusPropertyReadOnlyError, match="LlmAvailable"):
      await llm_available_proxy.set_async(False)
    with pytest.raises(DbusPropertyReadOnlyError, match="LlmActive"):
      await llm_active_proxy.set_async(False)

    for state in (
      ForegroundPhase.IDLE,
      ForegroundPhase.RECORDING,
      ForegroundPhase.RECONNECTING,
      ForegroundPhase.IDLE,
    ):
      await service.set_state(state)
      assert await state_proxy.get_async() == state.value

    fatal_error_iter = cast(StringSignalProxy, proxy.fatal_error).__aiter__()
    transcription_updated_iter = cast(
      TranscriptionUpdatedSignalProxy,
      proxy.transcription_updated,
    ).__aiter__()
    spectrum_updated_iter = cast(BytesSignalProxy, proxy.spectrum_updated).__aiter__()
    audio_archive_failed_iter = cast(StringSignalProxy, proxy.audio_archive_failed).__aiter__()
    pipeline_failed_iter = cast(PairSignalProxy, proxy.pipeline_failed).__aiter__()
    reconnecting_iter = cast(EmptySignalProxy, proxy.reconnecting).__aiter__()
    reconnected_iter = cast(EmptySignalProxy, proxy.reconnected).__aiter__()
    fatal_error_task = asyncio.create_task(receive_next_string_signal(fatal_error_iter))
    transcription_updated_task = asyncio.create_task(
      receive_next_transcription_update(transcription_updated_iter)
    )
    spectrum_updated_task = asyncio.create_task(receive_next_bytes_signal(spectrum_updated_iter))
    audio_archive_failed_task = asyncio.create_task(
      receive_next_string_signal(audio_archive_failed_iter)
    )
    pipeline_failed_task = asyncio.create_task(receive_next_pair_signal(pipeline_failed_iter))
    reconnecting_task = asyncio.create_task(receive_next_signal(reconnecting_iter))
    reconnected_task = asyncio.create_task(receive_next_signal(reconnected_iter))
    await asyncio.sleep(0.05)

    await service.fatal_error("boom")
    await service.transcription_updated(
      runs=[
        TextRun(text="alpha", is_command=False, is_complete=True),
        TextRun(text="bravo", is_command=True, is_complete=True),
        TextRun(text="draft", is_command=True, is_complete=False),
      ]
    )
    await service.spectrum_updated(bytes(range(50)))
    await service.audio_archive_failed("disk full")
    await service.pipeline_failed("rewrite_with_llm", "timed out")
    await service.reconnecting()
    await service.reconnected()

    assert await asyncio.wait_for(fatal_error_task, timeout=2) == "boom"
    assert await asyncio.wait_for(transcription_updated_task, timeout=2) == [
      ("alpha", False, True),
      ("bravo", True, True),
      ("draft", True, False),
    ]
    spectrum_payload = await asyncio.wait_for(spectrum_updated_task, timeout=2)
    assert isinstance(spectrum_payload, bytes)
    assert spectrum_payload == bytes(range(50))
    assert await asyncio.wait_for(audio_archive_failed_task, timeout=2) == "disk full"
    assert await asyncio.wait_for(pipeline_failed_task, timeout=2) == (
      "rewrite_with_llm",
      "timed out",
    )
    assert await asyncio.wait_for(reconnecting_task, timeout=2) is None
    assert await asyncio.wait_for(reconnected_task, timeout=2) is None
    assert await state_proxy.get_async() == ForegroundPhase.IDLE.value
    assert await llm_available_proxy.get_async() is True
    assert await llm_active_proxy.get_async() is True
  finally:
    await service.close()


@requires_user_bus
@pytest.mark.asyncio
async def test_start_or_finish_recording_returns_locked_results() -> None:
  service = await connect_test_service_or_skip()
  proxy = ActiveListenerDbusInterface.new_proxy(DBUS_BUS_NAME, DBUS_OBJECT_PATH, bus=service.bus)
  control = FakeRecordingControl(
    decisions=[
      AppActionDecision.START_RECORDING,
      AppActionDecision.FINISH_RECORDING,
      AppActionDecision.SUPPRESS_RECONNECTING_START,
    ]
  )
  service.attach_recording_control(control)

  try:
    method_proxy = cast(StartOrFinishProxy, proxy)

    assert await method_proxy.start_or_finish_recording() == StartOrFinishResult.STARTED.value
    assert await method_proxy.start_or_finish_recording() == StartOrFinishResult.FINISHED.value
    assert await method_proxy.start_or_finish_recording() == StartOrFinishResult.IGNORED.value
    assert control.actions == [
      AppAction.START_OR_FINISH,
      AppAction.START_OR_FINISH,
      AppAction.START_OR_FINISH,
    ]
  finally:
    await service.close()


@requires_user_bus
@pytest.mark.asyncio
async def test_set_llm_active_returns_effective_state_and_updates_property() -> None:
  service = await connect_test_service_or_skip()
  proxy = ActiveListenerDbusInterface.new_proxy(DBUS_BUS_NAME, DBUS_OBJECT_PATH, bus=service.bus)
  llm_runtime_control = FakeLlmRuntimeControl(llm_available=True, llm_active=True)
  service.attach_llm_runtime_control(llm_runtime_control)

  try:
    llm_active_proxy = cast(WritableBoolPropertyProxy, proxy.llm_active)
    method_proxy = cast(SetLlmActiveProxy, proxy)

    assert await llm_active_proxy.get_async() is True
    assert await method_proxy.set_llm_active(False) is False
    assert await llm_active_proxy.get_async() is False
    assert await method_proxy.set_llm_active(False) is False
    assert await llm_active_proxy.get_async() is False
    assert await method_proxy.set_llm_active(True) is True
    assert await llm_active_proxy.get_async() is True
    assert llm_runtime_control.calls == [False, False, True]
  finally:
    await service.close()


@requires_user_bus
@pytest.mark.asyncio
async def test_set_llm_active_fails_when_llm_is_unavailable() -> None:
  service = await connect_test_service_or_skip()
  proxy = ActiveListenerDbusInterface.new_proxy(DBUS_BUS_NAME, DBUS_OBJECT_PATH, bus=service.bus)
  llm_runtime_control = FakeLlmRuntimeControl(llm_available=False, llm_active=False)
  service.attach_llm_runtime_control(llm_runtime_control)

  try:
    method_proxy = cast(SetLlmActiveProxy, proxy)

    with pytest.raises(Exception, match="llm unavailable"):
      _ = await method_proxy.set_llm_active(True)
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
    assert '<method name="StartOrFinishRecording">' in introspection_xml
    assert '<method name="SetLlmActive">' in introspection_xml
    assert '<method name="StartOrFinishRecording">' in interface_block
    assert '<method name="SetLlmActive">' in interface_block
    assert 'name="result"' in interface_block
    assert 'type="s"' in interface_block
    assert 'direction="out"' in interface_block
    assert '<property name="State" type="s" access="read">' in introspection_xml
    assert '<property name="LlmAvailable" type="b" access="read">' in introspection_xml
    assert '<property name="LlmActive" type="b" access="read">' in introspection_xml
    assert '<signal name="TranscriptionUpdated">' in introspection_xml
    assert '<arg type="a(sbb)" name="runs"/>' in interface_block
    assert '<signal name="SpectrumUpdated">' in introspection_xml
    assert '<arg type="ay"' in interface_block
    assert '<signal name="RecordingAborted">' in introspection_xml
    assert '<signal name="AudioArchiveFailed">' in introspection_xml
    assert '<signal name="PipelineFailed">' in introspection_xml
    assert '<signal name="FatalError">' in introspection_xml
    assert '<signal name="Reconnecting">' in introspection_xml
    assert '<signal name="Reconnected">' in introspection_xml
    assert interface_block.count("<signal name=") == 8
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
