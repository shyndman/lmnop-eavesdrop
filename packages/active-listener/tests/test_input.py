"""Input-boundary contract tests for active-listener."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest
from evdev.events import InputEvent

from active_listener.app.state import KeyboardAction
from active_listener.infra.keyboard import (
  EvdevKeyboard,
  KeyboardResolutionError,
  action_from_event,
  resolve_keyboard,
)


@dataclass
class FakeDevice:
  """Evdev-device stand-in used by input-boundary tests."""

  path: str
  name: str
  events: list[InputEvent]
  grab_calls: int = 0
  ungrab_calls: int = 0
  close_calls: int = 0
  grabbed: bool = False
  kernel_events: list[tuple[int, int]] = field(default_factory=list)

  async def async_read_loop(self) -> AsyncIterator[InputEvent]:
    for event in self.events:
      if not self.grabbed:
        self.kernel_events.append((event.code, event.value))
      yield event

  def grab(self) -> None:
    self.grab_calls += 1
    self.grabbed = True

  def ungrab(self) -> None:
    self.ungrab_calls += 1
    self.grabbed = False

  def close(self) -> None:
    self.close_calls += 1


def _key_event(code: int, value: int) -> InputEvent:
  return InputEvent(0, 0, 1, code, value)


def test_action_from_event_filters_non_key_down_events() -> None:
  assert action_from_event(_key_event(58, 0)) is None
  assert action_from_event(_key_event(58, 2)) is None
  assert action_from_event(InputEvent(0, 0, 0, 58, 1)) is None


def test_action_from_event_maps_caps_lock_and_escape() -> None:
  assert action_from_event(_key_event(58, 1)) is KeyboardAction.START_OR_FINISH
  assert action_from_event(_key_event(1, 1)) is KeyboardAction.CANCEL


@pytest.mark.asyncio
async def test_evdev_keyboard_actions_only_emit_normalized_hotkeys() -> None:
  keyboard = EvdevKeyboard(
    device=FakeDevice(
      path="/dev/input/event1",
      name="Exact Keyboard",
      events=[
        _key_event(58, 1),
        _key_event(58, 0),
        _key_event(30, 1),
        _key_event(1, 1),
      ],
    )
  )

  actions = [action async for action in keyboard.actions()]

  assert actions == [KeyboardAction.START_OR_FINISH, KeyboardAction.CANCEL]


@pytest.mark.asyncio
async def test_grab_transitions_do_not_split_hotkey_press_release_pairs() -> None:
  device = FakeDevice(
    path="/dev/input/event1",
    name="Exact Keyboard",
    events=[
      _key_event(58, 1),
      _key_event(58, 0),
      _key_event(1, 1),
      _key_event(1, 0),
    ],
  )
  keyboard = EvdevKeyboard(device=device)

  action_iterator = keyboard.actions().__aiter__()

  assert await action_iterator.__anext__() is KeyboardAction.START_OR_FINISH
  keyboard.grab()

  assert await action_iterator.__anext__() is KeyboardAction.CANCEL
  keyboard.ungrab()

  with pytest.raises(StopAsyncIteration):
    _ = await action_iterator.__anext__()

  assert device.kernel_events == [
    (58, 1),
    (58, 0),
  ]


def test_evdev_keyboard_grab_and_ungrab_delegate_to_device() -> None:
  device = FakeDevice(path="/dev/input/event1", name="Exact Keyboard", events=[])
  keyboard = EvdevKeyboard(device=device)

  keyboard.grab()
  keyboard.ungrab()
  keyboard.close()

  assert device.grab_calls == 1
  assert device.ungrab_calls == 1
  assert device.close_calls == 1


@pytest.mark.asyncio
async def test_recording_grab_release_is_idempotent() -> None:
  device = FakeDevice(path="/dev/input/event1", name="Exact Keyboard", events=[])
  keyboard = EvdevKeyboard(device=device)

  async with keyboard.recording_grab() as release:
    assert device.grab_calls == 1
    assert device.grabbed is True

    release()
    release()

    assert device.ungrab_calls == 1
    assert device.grabbed is False

  assert device.grab_calls == 1
  assert device.ungrab_calls == 1


@pytest.mark.asyncio
async def test_close_is_safe_after_recording_grab_context_exits() -> None:
  device = FakeDevice(path="/dev/input/event1", name="Exact Keyboard", events=[])
  keyboard = EvdevKeyboard(device=device)

  async with keyboard.recording_grab():
    assert device.grabbed is True

  keyboard.close()

  assert device.ungrab_calls == 1
  assert device.close_calls == 1


def test_resolve_keyboard_requires_exact_single_match() -> None:
  devices = {
    "/dev/input/event1": FakeDevice("/dev/input/event1", "Office Keyboard", []),
    "/dev/input/event2": FakeDevice("/dev/input/event2", "Office Keyboard", []),
    "/dev/input/event3": FakeDevice("/dev/input/event3", "Other Keyboard", []),
  }

  def factory(path: str) -> FakeDevice:
    return devices[path]

  with pytest.raises(KeyboardResolutionError, match="Multiple keyboards matched"):
    _ = resolve_keyboard(
      "Office Keyboard",
      device_paths=devices.keys(),
      device_factory=factory,
    )

  assert devices["/dev/input/event1"].close_calls == 1
  assert devices["/dev/input/event2"].close_calls == 1
  assert devices["/dev/input/event3"].close_calls == 1


def test_resolve_keyboard_raises_when_no_exact_match_exists() -> None:
  devices = {"/dev/input/event3": FakeDevice("/dev/input/event3", "Other Keyboard", [])}

  with pytest.raises(KeyboardResolutionError, match="No keyboard matched"):
    _ = resolve_keyboard(
      "Office Keyboard",
      device_paths=devices.keys(),
      device_factory=lambda path: devices[path],
    )

  assert devices["/dev/input/event3"].close_calls == 1


def test_resolve_keyboard_returns_exact_match() -> None:
  device = FakeDevice("/dev/input/event7", "Office Keyboard", [])

  resolved = resolve_keyboard(
    "Office Keyboard",
    device_paths=[device.path],
    device_factory=lambda _path: device,
  )

  assert isinstance(resolved, EvdevKeyboard)
  assert resolved.device is device
