"""Evdev keyboard discovery and normalized hotkey input handling."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

import evdev
from evdev.events import InputEvent

from active_listener.state import KeyboardAction


class KeyboardInput(Protocol):
  """Protocol for the active-listener keyboard boundary."""

  def actions(self) -> AsyncIterator[KeyboardAction]:
    """Yield normalized hotkey actions from the workstation."""
    ...

  def grab(self) -> None:
    """Grab the keyboard device for exclusive foreground ownership."""
    ...

  def ungrab(self) -> None:
    """Release the keyboard device after foreground ownership ends."""
    ...

  def close(self) -> None:
    """Close any underlying workstation resources."""
    ...


class KeyboardResolutionError(RuntimeError):
  """Raised when the configured keyboard cannot be resolved exactly once."""


class SupportsEvdevInputDevice(Protocol):
  """Minimal evdev device API required by the input boundary."""

  path: str
  name: str

  def async_read_loop(self) -> AsyncIterator[InputEvent]:
    """Yield raw evdev events from the device."""
    ...

  def grab(self) -> None:
    """Grab the underlying evdev device."""
    ...

  def ungrab(self) -> None:
    """Release the underlying evdev device grab."""
    ...

  def close(self) -> None:
    """Close the underlying evdev device."""
    ...


@dataclass
class EvdevKeyboard:
  """Concrete keyboard boundary backed by one evdev device.

  :param device: Resolved evdev input device.
  :type device: SupportsEvdevInputDevice
  """

  device: SupportsEvdevInputDevice

  async def actions(self) -> AsyncIterator[KeyboardAction]:
    """Yield normalized key-down actions from the evdev stream.

    :returns: Async iterator of normalized hotkey actions.
    :rtype: AsyncIterator[KeyboardAction]
    """

    async for event in self.device.async_read_loop():
      action = action_from_event(event)
      if action is not None:
        yield action

  def grab(self) -> None:
    """Grab the concrete keyboard device.

    :returns: None
    :rtype: None
    """

    self.device.grab()

  def ungrab(self) -> None:
    """Release the concrete keyboard device grab.

    :returns: None
    :rtype: None
    """

    self.device.ungrab()

  def close(self) -> None:
    """Close the concrete keyboard device.

    :returns: None
    :rtype: None
    """

    self.device.close()


def action_from_event(event: InputEvent) -> KeyboardAction | None:
  """Translate one raw evdev event into a normalized hotkey action.

  :param event: Raw evdev input event.
  :type event: InputEvent
  :returns: Normalized hotkey action, or ``None`` for irrelevant events.
  :rtype: KeyboardAction | None
  """

  if event.type != evdev.ecodes.EV_KEY or event.value != 1:
    return None

  if event.code == evdev.ecodes.KEY_CAPSLOCK:
    return KeyboardAction.START_OR_FINISH
  if event.code == evdev.ecodes.KEY_ESC:
    return KeyboardAction.CANCEL
  return None


def resolve_keyboard(
  keyboard_name: str,
  *,
  device_paths: Iterable[str] | None = None,
  device_factory: Callable[[str], SupportsEvdevInputDevice] = evdev.InputDevice,
) -> KeyboardInput:
  """Resolve one readable keyboard by exact evdev device name.

  :param keyboard_name: Exact ``device.name`` value required for startup.
  :type keyboard_name: str
  :param device_paths: Optional iterable of event-device paths to probe.
  :type device_paths: Iterable[str] | None
  :param device_factory: Factory for opening evdev input devices.
  :type device_factory: Callable[[str], SupportsEvdevInputDevice]
  :returns: Ready keyboard boundary for the exact resolved device.
  :rtype: KeyboardInput
  :raises KeyboardResolutionError: If zero or multiple readable devices match.
  """

  listed_paths = evdev.list_devices() if device_paths is None else device_paths  # pyright: ignore[reportUnknownMemberType]
  candidate_paths = list(listed_paths)
  matches: list[SupportsEvdevInputDevice] = []

  for path in candidate_paths:
    try:
      device = device_factory(path)
    except OSError:
      continue

    if device.name == keyboard_name:
      matches.append(device)
      continue

    device.close()

  if not matches:
    raise KeyboardResolutionError(f"No keyboard matched exact device name: {keyboard_name}")

  if len(matches) > 1:
    matched_paths = ", ".join(sorted(device.path for device in matches))
    for device in matches:
      device.close()
    raise KeyboardResolutionError(
      f"Multiple keyboards matched exact device name {keyboard_name!r}: {matched_paths}"
    )

  return EvdevKeyboard(device=matches[0])
