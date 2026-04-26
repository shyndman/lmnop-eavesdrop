"""Evdev keyboard discovery and normalized hotkey input handling."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from enum import StrEnum
from types import TracebackType
from typing import Protocol

import evdev
from evdev.events import InputEvent
from typing_extensions import override

_CONTROL_KEY_CODES = frozenset({evdev.ecodes.KEY_CAPSLOCK, evdev.ecodes.KEY_ESC})
RecordingGrabRelease = Callable[[], None]


class KeyboardEventKind(StrEnum):
  """Low-level keyboard control events consumed by the service."""

  CAPSLOCK_DOWN = "capslock_down"
  CAPSLOCK_UP = "capslock_up"
  ESCAPE_DOWN = "escape_down"


@dataclass(frozen=True)
class KeyboardControlEvent:
  """One low-level keyboard control event with local receive timing."""

  kind: KeyboardEventKind
  received_monotonic_s: float


class KeyboardInput(Protocol):
  """Protocol for the active-listener keyboard boundary."""

  def events(self) -> AsyncIterator[KeyboardControlEvent]:
    """Yield low-level keyboard control events from the workstation."""
    ...

  def grab(self) -> None:
    """Grab the keyboard device for exclusive foreground ownership."""
    ...

  def ungrab(self) -> None:
    """Release the keyboard device after foreground ownership ends."""

  def recording_grab(self) -> AbstractAsyncContextManager[RecordingGrabRelease]:
    """Own one recording-scoped keyboard grab with guaranteed cleanup.

    Callers use this when they need exclusive foreground ownership for the
    full recording lifecycle instead of manually pairing ``grab()`` and
    ``ungrab()`` across multiple branches. The context manager acquires the
    grab on entry, yields an idempotent release callback for early handoff,
    and releases again on exit so exceptions do not strand the workstation in
    an exclusive-grab state.

    :returns: Async context manager yielding an idempotent release callback.
    :rtype: AbstractAsyncContextManager[RecordingGrabRelease]
    """
    ...
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

  This boundary owns two separate concerns that are easy to conflate during
  refactors. The first is policy: active-listener grabs the workstation
  keyboard while dictation is active so only the app consumes foreground
  hotkeys. The second is timing: evdev grab transitions must not split the
  press and release halves of Caps Lock or Escape across the kernel-visible
  boundary, or the desktop can observe an impossible key state.

  ``recording_grab()`` exists to make the policy side structurally safe. A
  leaked grab is operationally dangerous on a workstation because it can leave
  the desktop without normal keyboard input while the process is already in a
  failure path. ``grab()`` and ``ungrab()`` still implement the low-level
  deferred transition logic that keeps key press/release pairs balanced.

  :param device: Resolved evdev input device.
  :type device: SupportsEvdevInputDevice
  """

  device: SupportsEvdevInputDevice
  _grabbed: bool = False
  _pending_grab: bool = False
  _pending_ungrab: bool = False
  _pressed_hotkeys: set[int] = field(default_factory=set)

  async def events(self) -> AsyncIterator[KeyboardControlEvent]:
    """Yield low-level control events from the evdev stream.

    :returns: Async iterator of low-level keyboard control events.
    :rtype: AsyncIterator[KeyboardControlEvent]
    """

    async for event in self.device.async_read_loop():
      self._update_hotkey_state(event)
      self._apply_pending_transition_if_ready()
      control_event = control_event_from_input_event(event)
      if control_event is not None:
        yield control_event

  def grab(self) -> None:
    """Grab the concrete keyboard device.

    :returns: None
    :rtype: None
    """

    if self._grabbed or self._pending_grab:
      return

    if self._pressed_hotkeys:
      self._pending_grab = True
      self._pending_ungrab = False
      return

    self.device.grab()
    self._grabbed = True

  def ungrab(self) -> None:
    """Release the concrete keyboard device grab.

    :returns: None
    :rtype: None
    """

    if (not self._grabbed and not self._pending_grab) or self._pending_ungrab:
      return

    if self._pressed_hotkeys:
      self._pending_ungrab = True
      self._pending_grab = False
      return

    if self._grabbed:
      self.device.ungrab()
      self._grabbed = False

  def recording_grab(self) -> AbstractAsyncContextManager[RecordingGrabRelease]:
    """Acquire one recording-scoped foreground grab.

    The release callback returned by this context manager is intentionally
    idempotent. Active-listener uses that to drop exclusive keyboard ownership
    before awaited work like stopping the stream or flushing text, while still
    relying on ``__aexit__`` to release again if a later refactor inserts new
    code or an exception unwinds the stack early.

    This guards against Python-level failures and branch-ordering mistakes. It
    cannot protect against termination modes that bypass Python cleanup
    entirely, such as ``SIGKILL`` or sudden power loss.

    :returns: Async context manager yielding an idempotent release callback.
    :rtype: AbstractAsyncContextManager[RecordingGrabRelease]
    """

    return _RecordingGrab(keyboard=self)

  def close(self) -> None:
    """Close the concrete keyboard device.

    :returns: None
    :rtype: None
    """

    if self._grabbed:
      self.device.ungrab()
      self._grabbed = False

    self.device.close()

  def _update_hotkey_state(self, event: InputEvent) -> None:
    if event.type != evdev.ecodes.EV_KEY or event.code not in _CONTROL_KEY_CODES:
      return

    if event.value == 1:
      self._pressed_hotkeys.add(event.code)
      return

    if event.value == 0:
      self._pressed_hotkeys.discard(event.code)

  def _apply_pending_transition_if_ready(self) -> None:
    if self._pressed_hotkeys:
      return

    if self._pending_ungrab and self._grabbed:
      self.device.ungrab()
      self._grabbed = False
      self._pending_ungrab = False
      return

    if self._pending_grab and not self._grabbed:
      self.device.grab()
      self._grabbed = True
      self._pending_grab = False


@dataclass
class _RecordingGrab(AbstractAsyncContextManager[RecordingGrabRelease]):
  """Async context manager that owns one foreground keyboard grab.

  The active-listener service records across multiple awaited operations and
  exit branches. If those branches rely on a perfectly ordered manual
  ``grab()`` / ``ungrab()`` pairing, the next exception or refactor can leave
  the process holding the workstation keyboard exclusively. On a developer
  machine that is more than a minor resource leak: it can strand the operator
  without reliable local input while the service is already misbehaving.

  This helper makes release callable early and safely repeatable. Callers can
  release before slow or failure-prone work, and ``__aexit__`` still runs the
  same release path in ``finally``. The protection boundary is Python cleanup;
  it does not cover process death modes that skip ``__aexit__`` entirely.

  :param keyboard: Concrete keyboard boundary that owns the evdev device.
  :type keyboard: EvdevKeyboard
  """

  keyboard: EvdevKeyboard
  _released: bool = False

  @override
  async def __aenter__(self) -> RecordingGrabRelease:
    """Acquire the grab and expose an idempotent release callback.

    :returns: Callback that releases the foreground grab at most once.
    :rtype: RecordingGrabRelease
    """

    self.keyboard.grab()
    return self.release

  @override
  async def __aexit__(
    self,
    exc_type: type[BaseException] | None,
    exc_value: BaseException | None,
    traceback: TracebackType | None,
  ) -> None:
    """Release the grab while leaving exception propagation untouched.

    :param exc_type: Exception type raised inside the context, if any.
    :type exc_type: type[BaseException] | None
    :param exc_value: Exception instance raised inside the context, if any.
    :type exc_value: BaseException | None
    :param traceback: Traceback from the context body, if any.
    :type traceback: TracebackType | None
    :returns: None
    :rtype: None
    """

    self.release()

  def release(self) -> None:
    """Release the owned foreground grab at most once.

    :returns: None
    :rtype: None
    """

    if self._released:
      return

    self._released = True
    self.keyboard.ungrab()


def control_event_from_input_event(event: InputEvent) -> KeyboardControlEvent | None:
  """Translate one raw evdev event into a low-level control event.

  :param event: Raw evdev input event.
  :type event: InputEvent
  :returns: Keyboard control event, or ``None`` for irrelevant events.
  :rtype: KeyboardControlEvent | None
  """

  if event.type != evdev.ecodes.EV_KEY:
    return None

  if event.code == evdev.ecodes.KEY_CAPSLOCK:
    if event.value == 1:
      kind = KeyboardEventKind.CAPSLOCK_DOWN
    elif event.value == 0:
      kind = KeyboardEventKind.CAPSLOCK_UP
    else:
      return None
  elif event.code == evdev.ecodes.KEY_ESC:
    if event.value != 1:
      return None
    kind = KeyboardEventKind.ESCAPE_DOWN
  else:
    return None

  return KeyboardControlEvent(kind=kind, received_monotonic_s=time.monotonic())


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
