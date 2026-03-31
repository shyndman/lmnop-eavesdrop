"""Pure foreground state decisions for active-listener."""

from __future__ import annotations

from enum import StrEnum

from eavesdrop.client import ConnectedEvent, DisconnectedEvent, ReconnectedEvent, ReconnectingEvent


class KeyboardAction(StrEnum):
  """Normalized workstation actions consumed by app policy."""

  START_OR_FINISH = "start_or_finish"
  CANCEL = "cancel"


class ForegroundPhase(StrEnum):
  """Foreground service phases visible to local dictation policy."""

  STARTING = "starting"
  IDLE = "idle"
  RECORDING = "recording"
  RECONNECTING = "reconnecting"


class KeyboardDecision(StrEnum):
  """Pure keyboard decisions produced from the current foreground phase."""

  START_RECORDING = "start_recording"
  FINISH_RECORDING = "finish_recording"
  CANCEL_RECORDING = "cancel_recording"
  IGNORE = "ignore"
  SUPPRESS_RECONNECTING_START = "suppress_reconnecting_start"


class ConnectionDecision(StrEnum):
  """Pure client-event decisions produced from the current foreground phase."""

  CONNECTED = "connected"
  RECONNECTING = "reconnecting"
  RECONNECTED = "reconnected"
  DISCONNECTED = "disconnected"
  ABORT_RECORDING = "abort_recording"
  IGNORE = "ignore"


def decide_keyboard_action(phase: ForegroundPhase, action: KeyboardAction) -> KeyboardDecision:
  """Translate a normalized hotkey into a foreground policy decision.

  :param phase: Current foreground phase.
  :type phase: ForegroundPhase
  :param action: Normalized hotkey action.
  :type action: KeyboardAction
  :returns: Policy decision for the input action.
  :rtype: KeyboardDecision
  """

  if phase is ForegroundPhase.IDLE:
    if action is KeyboardAction.START_OR_FINISH:
      return KeyboardDecision.START_RECORDING
    return KeyboardDecision.IGNORE

  if phase is ForegroundPhase.RECORDING:
    if action is KeyboardAction.START_OR_FINISH:
      return KeyboardDecision.FINISH_RECORDING
    return KeyboardDecision.CANCEL_RECORDING

  if phase is ForegroundPhase.RECONNECTING and action is KeyboardAction.START_OR_FINISH:
    return KeyboardDecision.SUPPRESS_RECONNECTING_START

  return KeyboardDecision.IGNORE


def decide_client_event(
  phase: ForegroundPhase,
  event: ConnectedEvent | DisconnectedEvent | ReconnectingEvent | ReconnectedEvent,
) -> ConnectionDecision:
  """Translate a client lifecycle event into a foreground policy decision.

  :param phase: Current foreground phase.
  :type phase: ForegroundPhase
  :param event: Client lifecycle event.
  :type event: ConnectedEvent | DisconnectedEvent | ReconnectingEvent | ReconnectedEvent
  :returns: Policy decision for the client event.
  :rtype: ConnectionDecision
  """

  if isinstance(event, ConnectedEvent):
    if phase is ForegroundPhase.STARTING:
      return ConnectionDecision.CONNECTED
    return ConnectionDecision.IGNORE

  if isinstance(event, ReconnectedEvent):
    return ConnectionDecision.RECONNECTED

  if isinstance(event, ReconnectingEvent):
    return ConnectionDecision.RECONNECTING

  if phase is ForegroundPhase.RECORDING:
    return ConnectionDecision.ABORT_RECORDING

  return ConnectionDecision.DISCONNECTED
