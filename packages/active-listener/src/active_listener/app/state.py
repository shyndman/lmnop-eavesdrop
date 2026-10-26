"""Pure foreground state decisions for active-listener."""

from __future__ import annotations

from enum import StrEnum


class AppAction(StrEnum):
  """Normalized workstation actions consumed by app policy."""

  START_OR_FINISH = "start_or_finish"
  CANCEL = "cancel"


class ForegroundPhase(StrEnum):
  """Foreground service phases visible to local dictation policy."""

  STARTING = "starting"
  IDLE = "idle"
  RECORDING = "recording"
  RECONNECTING = "reconnecting"


class AppActionDecision(StrEnum):
  """Pure app-action decisions produced from the current foreground phase."""

  START_RECORDING = "start_recording"
  FINISH_RECORDING = "finish_recording"
  CANCEL_RECORDING = "cancel_recording"
  IGNORE = "ignore"
  SUPPRESS_RECONNECTING_START = "suppress_reconnecting_start"


class StartOrFinishResult(StrEnum):
  """Public start/finish outcomes exposed across the DBus boundary."""

  STARTED = "started"
  FINISHED = "finished"
  IGNORED = "ignored"


def decide_app_action(phase: ForegroundPhase, action: AppAction) -> AppActionDecision:
  """Translate a normalized app action into a foreground policy decision.

  :param phase: Current foreground phase.
  :type phase: ForegroundPhase
  :param action: Normalized app action.
  :type action: AppAction
  :returns: Policy decision for the input action.
  :rtype: AppActionDecision
  """

  if phase is ForegroundPhase.IDLE:
    if action is AppAction.START_OR_FINISH:
      return AppActionDecision.START_RECORDING
    return AppActionDecision.IGNORE

  if phase is ForegroundPhase.RECORDING:
    if action is AppAction.START_OR_FINISH:
      return AppActionDecision.FINISH_RECORDING
    return AppActionDecision.CANCEL_RECORDING

  if phase is ForegroundPhase.RECONNECTING and action is AppAction.START_OR_FINISH:
    return AppActionDecision.SUPPRESS_RECONNECTING_START

  return AppActionDecision.IGNORE
