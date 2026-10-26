"""State decisions for active-listener workstation actions."""

from active_listener.app.state import (
  AppAction,
  AppActionDecision,
  ForegroundPhase,
  decide_app_action,
)


def test_idle_to_recording_start_decision() -> None:
  assert decide_app_action(ForegroundPhase.IDLE, AppAction.START_OR_FINISH) is (
    AppActionDecision.START_RECORDING
  )


def test_recording_to_idle_cancel_decision() -> None:
  assert decide_app_action(ForegroundPhase.RECORDING, AppAction.CANCEL) is (
    AppActionDecision.CANCEL_RECORDING
  )


def test_recording_to_idle_finish_decision() -> None:
  assert decide_app_action(ForegroundPhase.RECORDING, AppAction.START_OR_FINISH) is (
    AppActionDecision.FINISH_RECORDING
  )


def test_idle_cancel_is_ignored() -> None:
  assert decide_app_action(ForegroundPhase.IDLE, AppAction.CANCEL) is AppActionDecision.IGNORE


def test_reconnecting_start_is_suppressed() -> None:
  assert decide_app_action(ForegroundPhase.RECONNECTING, AppAction.START_OR_FINISH) is (
    AppActionDecision.SUPPRESS_RECONNECTING_START
  )
