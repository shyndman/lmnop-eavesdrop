"""State decision tests for active-listener foreground policy."""

from __future__ import annotations

from active_listener.app.state import (
  AppAction,
  AppActionDecision,
  ConnectionDecision,
  ForegroundPhase,
  decide_app_action,
  decide_client_event,
)
from eavesdrop.client import ConnectedEvent, DisconnectedEvent, ReconnectedEvent, ReconnectingEvent


def test_idle_to_recording_start_decision() -> None:
  decision = decide_app_action(ForegroundPhase.IDLE, AppAction.START_OR_FINISH)

  assert decision is AppActionDecision.START_RECORDING


def test_recording_to_idle_cancel_decision() -> None:
  decision = decide_app_action(ForegroundPhase.RECORDING, AppAction.CANCEL)

  assert decision is AppActionDecision.CANCEL_RECORDING


def test_recording_to_idle_finish_decision() -> None:
  decision = decide_app_action(ForegroundPhase.RECORDING, AppAction.START_OR_FINISH)

  assert decision is AppActionDecision.FINISH_RECORDING


def test_idle_cancel_is_ignored() -> None:
  decision = decide_app_action(ForegroundPhase.IDLE, AppAction.CANCEL)

  assert decision is AppActionDecision.IGNORE


def test_reconnecting_start_is_suppressed() -> None:
  decision = decide_app_action(ForegroundPhase.RECONNECTING, AppAction.START_OR_FINISH)

  assert decision is AppActionDecision.SUPPRESS_RECONNECTING_START


def test_disconnected_while_idle_enters_reconnecting() -> None:
  decision = decide_client_event(
    ForegroundPhase.IDLE,
    DisconnectedEvent(stream="stream-1", reason="socket closed"),
  )

  assert decision is ConnectionDecision.DISCONNECTED


def test_disconnected_while_recording_aborts_recording() -> None:
  decision = decide_client_event(
    ForegroundPhase.RECORDING,
    DisconnectedEvent(stream="stream-1", reason="socket closed"),
  )

  assert decision is ConnectionDecision.ABORT_RECORDING


def test_reconnecting_event_keeps_reconnecting_phase() -> None:
  decision = decide_client_event(
    ForegroundPhase.RECONNECTING,
    ReconnectingEvent(stream="stream-1", attempt=2, retry_delay_s=10.0),
  )

  assert decision is ConnectionDecision.RECONNECTING


def test_reconnected_event_returns_to_idle() -> None:
  decision = decide_client_event(ForegroundPhase.RECONNECTING, ReconnectedEvent(stream="stream-1"))

  assert decision is ConnectionDecision.RECONNECTED


def test_connected_event_only_matters_during_startup() -> None:
  starting_decision = decide_client_event(
    ForegroundPhase.STARTING,
    ConnectedEvent(stream="stream-1"),
  )
  idle_decision = decide_client_event(
    ForegroundPhase.IDLE,
    ConnectedEvent(stream="stream-1"),
  )

  assert starting_decision is ConnectionDecision.CONNECTED
  assert idle_decision is ConnectionDecision.IGNORE
