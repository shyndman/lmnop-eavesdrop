"""State decision tests for active-listener foreground policy."""

from __future__ import annotations

from active_listener.state import (
  ConnectionDecision,
  ForegroundPhase,
  KeyboardAction,
  KeyboardDecision,
  decide_client_event,
  decide_keyboard_action,
)
from eavesdrop.client import ConnectedEvent, DisconnectedEvent, ReconnectedEvent, ReconnectingEvent


def test_idle_to_recording_start_decision() -> None:
  decision = decide_keyboard_action(ForegroundPhase.IDLE, KeyboardAction.START_OR_FINISH)

  assert decision is KeyboardDecision.START_RECORDING


def test_recording_to_idle_cancel_decision() -> None:
  decision = decide_keyboard_action(ForegroundPhase.RECORDING, KeyboardAction.CANCEL)

  assert decision is KeyboardDecision.CANCEL_RECORDING


def test_recording_to_idle_finish_decision() -> None:
  decision = decide_keyboard_action(ForegroundPhase.RECORDING, KeyboardAction.START_OR_FINISH)

  assert decision is KeyboardDecision.FINISH_RECORDING


def test_idle_cancel_is_ignored() -> None:
  decision = decide_keyboard_action(ForegroundPhase.IDLE, KeyboardAction.CANCEL)

  assert decision is KeyboardDecision.IGNORE


def test_reconnecting_start_is_suppressed() -> None:
  decision = decide_keyboard_action(ForegroundPhase.RECONNECTING, KeyboardAction.START_OR_FINISH)

  assert decision is KeyboardDecision.SUPPRESS_RECONNECTING_START


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
