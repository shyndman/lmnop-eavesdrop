"""Publish recording-only live transcript updates to the workstation overlay."""

from structlog.stdlib import BoundLogger

from active_listener.app.state import ForegroundPhase
from active_listener.infra.dbus import AppStateService
from active_listener.recording.session import RecordingSession
from eavesdrop.client import TranscriptionEvent


async def publish_live_transcription(
  event: TranscriptionEvent,
  *,
  phase: ForegroundPhase,
  recording: RecordingSession,
  dbus: AppStateService,
  logger: BoundLogger,
) -> None:
  """Reduce and publish only while foreground capture owns the overlay.

  :param event: Incoming live transcript, never a finalization result.
  :param phase: Foreground phase at sequential event dispatch.
  :param recording: Recording-local reducer and cursor owner.
  :param dbus: Workstation overlay publisher.
  :param logger: Sink for reduction and publication diagnostics.
  """
  message = event.message
  logger.debug(
    "live transcription event received",
    stream=message.stream,
    phase=phase.value,
    flush_complete=message.flush_complete is True,
    segment_count=len(message.segments),
  )
  if phase is not ForegroundPhase.RECORDING:
    logger.debug("live transcription event ignored", stream=message.stream, phase=phase.value)
    return
  transcription_update = recording.ingest_live_transcription_message(message)
  if transcription_update is None:
    logger.debug("live transcription event reduced without overlay update", stream=message.stream)
    return
  logger.debug(
    "publishing live transcription update", stream=message.stream, runs=transcription_update.runs
  )
  await dbus.transcription_updated(transcription_update.runs)
  logger.debug(
    "live transcription update published",
    stream=message.stream,
    run_count=len(transcription_update.runs),
  )
