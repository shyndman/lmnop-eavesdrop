"""Send recording transcript updates to the workstation D-Bus overlay."""

from structlog.stdlib import BoundLogger

from active_listener.app.state import ForegroundPhase
from active_listener.infra.dbus import AppStateService
from active_listener.recording.session import RecordingTranscript
from eavesdrop.wire import TranscriptionMessage


async def update_dbus_transcription(
  message: TranscriptionMessage,
  *,
  phase: ForegroundPhase,
  recording: RecordingTranscript,
  dbus: AppStateService,
  logger: BoundLogger,
) -> None:
  """Reduce and send updates only while foreground capture owns the overlay.

  :param message: Original wire transcription delivered by the client callback.
  :param phase: Foreground phase at sequential application dispatch.
  :param recording: Recording-local transcript and cursor owner.
  :param dbus: Workstation D-Bus overlay destination.
  :param logger: Reduction and delivery diagnostics.
  :raises Exception: If reduction or D-Bus delivery fails.
  """
  if phase is not ForegroundPhase.RECORDING:
    logger.debug("transcription ignored outside recording", session_id=message.stream)
    return
  transcription_update = recording.ingest_live_transcription_message(message)
  if transcription_update is None:
    return
  await dbus.transcription_updated(transcription_update.runs)
  logger.debug(
    "D-Bus transcription updated",
    session_id=message.stream,
    run_count=len(transcription_update.runs),
  )
