export const DBUS_PIPELINE_FAILED_SIGNAL = 'PipelineFailed';
export const DBUS_AUDIO_ARCHIVE_FAILED_SIGNAL = 'AudioArchiveFailed';
export const DBUS_RECORDING_ABORTED_SIGNAL = 'RecordingAborted';
export const DBUS_FATAL_ERROR_SIGNAL = 'FatalError';
export const ACTIVE_LISTENER_PIPELINE_FAILED_TITLE = 'Active Listener pipeline failed';
export const ACTIVE_LISTENER_AUDIO_ARCHIVE_FAILED_TITLE = 'Active Listener audio archive failed';
export const ACTIVE_LISTENER_RECORDING_ABORTED_TITLE = 'Active Listener recording aborted';
export const ACTIVE_LISTENER_FATAL_ERROR_TITLE = 'Active Listener fatal error';

type DbusPipelineFailedPayload = [string, string];
type DbusReasonPayload = [string];

export type ActiveListenerServiceErrorSignal = {
  title: string;
  detail: string;
};

export const resolveActiveListenerServiceErrorSignal = (
  signalName: string,
  parameters: unknown,
): ActiveListenerServiceErrorSignal | null => {
  if (signalName === DBUS_PIPELINE_FAILED_SIGNAL) {
    const [step, reason] = parameters as DbusPipelineFailedPayload;
    return {
      title: ACTIVE_LISTENER_PIPELINE_FAILED_TITLE,
      detail: `${step}: ${reason}`,
    };
  }

  if (signalName === DBUS_AUDIO_ARCHIVE_FAILED_SIGNAL) {
    const [reason] = parameters as DbusReasonPayload;
    return {
      title: ACTIVE_LISTENER_AUDIO_ARCHIVE_FAILED_TITLE,
      detail: reason,
    };
  }

  if (signalName === DBUS_RECORDING_ABORTED_SIGNAL) {
    const [reason] = parameters as DbusReasonPayload;
    return {
      title: ACTIVE_LISTENER_RECORDING_ABORTED_TITLE,
      detail: reason,
    };
  }

  if (signalName === DBUS_FATAL_ERROR_SIGNAL) {
    const [reason] = parameters as DbusReasonPayload;
    return {
      title: ACTIVE_LISTENER_FATAL_ERROR_TITLE,
      detail: reason,
    };
  }

  return null;
};
