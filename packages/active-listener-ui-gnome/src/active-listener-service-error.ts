export const DBUS_PIPELINE_FAILED_SIGNAL = 'PipelineFailed';
export const DBUS_AUDIO_ARCHIVE_FAILED_SIGNAL = 'AudioArchiveFailed';
export const ACTIVE_LISTENER_PIPELINE_FAILED_TITLE = 'Active Listener pipeline failed';
export const ACTIVE_LISTENER_AUDIO_ARCHIVE_FAILED_TITLE = 'Active Listener audio archive failed';

type DbusPipelineFailedPayload = [string, string];
type DbusAudioArchiveFailedPayload = [string];

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
    const [reason] = parameters as DbusAudioArchiveFailedPayload;
    return {
      title: ACTIVE_LISTENER_AUDIO_ARCHIVE_FAILED_TITLE,
      detail: reason,
    };
  }

  return null;
};
