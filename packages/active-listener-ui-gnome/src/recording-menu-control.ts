export type IndicatorState = 'absent' | 'idle' | 'recording';

export type MenuControlState = {
  label: string;
  enabled: boolean;
};

export type StartOrFinishCommandResponse =
  | { kind: 'success'; result: string }
  | { kind: 'failure'; title: string; detail: string };

export const ACTIVE_LISTENER_COMMAND_FAILED_TITLE = 'Active Listener command failed';

export const deriveIndicatorState = (
  servicePresent: boolean,
  phase: string | null,
): IndicatorState => {
  if (!servicePresent) {
    return 'absent';
  }

  return phase === 'recording' ? 'recording' : 'idle';
};

export const deriveMenuControlState = (
  servicePresent: boolean,
  phase: string | null,
): MenuControlState => {
  if (!servicePresent) {
    return { label: 'No Service', enabled: false };
  }

  if (phase === 'reconnecting') {
    return { label: 'Reconnecting', enabled: false };
  }

  if (phase === 'recording') {
    return { label: 'Stop Recording', enabled: true };
  }

  if (phase === 'idle') {
    return { label: 'Start Recording', enabled: true };
  }

  return { label: 'Start Recording', enabled: false };
};

export const resolveStartOrFinishCommandResponse = (
  callFinish: () => [string],
): StartOrFinishCommandResponse => {
  try {
    const [result] = callFinish();
    return { kind: 'success', result };
  } catch (error) {
    return {
      kind: 'failure',
      title: ACTIVE_LISTENER_COMMAND_FAILED_TITLE,
      detail: describeCommandError(error),
    };
  }
};

const describeCommandError = (error: unknown): string => {
  if (error instanceof Error && error.message.length > 0) {
    return error.message;
  }

  return String(error);
};
