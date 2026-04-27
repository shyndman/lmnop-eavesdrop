export type IndicatorState = 'absent' | 'idle' | 'recording';

export type MenuControlState = {
  label: string;
  enabled: boolean;
};

export type LlmRewriteMenuState = {
  visible: boolean;
  enabled: boolean;
  active: boolean;
};

export type CommandResponse<T> =
  | { kind: 'success'; result: T }
  | { kind: 'failure'; title: string; detail: string };

export type StartOrFinishCommandResponse = CommandResponse<string>;
export type SetLlmActiveCommandResponse = CommandResponse<boolean>;

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

export const deriveLlmRewriteMenuState = (
  servicePresent: boolean,
  llmAvailable: boolean,
  llmActive: boolean,
): LlmRewriteMenuState => {
  if (!servicePresent || !llmAvailable) {
    return { visible: false, enabled: false, active: false };
  }

  return {
    visible: true,
    enabled: true,
    active: llmActive,
  };
};

export const resolveStartOrFinishCommandResponse = (
  callFinish: () => [string],
): StartOrFinishCommandResponse => {
  return resolveCommandResponse(() => {
    const [result] = callFinish();
    return result;
  });
};

export const resolveSetLlmActiveCommandResponse = (
  callFinish: () => [boolean],
): SetLlmActiveCommandResponse => {
  return resolveCommandResponse(() => {
    const [result] = callFinish();
    return result;
  });
};

const resolveCommandResponse = <T>(
  callFinish: () => T,
): CommandResponse<T> => {
  try {
    const result = callFinish();
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
