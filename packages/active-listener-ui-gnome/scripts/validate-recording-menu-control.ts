import {
  ACTIVE_LISTENER_COMMAND_FAILED_TITLE,
  deriveIndicatorState,
  deriveMenuControlState,
  resolveStartOrFinishCommandResponse,
} from '../src/recording-menu-control.ts';

type MenuCase = {
  name: string;
  servicePresent: boolean;
  phase: string | null;
  expectedIndicator: 'absent' | 'idle' | 'recording';
  expectedLabel: string;
  expectedEnabled: boolean;
};

const assert = (condition: boolean, message: string): void => {
  if (!condition) {
    throw new Error(message);
  }
};

const cases: MenuCase[] = [
  {
    name: 'absent service',
    servicePresent: false,
    phase: null,
    expectedIndicator: 'absent',
    expectedLabel: 'No Service',
    expectedEnabled: false,
  },
  {
    name: 'reconnecting service',
    servicePresent: true,
    phase: 'reconnecting',
    expectedIndicator: 'idle',
    expectedLabel: 'Reconnecting',
    expectedEnabled: false,
  },
  {
    name: 'idle service',
    servicePresent: true,
    phase: 'idle',
    expectedIndicator: 'idle',
    expectedLabel: 'Start Recording',
    expectedEnabled: true,
  },
  {
    name: 'recording service',
    servicePresent: true,
    phase: 'recording',
    expectedIndicator: 'recording',
    expectedLabel: 'Stop Recording',
    expectedEnabled: true,
  },
  {
    name: 'starting service',
    servicePresent: true,
    phase: 'starting',
    expectedIndicator: 'idle',
    expectedLabel: 'Start Recording',
    expectedEnabled: false,
  },
  {
    name: 'unknown service phase',
    servicePresent: true,
    phase: 'mystery',
    expectedIndicator: 'idle',
    expectedLabel: 'Start Recording',
    expectedEnabled: false,
  },
];

for (const testCase of cases) {
  const indicatorState = deriveIndicatorState(testCase.servicePresent, testCase.phase);
  const menuState = deriveMenuControlState(testCase.servicePresent, testCase.phase);

  assert(
    indicatorState === testCase.expectedIndicator,
    `${testCase.name} indicator: expected ${testCase.expectedIndicator}, got ${indicatorState}`,
  );
  assert(
    menuState.label === testCase.expectedLabel,
    `${testCase.name} label: expected ${testCase.expectedLabel}, got ${menuState.label}`,
  );
  assert(
    menuState.enabled === testCase.expectedEnabled,
    `${testCase.name} enabled: expected ${String(testCase.expectedEnabled)}, got ${String(menuState.enabled)}`,
  );

  console.log(`PASS ${testCase.name}`);
}

{
  const response = resolveStartOrFinishCommandResponse(() => ['started']);

  assert(response.kind === 'success', 'success response should be tagged success');
  assert(response.result === 'started', 'success response should preserve the DBus result');
  console.log('PASS success response');
}

{
  const response = resolveStartOrFinishCommandResponse(() => {
    throw new Error('boom');
  });

  assert(response.kind === 'failure', 'error response should be tagged failure');
  assert(response.title === ACTIVE_LISTENER_COMMAND_FAILED_TITLE, 'failure response should use the locked title');
  assert(response.detail === 'boom', 'failure response should preserve the thrown message');
  console.log('PASS failure response');
}

{
  const response = resolveStartOrFinishCommandResponse(() => {
    throw 'plain failure';
  });

  assert(response.kind === 'failure', 'string failure should be tagged failure');
  assert(response.detail === 'plain failure', 'string failure should stringify the thrown value');
  console.log('PASS string failure response');
}

console.log('Recording menu control contracts hold.');
