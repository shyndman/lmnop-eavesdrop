import {
  ACTIVE_LISTENER_COMMAND_FAILED_TITLE,
  deriveLlmRewriteMenuState,
  deriveIndicatorState,
  deriveMenuControlState,
  resolveSetLlmActiveCommandResponse,
  resolveStartOrFinishCommandResponse,
} from '../src/recording-menu-control.ts';

type MenuCase = {
  name: string;
  servicePresent: boolean;
  phase: string | null;
  llmAvailable: boolean;
  llmActive: boolean;
  expectedIndicator: 'absent' | 'idle' | 'recording';
  expectedLabel: string;
  expectedEnabled: boolean;
  expectedLlmVisible: boolean;
  expectedLlmEnabled: boolean;
  expectedLlmActive: boolean;
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
    llmAvailable: false,
    llmActive: false,
    expectedIndicator: 'absent',
    expectedLabel: 'No Service',
    expectedEnabled: false,
    expectedLlmVisible: false,
    expectedLlmEnabled: false,
    expectedLlmActive: false,
  },
  {
    name: 'reconnecting service',
    servicePresent: true,
    phase: 'reconnecting',
    llmAvailable: true,
    llmActive: true,
    expectedIndicator: 'idle',
    expectedLabel: 'Reconnecting',
    expectedEnabled: false,
    expectedLlmVisible: true,
    expectedLlmEnabled: true,
    expectedLlmActive: true,
  },
  {
    name: 'idle service',
    servicePresent: true,
    phase: 'idle',
    llmAvailable: true,
    llmActive: true,
    expectedIndicator: 'idle',
    expectedLabel: 'Start Recording',
    expectedEnabled: true,
    expectedLlmVisible: true,
    expectedLlmEnabled: true,
    expectedLlmActive: true,
  },
  {
    name: 'recording service',
    servicePresent: true,
    phase: 'recording',
    llmAvailable: true,
    llmActive: false,
    expectedIndicator: 'recording',
    expectedLabel: 'Stop Recording',
    expectedEnabled: true,
    expectedLlmVisible: true,
    expectedLlmEnabled: true,
    expectedLlmActive: false,
  },
  {
    name: 'starting service',
    servicePresent: true,
    phase: 'starting',
    llmAvailable: false,
    llmActive: false,
    expectedIndicator: 'idle',
    expectedLabel: 'Start Recording',
    expectedEnabled: false,
    expectedLlmVisible: false,
    expectedLlmEnabled: false,
    expectedLlmActive: false,
  },
  {
    name: 'unknown service phase',
    servicePresent: true,
    phase: 'mystery',
    llmAvailable: true,
    llmActive: false,
    expectedIndicator: 'idle',
    expectedLabel: 'Start Recording',
    expectedEnabled: false,
    expectedLlmVisible: true,
    expectedLlmEnabled: true,
    expectedLlmActive: false,
  },
];

for (const testCase of cases) {
  const indicatorState = deriveIndicatorState(testCase.servicePresent, testCase.phase);
  const menuState = deriveMenuControlState(testCase.servicePresent, testCase.phase);
  const llmRewriteState = deriveLlmRewriteMenuState(
    testCase.servicePresent,
    testCase.llmAvailable,
    testCase.llmActive,
  );

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
  assert(
    llmRewriteState.visible === testCase.expectedLlmVisible,
    `${testCase.name} llm visible: expected ${String(testCase.expectedLlmVisible)}, got ${String(llmRewriteState.visible)}`,
  );
  assert(
    llmRewriteState.enabled === testCase.expectedLlmEnabled,
    `${testCase.name} llm enabled: expected ${String(testCase.expectedLlmEnabled)}, got ${String(llmRewriteState.enabled)}`,
  );
  assert(
    llmRewriteState.active === testCase.expectedLlmActive,
    `${testCase.name} llm active: expected ${String(testCase.expectedLlmActive)}, got ${String(llmRewriteState.active)}`,
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

{
  const response = resolveSetLlmActiveCommandResponse(() => [false]);

  assert(response.kind === 'success', 'llm toggle success should be tagged success');
  assert(response.result === false, 'llm toggle success should preserve the DBus result');
  console.log('PASS llm toggle success response');
}

{
  const response = resolveSetLlmActiveCommandResponse(() => {
    throw new Error('llm unavailable');
  });

  assert(response.kind === 'failure', 'llm toggle failure should be tagged failure');
  assert(response.title === ACTIVE_LISTENER_COMMAND_FAILED_TITLE, 'llm toggle failure should use the locked title');
  assert(response.detail === 'llm unavailable', 'llm toggle failure should preserve the thrown message');
  console.log('PASS llm toggle failure response');
}

console.log('Recording menu control contracts hold.');
