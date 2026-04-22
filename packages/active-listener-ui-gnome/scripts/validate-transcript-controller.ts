import {
  DEFAULT_TRANSITION_TIMING,
  PANGO_ALPHA_MAX,
  TranscriptAnimationController,
  type ByteAlphaRun,
  type TransitionPlan,
} from '../src/transcript-animation.ts';

type RenderEvent =
  | { kind: 'install'; text: string }
  | { kind: 'apply'; runs: ByteAlphaRun[] }
  | { kind: 'clear' }
  | { kind: 'beforeSwap'; sourceText: string; targetText: string }
  | { kind: 'afterSwap'; sourceText: string; targetText: string };

const assert = (condition: boolean, message: string): void => {
  if (!condition) {
    throw new Error(message);
  }
};

const durationToFinish = (): number =>
  DEFAULT_TRANSITION_TIMING.fadeDurationMs + DEFAULT_TRANSITION_TIMING.eraseStaggerMs * 32;

const createControllerHarness = () => {
  let installedText = '';
  let activeRuns: ByteAlphaRun[] = [];
  const events: RenderEvent[] = [];
  const controller = new TranscriptAnimationController({
    installText(text) {
      installedText = text;
      events.push({ kind: 'install', text });
    },
    applyAlphaRuns(runs) {
      activeRuns = runs.map((run) => ({ ...run }));
      events.push({ kind: 'apply', runs: activeRuns });
    },
    clearAlphaRuns() {
      activeRuns = [];
      events.push({ kind: 'clear' });
    },
    beforeSwap(plan: TransitionPlan) {
      events.push({ kind: 'beforeSwap', sourceText: plan.sourceText, targetText: plan.targetText });
    },
    afterSwap(plan: TransitionPlan) {
      events.push({ kind: 'afterSwap', sourceText: plan.sourceText, targetText: plan.targetText });
    },
  });

  const advance = (startMs: number, durationMs: number, stepMs: number = 24): number => {
    let currentMs = startMs;
    const endMs = startMs + durationMs;

    while (currentMs < endMs) {
      currentMs += stepMs;
      const shouldContinue = controller.tick(currentMs);
      if (!shouldContinue) {
        return currentMs;
      }
    }

    controller.tick(endMs);
    return endMs;
  };

  return {
    controller,
    events,
    advance,
    getInstalledText: () => installedText,
    getActiveRuns: () => activeRuns.map((run) => ({ ...run })),
  };
};

{
  const harness = createControllerHarness();
  harness.controller.installImmediate('alpha beta');
  const animating = harness.controller.setCanonicalText('alpha', 0);
  assert(animating, 'erase should start');
  harness.advance(0, durationToFinish());

  const installEvents = harness.events.filter((event): event is Extract<RenderEvent, { kind: 'install' }> => event.kind === 'install');
  assert(installEvents.at(-1)?.text === 'alpha', 'erase should converge to target text');
  console.log('PASS erase');
}

{
  const harness = createControllerHarness();
  harness.controller.installImmediate('hello');
  const animating = harness.controller.setCanonicalText('hello there', 0);
  assert(animating, 'swap case should animate');
  harness.advance(0, durationToFinish());

  const beforeSwapIndex = harness.events.findIndex((event) => event.kind === 'beforeSwap');
  const afterSwapIndex = harness.events.findIndex((event) => event.kind === 'afterSwap');
  assert(beforeSwapIndex !== -1, 'swap should emit beforeSwap');
  assert(afterSwapIndex > beforeSwapIndex, 'swap should emit afterSwap after beforeSwap');
  console.log('PASS swap');
}

{
  const harness = createControllerHarness();
  harness.controller.installImmediate('hello');
  harness.controller.setCanonicalText('hello there', 0);
  harness.controller.tick(0);

  const zeroAlphaRuns = harness.getActiveRuns();
  assert(zeroAlphaRuns.length > 0, 'reveal should hide the new tail at the swap');
  assert(zeroAlphaRuns.every((run) => run.alpha === 0), 'reveal should start with hidden new tail');

  harness.advance(0, durationToFinish());
  const snapshot = harness.controller.getSnapshot();
  assert(snapshot.phase === 'idle', 'reveal should finish in idle state');
  assert(harness.getInstalledText() === 'hello there', 'reveal should converge to the full target text');
  console.log('PASS reveal');
}

{
  const harness = createControllerHarness();
  harness.controller.installImmediate('hello');
  harness.controller.setCanonicalText('hello there', 0);
  harness.advance(0, durationToFinish());

  assert(harness.getActiveRuns().length === 0, 'controller should clear attributes after reveal');
  const clearCount = harness.events.filter((event) => event.kind === 'clear').length;
  assert(clearCount >= 2, 'controller should clear attributes during transition boundaries');
  console.log('PASS attribute clearing');
}

{
  const harness = createControllerHarness();
  harness.controller.installImmediate('alpha beta');
  harness.controller.setCanonicalText('alpha', 0);
  harness.controller.tick(48);
  harness.controller.setCanonicalText('alpha gamma', 48);
  harness.advance(48, durationToFinish());

  const snapshot = harness.controller.getSnapshot();
  assert(snapshot.canonicalText === 'alpha gamma', 'erase interruption should retarget to the newest canonical text');
  assert(snapshot.installedText === 'alpha gamma', 'erase interruption should converge to the newest canonical text');
  assert(snapshot.phase === 'idle', 'erase interruption should finish idle');
  assert(harness.getInstalledText() === 'alpha gamma', 'erase interruption should install the latest canonical text');
  assert(
    harness.events.some(
      (event) => event.kind === 'install' && event.text === 'alpha beta',
    ),
    'interruption during erase should restart from the still-installed source text',
  );

  const revealHarness = createControllerHarness();
  revealHarness.controller.installImmediate('alpha');
  revealHarness.controller.setCanonicalText('alpha beta', 0);
  revealHarness.controller.tick(0);
  revealHarness.controller.setCanonicalText('alpha betamax', 24);
  revealHarness.advance(24, durationToFinish());

  const revealSnapshot = revealHarness.controller.getSnapshot();
  assert(revealSnapshot.canonicalText === 'alpha betamax', 'reveal interruption should retarget to the newest canonical text');
  assert(revealSnapshot.installedText === 'alpha betamax', 'reveal interruption should converge to the newest canonical text');
  assert(
    revealHarness.events.some(
      (event) => event.kind === 'install' && event.text === 'alpha beta',
    ),
    'interruption during reveal should restart from the already installed text',
  );
  console.log('PASS interruption');
}

{
  const harness = createControllerHarness();
  harness.controller.installImmediate('one');
  harness.controller.setCanonicalText('one two', 0);
  harness.advance(0, durationToFinish());

  const alphaValues = harness.events
    .filter((event): event is Extract<RenderEvent, { kind: 'apply' }> => event.kind === 'apply')
    .flatMap((event) => event.runs.map((run) => run.alpha));
  assert(alphaValues.some((alpha) => alpha === 0), 'reveal should produce fully hidden graphemes');
  assert(alphaValues.some((alpha) => alpha > 0 && alpha < PANGO_ALPHA_MAX), 'reveal should produce partial alpha runs');
}

{
  const harness = createControllerHarness();
  harness.controller.installImmediate('hello');
  harness.controller.setCanonicalText('hello there', 0);
  harness.controller.tick(0);

  const beforeRepeat = harness.controller.getSnapshot();
  const stillAnimating = harness.controller.setCanonicalText('hello there', 24);
  const afterRepeat = harness.controller.getSnapshot();

  assert(stillAnimating, 'repeating the same target during reveal should keep the animation running');
  assert(afterRepeat.phase === beforeRepeat.phase, 'repeating the same target during reveal should preserve the current phase');
  assert(afterRepeat.plan !== null, 'repeating the same target during reveal should keep the transition plan');
  assert(
    JSON.stringify(afterRepeat.activeAlphaRuns) === JSON.stringify(beforeRepeat.activeAlphaRuns),
    'repeating the same target during reveal should preserve the active alpha runs',
  );
  console.log('PASS reveal dedupe');
}

{
  const harness = createControllerHarness();
  harness.controller.installImmediate('alpha beta');
  harness.controller.setCanonicalText('alpha', 0);
  harness.controller.tick(48);

  const beforeRepeat = harness.controller.getSnapshot();
  const stillAnimating = harness.controller.setCanonicalText('alpha', 72);
  const afterRepeat = harness.controller.getSnapshot();

  assert(stillAnimating, 'repeating the same target during erase should keep the animation running');
  assert(afterRepeat.phase === beforeRepeat.phase, 'repeating the same target during erase should preserve the current phase');
  assert(afterRepeat.plan !== null, 'repeating the same target during erase should keep the transition plan');
  assert(
    afterRepeat.phaseStartedAtMs === beforeRepeat.phaseStartedAtMs,
    'repeating the same target during erase should not restart the phase timer',
  );
  assert(
    JSON.stringify(afterRepeat.activeAlphaRuns) === JSON.stringify(beforeRepeat.activeAlphaRuns),
    'repeating the same target during erase should preserve the active alpha runs',
  );
  console.log('PASS erase dedupe');
}

{
  const loggedErrors: unknown[][] = [];
  const originalConsoleError = console.error;
  console.error = (...args: unknown[]) => {
    loggedErrors.push(args);
  };

  try {
    const harness = createControllerHarness();
    harness.controller.installImmediate('alpha');
    harness.controller.setCanonicalText('alpha beta', 0);
  } finally {
    console.error = originalConsoleError;
  }

  const changeLog = loggedErrors.find((args) => {
    if (args[0] !== 'Active Listener transcript animation canonical text changed') {
      return false;
    }

    const details = args[1] as { fromText?: unknown; toText?: unknown };
    return details.fromText === 'alpha' && details.toText === 'alpha beta';
  });
  assert(changeLog !== undefined, 'unequal canonical text updates should emit a transcript change log');
  const details = changeLog[1] as { fromText?: unknown; toText?: unknown };
  assert(details.fromText === 'alpha', 'transcript change log should include the full source transcript');
  assert(details.toText === 'alpha beta', 'transcript change log should include the full target transcript');
  console.log('PASS change logging');
}

console.log('Controller converges to latest canonical string.');
