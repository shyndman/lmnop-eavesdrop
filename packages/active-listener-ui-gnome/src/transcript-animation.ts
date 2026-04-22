export type TranscriptPhase = 'idle' | 'erasing-old-tail' | 'revealing-new-tail';

export type GraphemeSpan = {
  text: string;
  graphemeIndex: number;
  startByte: number;
  endByte: number;
};

export type TransitionPlan = {
  sourceText: string;
  targetText: string;
  sourceSpans: GraphemeSpan[];
  targetSpans: GraphemeSpan[];
  commonPrefixCount: number;
};

export type ByteAlphaRun = {
  startByte: number;
  endByte: number;
  alpha: number;
};

export type TransitionTiming = {
  eraseStaggerMs: number;
  revealStaggerMs: number;
  fadeDurationMs: number;
};

export type TranscriptAnimationState = {
  canonicalText: string;
  installedText: string;
  phase: TranscriptPhase;
  plan: TransitionPlan | null;
  frameSourceId: number | null;
  phaseStartedAtMs: number;
};

export type TranscriptAnimationSnapshot = TranscriptAnimationState & {
  activeAlphaRuns: ByteAlphaRun[];
};

export type TranscriptAnimationSink = {
  installText(text: string): void;
  applyAlphaRuns(runs: ByteAlphaRun[]): void;
  clearAlphaRuns(): void;
  beforeSwap?(plan: TransitionPlan): void;
  afterSwap?(plan: TransitionPlan): void;
};

export const PANGO_ALPHA_MAX = 65_535;
export const TRANSCRIPT_FRAME_INTERVAL_MS = 16;
export const TRANSCRIPT_LOG_SAMPLE_INTERVAL = 180;
const TRANSCRIPT_TIMING_SLOWDOWN_FACTOR = 20;
export const DEFAULT_TRANSITION_TIMING: TransitionTiming = {
  eraseStaggerMs: 18 * TRANSCRIPT_TIMING_SLOWDOWN_FACTOR,
  revealStaggerMs: 18 * TRANSCRIPT_TIMING_SLOWDOWN_FACTOR,
  fadeDurationMs: 180 * TRANSCRIPT_TIMING_SLOWDOWN_FACTOR,
};

const graphemeSegmenter = new Intl.Segmenter(undefined, { granularity: 'grapheme' });
const utf8Encoder = new TextEncoder();

const clamp = (value: number, minimum: number, maximum: number): number =>
  Math.min(Math.max(value, minimum), maximum);

const toAlpha = (value: number): number => Math.round(clamp(value, 0, 1) * PANGO_ALPHA_MAX);

const getPhaseDurationMs = (tailCount: number, staggerMs: number, fadeDurationMs: number): number => {
  if (tailCount === 0) {
    return 0;
  }

  return (tailCount - 1) * staggerMs + fadeDurationMs;
};

const buildTailAlphaRuns = (
  spans: GraphemeSpan[],
  commonPrefixCount: number,
  elapsedMs: number,
  staggerMs: number,
  fadeDurationMs: number,
  direction: 'forward' | 'reverse',
  alphaForProgress: (progress: number) => number,
): ByteAlphaRun[] => {
  const tailSpans = spans.slice(commonPrefixCount);
  if (tailSpans.length === 0) {
    return [];
  }

  const orderedTailSpans = direction === 'reverse' ? [...tailSpans].reverse() : tailSpans;
  const runEntries = orderedTailSpans
    .map((span, tailIndex) => {
      const startMs = tailIndex * staggerMs;
      const progress = clamp((elapsedMs - startMs) / fadeDurationMs, 0, 1);
      const alpha = alphaForProgress(progress);
      return {
        startByte: span.startByte,
        endByte: span.endByte,
        alpha,
      };
    })
    .filter((entry) => entry.alpha < PANGO_ALPHA_MAX)
    .sort((left, right) => left.startByte - right.startByte);

  if (runEntries.length === 0) {
    return [];
  }

  const coalescedRuns: ByteAlphaRun[] = [];
  for (const entry of runEntries) {
    const previousRun = coalescedRuns.at(-1);
    if (
      previousRun !== undefined &&
      previousRun.alpha === entry.alpha &&
      previousRun.endByte === entry.startByte
    ) {
      previousRun.endByte = entry.endByte;
      continue;
    }

    coalescedRuns.push({ ...entry });
  }

  return coalescedRuns;
};

export const buildGraphemeSpans = (text: string): GraphemeSpan[] => {
  const spans: GraphemeSpan[] = [];
  let startByte = 0;

  for (const { segment } of graphemeSegmenter.segment(text)) {
    const segmentByteLength = utf8Encoder.encode(segment).byteLength;
    spans.push({
      text: segment,
      graphemeIndex: spans.length,
      startByte,
      endByte: startByte + segmentByteLength,
    });
    startByte += segmentByteLength;
  }

  return spans;
};

export const computeCommonPrefixCount = (source: GraphemeSpan[], target: GraphemeSpan[]): number => {
  const sharedLength = Math.min(source.length, target.length);
  for (let index = 0; index < sharedLength; index += 1) {
    if (source[index]?.text !== target[index]?.text) {
      return index;
    }
  }

  return sharedLength;
};

export const buildTransitionPlan = (sourceText: string, targetText: string): TransitionPlan => {
  const sourceSpans = buildGraphemeSpans(sourceText);
  const targetSpans = buildGraphemeSpans(targetText);

  return {
    sourceText,
    targetText,
    sourceSpans,
    targetSpans,
    commonPrefixCount: computeCommonPrefixCount(sourceSpans, targetSpans),
  };
};

export const getErasePhaseDurationMs = (
  plan: TransitionPlan,
  timing: TransitionTiming = DEFAULT_TRANSITION_TIMING,
): number => getPhaseDurationMs(plan.sourceSpans.length - plan.commonPrefixCount, timing.eraseStaggerMs, timing.fadeDurationMs);

export const getRevealPhaseDurationMs = (
  plan: TransitionPlan,
  timing: TransitionTiming = DEFAULT_TRANSITION_TIMING,
): number => getPhaseDurationMs(plan.targetSpans.length - plan.commonPrefixCount, timing.revealStaggerMs, timing.fadeDurationMs);

export const buildEraseAlphaRuns = (
  plan: TransitionPlan,
  elapsedMs: number,
  timing: TransitionTiming = DEFAULT_TRANSITION_TIMING,
): ByteAlphaRun[] =>
  buildTailAlphaRuns(
    plan.sourceSpans,
    plan.commonPrefixCount,
    elapsedMs,
    timing.eraseStaggerMs,
    timing.fadeDurationMs,
    'reverse',
    (progress) => toAlpha(1 - progress),
  );

export const buildRevealAlphaRuns = (
  plan: TransitionPlan,
  elapsedMs: number,
  timing: TransitionTiming = DEFAULT_TRANSITION_TIMING,
): ByteAlphaRun[] =>
  buildTailAlphaRuns(
    plan.targetSpans,
    plan.commonPrefixCount,
    elapsedMs,
    timing.revealStaggerMs,
    timing.fadeDurationMs,
    'forward',
    (progress) => toAlpha(progress),
  );

const describeTranscriptTextForLogging = (text: string): Record<string, unknown> => ({
  text,
  characterCount: text.length,
  graphemeCount: buildGraphemeSpans(text).length,
  byteCount: utf8Encoder.encode(text).byteLength,
});

let transcriptFrameLoggingEnabled = false;
let transcriptEventCount = 0;

export const setTranscriptFrameLoggingEnabled = (enabled: boolean): void => {
  transcriptFrameLoggingEnabled = enabled;
};

export const isTranscriptFrameLoggingEnabled = (): boolean => transcriptFrameLoggingEnabled;

const logTranscriptAnimationEvent = (message: string, details: Record<string, unknown> = {}): void => {
  if (transcriptEventCount % TRANSCRIPT_LOG_SAMPLE_INTERVAL !== 0) {
    transcriptEventCount += 1;
    return;
  }

  transcriptEventCount += 1;
  console.error(`Active Listener transcript animation ${message}`, details);
};

const logTranscriptTextChange = (fromText: string, toText: string, nowMilliseconds: number | null, phase: TranscriptPhase): void => {
  console.error('Active Listener transcript animation canonical text changed', {
    fromText,
    toText,
    nowMilliseconds,
    phase,
  });
};

const logTranscriptAnimationFrameEvent = (message: string, details: Record<string, unknown> = {}): void => {
  if (!transcriptFrameLoggingEnabled) {
    return;
  }

  console.error(`Active Listener transcript animation ${message}`, details);
};

export class TranscriptAnimationController {
  private readonly sink: TranscriptAnimationSink;
  private readonly timing: TransitionTiming;
  private state: TranscriptAnimationState = {
    canonicalText: '',
    installedText: '',
    phase: 'idle',
    plan: null,
    frameSourceId: null,
    phaseStartedAtMs: 0,
  };
  private activeAlphaRuns: ByteAlphaRun[] = [];

  constructor(sink: TranscriptAnimationSink, timing: TransitionTiming = DEFAULT_TRANSITION_TIMING) {
    this.sink = sink;
    this.timing = timing;
  }

  getSnapshot(): TranscriptAnimationSnapshot {
    return {
      ...this.state,
      activeAlphaRuns: this.activeAlphaRuns.map((run) => ({ ...run })),
    };
  }

  isAnimating(): boolean {
    return this.state.phase !== 'idle' && this.state.plan !== null;
  }

  setFrameSourceId(frameSourceId: number | null): void {
    this.state.frameSourceId = frameSourceId;
  }

  installImmediate(text: string): void {
    if (text !== this.state.canonicalText) {
      logTranscriptTextChange(this.state.canonicalText, text, null, this.state.phase);
    }

    logTranscriptAnimationEvent('install immediate requested', {
      text: describeTranscriptTextForLogging(text),
      previousState: this.getSnapshot(),
    });
    this.clearAlphaRuns();
    this.sink.installText(text);
    this.state = {
      canonicalText: text,
      installedText: text,
      phase: 'idle',
      plan: null,
      frameSourceId: this.state.frameSourceId,
      phaseStartedAtMs: 0,
    };
    logTranscriptAnimationEvent('install immediate completed', {
      state: this.getSnapshot(),
    });
  }

  setCanonicalText(text: string, nowMs: number): boolean {
    if (text === this.state.canonicalText) {
      logTranscriptAnimationEvent('set canonical text ignored because text did not change', {
        nowMilliseconds: nowMs,
        text: describeTranscriptTextForLogging(text),
        state: this.getSnapshot(),
      });
      return this.isAnimating();
    }

    logTranscriptTextChange(this.state.canonicalText, text, nowMs, this.state.phase);

    logTranscriptAnimationEvent('set canonical text requested', {
      nowMilliseconds: nowMs,
      nextText: describeTranscriptTextForLogging(text),
      previousState: this.getSnapshot(),
    });
    this.state.canonicalText = text;
    this.restartTransition(this.state.installedText, text, nowMs);
    logTranscriptAnimationEvent('set canonical text completed', {
      state: this.getSnapshot(),
    });
    return this.isAnimating();
  }

  tick(nowMs: number): boolean {
    logTranscriptAnimationFrameEvent('tick started', {
      nowMilliseconds: nowMs,
      state: this.getSnapshot(),
    });
    while (this.state.plan !== null) {
      if (this.state.phase === 'erasing-old-tail') {
        const elapsedMs = nowMs - this.state.phaseStartedAtMs;
        const eraseRuns = buildEraseAlphaRuns(this.state.plan, elapsedMs, this.timing);
        this.applyAlphaRuns(eraseRuns);
        logTranscriptAnimationFrameEvent('erase phase tick applied alpha runs', {
          elapsedMilliseconds: elapsedMs,
          phaseDurationMilliseconds: getErasePhaseDurationMs(this.state.plan, this.timing),
          runs: eraseRuns,
          state: this.getSnapshot(),
        });

        if (elapsedMs < getErasePhaseDurationMs(this.state.plan, this.timing)) {
          return true;
        }

        this.swapToTarget(nowMs);
        continue;
      }

      if (this.state.phase === 'revealing-new-tail') {
        const elapsedMs = nowMs - this.state.phaseStartedAtMs;
        const revealRuns = buildRevealAlphaRuns(this.state.plan, elapsedMs, this.timing);
        this.applyAlphaRuns(revealRuns);
        logTranscriptAnimationFrameEvent('reveal phase tick applied alpha runs', {
          elapsedMilliseconds: elapsedMs,
          phaseDurationMilliseconds: getRevealPhaseDurationMs(this.state.plan, this.timing),
          runs: revealRuns,
          state: this.getSnapshot(),
        });

        if (elapsedMs < getRevealPhaseDurationMs(this.state.plan, this.timing)) {
          return true;
        }

        this.finishTransition();
        return false;
      }

      return false;
    }

    logTranscriptAnimationFrameEvent('tick finished with no active plan', {
      state: this.getSnapshot(),
    });
    return false;
  }

  private restartTransition(sourceText: string, targetText: string, nowMs: number): void {
    logTranscriptAnimationEvent('restart transition requested', {
      nowMilliseconds: nowMs,
      sourceText: describeTranscriptTextForLogging(sourceText),
      targetText: describeTranscriptTextForLogging(targetText),
    });
    this.clearAlphaRuns();

    const plan = buildTransitionPlan(sourceText, targetText);
    logTranscriptAnimationEvent('restart transition built plan', {
      plan,
    });
    if (plan.sourceText === plan.targetText) {
      this.sink.installText(plan.targetText);
      this.state.installedText = plan.targetText;
      this.state.plan = null;
      this.state.phase = 'idle';
      this.state.phaseStartedAtMs = nowMs;
      logTranscriptAnimationEvent('restart transition completed without animation', {
        state: this.getSnapshot(),
      });
      return;
    }

    this.sink.installText(plan.sourceText);
    this.state.installedText = plan.sourceText;
    this.state.plan = plan;
    this.state.phase = 'erasing-old-tail';
    this.state.phaseStartedAtMs = nowMs;
    logTranscriptAnimationEvent('restart transition entered erase phase', {
      state: this.getSnapshot(),
    });
  }

  private swapToTarget(nowMs: number): void {
    if (this.state.plan === null) {
      logTranscriptAnimationEvent('swap to target skipped because there is no active plan');
      return;
    }

    logTranscriptAnimationEvent('swap to target started', {
      nowMilliseconds: nowMs,
      plan: this.state.plan,
      stateBeforeSwap: this.getSnapshot(),
    });
    this.clearAlphaRuns();
    this.sink.beforeSwap?.(this.state.plan);
    this.sink.installText(this.state.plan.targetText);
    this.state.installedText = this.state.plan.targetText;
    this.sink.afterSwap?.(this.state.plan);
    this.state.phase = 'revealing-new-tail';
    this.state.phaseStartedAtMs = nowMs;
    logTranscriptAnimationEvent('swap to target completed', {
      state: this.getSnapshot(),
    });
  }

  private finishTransition(): void {
    logTranscriptAnimationEvent('finish transition started', {
      stateBeforeFinish: this.getSnapshot(),
    });
    this.clearAlphaRuns();
    this.state.plan = null;
    this.state.phase = 'idle';
    this.state.installedText = this.state.canonicalText;
    logTranscriptAnimationEvent('finish transition completed', {
      state: this.getSnapshot(),
    });
  }

  private applyAlphaRuns(runs: ByteAlphaRun[]): void {
    this.activeAlphaRuns = runs.map((run) => ({ ...run }));
    logTranscriptAnimationFrameEvent('apply alpha runs requested', {
      runs,
      state: this.getSnapshot(),
    });
    if (runs.length === 0) {
      this.sink.clearAlphaRuns();
      return;
    }

    this.sink.applyAlphaRuns(runs);
  }

  private clearAlphaRuns(): void {
    logTranscriptAnimationFrameEvent('clear alpha runs requested', {
      stateBeforeClear: this.getSnapshot(),
    });
    this.activeAlphaRuns = [];
    this.sink.clearAlphaRuns();
  }
}
