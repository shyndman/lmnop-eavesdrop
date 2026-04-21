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
export const DEFAULT_TRANSITION_TIMING: TransitionTiming = {
  eraseStaggerMs: 18,
  revealStaggerMs: 18,
  fadeDurationMs: 120,
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
  }

  setCanonicalText(text: string, nowMs: number): boolean {
    if (this.state.phase === 'idle' && text === this.state.canonicalText) {
      return false;
    }

    this.state.canonicalText = text;
    this.restartTransition(this.state.installedText, text, nowMs);
    return this.isAnimating();
  }

  tick(nowMs: number): boolean {
    while (this.state.plan !== null) {
      if (this.state.phase === 'erasing-old-tail') {
        const elapsedMs = nowMs - this.state.phaseStartedAtMs;
        const eraseRuns = buildEraseAlphaRuns(this.state.plan, elapsedMs, this.timing);
        this.applyAlphaRuns(eraseRuns);

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

        if (elapsedMs < getRevealPhaseDurationMs(this.state.plan, this.timing)) {
          return true;
        }

        this.finishTransition();
        return false;
      }

      return false;
    }

    return false;
  }

  private restartTransition(sourceText: string, targetText: string, nowMs: number): void {
    this.clearAlphaRuns();

    const plan = buildTransitionPlan(sourceText, targetText);
    if (plan.sourceText === plan.targetText) {
      this.sink.installText(plan.targetText);
      this.state.installedText = plan.targetText;
      this.state.plan = null;
      this.state.phase = 'idle';
      this.state.phaseStartedAtMs = nowMs;
      return;
    }

    this.sink.installText(plan.sourceText);
    this.state.installedText = plan.sourceText;
    this.state.plan = plan;
    this.state.phase = 'erasing-old-tail';
    this.state.phaseStartedAtMs = nowMs;
  }

  private swapToTarget(nowMs: number): void {
    if (this.state.plan === null) {
      return;
    }

    this.clearAlphaRuns();
    this.sink.beforeSwap?.(this.state.plan);
    this.sink.installText(this.state.plan.targetText);
    this.state.installedText = this.state.plan.targetText;
    this.sink.afterSwap?.(this.state.plan);
    this.state.phase = 'revealing-new-tail';
    this.state.phaseStartedAtMs = nowMs;
  }

  private finishTransition(): void {
    this.clearAlphaRuns();
    this.state.plan = null;
    this.state.phase = 'idle';
    this.state.installedText = this.state.canonicalText;
  }

  private applyAlphaRuns(runs: ByteAlphaRun[]): void {
    this.activeAlphaRuns = runs.map((run) => ({ ...run }));
    if (runs.length === 0) {
      this.sink.clearAlphaRuns();
      return;
    }

    this.sink.applyAlphaRuns(runs);
  }

  private clearAlphaRuns(): void {
    this.activeAlphaRuns = [];
    this.sink.clearAlphaRuns();
  }
}
