const PANGO_ALPHA_MAX = 65_535;
const INCOMPLETE_TRANSCRIPT_OPACITY = 0.54;
export const INCOMPLETE_TRANSCRIPT_ALPHA = Math.round(PANGO_ALPHA_MAX * INCOMPLETE_TRANSCRIPT_OPACITY);
const utf8Encoder = new TextEncoder();

export type TranscriptRun = {
  text: string;
  isCommand: boolean;
  isComplete: boolean;
};

export type TranscriptDisplayRun = {
  text: string;
  isCommand: boolean;
  isComplete: boolean;
  startByte: number;
  endByte: number;
};

export type TranscriptDisplay = {
  text: string;
  runs: TranscriptDisplayRun[];
};

export type TranscriptAttributeSpec =
  | {
      kind: 'foreground-alpha';
      startByte: number;
      endByte: number;
      alpha: number;
    }
  | {
      kind: 'foreground-color';
      startByte: number;
      endByte: number;
      red: number;
      green: number;
      blue: number;
    };

const clamp = (value: number, minimum: number, maximum: number): number =>
  Math.min(Math.max(value, minimum), maximum);

const TEXT_COLOR_PATTERN = /^#(?<red>[\da-fA-F]{2})(?<green>[\da-fA-F]{2})(?<blue>[\da-fA-F]{2})$/;
const toPangoChannel = (value: number): number => value * 257;

const parseTextColor = (colorHex: string): TranscriptAttributeSpec & { kind: 'foreground-color' } => {
  const match = TEXT_COLOR_PATTERN.exec(colorHex);
  if (match?.groups === undefined) {
    throw new Error(`invalid transcript color: ${colorHex}`);
  }

  return {
    kind: 'foreground-color',
    startByte: 0,
    endByte: 0,
    red: toPangoChannel(Number.parseInt(match.groups.red, 16)),
    green: toPangoChannel(Number.parseInt(match.groups.green, 16)),
    blue: toPangoChannel(Number.parseInt(match.groups.blue, 16)),
  };
};

export const normalizeTranscriptRuns = (runs: TranscriptRun[]): TranscriptRun[] => {
  const normalizedRuns: TranscriptRun[] = [];

  for (const run of runs) {
    const text = run.text.trim();
    if (text.length === 0) {
      continue;
    }

    const previousRun = normalizedRuns.at(-1);
    if (
      previousRun !== undefined &&
      previousRun.isCommand === run.isCommand &&
      previousRun.isComplete === run.isComplete
    ) {
      previousRun.text = `${previousRun.text} ${text}`;
      continue;
    }

    normalizedRuns.push({
      text,
      isCommand: run.isCommand,
      isComplete: run.isComplete,
    });
  }

  return normalizedRuns;
};

export const buildTranscriptDisplay = (runs: TranscriptRun[]): TranscriptDisplay => {
  const normalizedRuns = normalizeTranscriptRuns(runs);
  if (normalizedRuns.length === 0) {
    return {
      text: '',
      runs: [],
    };
  }

  let currentByte = 0;
  let text = '';
  const displayRuns: TranscriptDisplayRun[] = [];

  for (const run of normalizedRuns) {
    const separator = text.length === 0 ? '' : ' ';
    text += `${separator}${run.text}`;
    currentByte += separator.length;

    const startByte = currentByte;
    currentByte += utf8Encoder.encode(run.text).byteLength;
    displayRuns.push({
      text: run.text,
      isCommand: run.isCommand,
      isComplete: run.isComplete,
      startByte,
      endByte: currentByte,
    });
  }

  return {
    text,
    runs: displayRuns,
  };
};

export const buildTranscriptAttributeSpecs = (
  display: TranscriptDisplay,
  normalColorHex: string,
  commandColorHex: string,
): TranscriptAttributeSpec[] => {
  if (display.runs.length === 0) {
    return [];
  }

  const specs: TranscriptAttributeSpec[] = [];
  for (const run of display.runs) {
    const foregroundColor = parseTextColor(run.isCommand ? commandColorHex : normalColorHex);
    foregroundColor.startByte = run.startByte;
    foregroundColor.endByte = run.endByte;
    specs.push(foregroundColor);

    if (!run.isComplete) {
      specs.push({
        kind: 'foreground-alpha',
        startByte: run.startByte,
        endByte: run.endByte,
        alpha: clamp(INCOMPLETE_TRANSCRIPT_ALPHA, 0, PANGO_ALPHA_MAX),
      });
    }
  }

  return specs;
};
