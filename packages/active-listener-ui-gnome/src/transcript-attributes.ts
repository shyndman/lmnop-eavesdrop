type ByteAlphaRun = {
  startByte: number;
  endByte: number;
  alpha: number;
};

const PANGO_ALPHA_MAX = 65_535;

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

export const buildTranscriptAttributeSpecs = (
  text: string,
  runs: ByteAlphaRun[],
  colorHex: string,
): TranscriptAttributeSpec[] => {
  const textByteLength = new TextEncoder().encode(text).byteLength;
  if (textByteLength === 0) {
    return [];
  }

  const foregroundColor = parseTextColor(colorHex);
  foregroundColor.endByte = textByteLength;

  return [
    foregroundColor,
    ...runs.map((run) => ({
      kind: 'foreground-alpha' as const,
      startByte: run.startByte,
      endByte: run.endByte,
      alpha: clamp(run.alpha, 0, PANGO_ALPHA_MAX),
    })),
  ];
};
