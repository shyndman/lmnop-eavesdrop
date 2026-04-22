const PANGO_ALPHA_MAX = 65_535;
const INCOMPLETE_TRANSCRIPT_OPACITY = 0.54;
export const INCOMPLETE_TRANSCRIPT_ALPHA = Math.round(PANGO_ALPHA_MAX * INCOMPLETE_TRANSCRIPT_OPACITY);
const utf8Encoder = new TextEncoder();

export type TranscriptDisplay = {
  text: string;
  incompleteStartByte: number | null;
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

export const appendCompletedTranscript = (completedText: string, segmentText: string): string => {
  const normalizedSegmentText = segmentText.trim();
  if (normalizedSegmentText.length === 0) {
    return completedText;
  }

  if (completedText.length === 0) {
    return normalizedSegmentText;
  }

  return `${completedText} ${normalizedSegmentText}`;
};

export const buildTranscriptDisplay = (
  completedText: string,
  incompleteText: string,
): TranscriptDisplay => {
  if (incompleteText.length === 0) {
    return {
      text: completedText,
      incompleteStartByte: null,
    };
  }

  const separator = completedText.length === 0 ? '' : ' ';
  const prefix = `${completedText}${separator}`;
  return {
    text: `${prefix}${incompleteText}`,
    incompleteStartByte: utf8Encoder.encode(prefix).byteLength,
  };
};

export const buildTranscriptAttributeSpecs = (
  display: TranscriptDisplay,
  colorHex: string,
): TranscriptAttributeSpec[] => {
  const textByteLength = utf8Encoder.encode(display.text).byteLength;
  if (textByteLength === 0) {
    return [];
  }

  const foregroundColor = parseTextColor(colorHex);
  foregroundColor.endByte = textByteLength;

  const specs: TranscriptAttributeSpec[] = [foregroundColor];
  if (display.incompleteStartByte !== null) {
    specs.push({
      kind: 'foreground-alpha',
      startByte: display.incompleteStartByte,
      endByte: textByteLength,
      alpha: clamp(INCOMPLETE_TRANSCRIPT_ALPHA, 0, PANGO_ALPHA_MAX),
    });
  }

  return specs;
};
