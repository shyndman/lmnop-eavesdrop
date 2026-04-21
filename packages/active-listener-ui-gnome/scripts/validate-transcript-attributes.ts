import { buildTranscriptAttributeSpecs } from '../src/transcript-attributes.ts';

const assert = (condition: boolean, message: string): void => {
  if (!condition) {
    throw new Error(message);
  }
};

const specs = buildTranscriptAttributeSpecs(
  'hello world',
  [
    {
      startByte: 0,
      endByte: 11,
      alpha: 0,
    },
  ],
  '#F5F7FA',
);

const foregroundColor = specs.find(
  (spec): spec is Extract<(typeof specs)[number], { kind: 'foreground-color' }> => spec.kind === 'foreground-color',
);
const foregroundAlpha = specs.find(
  (spec): spec is Extract<(typeof specs)[number], { kind: 'foreground-alpha' }> => spec.kind === 'foreground-alpha',
);

assert(foregroundColor !== undefined, 'transcript attrs should preserve a foreground color for the full text range');
assert(foregroundColor.startByte === 0 && foregroundColor.endByte === 11, 'foreground color should cover the full UTF-8 byte range');
assert(
  foregroundColor.red === 62_965 && foregroundColor.green === 63_479 && foregroundColor.blue === 64_250,
  'foreground color should convert the overlay hex color into Pango 16-bit channels',
);
assert(foregroundAlpha !== undefined, 'transcript attrs should include per-run foreground alpha');
assert(foregroundAlpha.startByte === 0 && foregroundAlpha.endByte === 11, 'foreground alpha should preserve byte boundaries');
assert(foregroundAlpha.alpha === 0, 'foreground alpha should keep caller-provided alpha values');

console.log('Transcript attributes preserve the overlay foreground color.');
