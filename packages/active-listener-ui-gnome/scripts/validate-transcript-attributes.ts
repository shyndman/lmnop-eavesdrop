import {
  INCOMPLETE_TRANSCRIPT_ALPHA,
  appendCompletedTranscript,
  buildTranscriptAttributeSpecs,
  buildTranscriptDisplay,
} from '../src/transcript-attributes.ts';

const assert = (condition: boolean, message: string): void => {
  if (!condition) {
    throw new Error(message);
  }
};

assert(appendCompletedTranscript('', ' hello ') === 'hello', 'append should trim the first completed segment');
assert(
  appendCompletedTranscript('hello', ' there ') === 'hello there',
  'append should separate completed segments with a single space',
);
assert(appendCompletedTranscript('hello', '   ') === 'hello', 'append should ignore empty completed segments');

const completedOnlyDisplay = buildTranscriptDisplay('hello world', '');
assert(completedOnlyDisplay.text === 'hello world', 'completed-only display should preserve the combined text');
assert(completedOnlyDisplay.incompleteStartByte === null, 'completed-only display should not mark an incomplete range');

const mixedDisplay = buildTranscriptDisplay('hello there', 'pending');
assert(mixedDisplay.text === 'hello there pending', 'display should concatenate completed and incomplete text');
assert(mixedDisplay.incompleteStartByte === 12, 'display should start the incomplete range after the completed prefix');

const multibyteDisplay = buildTranscriptDisplay('A👍🏽', 'é');
assert(
  multibyteDisplay.incompleteStartByte === new TextEncoder().encode('A👍🏽 ').byteLength,
  'display should measure incomplete byte offsets in UTF-8 bytes',
);

const specs = buildTranscriptAttributeSpecs(mixedDisplay, '#F5F7FA');
const mixedDisplayByteLength = new TextEncoder().encode(mixedDisplay.text).byteLength;

const foregroundColor = specs.find(
  (spec): spec is Extract<(typeof specs)[number], { kind: 'foreground-color' }> => spec.kind === 'foreground-color',
);
const foregroundAlpha = specs.find(
  (spec): spec is Extract<(typeof specs)[number], { kind: 'foreground-alpha' }> => spec.kind === 'foreground-alpha',
);

assert(foregroundColor !== undefined, 'transcript attrs should preserve a foreground color for the full text range');
if (foregroundColor === undefined) {
  throw new Error('transcript attrs should preserve a foreground color for the full text range');
}
assert(
  foregroundColor.startByte === 0 && foregroundColor.endByte === mixedDisplayByteLength,
  'foreground color should cover the full UTF-8 byte range',
);
assert(
  foregroundColor.red === 62_965 && foregroundColor.green === 63_479 && foregroundColor.blue === 64_250,
  'foreground color should convert the overlay hex color into Pango 16-bit channels',
);
assert(foregroundAlpha !== undefined, 'transcript attrs should include an incomplete foreground alpha');
if (foregroundAlpha === undefined) {
  throw new Error('transcript attrs should include an incomplete foreground alpha');
}
assert(
  foregroundAlpha.startByte === 12 && foregroundAlpha.endByte === mixedDisplayByteLength,
  'foreground alpha should cover the incomplete UTF-8 byte range',
);
assert(foregroundAlpha.alpha === INCOMPLETE_TRANSCRIPT_ALPHA, 'foreground alpha should apply the configured incomplete opacity');

const baseSpecs = buildTranscriptAttributeSpecs(completedOnlyDisplay, '#F5F7FA');
assert(baseSpecs.length === 1, 'transcript attrs should preserve a base foreground color even without alpha runs');
const baseForegroundColor = baseSpecs[0];
assert(baseForegroundColor?.kind === 'foreground-color', 'base transcript attrs should still include a foreground color');
assert(baseForegroundColor.startByte === 0 && baseForegroundColor.endByte === 11, 'base foreground color should cover the full UTF-8 byte range');

console.log('Transcript display helpers preserve completed text and dim the incomplete tail.');
