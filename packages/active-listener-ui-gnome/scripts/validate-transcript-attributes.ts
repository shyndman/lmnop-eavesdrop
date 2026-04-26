import {
  INCOMPLETE_TRANSCRIPT_ALPHA,
  buildTranscriptAttributeSpecs,
  buildTranscriptDisplay,
  normalizeTranscriptRuns,
} from '../src/transcript-attributes.ts';

const assert = (condition: boolean, message: string): void => {
  if (!condition) {
    throw new Error(message);
  }
};

const normalizedRuns = normalizeTranscriptRuns([
  { text: ' hello ', isCommand: false, isComplete: true },
  { text: ' there ', isCommand: false, isComplete: true },
  { text: 'draft', isCommand: true, isComplete: false },
  { text: ' tail', isCommand: true, isComplete: false },
]);

assert(normalizedRuns.length === 2, 'normalization should merge adjacent runs with identical flags');
assert(normalizedRuns[0]?.text === 'hello there', 'normalization should trim and merge completed text');
assert(normalizedRuns[1]?.text === 'draft tail', 'normalization should trim and merge incomplete command text');

const completedOnlyDisplay = buildTranscriptDisplay([{ text: 'hello world', isCommand: false, isComplete: true }]);
assert(completedOnlyDisplay.text === 'hello world', 'completed-only display should preserve the combined text');
assert(completedOnlyDisplay.runs.length === 1, 'completed-only display should preserve one run');
assert(completedOnlyDisplay.runs[0]?.startByte === 0, 'completed-only display should start at byte zero');
assert(completedOnlyDisplay.runs[0]?.endByte === 11, 'completed-only display should measure UTF-8 byte length');

const mixedDisplay = buildTranscriptDisplay([
  { text: 'hello there', isCommand: false, isComplete: true },
  { text: 'pending', isCommand: true, isComplete: false },
]);
assert(mixedDisplay.text === 'hello there pending', 'display should concatenate ordered runs with single spaces');
assert(mixedDisplay.runs[1]?.startByte === 12, 'display should start the second run after the completed prefix');

const multibyteDisplay = buildTranscriptDisplay([
  { text: 'A👍🏽', isCommand: false, isComplete: true },
  { text: 'é', isCommand: true, isComplete: false },
]);
assert(
  multibyteDisplay.runs[1]?.startByte === new TextEncoder().encode('A👍🏽 ').byteLength,
  'display should measure byte offsets in UTF-8 bytes',
);

const specs = buildTranscriptAttributeSpecs(mixedDisplay, '#F5F7FA', '#C4B5FD');
const normalColor = specs.find(
  (spec): spec is Extract<(typeof specs)[number], { kind: 'foreground-color' }> => spec.kind === 'foreground-color' && spec.startByte === 0,
);
const commandColor = specs.find(
  (spec): spec is Extract<(typeof specs)[number], { kind: 'foreground-color' }> => spec.kind === 'foreground-color' && spec.startByte === 12,
);
const foregroundAlpha = specs.find(
  (spec): spec is Extract<(typeof specs)[number], { kind: 'foreground-alpha' }> => spec.kind === 'foreground-alpha',
);

assert(normalColor !== undefined, 'transcript attrs should include a normal-text foreground color');
assert(commandColor !== undefined, 'transcript attrs should include a command-text foreground color');
assert(foregroundAlpha !== undefined, 'transcript attrs should include an incomplete foreground alpha');

assert(
  commandColor?.red === 50_372 && commandColor.green === 46_517 && commandColor.blue === 65_021,
  'command foreground color should convert the command hex color into Pango 16-bit channels',
);
assert(
  foregroundAlpha?.startByte === 12 && foregroundAlpha.endByte === new TextEncoder().encode(mixedDisplay.text).byteLength,
  'foreground alpha should cover the incomplete UTF-8 byte range',
);
assert(foregroundAlpha?.alpha === INCOMPLETE_TRANSCRIPT_ALPHA, 'foreground alpha should apply the configured incomplete opacity');

console.log('Transcript display helpers preserve command color and dim incomplete runs.');
