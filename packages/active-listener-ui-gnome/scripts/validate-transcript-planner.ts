import {
  buildGraphemeSpans,
  buildTransitionPlan,
} from '../src/transcript-animation.ts';

type PlannerCase = {
  name: string;
  source: string;
  target: string;
  expectedCommonPrefixCount: number;
  expectedSourceTail: string[];
  expectedTargetTail: string[];
};

const cases: PlannerCase[] = [
  {
    name: 'identical strings',
    source: 'hello world',
    target: 'hello world',
    expectedCommonPrefixCount: 11,
    expectedSourceTail: [],
    expectedTargetTail: [],
  },
  {
    name: 'append',
    source: 'hello',
    target: 'hello there',
    expectedCommonPrefixCount: 5,
    expectedSourceTail: [],
    expectedTargetTail: [' ', 't', 'h', 'e', 'r', 'e'],
  },
  {
    name: 'delete',
    source: 'hello there',
    target: 'hello',
    expectedCommonPrefixCount: 5,
    expectedSourceTail: [' ', 't', 'h', 'e', 'r', 'e'],
    expectedTargetTail: [],
  },
  {
    name: 'shared-prefix divergence',
    source: 'alpha beta',
    target: 'alpha zeta',
    expectedCommonPrefixCount: 6,
    expectedSourceTail: ['b', 'e', 't', 'a'],
    expectedTargetTail: ['z', 'e', 't', 'a'],
  },
  {
    name: 'emoji',
    source: 'A👍🏽',
    target: 'A👍🏽!',
    expectedCommonPrefixCount: 2,
    expectedSourceTail: [],
    expectedTargetTail: ['!'],
  },
  {
    name: 'combining-mark',
    source: 'Cafe\u0301 time',
    target: 'Cafe\u0301 now',
    expectedCommonPrefixCount: 5,
    expectedSourceTail: ['t', 'i', 'm', 'e'],
    expectedTargetTail: ['n', 'o', 'w'],
  },
  {
    name: 'multiline',
    source: 'line one\nline two',
    target: 'line one\nline three',
    expectedCommonPrefixCount: 15,
    expectedSourceTail: ['w', 'o'],
    expectedTargetTail: ['h', 'r', 'e', 'e'],
  },
];

const assertEqual = <T>(actual: T, expected: T, message: string): void => {
  if (actual !== expected) {
    throw new Error(`${message}: expected ${String(expected)}, got ${String(actual)}`);
  }
};

const assertArrayEqual = (actual: string[], expected: string[], message: string): void => {
  const actualJson = JSON.stringify(actual);
  const expectedJson = JSON.stringify(expected);
  if (actualJson !== expectedJson) {
    throw new Error(`${message}: expected ${expectedJson}, got ${actualJson}`);
  }
};

for (const testCase of cases) {
  const sourceSpans = buildGraphemeSpans(testCase.source);
  const targetSpans = buildGraphemeSpans(testCase.target);
  const plan = buildTransitionPlan(testCase.source, testCase.target);

  assertEqual(plan.commonPrefixCount, testCase.expectedCommonPrefixCount, `${testCase.name} common prefix`);
  assertArrayEqual(
    sourceSpans.slice(plan.commonPrefixCount).map((span) => span.text),
    testCase.expectedSourceTail,
    `${testCase.name} source tail`,
  );
  assertArrayEqual(
    targetSpans.slice(plan.commonPrefixCount).map((span) => span.text),
    testCase.expectedTargetTail,
    `${testCase.name} target tail`,
  );

  for (const spans of [sourceSpans, targetSpans]) {
    for (let index = 1; index < spans.length; index += 1) {
      assertEqual(spans[index - 1]?.endByte, spans[index]?.startByte, `${testCase.name} byte continuity ${index}`);
    }
  }

  console.log(`PASS ${testCase.name}`);
}
