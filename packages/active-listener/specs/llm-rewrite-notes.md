# LLM rewrite notes

## Goal

Use an LLM to rewrite finalized dictation text before emission.

## Inputs

- Finalized transcript text
- Optional related words / hints
- Optional style or voice guidance

## Questions

- What exact input payload should the rewrite step receive?
- What constraints should the rewrite preserve?
- What should happen when the model fails or times out?
- How should related words be represented?
- What config should be user-editable?

## Prompt notes

Add your system prompt ideas here.

## API / integration notes

Add Pydantic AI and OpenAI-compatible endpoint notes here.

## Risks / unknowns

- Over-rewriting meaning
- Hallucinated corrections
- Latency after recording stops
- Handling proper nouns, acronyms, and rhyme resolution
