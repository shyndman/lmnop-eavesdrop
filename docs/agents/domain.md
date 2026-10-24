# Domain docs

## Before exploring a domain

Read the root `GLOSSARY-MAP.md`, then the package glossaries relevant
to the task. Follow the map's relationships for cross-package work.

Read relevant system-wide decisions in `docs/adr/` and package
decisions in `packages/<package>/docs/adr/`.

Proceed silently when a glossary or ADR directory is absent.
Create ADRs when an actual architectural decision is resolved.

## Layout

- `GLOSSARY-MAP.md`: context index and relationships.
- `packages/<package>/GLOSSARY.md`: vocabulary owned by that package.
- `docs/adr/`: system-wide decisions.
- `packages/<package>/docs/adr/`: package-scoped decisions.

## Vocabulary

Use glossary terms in issues, proposals, hypotheses, and tests.
Reference the owning glossary for shared terms.

When a needed term is missing, check existing project language.
Record resolved domain vocabulary through `/domain-modeling`.

## ADR conflicts

Explicitly identify any proposal that contradicts an existing ADR,
including the decision and the reason for reopening it.
