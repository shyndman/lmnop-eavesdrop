# Coding Guidelines

Apply these guidelines alongside [CODE_STYLE.md](CODE_STYLE.md). Consult the affected packages' `GLOSSARY.md` files before introducing domain language, and record new terms there.

Apply these rules to all code affected by a change, including existing code that is modified, moved, copied, split, or reorganized. Evaluate the resulting code and its interactions, rather than limiting consideration to newly written code.

## Names and domain language

- Use one term for each concept across types, fields, operations, and documentation. Give an identifier the name of what it identifies, rather than its transport or mechanism. Introduce a new name only for a genuinely new concept, and record it in the relevant `GLOSSARY.md`.
- Name actual behavior and ownership. Make the D-Bus destination explicit when it distinguishes an operation. Use qualifiers such as “live” only for meaningful, supported distinctions.
- Make ownership transfers legible in names and contracts: an operation that returns an observation and clears the stored reference must reveal both effects.

## Responsibilities and construction

- Give each class a cohesive responsibility, reflected in its constructor dependencies. Keep recording and transcript logic independent of keyboard hardware; place hardware interaction with its appropriate owner.
- Treat broad keyword arguments and callbacks exposing orchestration state as signals to reconsider ownership. Pass dependencies needed for the class’s responsibility rather than distributing another object’s coordination across callbacks.
- Use dataclasses to model data. Use ordinary classes to coordinate behavior and lifecycle.

## Modules and useful abstraction

- Keep non-test code files at or below 500 lines. Split by responsibility, cohesive behavior, and genuinely shared logic; meet the limit without manufacturing classes or forwarding layers.
- Prefer simple existing encapsulation. Add wrappers or setters when they enforce a real invariant, manage a resource, or perform a required operation.
- Remove obsolete protocol choices and callback obligations when their purpose disappears. Remove a dead callback requirement from the API rather than satisfying it with a no-op.

## Fit the design to the problem

- Start with the required behavior and the code that already supports it. Choose the simplest design that expresses that behavior clearly.
- Introduce a new system to solve a concrete problem. Explain what the existing design cannot handle cleanly and how the new system makes that work simpler.
- Let actual requirements determine the machinery: how many participants exist, what varies, what must be coordinated, and how long state lives.
- Reuse existing concepts and mechanisms before creating parallel ones. Every new layer must earn its place by reducing the complexity of the required behavior.
- Treat changes to how callers use or observe the code as explicit behavior changes, including when they arise during a structural refactor.

## State and lifetime

- Make each state value’s initializer, updater, and clearer discoverable. Keep transitions reachable and explain the scenario requiring each change, including observation resets; base defensive resets on an actual lifecycle scenario.
- Keep connection identity distinct from recording identity. Make task and observation ownership, completion responsibilities, and handoffs explicit.
- Introduce queue filtering, draining, or requeuing only for an actual requirement. Explain the lifetime of retained and discarded events and which owner enforces it.

## Scope reduction and invariants

- Preserve behavioral invariants while removing or reorganizing code. Keep the single-segment-open-for-revision invariant; discuss and obtain approval before changing it.
- Implement English-only scope by removing language-detection machinery. Remove a constant-valued `ClientType` distinction when its sole purpose was a deleted workflow, while retaining operations still needed by supported behavior.

## Design questions stay explicit

Explain and seek agreement on open design questions, especially any change to an established behavioral invariant. Distinguish questions from settled decisions. Resolve conflicts with existing requirements explicitly before implementing a changed contract.
