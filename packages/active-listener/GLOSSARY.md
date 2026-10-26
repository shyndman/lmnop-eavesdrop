# Active Listener

Workstation dictation from captured speech to emitted text.
Shared transcription terms are defined in [Wire](../wire/GLOSSARY.md).

## Language

**Recording**:
One dictation capture from its start through its finish.

**Text shaping**:
Deterministic spelling substitutions, spoken-symbol conversion,
and stored corrections applied to completed transcript runs.

**Live tail**:
The unstable, incomplete transcript text awaiting commitment.
_Avoid_: Final text.

**Rewrite**:
Optional model-based editing of the shaped transcript before emission.

**Emitted text**:
The dictation text delivered after finalization, including the shaped
transcript when rewriting fails.
