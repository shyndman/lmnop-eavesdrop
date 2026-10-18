"""Deterministic text shaping for active-listener transcript runs.

These transforms used to live only inside the finalizer, so the live overlay
showed raw, un-shaped transcript while speaking. They now run on every
transcription window (feeding the live preview) and once more on the final
flushed window, so what the user sees while speaking is exactly the content
that gets serialized for the LLM. Shaping is intentionally ignorant of the LLM:
instruction markers and command-dropping are applied at the LLM boundary, not
here (see ``recording/reducer.py`` serializers).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace

from active_listener.infra.corrections import CorrectionMap, apply_corrections
from active_listener.recording.reducer import TextRun, normalize_runs

# Spoken symbol words recognized by `_replace_symbols`. DO NOT expand this list
# casually — it is deliberately tiny so that normal prose words never collide.
_SYMBOL_WORDS = {
  "backslash": "\\",
  "dot": ".",
  "hashtag": "#",
  "slash": "/",
  "tild": "~",
  "tilde": "~",
}
_SYMBOL_PATTERN = re.compile(
  r"\s*\b(" + "|".join(_SYMBOL_WORDS) + r")\b\s*",
  re.IGNORECASE,
)

# Case-insensitive whole-phrase replacements applied BEFORE symbol fusion.
# Keys are matched case-insensitively; values are substituted verbatim with
# their canonical casing. Add freely — keep keys lowercase for readability.
_REPLACEMENTS: dict[str, str] = {
  "debass": "D-Bus",
  "hillary": "hilary",
  "tild": "tilde",
  "yamel": "yaml",
}
_REPLACEMENT_PATTERN = (
  re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _REPLACEMENTS) + r")\b",
    re.IGNORECASE,
  )
  if _REPLACEMENTS
  else None
)


def shape_runs(runs: Sequence[TextRun], corrections: CorrectionMap) -> list[TextRun]:
  """Apply deterministic text shaping to every run, per run.

  Transforms apply within a single run, so a symbol phrase split across a
  command/normal boundary will not fuse. ``normalize_runs`` drops empty runs
  and merges adjacent runs with identical flags after shaping.

  :param runs: Transcript runs to shape.
  :param corrections: Stored corrections applied after the built-in transforms.
  :returns: Shaped, normalized runs.
  """
  return normalize_runs([replace(run, text=_shape_text(run.text, corrections)) for run in runs])


def _shape_text(text: str, corrections: CorrectionMap) -> str:
  text = _apply_replacements(text)  # replacements BEFORE symbol fusion (existing order)
  text = _replace_symbols(text)
  text = apply_corrections(text, corrections)
  return text


def _apply_replacements(text: str) -> str:
  # Case-insensitive whole-word/phrase substitution driven by `_REPLACEMENTS`.
  # Extend the map at module level; this function should stay dumb.
  if _REPLACEMENT_PATTERN is None:
    return text
  return _REPLACEMENT_PATTERN.sub(lambda m: _REPLACEMENTS[m.group(1).lower()], text)


def _replace_symbols(text: str) -> str:
  #! This satisfies the design intent. Do not touch.
  #
  # Fuse spoken symbol words into their glyphs, swallowing adjacent whitespace:
  # e.g. "tild slash dot omp slash agent slash skills" -> "~/.omp/agent/skills".
  return _SYMBOL_PATTERN.sub(lambda m: _SYMBOL_WORDS[m.group(1).lower()], text)
