"""Pure helpers for reducing per-recording transcription windows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from eavesdrop.wire import Segment


@dataclass(frozen=True)
class TimeSpan:
  """Closed interval on the recording timeline."""

  start_s: float
  end_s: float


@dataclass(frozen=True)
class TimedWord:
  """Transcript word preserved on the recording timeline."""

  text: str
  start_s: float
  end_s: float
  is_complete: bool


@dataclass(frozen=True)
class ClassifiedWord:
  """Transcript word labeled as normal text or command text."""

  text: str
  is_command: bool
  is_complete: bool


@dataclass(frozen=True)
class TextRun:
  """Normalized transcript run used by display and rewrite boundaries."""

  text: str
  is_command: bool
  is_complete: bool


class CommandTextWordTimestampError(RuntimeError):
  """Raised when command-text classification lacks required word timestamps."""


@dataclass
class RecordingReducerState:
  """Per-recording reducer state consumed by app policy.

  :ivar last_id: Last committed segment identifier.
  :vartype last_id: int | None
  :ivar completed_words: Stable transcript prefix already committed by the server.
  :vartype completed_words: list[TimedWord]
  :ivar incomplete_words: Words from the current unstable tail only.
  :vartype incomplete_words: list[TimedWord]
  :ivar closed_command_spans: Closed command-text spans on the recording timeline.
  :vartype closed_command_spans: list[TimeSpan]
  :ivar open_command_start_s: Recording-relative start time for an open command span.
  :vartype open_command_start_s: float | None
  :ivar first_word_start: Earliest committed word start time.
  :vartype first_word_start: float | None
  :ivar last_word_end: Latest committed word end time.
  :vartype last_word_end: float | None
  """

  last_id: int | None = None
  completed_words: list[TimedWord] = field(default_factory=list)
  incomplete_words: list[TimedWord] = field(default_factory=list)
  closed_command_spans: list[TimeSpan] = field(default_factory=list)
  open_command_start_s: float | None = None
  first_word_start: float | None = None
  last_word_end: float | None = None

  @property
  def has_command_text(self) -> bool:
    return bool(self.closed_command_spans) or self.open_command_start_s is not None

  @property
  def duration_seconds(self) -> float | None:
    if self.first_word_start is None or self.last_word_end is None:
      return None

    return self.last_word_end - self.first_word_start


@dataclass(frozen=True)
class TranscriptionUpdate:
  """Live transcript snapshot surfaced to UI integrations.

  :ivar runs: Ordered normalized transcript runs for the current recording state.
  :vartype runs: list[TextRun]
  """

  runs: list[TextRun]


@dataclass(frozen=True)
class SegmentReduction:
  """Committed segments extracted from the latest window.

  :ivar segments: Newly committed non-tail segments.
  :vartype segments: list[Segment]
  :ivar incomplete_segment: Current in-progress tail segment.
  :vartype incomplete_segment: Segment | None
  :ivar last_id: Identifier to store for the next reduction.
  :vartype last_id: int | None
  :ivar missing_last_id: True when the previous sentinel was absent.
  :vartype missing_last_id: bool
  """

  segments: list[Segment]
  incomplete_segment: Segment | None
  last_id: int | None
  missing_last_id: bool


def reduce_new_segments(
  segments: Sequence[Segment],
  last_id: int | None,
) -> SegmentReduction:
  """Return newly committed segments from a transcription window.

  The transcriber keeps the final segment as the in-progress tail, so only the
  completed prefix is eligible for emission.

  :param segments: Window returned by the transcriber.
  :type segments: Sequence[Segment]
  :param last_id: Previously committed segment identifier.
  :type last_id: int | None
  :returns: Newly committed non-tail segments and sentinel status.
  :rtype: SegmentReduction
  """

  incomplete_segment = segments[-1] if segments else None
  committed_prefix = list(segments[:-1])
  if incomplete_segment is None:
    return SegmentReduction(
      segments=[],
      incomplete_segment=None,
      last_id=last_id,
      missing_last_id=False,
    )

  if not committed_prefix:
    return SegmentReduction(
      segments=[],
      incomplete_segment=incomplete_segment,
      last_id=last_id,
      missing_last_id=False,
    )

  missing_last_id = False
  new_segments = committed_prefix
  if last_id is not None:
    sentinel_index = next(
      (index for index, segment in enumerate(committed_prefix) if segment.id == last_id),
      None,
    )
    if sentinel_index is None:
      missing_last_id = True
    else:
      new_segments = committed_prefix[sentinel_index + 1 :]

  return SegmentReduction(
    segments=new_segments,
    incomplete_segment=incomplete_segment,
    last_id=committed_prefix[-1].id,
    missing_last_id=missing_last_id,
  )


def build_transcription_update(state: RecordingReducerState) -> TranscriptionUpdate | None:
  """Build a normalized transcript snapshot for live UI consumers.

  :param state: Recording reducer state after the latest reduction was applied.
  :type state: RecordingReducerState
  :returns: Ordered normalized transcript snapshot, or ``None`` when no text is present.
  :rtype: TranscriptionUpdate | None
  """

  runs = build_text_runs(state)
  if not runs:
    return None

  return TranscriptionUpdate(runs=runs)


def apply_segment_reduction(state: RecordingReducerState, reduction: SegmentReduction) -> None:
  """Apply a reduced transcription window to the recording state.

  :param state: Accumulated reducer state for the current recording.
  :type state: RecordingReducerState
  :param reduction: Reduced transcription window to ingest.
  :type reduction: SegmentReduction
  """

  if state.has_command_text:
    _require_word_timestamps(reduction.segments)
    incomplete_segment = reduction.incomplete_segment
    if incomplete_segment is not None:
      _require_word_timestamps([incomplete_segment])

  for segment in reduction.segments:
    for word in segment_words(segment, is_complete=True):
      state.completed_words.append(word)
      if state.first_word_start is None or word.start_s < state.first_word_start:
        state.first_word_start = word.start_s
      if state.last_word_end is None or word.end_s > state.last_word_end:
        state.last_word_end = word.end_s

  incomplete_segment = reduction.incomplete_segment
  if incomplete_segment is None:
    state.incomplete_words = []
    return

  state.incomplete_words = segment_words(incomplete_segment, is_complete=False)


def classify_word(word: TimedWord, timeline: RecordingReducerState) -> bool:
  """Classify one transcript word as command text by midpoint-in-span.

  :param word: Word to classify.
  :type word: TimedWord
  :param timeline: Recording timeline state.
  :type timeline: RecordingReducerState
  :returns: ``True`` when the word midpoint falls inside command text.
  :rtype: bool
  """

  midpoint = (word.start_s + word.end_s) / 2

  for span in timeline.closed_command_spans:
    if span.start_s <= midpoint <= span.end_s:
      return True

  open_start_s = timeline.open_command_start_s
  if open_start_s is not None and open_start_s <= midpoint:
    return True

  return False


def classify_words(
  words: Sequence[TimedWord],
  timeline: RecordingReducerState,
) -> list[ClassifiedWord]:
  """Classify ordered transcript words against the command-text timeline.

  :param words: Ordered transcript words to classify.
  :type words: Sequence[TimedWord]
  :param timeline: Recording timeline state.
  :type timeline: RecordingReducerState
  :returns: Classified transcript words.
  :rtype: list[ClassifiedWord]
  """

  return [
    ClassifiedWord(
      text=word.text,
      is_command=classify_word(word, timeline),
      is_complete=word.is_complete,
    )
    for word in words
  ]


def classify_recording_words(state: RecordingReducerState) -> list[ClassifiedWord]:
  """Classify both committed words and the current unstable tail.

  :param state: Recording reducer state to classify.
  :type state: RecordingReducerState
  :returns: Ordered classified words across the full reducer state.
  :rtype: list[ClassifiedWord]
  """

  return [
    *classify_words(state.completed_words, state),
    *classify_words(state.incomplete_words, state),
  ]


def group_words(words: Sequence[ClassifiedWord]) -> list[TextRun]:
  """Group adjacent classified words into normalized transcript runs.

  :param words: Ordered classified words.
  :type words: Sequence[ClassifiedWord]
  :returns: Grouped transcript runs.
  :rtype: list[TextRun]
  """

  runs: list[TextRun] = []
  current_words: list[str] = []
  current_is_command: bool | None = None
  current_is_complete: bool | None = None

  for word in words:
    if current_is_command == word.is_command and current_is_complete == word.is_complete:
      current_words.append(word.text)
      continue

    if current_words:
      runs.append(
        TextRun(
          text=" ".join(current_words),
          is_command=bool(current_is_command),
          is_complete=bool(current_is_complete),
        )
      )

    current_words = [word.text]
    current_is_command = word.is_command
    current_is_complete = word.is_complete

  if current_words:
    runs.append(
      TextRun(
        text=" ".join(current_words),
        is_command=bool(current_is_command),
        is_complete=bool(current_is_complete),
      )
    )

  return runs


def normalize_runs(runs: Sequence[TextRun]) -> list[TextRun]:
  """Drop empty runs and merge adjacent runs with identical flags.

  :param runs: Candidate transcript runs.
  :type runs: Sequence[TextRun]
  :returns: Normalized transcript runs.
  :rtype: list[TextRun]
  """

  normalized_runs: list[TextRun] = []

  for run in runs:
    text = run.text.strip()
    if not text:
      continue

    if normalized_runs:
      previous_run = normalized_runs[-1]
      if previous_run.is_command == run.is_command and previous_run.is_complete == run.is_complete:
        normalized_runs[-1] = TextRun(
          text=f"{previous_run.text} {text}",
          is_command=run.is_command,
          is_complete=run.is_complete,
        )
        continue

    normalized_runs.append(
      TextRun(
        text=text,
        is_command=run.is_command,
        is_complete=run.is_complete,
      )
    )

  return normalized_runs


def build_text_runs(state: RecordingReducerState) -> list[TextRun]:
  """Build normalized transcript runs from the reducer state.

  :param state: Recording reducer state to render.
  :type state: RecordingReducerState
  :returns: Ordered normalized transcript runs.
  :rtype: list[TextRun]
  """

  completed_runs = group_words(classify_words(state.completed_words, state))
  incomplete_runs = group_words(classify_words(state.incomplete_words, state))
  return normalize_runs([*completed_runs, *incomplete_runs])


def build_completed_text_runs(state: RecordingReducerState) -> list[TextRun]:
  """Build normalized transcript runs for the committed transcript only.

  :param state: Recording reducer state to render.
  :type state: RecordingReducerState
  :returns: Ordered normalized transcript runs for committed text.
  :rtype: list[TextRun]
  """

  return normalize_runs(group_words(classify_words(state.completed_words, state)))


def segment_words(segment: Segment, *, is_complete: bool) -> list[TimedWord]:
  """Convert one segment into ordered transcript words.

  :param segment: Transcript segment to convert.
  :type segment: Segment
  :param is_complete: Completeness flag inherited from the source segment.
  :type is_complete: bool
  :returns: Ordered words for the segment.
  :rtype: list[TimedWord]
  """

  if segment.words is not None:
    return [
      TimedWord(
        text=word.word.strip(),
        start_s=word.start,
        end_s=word.end,
        is_complete=is_complete,
      )
      for word in segment.words
      if word.word.strip()
    ]

  text = segment.text.strip()
  if not text:
    return []

  return [
    TimedWord(
      text=text,
      start_s=segment.absolute_start_time,
      end_s=segment.absolute_end_time,
      is_complete=is_complete,
    )
  ]


def render_text(words: Sequence[TimedWord]) -> str:
  """Render transcript words for emission.

  :param words: Ordered transcript words.
  :type words: Sequence[TimedWord]
  :returns: Space-joined transcript text.
  :rtype: str
  """

  return " ".join(word.text for word in words)


def serialize_text_runs(runs: Sequence[TextRun]) -> str:
  """Serialize normalized transcript runs for rewrite input.

  :param runs: Ordered normalized transcript runs.
  :type runs: Sequence[TextRun]
  :returns: Flat rewrite input string.
  :rtype: str
  """

  serialized_parts: list[str] = []
  for run in normalize_runs(runs):
    if run.is_command:
      serialized_parts.append(f"<instruction>{run.text}</instruction>")
      continue
    serialized_parts.append(run.text)

  return " ".join(serialized_parts)


def _require_word_timestamps(segments: Sequence[Segment]) -> None:
  for segment in segments:
    if segment.text.strip() and not segment.words:
      raise CommandTextWordTimestampError(
        f"command-text recording requires Segment.words for segment {segment.id}"
      )
