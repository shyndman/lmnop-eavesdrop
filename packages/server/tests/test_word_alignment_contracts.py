"""Contract tests for word-alignment integration with CTranslate2 APIs."""

from collections.abc import Callable
from typing import NamedTuple, override, cast

from eavesdrop.server.transcription.models import SegmentDict, WordTimingDict
from eavesdrop.server.transcription.word_alignment import (
  WordAlignmentProcessor,
  WordTimestampAligner,
)


class _FakeAlignmentResult(NamedTuple):
  text_token_probs: list[float]
  alignments: list[tuple[int, int]]


class _FakeStorageView:
  """Minimal stand-in for ctranslate2.StorageView in unit tests."""


class _FakeWhisperModel:
  """Fake model recording align() inputs for assertion."""

  def __init__(self) -> None:
    self.received_text_tokens: list[list[int]] | None = None
    self.received_num_frames: list[int] | None = None

  def align(
    self,
    _features: _FakeStorageView,
    _start_sequence: list[int],
    text_tokens: list[list[int]],
    num_frames: list[int],
    *,
    median_filter_width: int = 7,
  ) -> list[_FakeAlignmentResult]:
    self.received_text_tokens = text_tokens
    self.received_num_frames = num_frames

    assert median_filter_width == 7
    alignment_results: list[_FakeAlignmentResult] = []
    for token_sequence in text_tokens:
      token_count = len(token_sequence) + 1
      alignment_results.append(
        _FakeAlignmentResult(
          text_token_probs=[0.9] * token_count,
          alignments=[(index, index * 2) for index in range(token_count)],
        )
      )
    return alignment_results


class _FakeTokenizer:
  """Minimal tokenizer API needed by WordAlignmentProcessor."""

  sot_sequence: list[int] = [50257]
  eot: int = 50256

  def split_to_word_tokens(self, tokens: list[int]) -> tuple[list[str], list[list[int]]]:
    non_eot_tokens = tokens[:-1]
    words = [f"word_{token}" for token in non_eot_tokens]
    word_tokens = [[token] for token in non_eot_tokens] + [[self.eot]]
    return words, word_tokens


class _RecordingAlignmentProcessor(WordAlignmentProcessor):
  """Test double that records add_word_timestamps alignment inputs."""

  def __init__(self) -> None:
    super().__init__(frames_per_second=100.0, tokens_per_second=50.0)
    self.received_text_tokens: list[list[int]] | None = None
    self.received_num_frames: int | None = None

  @override
  def find_alignment(
    self,
    model: object,
    tokenizer: object,
    text_tokens: list[list[int]],
    encoder_output: object,
    num_frames: int,
  ) -> list[list[WordTimingDict]]:
    self.received_text_tokens = text_tokens
    self.received_num_frames = num_frames
    return [[] for _ in text_tokens]


def _find_alignment(
  processor: WordAlignmentProcessor,
  model: _FakeWhisperModel,
  tokenizer: _FakeTokenizer,
  text_tokens: list[list[int]],
  encoder_output: _FakeStorageView,
  num_frames: int,
) -> list[list[WordTimingDict]]:
  find_alignment = cast(
    Callable[
      [_FakeWhisperModel, _FakeTokenizer, list[list[int]], _FakeStorageView, int],
      list[list[WordTimingDict]],
    ],
    processor.find_alignment,
  )
  return find_alignment(model, tokenizer, text_tokens, encoder_output, num_frames)


def _add_word_timestamps(
  aligner: WordTimestampAligner,
  *,
  segments: list[list[SegmentDict]],
  model: _FakeWhisperModel,
  tokenizer: _FakeTokenizer,
  encoder_output: _FakeStorageView,
  num_frames: int,
  last_speech_timestamp: float,
) -> float:
  add_word_timestamps = cast(
    Callable[
      [
        list[list[SegmentDict]],
        _FakeWhisperModel,
        _FakeTokenizer,
        _FakeStorageView,
        int,
        str,
        str,
        float,
      ],
      float,
    ],
    aligner.add_word_timestamps,
  )
  return add_word_timestamps(
    segments,
    model,
    tokenizer,
    encoder_output,
    num_frames,
    "",
    "",
    last_speech_timestamp,
  )


def _segment(seek: int, start: float, end: float, tokens: list[int]) -> SegmentDict:
  return {
    "seek": seek,
    "start": start,
    "end": end,
    "tokens": tokens,
  }


def test_find_alignment_batches_segment_sequences_for_ctranslate2_signature() -> None:
  processor = WordAlignmentProcessor(frames_per_second=100.0, tokens_per_second=50.0)
  model = _FakeWhisperModel()
  tokenizer = _FakeTokenizer()

  alignments = _find_alignment(
    processor, model, tokenizer, [[770, 318], [42]], _FakeStorageView(), 467
  )

  assert model.received_text_tokens == [[770, 318], [42]]
  assert model.received_num_frames == [467, 467]
  assert len(alignments) == 2
  assert alignments[0][0]["word"] == "word_770"
  assert alignments[0][1]["word"] == "word_318"


def test_add_word_timestamps_builds_one_token_sequence_per_segment_group() -> None:
  aligner = WordTimestampAligner(frames_per_second=100.0, tokens_per_second=50.0)
  recording_processor = _RecordingAlignmentProcessor()
  aligner.alignment_processor = recording_processor
  tokenizer = _FakeTokenizer()

  segments = [
    [
      _segment(seek=0, start=0.0, end=1.0, tokens=[11, 12, tokenizer.eot]),
      _segment(seek=0, start=1.0, end=2.0, tokens=[13, tokenizer.eot]),
    ],
    [
      _segment(seek=200, start=2.0, end=3.0, tokens=[21, tokenizer.eot]),
      _segment(seek=200, start=3.0, end=4.0, tokens=[22, 23, tokenizer.eot]),
    ],
  ]

  last_speech_timestamp = _add_word_timestamps(
    aligner,
    segments=segments,
    model=_FakeWhisperModel(),
    tokenizer=tokenizer,
    encoder_output=_FakeStorageView(),
    num_frames=467,
    last_speech_timestamp=1.5,
  )

  assert recording_processor.received_text_tokens == [[11, 12, 13], [21, 22, 23]]
  assert recording_processor.received_num_frames == 467
  assert last_speech_timestamp == 1.5
