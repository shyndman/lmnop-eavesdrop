import zlib
from collections.abc import Iterable
from typing import cast

import numpy as np
from numpy.typing import NDArray

from eavesdrop.server.transcription.models import SpeechChunk, WordTimingDict
from eavesdrop.server.transcription.vendor_types import (
  SpeechTimestampsMapLike,
  StorageViewLike,
  TokenizerLike,
  load_speech_timestamps_map,
  load_storage_view,
)
from eavesdrop.wire import Segment, Word

NumericArray = NDArray[np.generic]
VadSpeechChunk = dict[str, int]
SpeechTimestampsMap = load_speech_timestamps_map()
StorageView = load_storage_view()


def summarize_array(name: str, array: NumericArray) -> dict[str, object]:
  summary: dict[str, object] = {
    f"{name}_shape": tuple(int(dim) for dim in array.shape),
    f"{name}_dtype": str(array.dtype),
    f"{name}_size": int(array.size),
    f"{name}_contiguous": bool(array.flags["C_CONTIGUOUS"]),
  }

  if array.size == 0:
    summary[f"{name}_min"] = None
    summary[f"{name}_max"] = None
    return summary

  summary[f"{name}_min"] = float(np.min(array))
  summary[f"{name}_max"] = float(np.max(array))
  return summary


def restore_speech_timestamps(
  segments: Iterable[Segment],
  speech_chunks: list[SpeechChunk],
  sampling_rate: int,
) -> Iterable[Segment]:
  ts_map: SpeechTimestampsMapLike = SpeechTimestampsMap(
    cast(list[VadSpeechChunk], speech_chunks), sampling_rate
  )

  for segment in segments:
    if segment.words:
      words: list[Word] = []
      for word in segment.words:
        # Ensure the word start and end times are resolved to the same chunk.
        middle = (word.start + word.end) / 2
        chunk_index = ts_map.get_chunk_index(middle)
        word.start = ts_map.get_original_time(word.start, chunk_index)
        word.end = ts_map.get_original_time(word.end, chunk_index)
        words.append(word)

      segment.start = words[0].start
      segment.end = words[-1].end
      segment.words = words

    else:
      segment.start = ts_map.get_original_time(segment.start)
      segment.end = ts_map.get_original_time(segment.end)
  return segments


def finalize_recording_timestamps(
  segments: Iterable[Segment],
  absolute_stream_start: float,
) -> Iterable[Segment]:
  """Ensures all segments and words have recording-relative timestamps.

  This adds the recording-level offset to chunk-local timestamps and updates
  the internal time_offset field for compatibility.

  :param segments: The transcribed segments to normalize.
  :type segments: Iterable[Segment]
  :param absolute_stream_start: The offset of the current audio chunk in the recording.
  :type absolute_stream_start: float
  :return: The segments with recording-relative timestamps.
  :rtype: Iterable[Segment]
  """
  for segment in segments:
    segment.start += absolute_stream_start
    segment.end += absolute_stream_start
    segment.time_offset = absolute_stream_start

    if segment.words:
      for word in segment.words:
        word.start += absolute_stream_start
        word.end += absolute_stream_start

  return segments


def get_ctranslate2_storage(segment: NumericArray) -> StorageViewLike:
  contiguous_segment = np.ascontiguousarray(segment)
  storage_view = StorageView.from_array(cast(NDArray[np.float32], contiguous_segment))
  return storage_view


def get_compression_ratio(text: str) -> float:
  text_bytes = text.encode("utf-8")
  return len(text_bytes) / len(zlib.compress(text_bytes))


def get_suppressed_tokens(
  tokenizer: TokenizerLike,
  suppress_tokens: list[int] | None,
) -> list[int]:
  if suppress_tokens is not None and -1 in suppress_tokens:
    suppress_tokens = [t for t in suppress_tokens if t >= 0]
    suppress_tokens.extend(tokenizer.non_speech_tokens)
  elif suppress_tokens is None or len(suppress_tokens) == 0:
    suppress_tokens = []  # interpret empty string as an empty list

  suppress_tokens.extend(
    [
      tokenizer.transcribe,
      tokenizer.translate,
      tokenizer.sot,
      tokenizer.sot_prev,
      tokenizer.sot_lm,
    ]
  )

  return list(sorted(set(suppress_tokens)))


def merge_punctuations(alignment: list[WordTimingDict], prepended: str, appended: str) -> None:
  # merge prepended punctuations
  i = len(alignment) - 2
  j = len(alignment) - 1
  while i >= 0:
    previous = alignment[i]
    following = alignment[j]
    if previous["word"].startswith(" ") and previous["word"].strip() in prepended:
      # prepend it to the following word
      following["word"] = previous["word"] + following["word"]
      following["tokens"] = previous["tokens"] + following["tokens"]
      previous["word"] = ""
      previous["tokens"] = []
    else:
      j = i
    i -= 1

  # merge appended punctuations
  i = 0
  j = 1
  while j < len(alignment):
    previous = alignment[i]
    following = alignment[j]
    if not previous["word"].endswith(" ") and following["word"] in appended:
      # append it to the previous word
      previous["word"] = previous["word"] + following["word"]
      previous["tokens"] = previous["tokens"] + following["tokens"]
      following["word"] = ""
      following["tokens"] = []
    else:
      i = j
    j += 1
