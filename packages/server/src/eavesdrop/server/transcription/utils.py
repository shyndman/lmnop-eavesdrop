from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from eavesdrop.wire import Segment

NumericArray = NDArray[np.generic]


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


def finalize_recording_timestamps(
  segments: Iterable[Segment],
  absolute_stream_start: float,
) -> Iterable[Segment]:
  """Ensures all segments and words have recording-relative timestamps.

  This is the final canonical timestamp boundary for WebSocket/live output. It
  adds the recording-level offset to chunk-local timestamps and updates the
  internal time_offset field for compatibility.

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
