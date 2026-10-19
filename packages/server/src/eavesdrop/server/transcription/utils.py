import os
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from eavesdrop.wire import Segment

SAMPLE_RATE = 16000
FloatAudio = NDArray[np.float32]
NumericArray = NDArray[np.generic]


def disable_runtime_telemetry() -> None:
  """Disable optional inference-library telemetry before importing their runtimes."""
  # Must be set BEFORE NeMo/Silero can import ONNX Runtime. The Python API alone
  # cannot prevent its initialization event. Official process-lifetime opt-out:
  # https://github.com/microsoft/onnxruntime/blob/main/docs/Privacy.md
  os.environ["ORT_DISABLE_TELEMETRY"] = "1"
  os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


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
