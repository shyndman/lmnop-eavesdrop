"""Run an explicit real-model smoke test against a local 16 kHz mono WAV."""

import argparse
import json
import time
from importlib import import_module
from math import isclose
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from eavesdrop.common import get_logger
from eavesdrop.server.config import TranscriptionConfig, load_config_from_file
from eavesdrop.server.transcription.orukeet import OrukeetTranscriber
from eavesdrop.server.transcription.session import create_session
from eavesdrop.server.transcription.utils import SAMPLE_RATE, FloatAudio
from eavesdrop.wire import Segment

logger = get_logger("orukeet/smoke")
RECORDING_OFFSET_SECONDS = 40.0
TIMESTAMP_TOLERANCE_SECONDS = 1e-6


class SoundFileReader(Protocol):
  def read(self, file: str, *, dtype: str) -> tuple[NDArray[np.float32], int]: ...


def _validate(segments: list[Segment], offset: float, duration: float) -> None:
  assert segments, "Speech sample produced no transcription"
  for segment in segments:
    assert segment.words, "Speech segment omitted native words"
    assert offset <= segment.start <= segment.end <= offset + duration
    for word in segment.words:
      assert segment.start <= word.start <= word.end <= segment.end


def _validate_recording_offset(
  baseline: list[Segment], shifted: list[Segment], offset: float
) -> None:
  for original, current in zip(baseline, shifted, strict=True):
    assert original.text == current.text, "Recording offset changed recognized text"
    assert original.words is not None and current.words is not None
    for first, second in zip(original.words, current.words, strict=True):
      assert first.word == second.word, "Recording offset changed recognized words"
      assert isclose(second.start - first.start, offset, abs_tol=TIMESTAMP_TOLERANCE_SECONDS), (
        "Word start lost its recording offset"
      )
      assert isclose(second.end - first.end, offset, abs_tol=TIMESTAMP_TOLERANCE_SECONDS), (
        "Word end lost its recording offset"
      )


def _validate_edges(
  transcriber: OrukeetTranscriber,
  waveform: FloatAudio,
  config: TranscriptionConfig,
  first_word_start: float,
) -> list[Segment]:
  excerpt_start = max(0, int((first_word_start - 0.08) * SAMPLE_RATE))
  excerpt = waveform[excerpt_start : excerpt_start + int(0.8 * SAMPLE_RATE)]
  short_segments, short_info = transcriber.transcribe(
    excerpt,
    config=config.model_copy(update={"use_vad": False}),
    absolute_stream_start=60.0,
    session=create_session("short-flush"),
  )
  assert short_info.duration < 1.0
  _validate(short_segments, 60.0, short_info.duration)

  silence, _ = transcriber.transcribe(
    np.zeros(SAMPLE_RATE, dtype=np.float32),
    config=config,
    absolute_stream_start=0.0,
    session=create_session("silence"),
  )
  assert not silence, "VAD silence should bypass recognition"

  return short_segments


def main() -> None:
  """Load Orukeet and verify real transcription, silence, and native timestamps."""
  parser = argparse.ArgumentParser(description=__doc__)
  _ = parser.add_argument("audio", type=Path)
  _ = parser.add_argument("--config", required=True, type=Path)
  args = parser.parse_args()
  audio_path = cast(Path, args.audio)
  reader = cast(SoundFileReader, cast(object, import_module("soundfile")))
  waveform, sample_rate = reader.read(str(audio_path), dtype="float32")
  if sample_rate != SAMPLE_RATE or waveform.ndim != 1:
    logger.error("Smoke input must be a 16 kHz mono WAV", path=str(audio_path))
    raise ValueError("Smoke input must be a 16 kHz mono WAV")
  configured = load_config_from_file(cast(Path, args.config)).transcription
  config = configured.model_copy(update={"word_timestamps": True, "use_vad": True})
  started = time.perf_counter()
  transcriber = OrukeetTranscriber.load(config)
  load_seconds = time.perf_counter() - started
  results: list[dict[str, object]] = []
  first_word_start = 0.0
  baseline: list[Segment] = []
  for offset in (0.0, RECORDING_OFFSET_SECONDS):
    started = time.perf_counter()
    segments, info = transcriber.transcribe(
      waveform, config=config, absolute_stream_start=offset, session=create_session("smoke")
    )
    elapsed = time.perf_counter() - started
    _validate(segments, offset, info.duration)
    if offset == 0.0:
      first_word_start = segments[0].start
      baseline = segments
    else:
      _validate_recording_offset(baseline, segments, offset)
    results.append(
      {
        "offset": offset,
        "seconds": elapsed,
        "audio_seconds": info.duration,
        "realtime_factor": elapsed / info.duration,
        "segments": [segment.model_dump() for segment in segments],
      }
    )
  short_segments = _validate_edges(transcriber, waveform, config, first_word_start)
  print(
    json.dumps(
      {
        "load_seconds": load_seconds,
        "device": config.device,
        "runs": results,
        "short_flush": [segment.model_dump() for segment in short_segments],
        "silence_passed": True,
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
