"""Run an explicit real-model smoke test against a local 16 kHz mono WAV."""

import argparse
import json
import time
from importlib import import_module
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from eavesdrop.server.config import TranscriptionConfig
from eavesdrop.server.transcription.parakeet import ParakeetTranscriber
from eavesdrop.server.transcription.session import create_session
from eavesdrop.wire import Segment


class SoundFileReader(Protocol):
  def read(self, file: str, *, dtype: str) -> tuple[NDArray[np.float32], int]: ...


def _validate(segments: list[Segment], offset: float, duration: float) -> None:
  assert segments, "Speech sample produced no transcription"
  for segment in segments:
    assert segment.words, "Speech segment omitted native words"
    assert offset <= segment.start <= segment.end <= offset + duration
    for word in segment.words:
      assert segment.start <= word.start <= word.end <= segment.end


def main() -> None:
  """Load Parakeet and verify real transcription, silence, and native timestamps."""
  parser = argparse.ArgumentParser(description=__doc__)
  _ = parser.add_argument("audio", type=Path)
  _ = parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
  _ = parser.add_argument("--device-index", type=int, default=0)
  args = parser.parse_args()
  audio_path = cast(Path, args.audio)
  device = cast(Literal["auto", "cpu", "cuda"], args.device)
  device_index = cast(int, args.device_index)
  reader = cast(SoundFileReader, cast(object, import_module("soundfile")))
  waveform, sample_rate = reader.read(str(audio_path), dtype="float32")
  if sample_rate != 16000 or waveform.ndim != 1:
    raise ValueError("Smoke input must be a 16 kHz mono WAV")
  config = TranscriptionConfig(device=device, device_index=device_index)
  started = time.perf_counter()
  transcriber = ParakeetTranscriber.load(config)
  load_seconds = time.perf_counter() - started
  results: list[dict[str, object]] = []
  first_word_start = 0.0
  for offset in (0.0, 40.0):
    started = time.perf_counter()
    segments, info = transcriber.transcribe(
      waveform, config=config, absolute_stream_start=offset, session=create_session("smoke")
    )
    elapsed = time.perf_counter() - started
    _validate(segments, offset, info.duration)
    if offset == 0.0:
      first_word_start = segments[0].start
    results.append(
      {
        "offset": offset,
        "seconds": elapsed,
        "audio_seconds": info.duration,
        "realtime_factor": elapsed / info.duration,
        "segments": [segment.model_dump() for segment in segments],
      }
    )
  # Exercise the sub-second flush path using a real speech excerpt. NeMo's
  # default symmetric padding must not introduce artificial leading silence.
  excerpt_start = max(0, int((first_word_start - 0.08) * sample_rate))
  excerpt = waveform[excerpt_start : excerpt_start + int(0.8 * sample_rate)]
  short_segments, short_info = transcriber.transcribe(
    excerpt,
    config=config.model_copy(update={"use_vad": False}),
    absolute_stream_start=60.0,
    session=create_session("short-flush"),
  )
  assert short_info.duration < 1.0
  _validate(short_segments, 60.0, short_info.duration)
  silence, _ = transcriber.transcribe(
    np.zeros(16000, dtype=np.float32),
    config=config,
    absolute_stream_start=0.0,
    session=create_session("silence"),
  )
  assert not silence, "VAD silence should bypass recognition"
  print(
    json.dumps(
      {
        "load_seconds": load_seconds,
        "runs": results,
        "short_flush": [segment.model_dump() for segment in short_segments],
        "silence_passed": True,
      },
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
