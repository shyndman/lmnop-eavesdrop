"""Model-independent Orukeet decoding, VAD, and serialization contracts."""

import hashlib
import tarfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from eavesdrop.server.config import TranscriptionConfig
from eavesdrop.server.transcription import orukeet
from eavesdrop.server.transcription.models import SpeechChunk, VadParameters
from eavesdrop.server.transcription.orukeet import OrukeetTranscriber
from eavesdrop.server.transcription.session import create_session
from eavesdrop.server.transcription.utils import FloatAudio


@dataclass
class Result:
  text: str = "Hello."
  tokens: list[str] = field(default_factory=lambda: ["▁Hel", "lo", "."])
  timestamps: list[float] = field(default_factory=lambda: [0.08, 0.16, 0.24])
  durations: list[float] = field(default_factory=lambda: [0.08, 0.08, 0.0])


@dataclass
class Recognizer:
  audio: list[FloatAudio] = field(default_factory=list)
  entered: Event | None = None
  release: Event | None = None
  error: bool = False

  def recognize(self, audio: FloatAudio) -> Result:
    self.audio.append(audio)
    if self.entered is not None:
      self.entered.set()
    if self.release is not None:
      assert self.release.wait(5)
    if self.error:
      raise RuntimeError("decoder failed")

    return Result()


@dataclass
class Detector:
  spans: list[SpeechChunk]

  def detect(self, audio: FloatAudio, parameters: VadParameters) -> list[SpeechChunk]:
    return self.spans


def test_vad_gaps_keep_original_recording_timeline() -> None:
  model = Recognizer()
  adapter = OrukeetTranscriber(
    model, Detector([{"start": 16000, "end": 32000}, {"start": 64000, "end": 80000}])
  )
  segments, info = adapter.transcribe(
    np.zeros(96000, dtype=np.float32),
    config=TranscriptionConfig(),
    absolute_stream_start=40,
    session=create_session("gaps"),
  )
  assert [segment.start for segment in segments] == [41.08, 44.08]
  assert info.duration == 6
  assert info.duration_after_vad == 2
  assert len(model.audio) == 2


def test_silence_bypasses_recognizer() -> None:
  model = Recognizer()
  segments, info = OrukeetTranscriber(model, Detector([])).transcribe(
    np.zeros(16000, dtype=np.float32),
    config=TranscriptionConfig(),
    absolute_stream_start=0,
    session=create_session("silence"),
  )
  assert segments == []
  assert info.duration_after_vad == 0
  assert model.audio == []


@pytest.mark.parametrize("samples", [4000, 8000, 16000])
def test_short_final_audio_is_not_shifted_or_padded(samples: int) -> None:
  model = Recognizer()
  segments, _ = OrukeetTranscriber(model, None).transcribe(
    np.ones(samples, dtype=np.float32),
    config=TranscriptionConfig(use_vad=False),
    absolute_stream_start=60,
    session=create_session("final"),
  )
  assert len(model.audio[0]) == samples
  assert segments[0].start == 60.08
  assert segments[0].end == 60.24


def test_shared_recognizer_serializes_concurrent_sessions() -> None:
  entered, release = Event(), Event()
  model = Recognizer(entered=entered, release=release)
  adapter = OrukeetTranscriber(model, None)

  def recognize(offset: float) -> float:
    segments, _ = adapter.transcribe(
      np.ones(16000, dtype=np.float32),
      config=TranscriptionConfig(use_vad=False),
      absolute_stream_start=offset,
      session=create_session(str(offset)),
    )
    return segments[0].start

  with ThreadPoolExecutor(max_workers=2) as executor:
    first = executor.submit(recognize, 10)
    assert entered.wait(5)
    second = executor.submit(recognize, 20)
    try:
      assert len(model.audio) == 1
    finally:
      release.set()
    assert first.result() == 10.08
    assert second.result() == 20.08


def test_decoder_error_releases_adapter_for_next_request() -> None:
  model = Recognizer(error=True)
  adapter = OrukeetTranscriber(model, None)
  with pytest.raises(RuntimeError, match="decoder failed"):
    adapter.transcribe(
      np.ones(16000, dtype=np.float32),
      config=TranscriptionConfig(use_vad=False),
      absolute_stream_start=0,
      session=create_session("error"),
    )
  model.error = False
  segments, _ = adapter.transcribe(
    np.ones(16000, dtype=np.float32),
    config=TranscriptionConfig(use_vad=False),
    absolute_stream_start=0,
    session=create_session("recovered"),
  )
  assert segments


def test_missing_detector_and_invalid_audio_are_rejected() -> None:
  adapter = OrukeetTranscriber(Recognizer(), None)
  with pytest.raises(RuntimeError, match="speech detector"):
    adapter.transcribe(
      np.ones(16000, dtype=np.float32),
      config=TranscriptionConfig(),
      absolute_stream_start=0,
      session=create_session("missing"),
    )
  with pytest.raises(ValueError, match="finite mono"):
    adapter.transcribe(
      np.array([np.nan], dtype=np.float32),
      config=TranscriptionConfig(use_vad=False),
      absolute_stream_start=0,
      session=create_session("invalid"),
    )


def test_word_output_can_be_disabled_without_losing_native_segment_bounds() -> None:
  segments, _ = OrukeetTranscriber(Recognizer(), None).transcribe(
    np.ones(16000, dtype=np.float32),
    config=TranscriptionConfig(use_vad=False, word_timestamps=False),
    absolute_stream_start=5,
    session=create_session("no-words"),
  )
  assert segments[0].words is None
  assert (segments[0].start, segments[0].end) == (5.08, 5.24)


def test_empty_final_chunk_does_not_decode() -> None:
  model = Recognizer()
  segments, info = OrukeetTranscriber(model, None).transcribe(
    np.zeros(0, dtype=np.float32),
    config=TranscriptionConfig(use_vad=False),
    absolute_stream_start=5,
    session=create_session("empty-final"),
  )
  assert segments == []
  assert info.duration == 0
  assert model.audio == []


def test_interrupted_asset_extraction_is_not_published_and_can_retry(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  source = tmp_path / orukeet.ARCHIVE
  source.mkdir()
  for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
    _ = (source / name).write_bytes(bytes(range(256)) * 256)
  archive = tmp_path / "model.tar.bz2"
  with tarfile.open(archive, "w:bz2") as output:
    output.add(source, arcname=orukeet.ARCHIVE)
  complete = archive.read_bytes()

  class Hub:
    def hf_hub_download(self, *, repo_id: str, filename: str, revision: str) -> str:
      return str(archive)

  monkeypatch.setattr(orukeet, "import_module", lambda name: Hub())
  interrupted = complete[: len(complete) // 2]
  _ = archive.write_bytes(interrupted)
  checksum = hashlib.sha256(interrupted).hexdigest()
  monkeypatch.setattr(orukeet, "SHA256", checksum)
  with pytest.raises((tarfile.ReadError, EOFError, OSError)):
    _ = orukeet.model_assets()
  assert not (tmp_path / f"{orukeet.ARCHIVE}-{checksum[:12]}").exists()
  _ = archive.write_bytes(complete)
  monkeypatch.setattr(orukeet, "SHA256", hashlib.sha256(complete).hexdigest())
  assets = orukeet.model_assets()
  assert (assets / "joiner.int8.onnx").read_bytes() == (source / "joiner.int8.onnx").read_bytes()
  assert orukeet.model_assets() == assets
