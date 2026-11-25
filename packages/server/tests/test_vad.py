"""Contracts for standalone sherpa-onnx Silero v5 ONNX VAD wrapper."""

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eavesdrop.server.transcription import vad
from eavesdrop.server.transcription.models import VadParameters
from eavesdrop.server.transcription.utils import FloatAudio


@dataclass
class _FakeSpeechSegment:
  start: int
  samples: list[float]


@dataclass
class _FakeDetector:
  config: object
  buffer_size_in_seconds: float
  accepted: list[FloatAudio] = field(default_factory=list)
  flushed: int = 0
  resets: int = 0
  raise_on_accept: bool = False
  segments: list[_FakeSpeechSegment] = field(default_factory=list)
  _queue: list[_FakeSpeechSegment] = field(default_factory=list, init=False)

  def accept_waveform(self, samples: FloatAudio) -> None:
    if self.raise_on_accept:
      raise RuntimeError("vad failure")
    self.accepted.append(samples.copy())

  def flush(self) -> None:
    self.flushed += 1
    self._queue = list(self.segments)

  def empty(self) -> bool:
    return len(self._queue) == 0

  @property
  def front(self) -> _FakeSpeechSegment:
    return self._queue[0]

  def pop(self) -> None:
    _ = self._queue.pop(0)

  def reset(self) -> None:
    self.resets += 1
    self._queue.clear()


@dataclass
class _FakeSherpa:
  silero_calls: list[dict[str, object]] = field(default_factory=list)
  vad_calls: list[dict[str, object]] = field(default_factory=list)
  detectors: list[_FakeDetector] = field(default_factory=list)
  next_segments: list[list[_FakeSpeechSegment]] = field(default_factory=list)

  def SileroVadModelConfig(self, **kwargs: object) -> SimpleNamespace:
    self.silero_calls.append(kwargs)
    return SimpleNamespace(**kwargs)

  def VadModelConfig(self, **kwargs: object) -> SimpleNamespace:
    self.vad_calls.append(kwargs)
    return SimpleNamespace(**kwargs)

  def VoiceActivityDetector(
    self, config: object, *, buffer_size_in_seconds: float
  ) -> _FakeDetector:
    segments = self.next_segments.pop(0) if self.next_segments else []
    detector = _FakeDetector(
      config=config,
      buffer_size_in_seconds=buffer_size_in_seconds,
      segments=segments,
    )
    self.detectors.append(detector)
    return detector


def test_vad_model_path_verifies_pinned_sha256(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model_file = tmp_path / vad.VAD_FILENAME
  payload = b"silero-v5-onnx"
  _ = model_file.write_bytes(payload)
  checksum = hashlib.sha256(payload).hexdigest()

  downloads: list[tuple[str, str, str]] = []

  def fake_download(*, repo_id: str, filename: str, revision: str) -> str:
    downloads.append((repo_id, filename, revision))
    return str(model_file)

  monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
  monkeypatch.setattr(vad, "VAD_SHA256", checksum)
  monkeypatch.setattr(
    vad,
    "import_module",
    lambda _name: SimpleNamespace(hf_hub_download=fake_download),
  )

  resolved = vad.vad_model_path()

  assert resolved == model_file
  assert os.environ["HF_HUB_DISABLE_XET"] == "1"
  assert downloads == [(vad.VAD_REPO_ID, vad.VAD_FILENAME, vad.VAD_REVISION)]


def test_vad_model_path_rejects_checksum_mismatch(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model_file = tmp_path / vad.VAD_FILENAME
  _ = model_file.write_bytes(b"corrupt")

  monkeypatch.setattr(
    vad,
    "import_module",
    lambda _name: SimpleNamespace(hf_hub_download=lambda **_kwargs: str(model_file)),
  )

  with pytest.raises(ValueError, match="SHA256"):
    _ = vad.vad_model_path()


def test_detector_feeds_512_sample_windows_pads_tail_and_caches_by_params(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model_file = tmp_path / vad.VAD_FILENAME
  _ = model_file.write_bytes(b"ok")

  fake_sherpa = _FakeSherpa(
    next_segments=[
      [_FakeSpeechSegment(start=512, samples=[0.1] * 400)],
      [_FakeSpeechSegment(start=0, samples=[0.2] * 300)],
    ]
  )
  telemetry_calls = 0

  def fake_disable_telemetry() -> None:
    nonlocal telemetry_calls
    telemetry_calls += 1

  monkeypatch.setattr(vad, "disable_runtime_telemetry", fake_disable_telemetry)
  monkeypatch.setattr(vad, "vad_model_path", lambda: model_file)
  monkeypatch.setattr(vad, "import_module", lambda _name: fake_sherpa)

  speech_detector = vad.SileroSpeechDetector()
  params_a = VadParameters(
    threshold=0.45,
    min_silence_duration_ms=800,
    min_speech_duration_ms=100,
    max_speech_duration_s=15.0,
  )
  params_b = VadParameters(
    threshold=0.45,
    min_silence_duration_ms=1200,
    min_speech_duration_ms=100,
    max_speech_duration_s=15.0,
  )

  audio = np.arange(700, dtype=np.float32)
  first = speech_detector.detect(audio, params_a)
  second = speech_detector.detect(np.ones(600, dtype=np.float32), params_a)
  third = speech_detector.detect(np.ones(512, dtype=np.float32), params_b)

  assert telemetry_calls == 1
  assert len(fake_sherpa.detectors) == 2

  assert fake_sherpa.silero_calls[0] == {
    "model": str(model_file),
    "threshold": 0.45,
    "min_silence_duration": 0.8,
    "min_speech_duration": 0.1,
    "window_size": vad.VAD_WINDOW_SIZE,
    "max_speech_duration": 15.0,
  }
  assert fake_sherpa.silero_calls[1]["min_silence_duration"] == 1.2
  assert fake_sherpa.vad_calls[0]["provider"] == "cpu"
  assert fake_sherpa.vad_calls[0]["num_threads"] == 1
  assert fake_sherpa.vad_calls[0]["sample_rate"] == 16000

  det_a = fake_sherpa.detectors[0]
  assert [len(win) for win in det_a.accepted] == [512, 512, 512, 512]
  assert all(det_a.accepted[1][i] == 0.0 for i in range(700 - 512, 512))
  assert det_a.flushed == 2
  assert det_a.resets == 4

  assert first == [{"start": 512, "end": 700}]
  assert second == [{"start": 512, "end": 600}]
  assert third == [{"start": 0, "end": 300}]


def test_detector_empty_audio_short_circuits_and_errors_reset_state(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  model_file = tmp_path / vad.VAD_FILENAME
  _ = model_file.write_bytes(b"ok")
  fake_sherpa = _FakeSherpa()

  monkeypatch.setattr(vad, "vad_model_path", lambda: model_file)
  monkeypatch.setattr(vad, "import_module", lambda _name: fake_sherpa)

  speech_detector = vad.SileroSpeechDetector()
  params = VadParameters()

  assert speech_detector.detect(np.zeros(0, dtype=np.float32), params) == []
  assert fake_sherpa.detectors == []

  # Force a detector creation that raises inside accept_waveform and verify reset() still runs.
  _ = speech_detector.detect(np.zeros(512, dtype=np.float32), params)
  detector = fake_sherpa.detectors[0]
  detector.raise_on_accept = True

  with pytest.raises(RuntimeError, match="vad failure"):
    _ = speech_detector.detect(np.ones(512, dtype=np.float32), params)

  assert detector.resets == 4
