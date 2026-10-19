"""Model-independent contracts for the standalone Silero VAD adapter."""

import os
from dataclasses import dataclass, field
from typing import NamedTuple
from unittest.mock import patch

import numpy as np
import pytest

from eavesdrop.server.transcription.models import SpeechChunk, VadParameters
from eavesdrop.server.transcription.utils import FloatAudio
from eavesdrop.server.transcription.vad import SileroSpeechDetector


@dataclass
class FakeTorchThreads:
  """Record changes to the host's Torch thread policy.

  :param count: Initial host thread count.
  """

  count: int = 8
  restored_counts: list[int] = field(default_factory=list)

  def get_num_threads(self) -> int:
    """Return the current host thread count."""
    return self.count

  def set_num_threads(self, count: int) -> None:
    """Record and apply a thread-policy restoration.

    :param count: Requested host thread count.
    """
    self.restored_counts.append(count)
    self.count = count


class DetectionCall(NamedTuple):
  """Capture the waveform, model, and exact arguments supplied to Silero."""

  audio: FloatAudio
  model: object
  sampling_rate: int
  threshold: float
  neg_threshold: float | None
  min_speech_duration_ms: int
  max_speech_duration_s: float
  min_silence_duration_ms: int
  speech_pad_ms: int


@dataclass
class FakeSileroRuntime:
  """Record model loading and return original-waveform speech intervals.

  :param speech: Prepared speech intervals in original input samples.
  """

  speech: list[SpeechChunk] = field(default_factory=list)
  model: object = field(default_factory=object)
  load_calls: list[bool] = field(default_factory=list)
  detect_calls: list[DetectionCall] = field(default_factory=list)

  def load_silero_vad(self, *, onnx: bool) -> object:
    """Record the chosen backend and return the prepared model.

    :param onnx: Whether to use the ONNX backend.
    :returns: Prepared model passed through to detection.
    """
    self.load_calls.append(onnx)
    return self.model

  def get_speech_timestamps(
    self,
    audio: FloatAudio,
    model: object,
    *,
    sampling_rate: int,
    threshold: float,
    neg_threshold: float | None,
    min_speech_duration_ms: int,
    max_speech_duration_s: float,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
  ) -> list[SpeechChunk]:
    """Record all detection arguments and return prepared sample intervals.

    :param audio: Original input waveform.
    :param model: Previously loaded VAD model.
    :param sampling_rate: Waveform sample rate.
    :param threshold: Speech probability threshold.
    :param neg_threshold: Silence probability threshold.
    :param min_speech_duration_ms: Minimum speech duration.
    :param max_speech_duration_s: Maximum speech duration.
    :param min_silence_duration_ms: Minimum silence duration.
    :param speech_pad_ms: Padding around detected speech.
    :returns: Speech boundaries in original input samples.
    """
    self.detect_calls.append(
      DetectionCall(
        audio,
        model,
        sampling_rate,
        threshold,
        neg_threshold,
        min_speech_duration_ms,
        max_speech_duration_s,
        min_silence_duration_ms,
        speech_pad_ms,
      )
    )
    return self.speech.copy()


class ImportCall(NamedTuple):
  """Capture telemetry settings at the moment a dependency is imported."""

  name: str
  ort_telemetry_disabled: str | None
  hub_telemetry_disabled: str | None


@dataclass
class FakeImports:
  """Simulate Silero's import-time Torch thread mutation without heavy imports.

  :param torch: Host thread-policy double.
  :param silero: Prepared Silero runtime.
  :param fail_silero_import: Whether the Silero import raises after changing threads.
  """

  torch: FakeTorchThreads
  silero: FakeSileroRuntime
  fail_silero_import: bool = False
  calls: list[ImportCall] = field(default_factory=list)

  def import_module(self, name: str) -> FakeTorchThreads | FakeSileroRuntime:
    """Record import-time environment and simulate dependency initialization.

    :param name: Requested dependency module.
    :returns: Prepared Torch or Silero runtime.
    :raises ImportError: If the simulated Silero import fails.
    :raises AssertionError: If an unexpected dependency is imported.
    """
    self.calls.append(
      ImportCall(
        name, os.environ.get("ORT_DISABLE_TELEMETRY"), os.environ.get("HF_HUB_DISABLE_TELEMETRY")
      )
    )
    if name == "torch":
      return self.torch
    if name == "silero_vad":
      self.torch.count = 1
      if self.fail_silero_import:
        raise ImportError("Simulated Silero import failure")
      return self.silero
    raise AssertionError(f"Unexpected runtime import: {name}")


def test_initialization_disables_telemetry_restores_threads_and_loads_onnx(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Imports see opt-outs, and Silero cannot leave Torch limited to one thread.

  :param monkeypatch: Restore environment variables after this test.
  """
  monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "0")
  monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "0")
  torch = FakeTorchThreads(count=8)
  silero = FakeSileroRuntime()
  imports = FakeImports(torch, silero)

  with patch("eavesdrop.server.transcription.vad.import_module", new=imports.import_module):
    _ = SileroSpeechDetector()

  assert imports.calls == [ImportCall("torch", "1", "1"), ImportCall("silero_vad", "1", "1")]
  assert torch.count == 8
  assert torch.restored_counts == [8]
  assert silero.load_calls == [True]


def test_failed_silero_import_restores_threads_and_propagates_error(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A partial dependency import cannot change the host's thread policy.

  :param monkeypatch: Restore environment variables after this test.
  """
  monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "0")
  monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "0")
  torch = FakeTorchThreads(count=6)
  silero = FakeSileroRuntime()
  imports = FakeImports(torch, silero, fail_silero_import=True)

  with patch("eavesdrop.server.transcription.vad.import_module", new=imports.import_module):
    with pytest.raises(ImportError, match="Simulated Silero import failure"):
      _ = SileroSpeechDetector()

  assert torch.count == 6
  assert torch.restored_counts == [6]
  assert silero.load_calls == []


def test_detect_passes_all_parameters_and_preserves_original_sample_offsets(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """VAD forwards its configuration and retains gaps on the input sample timeline.

  :param monkeypatch: Restore environment variables after this test.
  """
  monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "0")
  monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "0")
  speech: list[SpeechChunk] = [
    {"start": 1600, "end": 4800},
    {"start": 16000, "end": 24000},
  ]
  silero = FakeSileroRuntime(speech)
  imports = FakeImports(FakeTorchThreads(), silero)
  audio = np.linspace(-0.5, 0.5, num=32000, dtype=np.float32)
  parameters = VadParameters(
    threshold=0.72,
    neg_threshold=0.33,
    min_speech_duration_ms=120,
    max_speech_duration_s=4.5,
    min_silence_duration_ms=650,
    speech_pad_ms=240,
  )

  with patch("eavesdrop.server.transcription.vad.import_module", new=imports.import_module):
    detector = SileroSpeechDetector()
  result = detector.detect(audio, parameters)

  assert result == speech
  assert len(silero.detect_calls) == 1
  call = silero.detect_calls[0]
  assert call.audio is audio
  assert call.model is silero.model
  assert call.sampling_rate == 16000
  assert call.threshold == 0.72
  assert call.neg_threshold == 0.33
  assert call.min_speech_duration_ms == 120
  assert call.max_speech_duration_s == 4.5
  assert call.min_silence_duration_ms == 650
  assert call.speech_pad_ms == 240
