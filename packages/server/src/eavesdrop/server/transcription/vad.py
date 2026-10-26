"""Standalone Silero speech detection on original 16 kHz audio sample coordinates."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from threading import Lock
from typing import Literal, Protocol, cast

from eavesdrop.server.transcription.models import SpeechSegment, VadParameters
from eavesdrop.server.transcription.utils import SAMPLE_RATE, FloatAudio, disable_runtime_telemetry


class SileroOnnxModel(Protocol):
  """Stateful ONNX model returned by Silero's ONNX-only loader."""

  def reset_states(self, batch_size: int = 1) -> None: ...


class SileroLoader(Protocol):
  """Silero's ONNX-only model loader."""

  def __call__(self, *, onnx: Literal[True]) -> SileroOnnxModel: ...


class SpeechTimestamps(Protocol):
  """Silero's speech detection callable over original waveform sample coordinates."""

  def __call__(
    self,
    audio: FloatAudio,
    model: SileroOnnxModel,
    *,
    sampling_rate: int,
    threshold: float,
    neg_threshold: float | None,
    min_speech_duration_ms: int,
    max_speech_duration_s: float,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
  ) -> list[SpeechSegment]: ...


class SpeechDetector(Protocol):
  def detect(self, audio: FloatAudio, parameters: VadParameters) -> list[SpeechSegment]: ...


class SileroSpeechDetector:
  """Standalone CPU/ONNX Silero VAD, independent of the recognition model."""

  def __init__(self) -> None:
    disable_runtime_telemetry()
    torch = import_module("torch")
    get_num_threads = cast(Callable[[], int], torch.get_num_threads)
    set_num_threads = cast(Callable[[int], None], torch.set_num_threads)
    previous_threads = get_num_threads()
    try:
      silero = import_module("silero_vad")
      load = cast(SileroLoader, silero.load_silero_vad)
      self._speech_timestamps: SpeechTimestamps = cast(
        SpeechTimestamps, silero.get_speech_timestamps
      )
    finally:
      # Silero's module import sets PyTorch globally to one thread, even for
      # ONNX inference. Restore the host's policy so other CPU inference is unaffected.
      set_num_threads(previous_threads)
    self._model: SileroOnnxModel = load(onnx=True)
    self._lock: Lock = Lock()

  def detect(self, audio: FloatAudio, parameters: VadParameters) -> list[SpeechSegment]:
    """Return speech intervals as original audio sample offsets.

    :param audio: Mono float32 waveform at 16 kHz.
    :param parameters: Silence and speech thresholds.
    :returns: Ordered non-overlapping speech intervals in samples.
    """
    with self._lock:
      return self._speech_timestamps(
        audio,
        self._model,
        sampling_rate=SAMPLE_RATE,
        threshold=parameters.threshold,
        neg_threshold=parameters.neg_threshold,
        min_speech_duration_ms=parameters.min_speech_duration_ms,
        max_speech_duration_s=parameters.max_speech_duration_s,
        min_silence_duration_ms=parameters.min_silence_duration_ms,
        speech_pad_ms=parameters.speech_pad_ms,
      )
