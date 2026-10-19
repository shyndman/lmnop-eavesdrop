"""Standalone Silero speech detection on original 16 kHz audio sample coordinates."""

from importlib import import_module
from threading import Lock
from typing import Protocol, cast

from eavesdrop.server.transcription.models import SpeechChunk, VadParameters
from eavesdrop.server.transcription.utils import SAMPLE_RATE, FloatAudio, disable_runtime_telemetry


class TorchThreads(Protocol):
  def get_num_threads(self) -> int: ...
  def set_num_threads(self, count: int) -> None: ...


class SileroRuntime(Protocol):
  def load_silero_vad(self, *, onnx: bool) -> object: ...
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
  ) -> list[SpeechChunk]: ...


class SpeechDetector(Protocol):
  def detect(self, audio: FloatAudio, parameters: VadParameters) -> list[SpeechChunk]: ...


class SileroSpeechDetector:
  """Standalone CPU/ONNX Silero VAD, independent of the recognition model."""

  def __init__(self) -> None:
    disable_runtime_telemetry()
    threads = cast(TorchThreads, cast(object, import_module("torch")))
    previous_threads = threads.get_num_threads()
    try:
      self._runtime: SileroRuntime = cast(SileroRuntime, cast(object, import_module("silero_vad")))
    finally:
      # Silero's module import sets PyTorch globally to one thread, even for
      # ONNX inference. Restore the host's policy so CPU Parakeet is unaffected.
      threads.set_num_threads(previous_threads)
    self._model: object = self._runtime.load_silero_vad(onnx=True)
    self._lock: Lock = Lock()

  def detect(self, audio: FloatAudio, parameters: VadParameters) -> list[SpeechChunk]:
    """Return speech intervals as original audio sample offsets.

    :param audio: Mono float32 waveform at 16 kHz.
    :param parameters: Silence and speech thresholds.
    :returns: Ordered non-overlapping speech intervals in samples.
    """
    with self._lock:
      return self._runtime.get_speech_timestamps(
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
