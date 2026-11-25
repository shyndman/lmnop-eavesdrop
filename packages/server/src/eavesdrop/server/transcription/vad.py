"""Standalone sherpa-onnx Silero speech detection on original 16 kHz audio sample coordinates."""

import hashlib
import os
from importlib import import_module
from pathlib import Path
from threading import Lock
from typing import Protocol, cast

import numpy as np

from eavesdrop.common import get_logger
from eavesdrop.server.transcription.models import SpeechChunk, VadParameters
from eavesdrop.server.transcription.utils import SAMPLE_RATE, FloatAudio, disable_runtime_telemetry

VAD_REPO_ID = "csukuangfj/vad"
VAD_REVISION = "fba88cd2e921609e7675c3aaf51e0b9b295da4bc"
VAD_FILENAME = "silero_vad_v5.onnx"
VAD_SHA256 = "6b99cbfd39246b6706f98ec13c7c50c6b299181f2474fa05cbc8046acc274396"
VAD_WINDOW_SIZE = 512
VAD_BUFFER_SIZE_SECONDS = 120.0

logger = get_logger("vad")


class HubModule(Protocol):
  def hf_hub_download(self, *, repo_id: str, filename: str, revision: str) -> str: ...


class SpeechSegment(Protocol):
  @property
  def start(self) -> int: ...

  @property
  def samples(self) -> list[float]: ...


class VoiceActivityDetector(Protocol):
  def accept_waveform(self, samples: FloatAudio) -> None: ...

  def flush(self) -> None: ...

  def empty(self) -> bool: ...

  @property
  def front(self) -> SpeechSegment: ...

  def pop(self) -> None: ...

  def reset(self) -> None: ...


class SileroVadModelConfigFactory(Protocol):
  def __call__(
    self,
    *,
    model: str,
    threshold: float,
    min_silence_duration: float,
    min_speech_duration: float,
    window_size: int,
    max_speech_duration: float,
  ) -> object: ...


class VadModelConfigFactory(Protocol):
  def __call__(
    self,
    *,
    silero_vad: object,
    sample_rate: int,
    num_threads: int,
    provider: str,
  ) -> object: ...


class VoiceActivityDetectorFactory(Protocol):
  def __call__(
    self,
    config: object,
    *,
    buffer_size_in_seconds: float,
  ) -> VoiceActivityDetector: ...


class SherpaVadModule(Protocol):
  SileroVadModelConfig: SileroVadModelConfigFactory
  VadModelConfig: VadModelConfigFactory
  VoiceActivityDetector: VoiceActivityDetectorFactory


class SpeechDetector(Protocol):
  def detect(self, audio: FloatAudio, parameters: VadParameters) -> list[SpeechChunk]: ...


def vad_model_path() -> Path:
  """Fetch and verify the pinned Silero v5 ONNX model in Hugging Face's cache.

  :returns: Local path to the verified ``silero_vad_v5.onnx`` model file.
  :rtype: Path
  :raises ValueError: If the downloaded file SHA256 disagrees with the pinned release.
  """
  os.environ["HF_HUB_DISABLE_XET"] = "1"
  hub = cast(HubModule, cast(object, import_module("huggingface_hub")))
  model_path = Path(
    hub.hf_hub_download(
      repo_id=VAD_REPO_ID,
      filename=VAD_FILENAME,
      revision=VAD_REVISION,
    )
  )
  with model_path.open("rb") as source:
    checksum = hashlib.file_digest(source, "sha256").hexdigest()
  if checksum != VAD_SHA256:
    logger.error("Silero VAD model checksum mismatch", path=str(model_path), checksum=checksum)
    raise ValueError("Silero VAD model SHA256 does not match pinned release")

  logger.info("Using verified Silero v5 ONNX VAD model", path=str(model_path))
  return model_path


class SileroSpeechDetector:
  """Standalone CPU/ONNX Silero VAD via sherpa-onnx, independent of the recognizer."""

  def __init__(self) -> None:
    disable_runtime_telemetry()
    sherpa = import_module("sherpa_onnx")
    self._sherpa: SherpaVadModule = cast(SherpaVadModule, cast(object, sherpa))
    self._model_path: Path = vad_model_path()
    self._detectors: dict[tuple[float, int, int, float], VoiceActivityDetector] = {}
    self._lock: Lock = Lock()

  def _get_detector(self, parameters: VadParameters) -> VoiceActivityDetector:
    key = (
      parameters.threshold,
      parameters.min_silence_duration_ms,
      parameters.min_speech_duration_ms,
      parameters.max_speech_duration_s,
    )
    detector = self._detectors.get(key)
    if detector is not None:
      return detector

    silero_config = self._sherpa.SileroVadModelConfig(
      model=str(self._model_path),
      threshold=parameters.threshold,
      min_silence_duration=parameters.min_silence_duration_ms / 1000.0,
      min_speech_duration=parameters.min_speech_duration_ms / 1000.0,
      window_size=VAD_WINDOW_SIZE,
      max_speech_duration=parameters.max_speech_duration_s,
    )
    vad_config = self._sherpa.VadModelConfig(
      silero_vad=silero_config,
      sample_rate=SAMPLE_RATE,
      num_threads=1,
      provider="cpu",
    )
    detector = self._sherpa.VoiceActivityDetector(
      vad_config,
      buffer_size_in_seconds=VAD_BUFFER_SIZE_SECONDS,
    )
    self._detectors[key] = detector
    return detector

  def detect(self, audio: FloatAudio, parameters: VadParameters) -> list[SpeechChunk]:
    """Return speech intervals as original audio sample offsets.

    :param audio: Mono float32 waveform at 16 kHz.
    :type audio: FloatAudio
    :param parameters: Silence and speech thresholds.
    :type parameters: VadParameters
    :returns: Ordered non-overlapping speech intervals in samples.
    :rtype: list[SpeechChunk]
    """
    if len(audio) == 0:
      return []

    with self._lock:
      vad = self._get_detector(parameters)
      vad.reset()
      try:
        for offset in range(0, len(audio), VAD_WINDOW_SIZE):
          window = audio[offset : offset + VAD_WINDOW_SIZE]
          if len(window) < VAD_WINDOW_SIZE:
            window = np.pad(window, (0, VAD_WINDOW_SIZE - len(window)))
          vad.accept_waveform(window)

        vad.flush()
        chunks: list[SpeechChunk] = []
        while not vad.empty():
          segment = vad.front
          start = segment.start
          end = min(start + len(segment.samples), len(audio))
          if end > start:
            chunks.append({"start": start, "end": end})
          vad.pop()
        return chunks
      finally:
        vad.reset()
