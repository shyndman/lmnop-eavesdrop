"""Config-selected Orukeet CPU or Vulkan inference with standalone Silero VAD."""

import hashlib
import os
import tarfile
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Protocol, cast

import numpy as np

from eavesdrop.common import get_logger
from eavesdrop.server.config import TranscriptionConfig
from eavesdrop.server.transcription.models import SpeechChunk, TranscriptionInfo
from eavesdrop.server.transcription.orukeet_timestamps import Hypothesis, native_segments
from eavesdrop.server.transcription.orukeet_vulkan import VulkanRecognizer
from eavesdrop.server.transcription.session import TranscriptionSession
from eavesdrop.server.transcription.utils import (
  SAMPLE_RATE,
  FloatAudio,
  disable_runtime_telemetry,
  finalize_recording_timestamps,
)
from eavesdrop.server.transcription.vad import SileroSpeechDetector, SpeechDetector
from eavesdrop.wire import Segment

MODEL_NAME = "oruk/orukeet"
REVISION = "55a984d46f68323301837194ce647c702f55facc"
ARCHIVE = "sherpa-onnx-orukeet-v0.1.0-int8"
SHA256 = "f9191f30178cc9122ce2f023bf9fefafc822028307b0efa4caff645ba3fe8d0a"
logger = get_logger("orukeet")
_asset_lock = Lock()


class HubModule(Protocol):
  def hf_hub_download(self, *, repo_id: str, filename: str, revision: str) -> str: ...


class RecognitionStream(Protocol):
  @property
  def result(self) -> Hypothesis: ...

  def accept_waveform(self, sample_rate: int, waveform: FloatAudio) -> None: ...


class Recognizer(Protocol):
  def create_stream(self) -> RecognitionStream: ...
  def decode_stream(self, stream: RecognitionStream) -> None: ...


class RecognizerFactory(Protocol):
  def from_transducer(
    self,
    *,
    encoder: str,
    decoder: str,
    joiner: str,
    tokens: str,
    model_type: str,
    feature_dim: int,
    sample_rate: int,
    provider: str,
    decoding_method: str,
  ) -> Recognizer: ...


class SherpaModule(Protocol):
  OfflineRecognizer: RecognizerFactory


class RecognitionBackend(Protocol):
  """Offline recognition with waveform-local native timing."""

  def recognize(self, audio: FloatAudio) -> Hypothesis:
    """Decode one contiguous mono waveform.

    :param audio: Float32 samples at the model sample rate.
    :returns: Recognized text and native timed pieces.
    """
    ...


class _CPURecognizer:
  def __init__(self, model: Recognizer) -> None:
    self._model: Recognizer = model

  def recognize(self, audio: FloatAudio) -> Hypothesis:
    """Decode through sherpa-onnx with native token timing.

    :param audio: Contiguous mono float32 samples at 16 kHz.
    :returns: Native sherpa token hypothesis.
    """
    stream = self._model.create_stream()
    stream.accept_waveform(SAMPLE_RATE, audio)
    self._model.decode_stream(stream)

    return stream.result


def model_assets() -> Path:
  """Fetch and safely extract the hash-verified INT8 release in Hugging Face's cache.

  :returns: Directory containing the encoder, decoder, joiner, and token vocabulary.
  :raises ValueError: If the archive checksum or required model files disagree.
  """
  # Xet reconstruction returned HTTP 416 for this pinned release; use Hub's HTTP transport.
  os.environ["HF_HUB_DISABLE_XET"] = "1"
  hub = cast(HubModule, cast(object, import_module("huggingface_hub")))
  archive = Path(
    hub.hf_hub_download(repo_id=MODEL_NAME, filename=f"onnx/{ARCHIVE}.tar.bz2", revision=REVISION)
  )
  target = archive.parent / f"{ARCHIVE}-{SHA256[:12]}"
  required = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")
  with _asset_lock:
    if all((target / ARCHIVE / name).is_file() for name in required):
      logger.info("Reusing cached Orukeet INT8 model", path=str(target / ARCHIVE))
      return target / ARCHIVE
    with archive.open("rb") as source:
      checksum = hashlib.file_digest(source, "sha256").hexdigest()
    if checksum != SHA256:
      raise ValueError("Orukeet archive SHA256 does not match pinned release")
    logger.info("Extracting pinned Orukeet INT8 model", revision=REVISION)
    with TemporaryDirectory(prefix=f".{ARCHIVE}-", dir=archive.parent) as temporary:
      staging = Path(temporary) / "assets"
      staging.mkdir()
      with tarfile.open(archive, "r:bz2") as source:
        source.extractall(staging, filter="data")
      nested = staging / ARCHIVE
      if not all((nested / name).is_file() for name in required):
        raise ValueError("Orukeet archive is missing required model files")
      try:
        _ = staging.rename(target)
      except OSError:
        # Another process published its complete extraction first.
        if not all((target / ARCHIVE / name).is_file() for name in required):
          raise
    return target / ARCHIVE


class OrukeetTranscriber:
  """Serialize recognition while preserving recording time across VAD gaps.

  :param model: Loaded CPU or Vulkan recognition backend.
  :param detector: Standalone VAD detector, or None when VAD is disabled.
  """

  def __init__(self, model: RecognitionBackend, detector: SpeechDetector | None) -> None:
    self._model: RecognitionBackend = model
    self._detector: SpeechDetector | None = detector
    self._lock: Lock = Lock()

  @classmethod
  def load(cls, config: TranscriptionConfig) -> "OrukeetTranscriber":
    """Load the configured CPU INT8 ONNX or Vulkan Q8 GGUF recognizer.

    :param config: Validated model and VAD settings.
    :returns: Ready shared inference adapter.
    """
    disable_runtime_telemetry()
    try:
      logger.info("Loading Orukeet", model=config.model, device=config.device)
      model: RecognitionBackend
      if config.device == "vulkan":
        model = VulkanRecognizer()
      else:
        sherpa = cast(SherpaModule, cast(object, import_module("sherpa_onnx")))
        assets = model_assets()
        model = _CPURecognizer(
          sherpa.OfflineRecognizer.from_transducer(
            encoder=str(assets / "encoder.int8.onnx"),
            decoder=str(assets / "decoder.int8.onnx"),
            joiner=str(assets / "joiner.int8.onnx"),
            tokens=str(assets / "tokens.txt"),
            model_type="nemo_transducer",
            feature_dim=128,
            sample_rate=SAMPLE_RATE,
            provider="cpu",
            decoding_method="greedy_search",
          )
        )
      adapter = cls(model, SileroSpeechDetector() if config.use_vad else None)
      logger.info("Orukeet ready", model=config.model, device=config.device)
      return adapter
    except Exception:
      logger.exception("Failed to load Orukeet", device=config.device)
      raise

  def transcribe(
    self,
    audio: FloatAudio,
    *,
    config: TranscriptionConfig,
    absolute_stream_start: float,
    session: TranscriptionSession,
  ) -> tuple[list[Segment], TranscriptionInfo]:
    """Decode each VAD span and add its recording offset exactly once.

    :param audio: Finite mono float32 audio at 16 kHz.
    :param config: Session output and VAD settings.
    :param absolute_stream_start: Recording-relative start of this audio chunk.
    :param session: Per-session tracing context.
    :returns: Recording-relative segments and original-chunk speech metadata.
    """
    if audio.ndim != 1 or not np.all(np.isfinite(audio)):
      logger.error("Rejecting non-finite or non-mono Orukeet input")
      raise ValueError("Orukeet requires finite mono audio")
    duration = len(audio) / SAMPLE_RATE
    session.update_audio_context(start_offset=absolute_stream_start, duration=duration)
    if config.use_vad and self._detector is None:
      logger.error("Orukeet VAD requested without a speech detector")
      raise RuntimeError("VAD was requested without a speech detector")
    with self._lock, session.trace_pipeline():
      with session.trace_vad_stage() as tracer:
        speech: list[SpeechChunk] = (
          self._detector.detect(audio, config.vad_parameters)
          if config.use_vad and self._detector
          else [{"start": 0, "end": len(audio)}]
        )
        tracer(
          speech if config.use_vad else None, sample_rate=SAMPLE_RATE, total_samples=len(audio)
        )
      segments: list[Segment] = []
      with session.trace_inference_stage() as tracer:
        for span in speech:
          if span["end"] <= span["start"]:
            continue
          segments.extend(self._transcribe_span(audio, span, absolute_stream_start))
        tracer()
      with session.trace_segment_stage() as tracer:
        tracer(segments)
    return segments, TranscriptionInfo(
      duration=duration,
      duration_after_vad=sum(s["end"] - s["start"] for s in speech) / SAMPLE_RATE,
      speech_chunks=speech if config.use_vad else [],
    )

  def _transcribe_span(self, audio: FloatAudio, span: SpeechChunk, offset: float) -> list[Segment]:
    waveform = np.ascontiguousarray(audio[span["start"] : span["end"]])
    segments = native_segments(self._recognize(waveform), len(waveform) / SAMPLE_RATE)
    _ = finalize_recording_timestamps(segments, offset + span["start"] / SAMPLE_RATE)

    return segments

  def _recognize(self, audio: FloatAudio) -> Hypothesis:
    try:
      return self._model.recognize(audio)
    except Exception:
      logger.exception("Orukeet decoding failed", samples=len(audio))
      raise
