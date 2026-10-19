"""English Parakeet TDT inference with native word and segment timestamps.

NeMo is loaded only at model initialization, keeping protocol tests lightweight.
A model lock covers decoding because NeMo's transcribe mutates model state.
"""

from collections.abc import Sequence
from contextlib import AbstractContextManager
from importlib import import_module
from threading import Lock
from typing import Literal, Protocol, cast

import numpy as np

from eavesdrop.server.config import TranscriptionConfig
from eavesdrop.server.transcription.models import SpeechChunk, TranscriptionInfo
from eavesdrop.server.transcription.parakeet_timestamps import Hypothesis, native_segments
from eavesdrop.server.transcription.session import TranscriptionSession
from eavesdrop.server.transcription.utils import (
  SAMPLE_RATE,
  FloatAudio,
  disable_runtime_telemetry,
  finalize_recording_timestamps,
)
from eavesdrop.server.transcription.vad import SileroSpeechDetector, SpeechDetector
from eavesdrop.wire import Segment

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"


class NemoModel(Protocol):
  """Narrow typed boundary for NeMo's inference-only model API."""

  def eval(self) -> object: ...
  def transcribe(
    self,
    audio: list[FloatAudio],
    *,
    batch_size: int,
    return_hypotheses: bool,
    timestamps: bool,
    num_workers: int,
    verbose: bool,
  ) -> list[Hypothesis]: ...


class ModelFactory(Protocol):
  def from_pretrained(
    self,
    model_name: str,
    *,
    map_location: str,
    return_config: bool = False,
    override_config_path: object = None,
  ) -> object: ...


class NemoModels(Protocol):
  ASRModel: ModelFactory


class OmegaConfAPI(Protocol):
  def update(self, config: object, key: str, value: object, *, force_add: bool) -> None: ...


class OmegaConfModule(Protocol):
  OmegaConf: OmegaConfAPI


class CudaRuntime(Protocol):
  def is_available(self) -> bool: ...
  def device_count(self) -> int: ...


class TorchRuntime(Protocol):
  cuda: CudaRuntime

  def inference_mode(self) -> AbstractContextManager[object]: ...


def resolve_device(torch: TorchRuntime, device: Literal["auto", "cpu", "cuda"], index: int) -> str:
  """Resolve an explicit device without silently falling back from requested GPU.

  :param torch: PyTorch runtime, including ROCm's CUDA-compatible namespace.
  :param device: Requested execution device.
  :param index: GPU device index.
  :returns: PyTorch device string.
  :raises RuntimeError: If a requested GPU is unavailable.
  """
  if device == "cpu" or (device == "auto" and not torch.cuda.is_available()):
    return "cpu"
  if not torch.cuda.is_available() or index >= torch.cuda.device_count():
    raise RuntimeError(f"Requested GPU cuda:{index} is unavailable")
  return f"cuda:{index}"


class ParakeetTranscriber:
  """Serialize a shared model and preserve original audio time across VAD gaps.

  :param model: Loaded NeMo model or an injectable test double.
  :param detector: Optional standalone VAD detector.
  :param torch: Optional runtime providing inference_mode for the real model.
  """

  def __init__(
    self, model: NemoModel, detector: SpeechDetector | None, torch: TorchRuntime | None = None
  ) -> None:
    self._model: NemoModel = model
    self._detector: SpeechDetector | None = detector
    self._torch: TorchRuntime | None = torch
    self._lock: Lock = Lock()

  @classmethod
  def load(cls, config: TranscriptionConfig) -> "ParakeetTranscriber":
    """Load the official v2 checkpoint with GPU-graph paths disabled before construction.

    :param config: Validated model/device configuration.
    :returns: Ready inference adapter.
    """
    disable_runtime_telemetry()
    try:
      torch = cast(TorchRuntime, cast(object, import_module("torch")))
      models = cast(NemoModels, cast(object, import_module("nemo.collections.asr.models")))
      omega = cast(OmegaConfModule, cast(object, import_module("omegaconf")))
    except ImportError as error:
      raise RuntimeError("Install the server's 'transcription' extra to load Parakeet") from error
    device = resolve_device(torch, config.device, config.device_index)
    model_config = models.ASRModel.from_pretrained(
      MODEL_NAME, map_location="cpu", return_config=True
    )
    # NeMo's optional NVIDIA CUDA graph probe can fail even before .to(device)
    # on ROCm/CPU hosts. Override the checkpoint config BEFORE model construction.
    # Ordinary greedy TDT works on all three paths without CUDA Python graphs.
    omega.OmegaConf.update(
      model_config, "decoding.greedy.use_cuda_graph_decoder", False, force_add=True
    )
    omega.OmegaConf.update(model_config, "decoding.strategy", "greedy_batch", force_add=True)
    model = cast(
      NemoModel,
      models.ASRModel.from_pretrained(
        MODEL_NAME,
        map_location=device,
        override_config_path=model_config,
      ),
    )
    _ = model.eval()
    return cls(model, SileroSpeechDetector() if config.use_vad else None, torch)

  def transcribe(
    self,
    audio: FloatAudio,
    *,
    config: TranscriptionConfig,
    absolute_stream_start: float,
    session: TranscriptionSession,
  ) -> tuple[list[Segment], TranscriptionInfo]:
    """Recognize speech spans and add each recording offset exactly once.

    :param audio: Mono finite float32 waveform at 16 kHz.
    :param config: Session output and VAD settings.
    :param absolute_stream_start: Recording-relative start of the full input chunk.
    :param session: Per-session tracing context.
    :returns: Recording-relative segments and original-chunk speech metadata.
    """
    if audio.ndim != 1 or not np.all(np.isfinite(audio)):
      raise ValueError("Parakeet requires finite mono audio")
    duration = len(audio) / SAMPLE_RATE
    session.update_audio_context(start_offset=absolute_stream_start, duration=duration)
    if config.use_vad and self._detector is None:
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
          waveform = np.ascontiguousarray(audio[span["start"] : span["end"]])
          hypotheses = self._recognize(waveform)
          if len(hypotheses) != 1:
            raise ValueError("Parakeet must return one hypothesis per input waveform")
          local = native_segments(hypotheses[0], len(waveform) / SAMPLE_RATE)
          _ = finalize_recording_timestamps(
            local, absolute_stream_start + span["start"] / SAMPLE_RATE
          )
          if not config.word_timestamps:
            for segment in local:
              segment.words = None
          segments.extend(local)
        tracer()
      with session.trace_segment_stage() as tracer:
        tracer(segments)
    return segments, TranscriptionInfo(
      duration=duration,
      duration_after_vad=sum(span["end"] - span["start"] for span in speech) / SAMPLE_RATE,
      speech_chunks=speech if config.use_vad else [],
    )

  def _recognize(self, audio: FloatAudio) -> Sequence[Hypothesis]:
    # NeMo centers ndarray inputs shorter than one second by default. Right-pad
    # explicitly so short VAD/flush spans keep their native zero point; callers
    # still validate/clamp results against the ORIGINAL waveform duration.
    if len(audio) < SAMPLE_RATE:
      audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)))
    if self._torch is None:
      return self._model.transcribe(
        [audio], batch_size=1, return_hypotheses=True, timestamps=True, num_workers=0, verbose=False
      )
    with self._torch.inference_mode():
      return self._model.transcribe(
        [audio], batch_size=1, return_hypotheses=True, timestamps=True, num_workers=0, verbose=False
      )
