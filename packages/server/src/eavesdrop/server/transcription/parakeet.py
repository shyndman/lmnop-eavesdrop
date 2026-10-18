"""English Parakeet TDT inference with native word and segment timestamps.

NeMo is loaded only at model initialization, keeping protocol tests lightweight.
A model lock covers decoding because NeMo's transcribe mutates model state.
"""

import os
from collections.abc import Sequence
from contextlib import AbstractContextManager
from importlib import import_module
from threading import Lock
from typing import Literal, Protocol, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from eavesdrop.server.config import TranscriptionConfig
from eavesdrop.server.transcription.models import SpeechChunk, TranscriptionInfo, VadParameters
from eavesdrop.server.transcription.session import TranscriptionSession
from eavesdrop.server.transcription.utils import finalize_recording_timestamps
from eavesdrop.wire import Segment, Word

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
SAMPLE_RATE = 16000
# v2 TDT durations are [0, 1, 2, 3, 4] at an 80 ms encoder stride.
MAX_TOKEN_DURATION = 0.32
FloatAudio = NDArray[np.float32]


def _disable_runtime_telemetry() -> None:
  # Must be set BEFORE NeMo/Silero can import ONNX Runtime. The Python API alone
  # cannot prevent its initialization event. Official process-lifetime opt-out:
  # https://github.com/microsoft/onnxruntime/blob/main/docs/Privacy.md
  os.environ["ORT_DISABLE_TELEMETRY"] = "1"
  os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


class NativeWord(TypedDict):
  """Native word boundaries in seconds relative to the input waveform."""

  word: str
  start: float
  end: float


class NativeSegment(TypedDict):
  """Native sentence boundaries in seconds relative to the input waveform."""

  segment: str
  start: float
  end: float


class NativeTimestamps(TypedDict):
  word: list[NativeWord]
  segment: list[NativeSegment]


class Hypothesis(Protocol):
  text: str
  timestamp: NativeTimestamps


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


class SileroSpeechDetector:
  """Standalone CPU/ONNX Silero VAD, independent of the recognition model."""

  def __init__(self) -> None:
    _disable_runtime_telemetry()
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


def native_segments(hypothesis: Hypothesis, duration: float) -> list[Segment]:
  """Translate native NeMo boundaries without alignment or invented confidence.

  :param hypothesis: NeMo result with word and segment timestamps enabled.
  :param duration: Input waveform duration, used to validate native bounds.
  :returns: Chunk-local segments with punctuation and original whitespace preserved.
  :raises ValueError: If native timing/text is missing or inconsistent.
  """
  text = hypothesis.text
  if not text.strip():
    return []
  native_words = hypothesis.timestamp["word"]
  if not native_words:
    raise ValueError("Parakeet returned text without native word timestamps")
  words: list[Word] = []
  cursor = 0
  previous_start = 0.0
  previous_end = 0.0
  for stamp in native_words:
    start, end = stamp["start"], stamp["end"]
    if not (
      np.isfinite(start)
      and np.isfinite(end)
      and previous_start <= start <= end <= duration + MAX_TOKEN_DURATION
      and start <= duration
      and previous_end <= end
    ):
      raise ValueError("Parakeet returned invalid native word timestamps")
    token = stamp["word"]
    index = text.find(token, cursor)
    if not token or index < 0 or text[cursor:index].strip():
      raise ValueError("Parakeet native words do not match the recognized text")
    stop = index + len(token)
    words.append(Word(start=start, end=min(end, duration), word=text[cursor:stop]))
    cursor = stop
    previous_start = start
    previous_end = end
  if text[cursor:].strip():
    raise ValueError("Parakeet native words omit recognized text")
  words[-1].word += text[cursor:]

  output: list[Segment] = []
  word_index = 0
  previous_segment_end = 0.0
  for stamp in hypothesis.timestamp["segment"]:
    start, end = stamp["start"], stamp["end"]
    if not (
      np.isfinite(start)
      and np.isfinite(end)
      and previous_segment_end <= start <= end <= duration + MAX_TOKEN_DURATION
      and start <= duration
    ):
      raise ValueError("Parakeet returned invalid native segment timestamps")
    selected: list[Word] = []
    expected = " ".join(stamp["segment"].split())
    joined = ""
    while word_index < len(words):
      word = words[word_index]
      if word.start < start - 1e-6 or word.end > min(end, duration) + 1e-6:
        raise ValueError("Parakeet native words fall outside their segment")
      selected.append(word)
      word_index += 1
      # TDT can emit zero-duration words at the previous sentence's exact end.
      # Match native sentence text rather than consuming every tied timestamp.
      joined = " ".join("".join(item.word for item in selected).split())
      if joined == expected:
        break
      if not expected.startswith(joined):
        raise ValueError("Parakeet native segment text disagrees with words")
    if not selected or joined != expected:
      raise ValueError("Parakeet native segment text or boundaries disagree with words")
    if selected:
      output.append(
        Segment(
          id=0,
          start=start,
          end=min(end, duration),
          text="".join(word.word for word in selected),
          words=selected,
        )
      )
    previous_segment_end = end
  # Some NeMo versions omit segment metadata for short utterances. Word timestamps
  # are sufficient to form the remaining segment without estimating any boundary.
  if word_index < len(words):
    if hypothesis.timestamp["segment"]:
      raise ValueError("Parakeet native words fall outside segment metadata")
    selected = words[word_index:]
    output.append(
      Segment(
        id=0,
        start=selected[0].start,
        end=selected[-1].end,
        text="".join(word.word for word in selected),
        words=selected,
      )
    )
  return output


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
    _disable_runtime_telemetry()
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
