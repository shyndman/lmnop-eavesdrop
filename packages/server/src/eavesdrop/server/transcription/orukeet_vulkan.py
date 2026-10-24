"""Pinned NeMo-Speech.cpp Vulkan inference with native word-level timestamps."""

import ctypes as c
import hashlib
import os
import platform
import shutil
import tarfile
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from http.client import HTTPResponse
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Protocol, cast, final
from urllib.request import urlopen

from eavesdrop.common import get_logger
from eavesdrop.server.constants import CACHE_PATH
from eavesdrop.server.transcription.orukeet_timestamps import Hypothesis
from eavesdrop.server.transcription.utils import SAMPLE_RATE, FloatAudio

logger = get_logger("orukeet/vulkan")
_asset_lock = Lock()
MODEL_FILE = "orukeet-v0.1.0-q8.gguf"
MODEL_REVISION = "555136b50265a132d4cea0d35560c26fc4f657ab"
MODEL_SHA256 = "93ce19c6d8244acbfea980eeaf970531d4f216171578ef8e041dcc2d070a45bd"
MODEL_BYTES = 714456704
SDK_NAME = "nemo-speech-0.1.0-linux-x86_64-vulkan"
SDK_SHA256 = "ce7b7c3c8771cb7450b26e6d4bd8fb2c5e35bcd9fe0076387f35052e9b9523ae"
SDK_BYTES = 18014113
SDK_URL = f"https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v0.1.0/{SDK_NAME}.tar.gz"


class HubModule(Protocol):
  def hf_hub_download(self, *, repo_id: str, filename: str, revision: str) -> str: ...


def _verify(path: Path, size: int, checksum: str) -> None:
  """Reject incomplete or changed pinned assets before native code loads them."""
  if path.stat().st_size != size:
    raise ValueError(f"Pinned Vulkan asset has an unexpected size: {path.name}")
  with path.open("rb") as source:
    actual = hashlib.file_digest(source, "sha256").hexdigest()
  if actual != checksum:
    raise ValueError(f"Pinned Vulkan asset SHA256 mismatch: {path.name}")


def model_assets() -> Path:
  """Download and verify the official Q8 GGUF using the standard Hub cache.

  :returns: Verified model file in Hugging Face's pinned snapshot cache.
  :raises ValueError: If model bytes disagree with the pinned release.
  """
  os.environ["HF_HUB_DISABLE_XET"] = "1"
  hub = cast(HubModule, cast(object, import_module("huggingface_hub")))
  with _asset_lock:
    path = Path(
      hub.hf_hub_download(repo_id="oruk/orukeet", filename=MODEL_FILE, revision=MODEL_REVISION)
    )
    _verify(path, MODEL_BYTES, MODEL_SHA256)
  logger.info("Verified Orukeet Q8 model", path=str(path), revision=MODEL_REVISION)
  return path


def _sdk_complete(root: Path) -> bool:
  required = (
    "include/nemo_speech/asr.h",
    "lib/libnemo_speech_asr_c.so",
    "lib/libggml-vulkan.so",
    "lib/libggml-cpu.so",
    "lib/libggml-base.so",
  )
  return all((root / name).is_file() for name in required)


def runtime_assets() -> Path:
  """Atomically publish the checksum-verified Linux x86_64 Vulkan SDK.

  :returns: SDK root containing its public header and shared libraries.
  :raises ValueError: If the pinned archive is corrupt or incomplete.
  """
  cache = Path(CACHE_PATH)
  target = cache / f"{SDK_NAME}-{SDK_SHA256[:12]}"
  with _asset_lock:
    root = target / SDK_NAME
    if _sdk_complete(root):
      logger.info("Reusing cached NeMo Vulkan SDK", path=str(root))
      return root
    cache.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading pinned NeMo Vulkan SDK", url=SDK_URL)
    with TemporaryDirectory(prefix=f".{SDK_NAME}-", dir=cache) as temporary:
      archive = Path(temporary) / "sdk.tar.gz"
      with (
        cast(HTTPResponse, urlopen(SDK_URL, timeout=120)) as source,
        archive.open("wb") as destination,
      ):
        shutil.copyfileobj(source, destination)
      _verify(archive, SDK_BYTES, SDK_SHA256)
      staging = Path(temporary) / "assets"
      staging.mkdir()
      with tarfile.open(archive, "r:gz") as source:
        source.extractall(staging, filter="data")
      if not _sdk_complete(staging / SDK_NAME):
        raise ValueError("Pinned NeMo Vulkan SDK is missing required runtime files")
      try:
        _ = staging.rename(target)
      except OSError:
        # A concurrent process may have published the same verified SDK first.
        if not _sdk_complete(root):
          raise
    return root


@final
class _BackendConfig(c.Structure):
  _fields_ = [("size", c.c_size_t), ("gpu", c.c_int32)]


@final
class _ModelConfig(c.Structure):
  _fields_ = [("size", c.c_size_t), ("path", c.c_char_p), ("name", c.c_char_p)]


@final
class _RecognizerConfig(c.Structure):
  _fields_ = [
    ("size", c.c_size_t),
    ("backend", c.c_void_p),
    ("model", c.c_void_p),
    ("streaming", c.c_void_p),
    ("decoder", c.c_void_p),
    ("vad", c.c_void_p),
    ("endpointing", c.c_void_p),
    ("postproc", c.c_void_p),
    ("diar", c.c_void_p),
    ("batching", c.c_void_p),
  ]


@final
class _Options(c.Structure):
  _fields_ = [
    ("size", c.c_size_t),
    ("request_id", c.c_char_p),
    ("language_code", c.c_char_p),
    ("interim_results", c.c_bool),
    ("enable_word_time_offsets", c.c_bool),
    ("enable_automatic_punctuation", c.c_bool),
    ("verbatim_transcripts", c.c_bool),
    ("profanity_filter", c.c_bool),
    ("stop_history_eou_ms", c.c_int32),
    ("speech_contexts", c.c_void_p),
    ("speech_context_count", c.c_size_t),
    ("max_alternatives", c.c_int32),
    ("enable_speaker_diarization", c.c_bool),
    ("max_speaker_count", c.c_int32),
  ]


def _address(value: c.Structure | c.c_void_p) -> c.c_void_p:
  return c.c_void_p(c.addressof(value))


@final
class _NativeApi:
  """Declare only the installed, public v1 C symbols used for offline decoding."""

  def __init__(self, root: Path) -> None:
    os.environ["GGML_BACKEND_DL_PATH"] = str(root / "lib")
    # The SDK bundles an older libstdc++ that cannot load current Mesa Vulkan drivers.
    # Load the host runtime first so both the SDK and driver resolve against it.
    self._host_cpp_runtime: c.CDLL = c.CDLL("libstdc++.so.6", mode=c.RTLD_GLOBAL)
    logger.debug("Loaded host C++ runtime for Vulkan driver compatibility")

    self.library = c.CDLL(str(root / "lib/libnemo_speech_asr_c.so"))
    self.create = cast(
      Callable[[c.c_void_p, c.c_void_p], int],
      c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_void_p)(("nemo_speech_asr_create", self.library)),
    )
    self.destroy = cast(
      Callable[[c.c_void_p], None],
      c.CFUNCTYPE(None, c.c_void_p)(("nemo_speech_asr_destroy", self.library)),
    )
    self.options = cast(
      Callable[[], _Options],
      c.CFUNCTYPE(_Options)(("nemo_speech_asr_recognition_options_default", self.library)),
    )
    self.recognize = cast(
      Callable[[c.c_void_p, c.c_void_p, c.c_void_p, int, int, c.c_void_p], int],
      c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_void_p, c.c_void_p, c.c_size_t, c.c_int32, c.c_void_p)(
        ("nemo_speech_asr_recognize_f32", self.library)
      ),
    )
    self.last_error = cast(
      Callable[[], bytes | None],
      c.CFUNCTYPE(c.c_char_p)(("nemo_speech_asr_last_error", self.library)),
    )
    self._bind_results()

  def _bind_results(self) -> None:
    self.transcript = cast(
      Callable[[c.c_void_p, int], bytes | None],
      c.CFUNCTYPE(c.c_char_p, c.c_void_p, c.c_size_t)(
        ("nemo_speech_asr_result_transcript", self.library)
      ),
    )
    self.word_count = cast(
      Callable[[c.c_void_p, int], int],
      c.CFUNCTYPE(c.c_size_t, c.c_void_p, c.c_size_t)(
        ("nemo_speech_asr_result_word_count", self.library)
      ),
    )
    self.word_text = cast(
      Callable[[c.c_void_p, int, int], bytes | None],
      c.CFUNCTYPE(c.c_char_p, c.c_void_p, c.c_size_t, c.c_size_t)(
        ("nemo_speech_asr_result_word_text", self.library)
      ),
    )
    self.word_start = cast(
      Callable[[c.c_void_p, int, int], int],
      c.CFUNCTYPE(c.c_int32, c.c_void_p, c.c_size_t, c.c_size_t)(
        ("nemo_speech_asr_result_word_start_time", self.library)
      ),
    )
    self.word_end = cast(
      Callable[[c.c_void_p, int, int], int],
      c.CFUNCTYPE(c.c_int32, c.c_void_p, c.c_size_t, c.c_size_t)(
        ("nemo_speech_asr_result_word_end_time", self.library)
      ),
    )
    self.result_destroy = cast(
      Callable[[c.c_void_p], None],
      c.CFUNCTYPE(None, c.c_void_p)(("nemo_speech_asr_result_destroy", self.library)),
    )

  def check(self, status: int) -> None:
    """Raise the thread-local native error before another call overwrites it."""
    if status:
      message = self.last_error()
      detail = message.decode("utf-8", "replace") if message else f"status {status}"
      raise RuntimeError(f"NeMo Vulkan recognition failed: {detail}")


@dataclass
class VulkanHypothesis:
  """Native words and recording-local seconds, without synthesized boundaries."""

  text: str
  tokens: list[str]
  timestamps: list[float]
  durations: list[float]


@final
class VulkanRecognizer:
  """Own a Q8 Vulkan model for the lifetime of the shared transcription backend."""

  def __init__(self) -> None:
    try:
      if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError(
          "Orukeet Vulkan requires Linux x86_64 and an installed Vulkan driver; "
          + "select transcription.device=cpu on unsupported platforms"
        )
      logger.info("Loading Orukeet Q8 on Vulkan", gpu=0)
      model = model_assets()
      self._api = _NativeApi(runtime_assets())
      self._handle = c.c_void_p()
      # Register ownership before creation so even a partially created handle is released.
      self._finalizer = weakref.finalize(self, self._api.destroy, self._handle)
      self._create(model)
      logger.info("Orukeet Q8 Vulkan ready", model=str(model), gpu=0)
    except Exception:
      logger.exception("Failed to load Orukeet Q8 Vulkan")
      raise

  def _create(self, path: Path) -> None:
    backend = _BackendConfig(c.sizeof(_BackendConfig), 0)
    model = _ModelConfig(c.sizeof(_ModelConfig), os.fsencode(path), None)
    config = _RecognizerConfig()
    config.size = c.sizeof(config)
    config.backend = c.addressof(backend)
    config.model = c.addressof(model)
    try:
      self._api.check(self._api.create(_address(config), _address(self._handle)))
      if not self._handle.value:
        raise RuntimeError("NeMo Vulkan returned no model handle")
    except Exception:
      _ = self._finalizer()
      raise

  def recognize(self, audio: FloatAudio) -> Hypothesis:
    """Decode contiguous float32 audio with true word times in waveform-local seconds.

    :param audio: Contiguous mono float32 samples at the server sample rate.
    :returns: Exact native transcript, words, starts, and durations.
    :raises RuntimeError: If the Vulkan SDK cannot decode the utterance.
    """
    result = c.c_void_p()
    try:
      options = self._api.options()
      options.language_code = b"en"
      options.enable_word_time_offsets = True
      options.enable_automatic_punctuation = True
      self._api.check(
        self._api.recognize(
          self._handle,
          _address(options),
          c.c_void_p(audio.ctypes.data),
          len(audio),
          SAMPLE_RATE,
          _address(result),
        )
      )
      if not result.value:
        raise RuntimeError("NeMo Vulkan returned no recognition result")
      hypothesis = self._convert(result)
      logger.debug("Orukeet Vulkan recognition complete", words=len(hypothesis.tokens))
      return hypothesis
    except Exception:
      logger.exception("Orukeet Vulkan recognition failed")
      raise
    finally:
      if result.value:
        self._api.result_destroy(result)

  def _convert(self, result: c.c_void_p) -> VulkanHypothesis:
    text = (self._api.transcript(result, 0) or b"").decode("utf-8")
    tokens: list[str] = []
    timestamps: list[float] = []
    durations: list[float] = []
    for index in range(self._api.word_count(result, 0)):
      tokens.append((self._api.word_text(result, 0, index) or b"").decode("utf-8"))
      start = self._api.word_start(result, 0, index) / 1000
      end = self._api.word_end(result, 0, index) / 1000
      timestamps.append(start)
      durations.append(end - start)
    return VulkanHypothesis(text, tokens, timestamps, durations)
