"""Whisper model management and configuration module.

This module handles the loading, initialization, and configuration of Whisper models
using CTranslate2. It provides a clean API for setting up models and feature extractors
without handling transcription logic.
"""

import json
import os
from inspect import signature
from typing import TypedDict, cast

import ctranslate2
import tokenizers
from faster_whisper.feature_extractor import FeatureExtractor
from faster_whisper.tokenizer import _LANGUAGE_CODES
from faster_whisper.utils import download_model

from eavesdrop.server.logs import get_logger
from eavesdrop.server.transcription.models import FeatureExtractorConfig

# Private module constants
_DEFAULT_INPUT_STRIDE = 2
_DEFAULT_TIME_PRECISION = 0.02
_DEFAULT_MAX_LENGTH = 448
_DEFAULT_CPU_THREADS = 0
_DEFAULT_NUM_WORKERS = 1
_DEFAULT_DEVICE_INDEX = 0


class ModelPathResolution(TypedDict):
  """Result of model path resolution and file extraction."""

  model_path: str
  tokenizer_bytes: bytes | None
  preprocessor_bytes: bytes | None


class WhisperModelConfig:
  """Configuration class for Whisper model initialization."""

  def __init__(
    self,
    model_size_or_path: str,
    device: str = "auto",
    device_index: int | list[int] = _DEFAULT_DEVICE_INDEX,
    compute_type: str = "default",
    cpu_threads: int = _DEFAULT_CPU_THREADS,
    num_workers: int = _DEFAULT_NUM_WORKERS,
    download_root: str | None = None,
    local_files_only: bool = False,
    files: dict | None = None,
    **model_kwargs,
  ):
    """Initialize Whisper model configuration.

    :param model_size_or_path: Size of the model to use (tiny, tiny.en, base, base.en,
        small, small.en, distil-small.en, medium, medium.en, distil-medium.en,
        large-v1, large-v2, large-v3, large, distil-large-v2, distil-large-v3,
        large-v3-turbo, or turbo), a path to a converted model directory, or a
        CTranslate2-converted Whisper model ID from the HF Hub.
    :type model_size_or_path: str
    :param device: Device to use for computation ("cpu", "cuda", "auto").
    :type device: str
    :param device_index: Device ID to use. Can be a list for multi-GPU setups.
    :type device_index: int | list[int]
    :param compute_type: Type to use for computation.
        See https://opennmt.net/CTranslate2/quantization.html.
    :type compute_type: str
    :param cpu_threads: Number of threads to use when running on CPU.
    :type cpu_threads: int
    :param num_workers: Number of workers for parallel transcription.
    :type num_workers: int
    :param download_root: Directory where models should be saved.
    :type download_root: str | None
    :param local_files_only: If True, avoid downloading and use local cached files.
    :type local_files_only: bool
    :param files: Load model files from memory (dict mapping filenames to contents).
    :type files: dict | None
    :param model_kwargs: Additional arguments passed to CTranslate2 model.
    """
    self.model_size_or_path = model_size_or_path
    self.device = device
    self.device_index = device_index
    self.compute_type = compute_type
    self.cpu_threads = cpu_threads
    self.num_workers = num_workers
    self.download_root = download_root
    self.local_files_only = local_files_only
    self.files = files
    self.model_kwargs = model_kwargs


class WhisperModelBundle:
  """Bundle containing a loaded Whisper model and its associated components."""

  def __init__(
    self,
    model: ctranslate2.models.Whisper,
    feature_extractor: FeatureExtractor,
    hf_tokenizer: tokenizers.Tokenizer,
    model_path: str,
    feature_kwargs: FeatureExtractorConfig,
  ):
    """Initialize the model bundle.

    :param model: The loaded CTranslate2 Whisper model.
    :type model: ctranslate2.models.Whisper
    :param feature_extractor: The configured feature extractor.
    :type feature_extractor: FeatureExtractor
    :param hf_tokenizer: The HuggingFace tokenizer.
    :type hf_tokenizer: tokenizers.Tokenizer
    :param model_path: Path to the model directory.
    :type model_path: str
    :param feature_kwargs: Configuration used for the feature extractor.
    :type feature_kwargs: FeatureExtractorConfig
    """
    self.model = model
    self.feature_extractor = feature_extractor
    self.hf_tokenizer = hf_tokenizer
    self.model_path = model_path
    self.feature_kwargs = feature_kwargs

    # Computed properties derived from feature extractor
    self.input_stride = _DEFAULT_INPUT_STRIDE
    self.num_samples_per_token = self.feature_extractor.hop_length * self.input_stride
    self.frames_per_second = (
      self.feature_extractor.sampling_rate // self.feature_extractor.hop_length
    )
    self.tokens_per_second = self.feature_extractor.sampling_rate // self.num_samples_per_token
    self.time_precision = _DEFAULT_TIME_PRECISION
    self.max_length = _DEFAULT_MAX_LENGTH

  @property
  def supported_languages(self) -> list[str]:
    """The languages supported by the model."""
    return list(_LANGUAGE_CODES) if self.model.is_multilingual else ["en"]


def _resolve_model_path(config: WhisperModelConfig) -> ModelPathResolution:
  """Resolve the model path and extract tokenizer/preprocessor bytes if available.

  :param config: The model configuration.
  :type config: WhisperModelConfig
  :returns: ModelPathResolution containing model path and extracted file bytes.
  :rtype: ModelPathResolution
  """
  tokenizer_bytes: bytes | None = None
  preprocessor_bytes: bytes | None = None

  if config.files:
    model_path = config.model_size_or_path
    tokenizer_bytes = config.files.pop("tokenizer.json", None)
    preprocessor_bytes = config.files.pop("preprocessor_config.json", None)
  elif os.path.isdir(config.model_size_or_path):
    model_path = config.model_size_or_path
  else:
    model_path = download_model(
      config.model_size_or_path,
      local_files_only=config.local_files_only,
      cache_dir=config.download_root,
    )

  return {
    "model_path": model_path,
    "tokenizer_bytes": tokenizer_bytes,
    "preprocessor_bytes": preprocessor_bytes,
  }


def _load_ctranslate2_model(
  config: WhisperModelConfig, model_path: str
) -> ctranslate2.models.Whisper:
  """Load the CTranslate2 Whisper model.

  :param config: The model configuration.
  :type config: WhisperModelConfig
  :param model_path: Path to the model directory.
  :type model_path: str
  :returns: The loaded CTranslate2 Whisper model.
  :rtype: ctranslate2.models.Whisper
  """
  return ctranslate2.models.Whisper(
    model_path,
    device=config.device,
    device_index=config.device_index,
    compute_type=config.compute_type,
    intra_threads=config.cpu_threads,
    inter_threads=config.num_workers,
    files=config.files,
    **config.model_kwargs,
  )


def _load_hf_tokenizer(
  model_path: str, tokenizer_bytes: bytes | None, is_multilingual: bool
) -> tokenizers.Tokenizer:
  """Load the HuggingFace tokenizer.

  :param model_path: Path to the model directory.
  :type model_path: str
  :param tokenizer_bytes: Pre-loaded tokenizer bytes, if available.
  :type tokenizer_bytes: bytes | None
  :param is_multilingual: Whether the model is multilingual.
  :type is_multilingual: bool
  :returns: The loaded HuggingFace tokenizer.
  :rtype: tokenizers.Tokenizer
  """
  tokenizer_file = os.path.join(model_path, "tokenizer.json")

  if tokenizer_bytes:
    return tokenizers.Tokenizer.from_buffer(tokenizer_bytes)
  elif os.path.isfile(tokenizer_file):
    return tokenizers.Tokenizer.from_file(tokenizer_file)
  else:
    model_name = "openai/whisper-tiny" + ("" if is_multilingual else ".en")
    return tokenizers.Tokenizer.from_pretrained(model_name)


def _get_feature_extractor_config(
  model_path: str, preprocessor_bytes: bytes | None = None
) -> FeatureExtractorConfig:
  """Extract feature extractor configuration from model directory.

  :param model_path: Path to the model directory.
  :type model_path: str
  :param preprocessor_bytes: Pre-loaded preprocessor config bytes, if available.
  :type preprocessor_bytes: bytes | None
  :returns: Configuration dictionary for the FeatureExtractor.
  :rtype: FeatureExtractorConfig
  :raises json.JSONDecodeError: If the preprocessor config cannot be parsed.
  """
  logger = get_logger("whisper.config")
  config = {}

  try:
    config_path = os.path.join(model_path, "preprocessor_config.json")
    if preprocessor_bytes:
      config = json.loads(preprocessor_bytes)
    elif os.path.isfile(config_path):
      with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    else:
      return cast(FeatureExtractorConfig, config)

    # Filter to only valid FeatureExtractor parameters
    valid_keys = signature(FeatureExtractor.__init__).parameters.keys()
    filtered_config = {k: v for k, v in config.items() if k in valid_keys}
    return cast(FeatureExtractorConfig, filtered_config)

  except json.JSONDecodeError:
    logger.exception("Could not load preprocessor config")
    raise


def _create_feature_extractor(feature_kwargs: FeatureExtractorConfig) -> FeatureExtractor:
  """Create and configure the feature extractor.

  :param feature_kwargs: Configuration for the feature extractor.
  :type feature_kwargs: FeatureExtractorConfig
  :returns: The configured FeatureExtractor instance.
  :rtype: FeatureExtractor
  """
  return FeatureExtractor(**feature_kwargs)


def load_whisper_model(config: WhisperModelConfig) -> WhisperModelBundle:
  """Load and initialize a Whisper model with all its components.

  This is the main entry point for loading Whisper models. It handles model
  downloading, CTranslate2 setup, tokenizer loading, and feature extractor
  configuration.

  :param config: Configuration for the model to load.
  :type config: WhisperModelConfig
  :returns: A WhisperModelBundle containing the loaded model and components.
  :rtype: WhisperModelBundle
  :raises: Various exceptions related to model loading, file I/O, or JSON parsing.
  """
  logger = get_logger("whisper.loader")

  # Resolve model path and extract embedded files
  path_resolution = _resolve_model_path(config)

  # Load the CTranslate2 model
  model = _load_ctranslate2_model(config, path_resolution["model_path"])

  logger.info(
    "Initialized CTranslate2 Whisper model: path='%s', device='%s', device_index=%s, "
    "compute_type='%s', cpu_threads=%d, num_workers=%d, is_multilingual=%s",
    path_resolution["model_path"],
    config.device,
    config.device_index,
    config.compute_type,
    config.cpu_threads,
    config.num_workers,
    model.is_multilingual,
  )

  # Load the HuggingFace tokenizer
  hf_tokenizer = _load_hf_tokenizer(
    path_resolution["model_path"], path_resolution["tokenizer_bytes"], model.is_multilingual
  )

  # Get feature extractor configuration
  feature_kwargs = _get_feature_extractor_config(
    path_resolution["model_path"], path_resolution["preprocessor_bytes"]
  )

  # Create the feature extractor
  feature_extractor = _create_feature_extractor(feature_kwargs)

  logger.info(
    "Initialized feature extractor: sampling_rate=%d, n_fft=%d, hop_length=%d, chunk_length=%d",
    feature_extractor.sampling_rate,
    feature_extractor.n_fft,
    feature_extractor.hop_length,
    feature_extractor.chunk_length,
  )

  return WhisperModelBundle(
    model=model,
    feature_extractor=feature_extractor,
    hf_tokenizer=hf_tokenizer,
    model_path=path_resolution["model_path"],
    feature_kwargs=feature_kwargs,
  )


def create_whisper_model(
  model_size_or_path: str,
  device: str = "auto",
  device_index: int | list[int] = _DEFAULT_DEVICE_INDEX,
  compute_type: str = "default",
  cpu_threads: int = _DEFAULT_CPU_THREADS,
  num_workers: int = _DEFAULT_NUM_WORKERS,
  download_root: str | None = None,
  local_files_only: bool = False,
  files: dict | None = None,
  **model_kwargs,
) -> WhisperModelBundle:
  """Convenience function to create a Whisper model with default configuration.

  This is a simplified interface that creates a WhisperModelConfig and loads
  the model in one step.

  :param model_size_or_path: Model size or path (same as WhisperModelConfig).
  :type model_size_or_path: str
  :param device: Device to use for computation.
  :type device: str
  :param device_index: Device ID or list of IDs.
  :type device_index: int | list[int]
  :param compute_type: Computation type.
  :type compute_type: str
  :param cpu_threads: Number of CPU threads.
  :type cpu_threads: int
  :param num_workers: Number of workers.
  :type num_workers: int
  :param download_root: Model download directory.
  :type download_root: str | None
  :param local_files_only: Whether to use only local files.
  :type local_files_only: bool
  :param files: In-memory model files.
  :type files: dict | None
  :param model_kwargs: Additional model arguments.
  :returns: A WhisperModelBundle with the loaded model and components.
  :rtype: WhisperModelBundle
  """
  config = WhisperModelConfig(
    model_size_or_path=model_size_or_path,
    device=device,
    device_index=device_index,
    compute_type=compute_type,
    cpu_threads=cpu_threads,
    num_workers=num_workers,
    download_root=download_root,
    local_files_only=local_files_only,
    files=files,
    **model_kwargs,
  )
  return load_whisper_model(config)
