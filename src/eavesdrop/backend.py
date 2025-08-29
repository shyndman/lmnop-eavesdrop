import json
import os
import threading

import ctranslate2
import torch
from faster_whisper.vad import VadOptions
from huggingface_hub import snapshot_download
from numpy import ndarray

from .base import ServeClientBase
from .logs import get_logger
from .transcription.whisper_model import WhisperModel


class ServeClientFasterWhisper(ServeClientBase):
  SINGLE_MODEL = None
  SINGLE_MODEL_LOCK = threading.Lock()

  def __init__(
    self,
    websocket,
    task="transcribe",
    device=None,
    language=None,
    client_uid=None,
    model="distil-small.en",
    initial_prompt=None,
    vad_parameters=None,
    use_vad=True,
    single_model=False,
    send_last_n_segments=10,
    no_speech_thresh=0.45,
    clip_audio=False,
    same_output_threshold=7,
    cache_path="~/.cache/whisper-live/",
    translation_queue=None,
    device_index: int = 0,
  ):
    """
    Initialize a ServeClient instance.
    The Whisper model is initialized based on the client's language and device availability.
    The transcription thread is started upon initialization. A "SERVER_READY" message is sent
    to the client to indicate that the server is ready.

    Args:
    websocket: The WebSocket connection for the client.
    task: The task type, e.g., "transcribe".
    device: The device type for Whisper, "cuda" or "cpu".
    language: The language for transcription.
    client_uid: A unique identifier for the client.
    model: The whisper model size.
    initial_prompt: Prompt for whisper inference.
    single_model: Whether to instantiate a new model for each client connection.
    send_last_n_segments: Number of most recent segments to send to the client.
    no_speech_thresh: Segments with no speech probability above this threshold will be discarded.
    clip_audio: Whether to clip audio with no valid segments.
    same_output_threshold: Number of repeated outputs before considering it as a valid segment.

    """
    self.logger = get_logger("faster_whisper_client")
    self.logger.debug(
      "Initializing ServeClientFasterWhisper",
      client_uid=client_uid,
      model=model,
      task=task,
      device=device,
    )
    super().__init__(
      client_uid,
      websocket,
      send_last_n_segments,
      no_speech_thresh,
      clip_audio,
      same_output_threshold,
      translation_queue,
    )
    self.cache_path = cache_path
    self.device_index = device_index
    self.model_sizes = [
      "tiny",
      "tiny.en",
      "base",
      "base.en",
      "small",
      "small.en",
      "medium",
      "medium.en",
      "large-v2",
      "large-v3",
      "distil-small.en",
      "distil-medium.en",
      "distil-large-v2",
      "distil-large-v3",
      "large-v3-turbo",
      "turbo",
    ]

    self.model_size_or_path = model
    self.language = "en" if self.model_size_or_path.endswith("en") else language
    self.task = task
    self.initial_prompt = initial_prompt
    self.vad_parameters = (
      VadOptions()
      if vad_parameters is None
      else (
        vad_parameters if isinstance(vad_parameters, VadOptions) else VadOptions(**vad_parameters)
      )
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    self.logger.debug("Selected device", device=device)
    if device == "cuda":
      major, _ = torch.cuda.get_device_capability(device)
      self.compute_type = "float16" if major >= 7 else "float32"
      self.logger.debug("CUDA device capability", major=major, compute_type=self.compute_type)
    else:
      self.compute_type = "int8"
      self.logger.debug("Using CPU with compute_type", compute_type=self.compute_type)

    if self.model_size_or_path is None:
      return
    self.logger.info("Using Device with precision", device=device, precision=self.compute_type)

    try:
      if single_model:
        self.logger.debug("Using single model mode", client_uid=client_uid)
        if ServeClientFasterWhisper.SINGLE_MODEL is None:
          self.logger.debug("Creating new single model instance")
          self.create_model(device)
          ServeClientFasterWhisper.SINGLE_MODEL = self.transcriber
        else:
          self.logger.debug("Reusing existing single model instance")
          self.transcriber = ServeClientFasterWhisper.SINGLE_MODEL
      else:
        self.logger.debug("Creating dedicated model", client_uid=client_uid)
        self.create_model(device)
    except Exception:
      self.logger.exception("Failed to load model")
      self.websocket.send(
        json.dumps(
          {
            "uid": self.client_uid,
            "status": "ERROR",
            "message": f"Failed to load model: {str(self.model_size_or_path)}",
          }
        )
      )
      self.websocket.close()
      return

    self.use_vad = use_vad

    # threading
    self.trans_thread = threading.Thread(target=self.speech_to_text)
    self.trans_thread.start()
    self.websocket.send(
      json.dumps(
        {
          "uid": self.client_uid,
          "message": self.SERVER_READY,
          "backend": "faster_whisper",
        }
      )
    )

  def create_model(self, device):
    """
    Instantiates a new model, sets it as the transcriber. If model is a huggingface model_id
    then it is automatically converted to ctranslate2(faster_whisper) format.
    """
    self.logger.debug("Creating model", model_reference=self.model_size_or_path)
    model_ref = self.model_size_or_path

    if model_ref in self.model_sizes:
      self.logger.debug("Model found in standard model sizes", model=model_ref)
      model_to_load = model_ref
    else:
      self.logger.debug("Model not in standard model sizes, checking if custom", model=model_ref)
      if os.path.isdir(model_ref) and ctranslate2.contains_model(model_ref):
        self.logger.debug("Found local CTranslate2 model", path=model_ref)
        model_to_load = model_ref
      else:
        self.logger.debug("Downloading model from HuggingFace", model=model_ref)
        local_snapshot = snapshot_download(
          repo_id=model_ref,
          repo_type="model",
        )
        self.logger.debug("Downloaded model", path=local_snapshot)
        if ctranslate2.contains_model(local_snapshot):
          self.logger.debug("Downloaded model is already in CTranslate2 format")
          model_to_load = local_snapshot
        else:
          cache_root = os.path.expanduser(os.path.join(self.cache_path, "whisper-ct2-models/"))
          os.makedirs(cache_root, exist_ok=True)
          safe_name = model_ref.replace("/", "--")
          ct2_dir = os.path.join(cache_root, safe_name)
          self.logger.debug("CTranslate2 cache directory", path=ct2_dir)

          if not ctranslate2.contains_model(ct2_dir):
            self.logger.debug(
              "Converting to CTranslate2 format",
              model=model_ref,
              output_dir=ct2_dir,
            )
            self.logger.info(
              "Converting to CTranslate2",
              model=model_ref,
              output_dir=ct2_dir,
            )
            ct2_converter = ctranslate2.converters.TransformersConverter(
              local_snapshot,
              copy_files=["tokenizer.json", "preprocessor_config.json"],
            )
            ct2_converter.convert(
              output_dir=ct2_dir,
              quantization=self.compute_type,
              force=False,  # skip if already up-to-date
            )
            self.logger.debug("Model conversion completed")
          else:
            self.logger.debug("CTranslate2 model already exists in cache")
          model_to_load = ct2_dir

    self.logger.info("Loading model", model=model_to_load)
    self.logger.debug(
      "Model loading parameters",
      device=device,
      compute_type=self.compute_type,
      device_index=self.device_index,
    )
    self.transcriber = WhisperModel(
      model_to_load,
      device=device,
      device_index=self.device_index,
      compute_type=self.compute_type,
      download_root=self.cache_path,
      local_files_only=False,
    )
    self.logger.debug("Model loaded successfully")

  def set_language(self, info):
    """
    Updates the language attribute based on the detected language information.

    Args:
        info (object): An object containing the detected language and its probability. This object
                    must have at least two attributes: `language`, a string indicating the detected
                    language, and `language_probability`, a float representing the confidence level
                    of the language detection.
    """
    self.logger.debug(
      "Language detection info",
      language=info.language,
      probability=info.language_probability,
    )
    if info.language_probability > 0.5:
      self.logger.debug(
        "Language probability > 0.5, setting language",
        probability=info.language_probability,
        language=info.language,
      )
      self.language = info.language
      self.logger.info(
        "Detected language",
        language=self.language,
        probability=info.language_probability,
      )
      self.websocket.send(
        json.dumps(
          {
            "uid": self.client_uid,
            "language": self.language,
            "language_prob": info.language_probability,
          }
        )
      )
    else:
      self.logger.debug(
        "Language probability <= 0.5, not setting language",
        probability=info.language_probability,
      )

  def transcribe_audio(self, input_sample: ndarray):
    """
    Transcribes the provided audio sample using the configured transcriber instance.

    If the language has not been set, it updates the session's language based on the transcription
    information.

    Args:
        input_sample (np.array): The audio chunk to be transcribed. This should be a NumPy
                                array representing the audio data.

    Returns:
        The transcription result from the transcriber. The exact format of this result
        depends on the implementation of the `transcriber.transcribe` method but typically
        includes the transcribed text.
    """
    shape = input_sample.shape if hasattr(input_sample, "shape") else "unknown"
    self.logger.debug("Transcribing audio sample", shape=shape)

    if ServeClientFasterWhisper.SINGLE_MODEL:
      self.logger.debug("Acquiring single model lock")
      ServeClientFasterWhisper.SINGLE_MODEL_LOCK.acquire()

    self.logger.debug(
      "Starting transcription",
      language=self.language,
      task=self.task,
      vad=self.use_vad,
    )
    result, info = self.transcriber.transcribe(
      input_sample,
      initial_prompt=self.initial_prompt,
      language=self.language,
      task=self.task,
      vad_filter=self.use_vad,
      vad_parameters=self.vad_parameters,
    )

    if ServeClientFasterWhisper.SINGLE_MODEL:
      self.logger.debug("Releasing single model lock")
      ServeClientFasterWhisper.SINGLE_MODEL_LOCK.release()

    result_count = len(list(result)) if result else 0
    self.logger.debug("Transcription completed", segments=result_count)
    if self.language is None and info is not None:
      self.set_language(info)
    return result

  def handle_transcription_output(self, result, duration):
    """
    Handle the transcription output, updating the transcript and sending data to the client.

    Args:
        result (str): The result from whisper inference i.e. the list of segments.
        duration (float): Duration of the transcribed audio chunk.
    """
    result_count = len(result) if result else 0
    self.logger.debug(
      "Handling transcription output",
      duration=f"{duration:.2f}s",
      segments=result_count,
    )
    segments = []
    if len(result):
      self.logger.debug("Processing transcription segments")
      self.t_start = None
      last_segment = self.update_segments(result, duration)
      segments = self.prepare_segments(last_segment)
      self.logger.debug("Prepared segments for client", segment_count=len(segments))

    if len(segments):
      self.logger.debug("Sending segments to client", segment_count=len(segments))
      self.send_transcription_to_client(segments)
    else:
      self.logger.debug("No segments to send to client")
