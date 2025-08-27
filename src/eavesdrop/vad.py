import os
import subprocess
import warnings

import numpy as np
import onnxruntime
import torch

from .logs import get_logger


class VoiceActivityDetection:
  def __init__(self, force_onnx_cpu=True):
    self.logger = get_logger("voice_activity_detection")
    self.logger.debug("Initializing VoiceActivityDetection", force_onnx_cpu=force_onnx_cpu)
    path = self.download()
    self.logger.debug("Using VAD model", model_path=path)

    opts = onnxruntime.SessionOptions()
    opts.log_severity_level = 3

    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1

    if force_onnx_cpu and "CPUExecutionProvider" in onnxruntime.get_available_providers():
      self.logger.debug("Using CPU execution provider for VAD")
      self.session = onnxruntime.InferenceSession(
        path, providers=["CPUExecutionProvider"], sess_options=opts
      )
      provider = "CPUExecutionProvider"
    else:
      self.logger.debug("Using CUDA execution provider for VAD")
      self.session = onnxruntime.InferenceSession(
        path, providers=["CUDAExecutionProvider"], sess_options=opts
      )
      provider = "CUDAExecutionProvider"

    self.reset_states()
    if "16k" in path:
      warnings.warn("This model support only 16000 sampling rate!")
      self.sample_rates = [16000]
      self.logger.debug("VAD model supports only 16kHz sampling rate")
    else:
      self.sample_rates = [8000, 16000]
      self.logger.debug("VAD model supports 8kHz and 16kHz sampling rates")

    self.logger.info(
      "Initialized VoiceActivityDetection",
      model_path=path,
      provider=provider,
      force_onnx_cpu=force_onnx_cpu,
      supported_sample_rates=self.sample_rates,
      inter_op_threads=opts.inter_op_num_threads,
      intra_op_threads=opts.intra_op_num_threads,
    )

  def _validate_input(self, x, sr: int):
    self.logger.debug("Validating VAD input", shape=x.shape, sample_rate=sr)
    if x.dim() == 1:
      x = x.unsqueeze(0)
    if x.dim() > 2:
      raise ValueError(f"Too many dimensions for input audio chunk {x.dim()}")

    if sr != 16000 and (sr % 16000 == 0):
      step = sr // 16000
      x = x[:, ::step]
      sr = 16000

    if sr not in self.sample_rates:
      raise ValueError(f"Supported sampling rates: {self.sample_rates} (or multiply of 16000)")
    if sr / x.shape[1] > 31.25:
      raise ValueError("Input audio chunk is too short")

    return x, sr

  def reset_states(self, batch_size=1):
    self.logger.debug("Resetting VAD states", batch_size=batch_size)
    self._state = torch.zeros((2, batch_size, 128)).float()
    self._context = torch.zeros(0)
    self._last_sr = 0
    self._last_batch_size = 0

  def __call__(self, x, sr: int):
    shape = x.shape if hasattr(x, "shape") else len(x)
    self.logger.debug("VAD processing audio chunk", shape=shape, sr=sr)
    x, sr = self._validate_input(x, sr)
    num_samples = 512 if sr == 16000 else 256
    self.logger.debug("Expected num_samples", num_samples=num_samples)

    if x.shape[-1] != num_samples:
      raise ValueError(
        f"Provided number of samples is {x.shape[-1]} (Supported values: 256 for 8000 sample rate, 512 for 16000)"
      )

    batch_size = x.shape[0]
    context_size = 64 if sr == 16000 else 32

    if not self._last_batch_size:
      self.reset_states(batch_size)
    if (self._last_sr) and (self._last_sr != sr):
      self.reset_states(batch_size)
    if (self._last_batch_size) and (self._last_batch_size != batch_size):
      self.reset_states(batch_size)

    if not len(self._context):
      self._context = torch.zeros(batch_size, context_size)

    x = torch.cat([self._context, x], dim=1)
    if sr in [8000, 16000]:
      ort_inputs = {
        "input": x.numpy(),
        "state": self._state.numpy(),
        "sr": np.array(sr, dtype="int64"),
      }
      ort_outs = self.session.run(None, ort_inputs)
      out, state = ort_outs
      self._state = torch.from_numpy(state)
    else:
      raise ValueError()

    self._context = x[..., -context_size:]
    self._last_sr = sr
    self._last_batch_size = batch_size

    out = torch.from_numpy(out)
    self.logger.debug("VAD output", output=out)
    return out

  def audio_forward(self, x, sr: int):
    outs = []
    x, sr = self._validate_input(x, sr)
    self.reset_states()
    num_samples = 512 if sr == 16000 else 256

    if x.shape[1] % num_samples:
      pad_num = num_samples - (x.shape[1] % num_samples)
      x = torch.nn.functional.pad(x, (0, pad_num), "constant", value=0.0)

    for i in range(0, x.shape[1], num_samples):
      wavs_batch = x[:, i : i + num_samples]
      out_chunk = self.__call__(wavs_batch, sr)
      outs.append(out_chunk)

    stacked = torch.cat(outs, dim=1)
    return stacked.cpu()

  @staticmethod
  def download(
    model_url="https://github.com/snakers4/silero-vad/raw/v5.0/files/silero_vad.onnx",
  ):
    logger = get_logger("vad_download")
    target_dir = os.path.expanduser("~/.cache/whisper-live/")
    logger.debug("VAD model target directory", target_dir=target_dir)

    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Define the target file path
    model_filename = os.path.join(target_dir, "silero_vad.onnx")
    logger.debug("VAD model filename", filename=model_filename)

    # Check if the model file already exists
    if not os.path.exists(model_filename):
      logger.debug("VAD model not found, downloading", url=model_url)
      # If it doesn't exist, download the model using wget
      try:
        subprocess.run(["wget", "-O", model_filename, model_url], check=True)
        logger.debug("VAD model downloaded successfully")
      except subprocess.CalledProcessError:
        logger.exception("Failed to download the VAD model using wget")
    else:
      logger.debug("VAD model already exists, skipping download")
    return model_filename


class VoiceActivityDetector:
  def __init__(self, threshold=0.5, frame_rate=16000):
    """
    Initializes the VoiceActivityDetector with a voice activity detection model and a threshold.

    Args:
        threshold (float, optional): The probability threshold for detecting voice activity. Defaults to 0.5.
    """
    self.logger = get_logger("voice_activity_detector")
    self.logger.debug(
      "Initializing VoiceActivityDetector", threshold=threshold, frame_rate=frame_rate
    )
    self.model = VoiceActivityDetection()
    self.threshold = threshold
    self.frame_rate = frame_rate

    self.logger.info(
      "Initialized VoiceActivityDetector",
      threshold=self.threshold,
      frame_rate=self.frame_rate,
    )

  def __call__(self, audio_frame):
    """
    Determines if the given audio frame contains speech by comparing the detected speech probability against
    the threshold.

    Args:
        audio_frame (np.ndarray): The audio frame to be analyzed for voice activity. It is expected to be a
                                  NumPy array of audio samples.

    Returns:
        bool: True if the speech probability exceeds the threshold, indicating the presence of voice activity;
              False otherwise.
    """
    self.logger.debug(
      "VoiceActivityDetector processing frame", shape=audio_frame.shape, dtype=audio_frame.dtype
    )
    speech_probs = self.model.audio_forward(torch.from_numpy(audio_frame.copy()), self.frame_rate)[
      0
    ]
    voice_detected = torch.any(speech_probs > self.threshold).item()
    self.logger.debug(
      "Voice activity result",
      probabilities=speech_probs,
      threshold=self.threshold,
      detected=voice_detected,
    )
    return voice_detected
