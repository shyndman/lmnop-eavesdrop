import asyncio

from .backend import ServeClientFasterWhisper
from .logs import get_logger


class RTSPModelManager:
  """
  Centralized model management for RTSP stream transcription.

  Manages the creation and sharing of transcription models across multiple
  RTSP streams. Ensures consistent model parameters and efficient resource
  usage by sharing a single model instance when possible.
  """

  def __init__(self, backend_params: dict):
    """
    Initialize the RTSP model manager.

    Args:
        backend_params: Dictionary of parameters for model creation
    """
    self.backend_params = backend_params
    self.shared_transcriber: ServeClientFasterWhisper | None = None
    self.model_lock = asyncio.Lock()
    self.logger = get_logger("rtsp_model_manager")

    # Track model usage
    self.model_creation_count = 0
    self.transcriber_requests = 0

  async def get_transcriber(self) -> ServeClientFasterWhisper:
    """
    Get or create a shared transcriber instance.

    Returns:
        ServeClientFasterWhisper instance configured for RTSP transcription

    Raises:
        Exception: If model creation fails
    """
    async with self.model_lock:
      self.transcriber_requests += 1

      if self.shared_transcriber is None:
        self.logger.info("Creating shared transcription model for RTSP streams")
        self.shared_transcriber = await self._create_transcriber()
        self.model_creation_count += 1

        self.logger.info(
          "Shared transcription model created successfully",
          model_type=self.backend_params.get("backend", "unknown"),
          model_path=self.backend_params.get("faster_whisper_custom_model_path", "default"),
          single_model=self.backend_params.get("single_model", False),
        )
      else:
        self.logger.debug(
          "Reusing existing shared transcription model", requests=self.transcriber_requests
        )

      return self.shared_transcriber

  async def _create_transcriber(self) -> ServeClientFasterWhisper:
    """
    Create a new transcriber instance with RTSP-appropriate parameters.

    Returns:
        Configured ServeClientFasterWhisper instance

    Raises:
        Exception: If model creation or initialization fails
    """
    try:
      # Create transcriber with dummy websocket (None) since RTSP doesn't use WebSocket
      transcriber = ServeClientFasterWhisper(
        websocket=None,  # RTSP streams don't have WebSocket connections
        task=self.backend_params.get("task", "transcribe"),
        device=None,  # Will be auto-detected during initialization
        language=self.backend_params.get("language", None),
        client_uid="rtsp_shared_model",  # Unique identifier for logging
        model=self.backend_params.get("model", "distil-small.en"),
        initial_prompt=self.backend_params.get("initial_prompt", None),
        vad_parameters=self.backend_params.get("vad_parameters", None),
        use_vad=self.backend_params.get("use_vad", True),
        single_model=self.backend_params.get("single_model", True),
        send_last_n_segments=10,  # Not used for RTSP but required parameter
        no_speech_thresh=self.backend_params.get("no_speech_thresh", 0.45),
        clip_audio=self.backend_params.get("clip_audio", False),
        same_output_threshold=self.backend_params.get("same_output_threshold", 10),
        cache_path=self.backend_params.get("cache_path", "~/.cache/eavesdrop/"),
        device_index=self.backend_params.get("device_index", 0),
      )

      # Initialize the transcriber (loads model, detects device, etc.)
      self.logger.debug("Initializing transcriber model")
      await transcriber.initialize()

      return transcriber

    except Exception:
      self.logger.exception("Failed to create transcriber")
      raise Exception("Model creation failed")

  async def cleanup(self) -> None:
    """
    Clean up model resources.

    Performs any necessary cleanup of the shared transcriber instance.
    Should be called during server shutdown.
    """
    async with self.model_lock:
      if self.shared_transcriber is not None:
        self.logger.info("Cleaning up shared transcription model")

        # ServeClientFasterWhisper doesn't have explicit cleanup method
        # but we can clear our reference
        self.shared_transcriber = None

        self.logger.info(
          "Model cleanup completed",
          creation_count=self.model_creation_count,
          total_requests=self.transcriber_requests,
        )

  def get_model_info(self) -> dict[str, str | int | bool]:
    """
    Get information about the current model state.

    Returns:
        Dictionary with model status information
    """
    return {
      "model_created": self.shared_transcriber is not None,
      "creation_count": self.model_creation_count,
      "transcriber_requests": self.transcriber_requests,
      "backend": self.backend_params.get("backend", "unknown"),
      "model_path": self.backend_params.get("faster_whisper_custom_model_path", "default"),
      "single_model": self.backend_params.get("single_model", False),
      "device_index": self.backend_params.get("device_index", 0),
    }
