import asyncio
import json
import os
from enum import Enum

import numpy as np
from websockets.exceptions import ConnectionClosed, InvalidMessage

from .backend import ServeClientFasterWhisper
from .config import RTSPConfig
from .gpu import resolve_gpu_index
from .logs import get_logger
from .rtsp_manager import RTSPClientManager
from .rtsp_models import RTSPModelManager
from .websocket import ClientManager, WebSocketServer


class BackendType(Enum):
  FASTER_WHISPER = "faster_whisper"

  @staticmethod
  def valid_types() -> list[str]:
    return [backend_type.value for backend_type in BackendType]

  @staticmethod
  def is_valid(backend: str) -> bool:
    return backend in BackendType.valid_types()

  def is_faster_whisper(self) -> bool:
    return self == BackendType.FASTER_WHISPER


class TranscriptionServer:
  RATE = 16000

  def __init__(self):
    self.client_manager = ClientManager(10, 60)
    self.no_voice_activity_chunks = 0
    self.use_vad = True
    self.single_model = False
    self.logger = get_logger("transcription_server")

    # RTSP-related components
    self.rtsp_client_manager: RTSPClientManager | None = None

  async def initialize_client(
    self,
    websocket,
    options,
    faster_whisper_custom_model_path,
  ):
    self.logger.debug(
      f"initialize_client: Starting initialization for client {options['uid']} with "
      f"backend: {self.backend.value}"
    )
    client = None

    try:
      if self.backend.is_faster_whisper():
        self.logger.debug("initialize_client: Initializing faster_whisper backend")

        if faster_whisper_custom_model_path is not None:
          self.logger.info(
            f"initialize_client: Using custom model {faster_whisper_custom_model_path}"
          )
          options["model"] = faster_whisper_custom_model_path

        self.logger.debug("initialize_client: Creating faster_whisper client")
        client = ServeClientFasterWhisper(
          websocket,
          language=options["language"],
          task=options["task"],
          client_uid=options["uid"],
          model=options["model"],
          initial_prompt=options.get("initial_prompt"),
          vad_parameters=options.get("vad_parameters"),
          use_vad=self.use_vad,
          single_model=self.single_model,
          send_last_n_segments=options.get("send_last_n_segments", 10),
          no_speech_thresh=options.get("no_speech_thresh", 0.45),
          clip_audio=options.get("clip_audio", False),
          same_output_threshold=options.get("same_output_threshold", 10),
          cache_path=self.cache_path,
          device_index=self.device_index,
        )
        await client.initialize()
        self.logger.info("initialize_client: Running faster_whisper backend.")
        self.logger.debug("initialize_client: faster_whisper client created successfully")
    except Exception:
      self.logger.exception("initialize_client: Error creating faster_whisper client")
      return None

    if client is None:
      self.logger.error(f"initialize_client: Client is None for backend {self.backend.value}")
      raise ValueError(f"Backend type {self.backend.value} not recognised or not handled.")

    self.logger.debug("initialize_client: Client created successfully")

    self.logger.debug("initialize_client: Adding client to client manager")
    self.client_manager.add_client(websocket, client)
    self.logger.debug("initialize_client: Client initialization completed")
    return client

  async def get_audio_from_websocket(self, websocket):
    """
    Receives audio buffer from websocket and creates a numpy array out of it.
    """
    self.logger.debug("get_audio_from_websocket: About to receive data")
    frame_data = await websocket.recv()
    byte_count = len(frame_data) if frame_data != b"END_OF_AUDIO" else 0
    self.logger.debug(f"get_audio_from_websocket: Received {byte_count} bytes")

    if frame_data == b"END_OF_AUDIO":
      self.logger.debug("get_audio_from_websocket: Received END_OF_AUDIO signal")
      return False

    self.logger.debug("get_audio_from_websocket: Converting to numpy array")
    audio_array = np.frombuffer(frame_data, dtype=np.float32)
    self.logger.debug(
      f"get_audio_from_websocket: Created numpy array with {len(audio_array)} samples"
    )

    # Debug audio capture
    if self.debug_audio_path and audio_array is not False:
      self._capture_debug_audio(websocket, audio_array)

    return audio_array

  def _capture_debug_audio(self, websocket, audio_array):
    """
    Captures audio data to debug .wav files for analysis.
    """
    import os
    import time

    import soundfile as sf

    if websocket not in self.debug_audio_files:
      client = self.client_manager.get_client(websocket) if self.client_manager else None
      client_uid = client.client_uid if client else "unknown"
      timestamp = int(time.time())

      filename = f"{self.debug_audio_path}_{client_uid}_{timestamp}.wav"

      os.makedirs(
        os.path.dirname(filename) if os.path.dirname(filename) else ".",
        exist_ok=True,
      )

      file_handle = sf.SoundFile(filename, mode="w", samplerate=self.RATE, channels=1, format="WAV")
      self.debug_audio_files[websocket] = (file_handle, filename)

      self.logger.info(f"Debug audio capture started: {filename}")

    file_handle, filename = self.debug_audio_files[websocket]
    try:
      file_handle.write(audio_array)
      self.logger.debug(f"Debug audio: wrote {len(audio_array)} samples to {filename}")
    except Exception:
      self.logger.exception(f"Error writing debug audio to {filename}")

  async def handle_new_connection(
    self,
    websocket,
    faster_whisper_custom_model_path,
  ):
    try:
      self.logger.info("handle_new_connection: New client connected")
      options = await websocket.recv()
      options = json.loads(options)
      self.logger.debug(f"handle_new_connection: Parsed client options: {options}")

      if options.get("type") == "health_check":
        self.logger.info("handle_new_connection: Health check successful")
        await websocket.close()
        return None

      self.use_vad = options.get("use_vad")

      if await self.client_manager.is_server_full(websocket, options):
        await websocket.close()
        return None

      client = await self.initialize_client(
        websocket,
        options,
        faster_whisper_custom_model_path,
      )
      return client
    except json.JSONDecodeError:
      self.logger.exception("handle_new_connection: Failed to decode JSON from client")
      return None
    except ConnectionClosed:
      self.logger.info("handle_new_connection: Connection closed by client")
      return None
    except (KeyboardInterrupt, SystemExit):
      raise
    except Exception:
      self.logger.exception("handle_new_connection: Error during new connection initialization")
      return None

  async def process_audio_frames(self, websocket):
    self.logger.debug("process_audio_frames: Starting")

    self.logger.debug("process_audio_frames: Getting audio from websocket")
    frame_np = await self.get_audio_from_websocket(websocket)
    self.logger.debug("process_audio_frames: Got audio from websocket")

    self.logger.debug("process_audio_frames: Getting client from manager")
    client = self.client_manager.get_client(websocket)
    self.logger.debug(f"process_audio_frames: Got client: {client.client_uid if client else None}")

    if frame_np is False:
      client_uid = client.client_uid if client else "unknown"
      self.logger.debug(f"process_audio_frames: End of audio received for client {client_uid}")
      self.logger.debug("process_audio_frames: Returning False for end of audio")
      return False

    self.logger.debug(f"process_audio_frames: Processing {len(frame_np)} audio samples")

    if client:
      self.logger.debug(
        f"process_audio_frames: Adding {len(frame_np)} audio samples to client {client.client_uid}"
      )
      client.add_frames(frame_np)
      self.logger.debug(
        f"process_audio_frames: Successfully added frames to client {client.client_uid}"
      )
    else:
      self.logger.debug(
        "process_audio_frames: No client found for websocket when processing audio frames"
      )

    self.logger.debug("process_audio_frames: Returning True")
    return True

  async def recv_audio(
    self,
    websocket,
    backend: BackendType = BackendType.FASTER_WHISPER,
    faster_whisper_custom_model_path=None,
  ):
    """
    Receive audio chunks from a client in an infinite loop.
    """
    self.logger.debug(f"recv_audio: Starting for backend {backend.value}")
    self.backend = backend

    self.logger.debug("recv_audio: About to handle new connection")
    client = await self.handle_new_connection(
      websocket,
      faster_whisper_custom_model_path,
    )
    if not client:
      self.logger.debug("recv_audio: handle_new_connection returned False, exiting")
      return

    asyncio.create_task(client.speech_to_text())
    self.logger.debug("recv_audio: Entering main audio processing loop")
    assert self.client_manager is not None
    try:
      loop_count = 0
      while not self.client_manager.is_client_timeout(websocket):
        loop_count += 1
        if loop_count % 100 == 0:  # Log every 100 iterations to avoid spam
          self.logger.debug(f"recv_audio: Main loop iteration {loop_count}")

        self.logger.debug("recv_audio: About to process audio frames")
        if not await self.process_audio_frames(websocket):
          self.logger.debug("recv_audio: process_audio_frames returned False, breaking loop")
          break
        self.logger.debug("recv_audio: process_audio_frames completed successfully")
    except (ConnectionClosed, InvalidMessage):
      self.logger.info("recv_audio: Connection closed by client")
    except (KeyboardInterrupt, SystemExit):
      self.logger.info("recv_audio: Shutdown signal received, exiting client loop.")
      raise
    except Exception:
      self.logger.exception("recv_audio: Unexpected error")
    finally:
      self.logger.debug("recv_audio: Entering cleanup phase")
      if self.client_manager.get_client(websocket):
        self.logger.debug("recv_audio: Calling cleanup")
        self.cleanup(websocket)
        self.logger.debug("recv_audio: Closing websocket")
        await websocket.close()
      self.logger.debug("recv_audio: Deleting websocket reference")
      del websocket

  async def run(
    self,
    host,
    port=9090,
    backend="faster_whisper",
    faster_whisper_custom_model_path=None,
    single_model=False,
    max_clients=4,
    max_connection_time=600,
    cache_path="~/.cache/eavesdrop/",
    debug_audio_path=None,
    gpu_name: str | None = None,
    config: str | None = None,
  ):
    """
    Run the transcription server.
    """
    self.device_index = resolve_gpu_index(gpu_name)
    self.cache_path = cache_path
    self.debug_audio_path = debug_audio_path
    self.debug_audio_files = {}  # websocket -> (file_handle, filename)

    self.client_manager = ClientManager(max_clients, max_connection_time)
    if faster_whisper_custom_model_path is not None and not os.path.exists(
      faster_whisper_custom_model_path
    ):
      raise ValueError(
        f"Custom faster_whisper model '{faster_whisper_custom_model_path}' is not a valid path."
      )
    if single_model:
      if faster_whisper_custom_model_path:
        self.logger.info("Custom model option was provided. Switching to single model mode.")
        self.single_model = True
      else:
        self.logger.info("Single model mode currently only works with custom models.")
    if not BackendType.is_valid(backend):
      raise ValueError(
        f"{backend} is not a valid backend type. Choose backend from {BackendType.valid_types()}"
      )

    # Load and initialize RTSP streams if config provided
    if config:
      rtsp_streams = await self._load_rtsp_config(config)
      if rtsp_streams:
        await self._initialize_rtsp_streams(
          rtsp_streams, backend, faster_whisper_custom_model_path, single_model, cache_path
        )

    async def connection_handler(websocket):
      await self.recv_audio(
        websocket,
        backend=BackendType(backend),
        faster_whisper_custom_model_path=faster_whisper_custom_model_path,
      )

    websocket_server = WebSocketServer(connection_handler, host, port)

    # Run WebSocket server with or without RTSP streams
    if self.rtsp_client_manager:
      await self._run_with_rtsp_streams(websocket_server)
    else:
      await websocket_server.start()

  def cleanup(self, websocket):
    """
    Cleans up resources associated with a given client's websocket.
    """
    self.logger.debug("cleanup: Starting cleanup process")
    client = self.client_manager.get_client(websocket)
    if client:
      self.logger.debug(f"cleanup: Starting cleanup for client {client.client_uid}")

      self.logger.debug(f"cleanup: Removing client {client.client_uid} from client manager")
      self.client_manager.remove_client(websocket)
      self.logger.debug(f"cleanup: Cleanup completed for client {client.client_uid}")

    # Clean up debug audio file if exists
    if websocket in self.debug_audio_files:
      file_handle, filename = self.debug_audio_files.pop(websocket)
      try:
        file_handle.close()
        self.logger.info(f"Debug audio capture finished: {filename}")
      except Exception:
        self.logger.exception(f"Error closing debug audio file {filename}")

  async def _load_rtsp_config(self, config_path: str) -> dict[str, str]:
    """
    Load and validate RTSP configuration from file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary mapping stream names to RTSP URLs
    """
    try:
      rtsp_config = RTSPConfig(config_path)
      streams = rtsp_config.load_and_validate()

      if streams:
        self.logger.info(
          "RTSP configuration loaded successfully",
          config_path=config_path,
          stream_count=len(streams),
        )
      else:
        self.logger.info("No RTSP streams configured")

      return streams

    except Exception as e:
      self.logger.error("Failed to load RTSP configuration", config_path=config_path, error=str(e))
      # Return empty dict to continue without RTSP streams
      return {}

  async def _initialize_rtsp_streams(
    self,
    rtsp_streams: dict[str, str],
    backend: str,
    faster_whisper_custom_model_path: str | None,
    single_model: bool,
    cache_path: str,
  ) -> None:
    """
    Initialize RTSP client manager and model manager.

    Args:
        rtsp_streams: Dictionary of stream names to RTSP URLs
        backend: Backend type for transcription
        faster_whisper_custom_model_path: Custom model path if provided
        single_model: Whether to use single model mode
        cache_path: Path for model caching
    """
    try:
      self.logger.info("Initializing RTSP transcription system")

      # Create backend parameters for model manager
      backend_params = {
        "backend": backend,
        "faster_whisper_custom_model_path": faster_whisper_custom_model_path,
        "single_model": single_model,
        "cache_path": cache_path,
        "device_index": self.device_index,
        "task": "transcribe",
        "language": None,  # Auto-detect
        "model": faster_whisper_custom_model_path or "distil-small.en",
        "initial_prompt": None,
        "vad_parameters": None,
        "use_vad": self.use_vad,
        "no_speech_thresh": 0.45,
        "clip_audio": False,
        "same_output_threshold": 10,
      }

      # Create model manager
      model_manager = RTSPModelManager(backend_params)

      # Create RTSP client manager
      self.rtsp_client_manager = RTSPClientManager(model_manager)

      # Start all configured streams
      await self.rtsp_client_manager.start_all_streams(rtsp_streams)

      self.logger.info(
        "RTSP transcription system initialized",
        active_streams=self.rtsp_client_manager.get_stream_count(),
      )

    except Exception as e:
      self.logger.error("Failed to initialize RTSP system", error=str(e))
      # Clean up on failure
      self.rtsp_client_manager = None

  async def _run_with_rtsp_streams(self, websocket_server: WebSocketServer) -> None:
    """
    Run WebSocket server concurrently with RTSP streams.

    Args:
        websocket_server: WebSocket server instance to run
    """
    self.logger.info("Starting server with RTSP and WebSocket support")

    try:
      # Create WebSocket server task
      websocket_task = asyncio.create_task(websocket_server.start())
      websocket_task.set_name("websocket_server")

      # The RTSP streams are already running via RTSPClientManager
      # We just need to wait for the WebSocket server and handle shutdown

      await websocket_task

    except (KeyboardInterrupt, SystemExit):
      self.logger.info("Shutdown signal received")
    except Exception as e:
      self.logger.error("Error in server execution", error=str(e))
    finally:
      # Clean up RTSP streams
      if self.rtsp_client_manager:
        self.logger.info("Shutting down RTSP streams")
        await self.rtsp_client_manager.stop_all_streams()
        self.rtsp_client_manager = None
