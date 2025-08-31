import asyncio
import json
import os

import numpy as np
from websockets.exceptions import ConnectionClosed, InvalidMessage

from .config import EavesdropConfig
from .logs import get_logger
from .rtsp_manager import RTSPClientManager
from .streaming import (
  BufferConfig,
  TranscriptionConfig,
  WebSocketStreamingClient,
)
from .websocket import WebSocketClientManager, WebSocketServer


class TranscriptionServer:
  RATE = 16000

  def __init__(self):
    self.client_manager = WebSocketClientManager()
    self.no_voice_activity_chunks = 0
    self.logger = get_logger("transcription_server")

    # RTSP-related components
    self.rtsp_client_manager: RTSPClientManager | None = None

  async def initialize_client(self, websocket, options):
    self.logger.debug(f"initialize_client: Starting initialization for client {options['uid']}")

    try:
      self.logger.debug("initialize_client: Initializing faster_whisper client")

      # Validate that clients cannot override immutable settings
      if "cache_path" in options:
        error_msg = (
          "WebSocket clients cannot override 'cache_path' - this is a server constant. "
          "Remove this option from your connection request."
        )
        self.logger.error(
          "Client attempted to override immutable setting",
          client_uid=options["uid"],
          setting="cache_path",
        )
        await websocket.send(json.dumps({"error": error_msg}))
        await websocket.close()
        return None

      if "single_model" in options:
        error_msg = (
          "WebSocket clients cannot override 'single_model' - this is a server constant. "
          "Remove this option from your connection request."
        )
        self.logger.error(
          "Client attempted to override immutable setting",
          client_uid=options["uid"],
          setting="single_model",
        )
        await websocket.send(json.dumps({"error": error_msg}))
        await websocket.close()
        return None

      # Use config transcription settings as defaults, allow most overrides
      # Create configurations for the streaming client
      buffer_config = BufferConfig(
        clip_audio=options.get("clip_audio", self.transcription_config.clip_audio)
      )

      transcription_config = TranscriptionConfig(
        send_last_n_segments=options.get(
          "send_last_n_segments", self.transcription_config.send_last_n_segments
        ),
        no_speech_thresh=options.get(
          "no_speech_thresh", self.transcription_config.no_speech_thresh
        ),
        same_output_threshold=options.get(
          "same_output_threshold", self.transcription_config.same_output_threshold
        ),
        use_vad=options.get("use_vad", self.transcription_config.use_vad),
        clip_audio=options.get("clip_audio", self.transcription_config.clip_audio),
        model=options.get("model", self.transcription_config.model),
        language=options.get("language", self.transcription_config.language),
        initial_prompt=options.get("initial_prompt", self.transcription_config.initial_prompt),
        vad_parameters=options.get("vad_parameters", self.transcription_config.vad_parameters),
        device_index=self.transcription_config.device_index,
      )

      client = WebSocketStreamingClient(
        websocket=websocket,
        client_uid=options["uid"],
        get_audio_func=self.get_audio_from_websocket,
        transcription_config=transcription_config,
        buffer_config=buffer_config,
      )

      # Start the client and get the completion task
      completion_task = await client.start()

      # Store the completion task with the client so we can await it later
      client._completion_task = completion_task
      self.logger.info("initialize_client: Running faster_whisper.")
      self.logger.debug("initialize_client: faster_whisper client created successfully")
    except Exception:
      self.logger.exception("initialize_client: Error creating faster_whisper client")
      return None

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

  async def handle_new_connection(self, websocket):
    try:
      self.logger.info("handle_new_connection: New client connected")
      options = await websocket.recv()
      options = json.loads(options)
      self.logger.debug(f"handle_new_connection: Parsed client options: {options}")

      if options.get("type") == "health_check":
        self.logger.info("handle_new_connection: Health check successful")
        await websocket.close()
        return None

      client = await self.initialize_client(websocket, options)
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

  async def recv_audio(self, websocket):
    """
    Receive audio chunks from a client in an infinite loop.
    """
    self.logger.debug("recv_audio: Starting")

    self.logger.debug("recv_audio: About to handle new connection")
    client = await self.handle_new_connection(websocket)
    if not client:
      self.logger.debug("recv_audio: handle_new_connection returned False, exiting")
      return

    # The new WebSocketStreamingClient handles the audio processing internally
    # We await its completion task which will finish when the client connection ends
    self.logger.debug("recv_audio: Awaiting client completion")
    assert self.client_manager is not None
    try:
      # Await the client's completion task - this will naturally end when
      # the WebSocket closes, encounters an error, or times out
      await client._completion_task
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
    config_path: str,
    port=9090,
    debug_audio_path=None,
  ):
    """
    Run the transcription server.
    """
    # Load and validate configuration file
    try:
      eavesdrop_config = EavesdropConfig(config_path)
      rtsp_streams, self.transcription_config = eavesdrop_config.load_and_validate()
    except ValueError as e:
      self.logger.error("Configuration validation failed", error=str(e), config_path=config_path)
      self.logger.error(
        "Please check your configuration file format and values. "
        "See documentation for valid configuration structure."
      )
      raise
    except Exception as e:
      self.logger.error("Failed to load configuration file", error=str(e), config_path=config_path)
      self.logger.error("Ensure the configuration file exists and is readable.")
      raise

    self.debug_audio_path = debug_audio_path
    self.debug_audio_files = {}  # websocket -> (file_handle, filename)

    self.client_manager = WebSocketClientManager()

    # Initialize RTSP streams if configured
    if rtsp_streams:
      await self._initialize_rtsp_streams(rtsp_streams, self.transcription_config)

    async def connection_handler(websocket):
      await self.recv_audio(websocket)

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

      # The WebSocketStreamingClient handles its own cleanup in the stop() method
      # which is called from _wait_for_completion(), so we don't need to do much here

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

  async def _initialize_rtsp_streams(
    self,
    rtsp_streams: dict[str, str],
    transcription_config: TranscriptionConfig,
  ) -> None:
    """
    Initialize RTSP client manager.

    Args:
        rtsp_streams: Dictionary of stream names to RTSP URLs
        transcription_config: Global transcription configuration
    """
    try:
      self.logger.info("Initializing RTSP transcription system")

      # Create RTSP client manager
      self.rtsp_client_manager = RTSPClientManager(transcription_config)

      # Start all configured streams
      await self.rtsp_client_manager.start_all_streams(rtsp_streams)

      self.logger.info(
        "RTSP transcription system initialized",
        active_streams=self.rtsp_client_manager.get_stream_count(),
      )

    except Exception:
      self.logger.exception("Failed to initialize RTSP system")
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
    except Exception:
      self.logger.exception("Error in server execution")
    finally:
      # Clean up RTSP streams
      if self.rtsp_client_manager:
        self.logger.info("Shutting down RTSP streams")
        await self.rtsp_client_manager.stop_all_streams()
        self.rtsp_client_manager = None
