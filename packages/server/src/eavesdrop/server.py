import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed, InvalidMessage

from eavesdrop.config import RTSPConfig, load_config_from_file
from eavesdrop.constants import SAMPLE_RATE
from eavesdrop.logs import get_logger
from eavesdrop.messages import ClientType, ErrorMessage, WebSocketHeaders
from eavesdrop.rtsp.cache import RTSPTranscriptionCache
from eavesdrop.rtsp.manager import RTSPClientManager
from eavesdrop.rtsp.subscriber import RTSPSubscriberManager
from eavesdrop.streaming import (
  TranscriptionConfig,
  WebSocketStreamingClient,
)
from eavesdrop.websocket import WebSocketClientManager, WebSocketServer


@dataclass
class TranscriberConnection:
  client: WebSocketStreamingClient


@dataclass
class SubscriberConnection:
  pass


ConnectionResult = TranscriberConnection | SubscriberConnection | None


class TranscriptionServer:
  RATE = 16000

  def __init__(self):
    self.client_manager = WebSocketClientManager()
    self.no_voice_activity_chunks = 0
    self.logger = get_logger("server")

    # RTSP-related components
    self.rtsp_client_manager: RTSPClientManager | None = None
    self.subscriber_manager: RTSPSubscriberManager | None = None

  async def initialize_client(
    self, websocket: ServerConnection, options: dict
  ) -> WebSocketStreamingClient | None:
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
      # Define which fields WebSocket clients can override
      WEBSOCKET_CONFIGURABLE_FIELDS = {
        "send_last_n_segments",
        "no_speech_thresh",
        "same_output_threshold",
        "use_vad",
        "model",
        "language",
        "initial_prompt",
        "hotwords",
        "vad_parameters",
      }

      # Filter client options to only allowed fields
      client_overrides = {
        key: value for key, value in options.items() if key in WEBSOCKET_CONFIGURABLE_FIELDS
      }

      # Create configuration for the streaming client with client overrides
      transcription_config = self.transcription_config.model_copy(update=client_overrides)

      client = WebSocketStreamingClient(
        websocket=websocket,
        client_uid=options["uid"],
        get_audio_func=self.get_audio_from_websocket,
        transcription_config=transcription_config,
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

  async def get_audio_from_websocket(self, websocket: ServerConnection) -> np.ndarray | bool:
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
    # Ensure frame_data is bytes for numpy processing
    if isinstance(frame_data, str):
      frame_data = frame_data.encode()
    audio_array = np.frombuffer(frame_data, dtype=np.float32)
    self.logger.debug(
      f"get_audio_from_websocket: Created numpy array with {len(audio_array)} samples"
    )

    # Debug audio capture
    if self.debug_audio_path and audio_array is not False:
      self._capture_debug_audio(websocket, audio_array)

    return audio_array

  def _capture_debug_audio(self, websocket: ServerConnection, audio_array: np.ndarray) -> None:
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

      file_handle = sf.SoundFile(
        filename, mode="w", samplerate=SAMPLE_RATE, channels=1, format="WAV"
      )
      self.debug_audio_files[websocket] = (file_handle, filename)

      self.logger.info(f"Debug audio capture started: {filename}")

    file_handle, filename = self.debug_audio_files[websocket]
    try:
      file_handle.write(audio_array)
      self.logger.debug(f"Debug audio: wrote {len(audio_array)} samples to {filename}")
    except Exception:
      self.logger.exception(f"Error writing debug audio to {filename}")

  async def handle_new_connection(self, websocket: ServerConnection) -> ConnectionResult:
    try:
      self.logger.info("handle_new_connection: New client connected")

      # Check WebSocket headers to determine client type
      headers = websocket.request.headers if websocket.request else {}
      client_type = headers.get(WebSocketHeaders.CLIENT_TYPE, ClientType.TRANSCRIBER)

      self.logger.debug(f"handle_new_connection: Client type: {client_type}")

      if client_type == ClientType.RTSP_SUBSCRIBER:
        # Convert headers to dict for easier handling
        headers_dict = dict(headers)
        result = await self._handle_subscriber_connection(websocket, headers_dict)
        return SubscriberConnection() if result else None
      else:
        client = await self._handle_transcriber_connection(websocket)
        return TranscriberConnection(client) if client else None

    except ConnectionClosed:
      self.logger.info("handle_new_connection: Connection closed by client")
      return None
    except (KeyboardInterrupt, SystemExit):
      raise
    except Exception:
      self.logger.exception("handle_new_connection: Error during new connection initialization")
      return None

  async def _handle_transcriber_connection(
    self, websocket: ServerConnection
  ) -> WebSocketStreamingClient | None:
    """Handle traditional transcriber client connections."""
    try:
      options = await websocket.recv()
      options = json.loads(options)
      self.logger.debug(f"_handle_transcriber_connection: Parsed client options: {options}")

      if options.get("type") == "health_check":
        self.logger.info("_handle_transcriber_connection: Health check successful")
        await websocket.close()
        return None

      client = await self.initialize_client(websocket, options)
      return client
    except json.JSONDecodeError:
      self.logger.exception("_handle_transcriber_connection: Failed to decode JSON from client")
      return None

  async def _handle_subscriber_connection(self, websocket: ServerConnection, headers: dict) -> bool:
    """Handle RTSP subscriber client connections."""
    if not self.subscriber_manager:
      error_msg = "RTSP subscription not available: no RTSP streams configured"
      await self._send_error_and_close(websocket, error_msg)
      return False

    # Parse stream names from header
    stream_names_header = headers.get(WebSocketHeaders.STREAM_NAMES, "")
    if not stream_names_header.strip():
      error_msg = f"{WebSocketHeaders.STREAM_NAMES} header is required for RTSP subscribers"
      await self._send_error_and_close(websocket, error_msg)
      return False

    stream_names = [name.strip() for name in stream_names_header.split(",") if name.strip()]
    if not stream_names:
      error_msg = f"No valid stream names provided in {WebSocketHeaders.STREAM_NAMES} header"
      await self._send_error_and_close(websocket, error_msg)
      return False

    self.logger.info(
      "handle_subscriber_connection: Attempting to subscribe to streams",
      streams=stream_names,
      client=id(websocket),
    )

    # Subscribe the client
    success, error_message = await self.subscriber_manager.subscribe_client(websocket, stream_names)

    if not success:
      await self._send_error_and_close(websocket, error_message or "Subscription failed")
      return False

    self.logger.info(
      "handle_subscriber_connection: Client subscribed successfully",
      streams=stream_names,
      client=id(websocket),
    )

    return True

  async def _send_error_and_close(self, websocket: ServerConnection, error_message: str) -> None:
    """Send error message and close WebSocket connection."""
    try:
      error_msg = ErrorMessage(message=error_message)
      await websocket.send(error_msg.model_dump_json())
      await websocket.close()
      self.logger.warning("Sent error and closed connection", error=error_message)
    except Exception:
      self.logger.exception("Error sending error message and closing connection")

  async def recv_audio(self, websocket: ServerConnection) -> None:
    """
    Handle WebSocket client connection (transcriber or subscriber).
    """
    self.logger.debug("recv_audio: Starting connection handling")

    self.logger.debug("recv_audio: About to handle new connection")
    connection_result = await self.handle_new_connection(websocket)
    if not connection_result:
      self.logger.debug("recv_audio: handle_new_connection returned None, exiting")
      return

    # Check if this is a subscriber client
    if isinstance(connection_result, SubscriberConnection):
      await self._handle_subscriber_lifecycle(websocket)
      return

    # This is a transcriber client
    assert isinstance(connection_result, TranscriberConnection)
    client = connection_result.client
    self.logger.debug("recv_audio: Awaiting transcriber client completion")
    assert self.client_manager is not None
    try:
      # Await the client's completion task - this will naturally end when
      # the WebSocket closes, encounters an error, or times out
      assert client._completion_task is not None
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

  async def _handle_subscriber_lifecycle(self, websocket: ServerConnection) -> None:
    """Handle the lifecycle of an RTSP subscriber client."""
    self.logger.debug("_handle_subscriber_lifecycle: Starting subscriber lifecycle management")

    try:
      # Wait for the WebSocket connection to close
      # Subscribers don't send data, they just receive
      async for message in websocket:
        # Log any unexpected messages from subscribers
        self.logger.warning(
          "Received unexpected message from subscriber",
          client=id(websocket),
          message=message[:100] if isinstance(message, str) else str(type(message)),
        )

    except ConnectionClosed:
      self.logger.info("_handle_subscriber_lifecycle: Subscriber connection closed")
    except (KeyboardInterrupt, SystemExit):
      self.logger.info("_handle_subscriber_lifecycle: Shutdown signal received")
      raise
    except Exception:
      self.logger.exception("_handle_subscriber_lifecycle: Unexpected error")
    finally:
      self.logger.debug("_handle_subscriber_lifecycle: Cleaning up subscriber")
      if self.subscriber_manager:
        await self.subscriber_manager.unsubscribe_client(websocket)
      self.logger.debug("_handle_subscriber_lifecycle: Subscriber cleanup complete")

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
      eavesdrop_config = load_config_from_file(Path(config_path))
      rtsp_config = eavesdrop_config.rtsp
      self.transcription_config = eavesdrop_config.transcription

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
    if rtsp_config.streams:
      await self._initialize_rtsp_streams(
        rtsp_config,
        self.transcription_config,
      )

    async def connection_handler(websocket: ServerConnection) -> None:
      await self.recv_audio(websocket)

    websocket_server = WebSocketServer(connection_handler, host, port)

    # Run WebSocket server with or without RTSP streams
    if self.rtsp_client_manager:
      await self._run_with_rtsp_streams(websocket_server)
    else:
      await websocket_server.start()

  def cleanup(self, websocket: ServerConnection) -> None:
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
    rtsp_config: RTSPConfig,
    transcription_config: TranscriptionConfig,
  ) -> None:
    """
    Initialize RTSP client manager and subscriber manager.

    Args:
        rtsp_config: RTSP configuration object
        transcription_config: Global transcription configuration
    """
    try:
      self.logger.info("Initializing RTSP transcription system")

      # Create transcription cache
      transcription_cache = RTSPTranscriptionCache(rtsp_config.cache)

      # Create RTSP subscriber manager with cache
      available_streams = set(rtsp_config.streams.keys())
      self.subscriber_manager = RTSPSubscriberManager(available_streams, transcription_cache)

      # Create RTSP client manager with subscriber manager and cache
      self.rtsp_client_manager = RTSPClientManager(
        transcription_config, self.subscriber_manager, transcription_cache
      )

      self.logger.info("RTSP subscriber manager created", available_streams=list(available_streams))

      # Start all configured streams
      await self.rtsp_client_manager.start_all_streams(rtsp_config.streams)

      self.logger.info(
        "RTSP transcription system initialized",
        active_streams=self.rtsp_client_manager.get_stream_count(),
      )

    except Exception:
      self.logger.exception("Failed to initialize RTSP system")
      # Clean up on failure
      self.rtsp_client_manager = None
      self.subscriber_manager = None

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
