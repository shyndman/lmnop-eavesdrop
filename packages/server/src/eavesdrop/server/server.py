import asyncio
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed, InvalidMessage

from eavesdrop.server.config import RTSPConfig, load_config_from_file
from eavesdrop.server.constants import SAMPLE_RATE
from eavesdrop.server.logs import get_logger
from eavesdrop.server.rtsp.cache import RTSPTranscriptionCache
from eavesdrop.server.rtsp.manager import RTSPClientManager
from eavesdrop.server.rtsp.subscriber import RTSPSubscriberManager
from eavesdrop.server.streaming import (
  TranscriptionConfig,
  WebSocketStreamingClient,
)
from eavesdrop.server.websocket import WebSocketClientManager, WebSocketServer
from eavesdrop.wire import (
  ClientType,
  ErrorMessage,
  MessageCodec,
  OutboundMessage,
  TranscriptionSetupMessage,
  WebSocketHeaders,
)
from eavesdrop.wire.messages import HealthCheckRequest


@dataclass
class TranscriberConnection:
  client: WebSocketStreamingClient


@dataclass
class SubscriberConnection:
  pass


ConnectionResult = TranscriberConnection | SubscriberConnection | None


# TODO: Introduce a common pathway for message deserialization
class TranscriptionServer:
  def __init__(self):
    self.client_manager = WebSocketClientManager()
    self.no_voice_activity_chunks = 0
    self.logger = get_logger("server")

    # RTSP-related components
    self.rtsp_client_manager: RTSPClientManager | None = None
    self.subscriber_manager: RTSPSubscriberManager | None = None

  async def initialize_client(
    self, websocket: ServerConnection, msg: TranscriptionSetupMessage
  ) -> WebSocketStreamingClient:
    self.logger.debug(f"initialize_client: Starting initialization for client {msg.stream}")

    # Use config transcription settings as defaults, allow client overrides
    # Parse and validate client options using wire protocol data structure
    user_options = msg.options
    client_overrides = dict(user_options)

    # Create configuration for the streaming client with client overrides
    transcription_config = self.transcription_config.model_copy(update=client_overrides)
    client = WebSocketStreamingClient(
      websocket=websocket,
      stream_name=msg.stream,
      get_audio_func=self.get_audio_from_websocket,
      transcription_config=transcription_config,
    )

    # Start the client and get the completion task
    completion_task = await client.start()
    client._completion_task = completion_task
    self.client_manager.add_client(websocket, client)

    return client

  async def get_audio_from_websocket(self, websocket: ServerConnection) -> np.ndarray | bool:
    """
    Receives audio buffer from websocket and creates a numpy array out of it.
    """
    frame_data = await websocket.recv()
    byte_count = len(frame_data) if frame_data != b"END_OF_AUDIO" else 0
    self.logger.debug("Received audio", bytes_received=byte_count)

    if frame_data == b"END_OF_AUDIO":
      return False

    # Ensure frame_data is bytes for numpy processing
    if isinstance(frame_data, str):
      frame_data = frame_data.encode()
    audio_array = np.frombuffer(frame_data, dtype=np.float32)

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
      stream_name = client.stream_name if client else "unknown"
      timestamp = int(time.time())

      filename = f"{self.debug_audio_path}_{stream_name}_{timestamp}.wav"

      os.makedirs(
        os.path.dirname(filename) if os.path.dirname(filename) else ".",
        exist_ok=True,
      )

      file_handle = sf.SoundFile(
        filename, mode="w", samplerate=SAMPLE_RATE, channels=1, format="WAV"
      )
      self.debug_audio_files[websocket] = (file_handle, filename)

    file_handle, filename = self.debug_audio_files[websocket]
    try:
      file_handle.write(audio_array)
    except Exception:
      self.logger.exception(f"Error writing debug audio to {filename}")

  async def handle_new_connection(self, websocket: ServerConnection) -> ConnectionResult:
    try:
      self.logger.info("handle_new_connection: New client connected")

      # Check WebSocket headers to determine client type
      headers = websocket.request.headers if websocket.request else {}
      client_type = headers.get(WebSocketHeaders.CLIENT_TYPE, ClientType.TRANSCRIBER)

      if client_type == ClientType.RTSP_SUBSCRIBER:
        result = await self._handle_subscriber_connection(websocket, dict(headers))
        return SubscriberConnection() if result else None

      raw_msg = json.loads(await websocket.recv())
      message = MessageCodec.model_validate({"message": raw_msg}).message

      match (client_type, message):
        case (ClientType.TRANSCRIBER, TranscriptionSetupMessage()):
          client = await self._handle_transcriber_connection(websocket, message)
          return TranscriberConnection(client) if client else None

        case (ClientType.HEALTH_CHECK, HealthCheckRequest()):
          return await self._handle_health_check(websocket, message)

    except ConnectionClosed:
      self.logger.info("handle_new_connection: Connection closed by client")
      return None

    except (KeyboardInterrupt, SystemExit):
      raise

  async def _handle_health_check(
    self, websocket: ServerConnection, message: HealthCheckRequest
  ) -> None:
    """Handle traditional transcriber client connections."""
    self.logger.info("Health check successful", websocket_id=websocket.id)
    self._send_error_and_close
    return None

  async def _handle_transcriber_connection(
    self, websocket: ServerConnection, message: TranscriptionSetupMessage
  ) -> WebSocketStreamingClient | None:
    """Handle traditional transcriber client connections."""
    return await self.initialize_client(websocket, message)

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

    # Subscribe the client
    success, error_message = await self.subscriber_manager.subscribe_client(websocket, stream_names)

    if not success:
      await self._send_error_and_close(websocket, error_message or "Subscription failed")
      return False

    return True

  async def _send_error_and_close(self, websocket: ServerConnection, error_message: str) -> None:
    """Send error message and close WebSocket connection."""
    await self._send_and_close(websocket, ErrorMessage(message=error_message))
    self.logger.warning("Sent error and closed connection", error=error_message)

  async def _send_and_close(self, websocket: ServerConnection, message: OutboundMessage) -> None:
    """Send error message and close WebSocket connection."""
    try:
      await websocket.send(json.dumps(asdict(message)))
      await websocket.close()
    except Exception:
      self.logger.exception("Error sending error message and closing connection")

  async def recv_audio(self, websocket: ServerConnection) -> None:
    """
    Handle WebSocket client connection (transcriber or subscriber).
    """
    connection_res = await self.handle_new_connection(websocket)
    if not connection_res:
      return

    try:
      match connection_res:
        case SubscriberConnection():
          await self._handle_subscriber_lifecycle(websocket)

        case TranscriberConnection(client):
          assert client._completion_task is not None
          await client._completion_task
    except (ConnectionClosed, InvalidMessage):
      self.logger.info("Connection closed by client")
    except (KeyboardInterrupt, SystemExit):
      self.logger.info("Shutdown signal received, exiting client loop.")
      raise
    except Exception:
      self.logger.exception("Unexpected error")
    finally:
      self.logger.debug("Entering cleanup phase")
      if self.client_manager.get_client(websocket):
        self.logger.debug("Calling cleanup")
        self.cleanup(websocket)
        self.logger.debug("Closing websocket")
        await websocket.close()
      self.logger.debug("Deleting websocket reference")
      del websocket

  async def _handle_subscriber_lifecycle(self, websocket: ServerConnection) -> None:
    """Handle the lifecycle of an RTSP subscriber client."""
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
      # The WebSocketStreamingClient handles its own cleanup in the stop() method
      # which is called from _wait_for_completion(), so we don't need to do much here
      self.client_manager.remove_client(websocket)

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
