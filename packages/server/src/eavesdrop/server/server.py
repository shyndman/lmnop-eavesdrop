import asyncio
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from structlog.stdlib import BoundLogger
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed, InvalidMessage

from eavesdrop.common import get_logger
from eavesdrop.server.config import TranscriptionConfig, load_config_from_file
from eavesdrop.server.connection_handler import (
  SubscriberConnection,
  TranscriberConnection,
  WebSocketConnectionHandler,
)
from eavesdrop.server.rtsp import RTSPStreamCoordinator
from eavesdrop.server.streaming import WebSocketStreamingClient
from eavesdrop.server.streaming.interfaces import Float32Audio
from eavesdrop.server.websocket import WebSocketClientManager, WebSocketServer
from eavesdrop.wire import (
  ErrorMessage,
  FlushControlMessage,
  RecordingStartedMessage,
  TranscriptionSetupMessage,
  UtteranceCancelledMessage,
  deserialize_message,
  serialize_message,
)

RTSP_FLUSH_REJECTION = "Flush rejected: control_flush is unsupported for RTSP subscriber sessions"
RTSP_RECORDING_START_REJECTION = (
  "Recording start rejected: control_recording_started is unsupported for RTSP subscriber sessions"
)
RTSP_UTTERANCE_CANCEL_REJECTION = (
  "Utterance cancel rejected: control_utterance_cancelled is unsupported for RTSP "
  "subscriber sessions"
)


class ClientManagerProtocol(Protocol):
  def add_client(self, websocket: ServerConnection, client: WebSocketStreamingClient) -> None: ...

  def get_client(self, websocket: ServerConnection) -> WebSocketStreamingClient | None: ...

  def remove_client(self, websocket: ServerConnection) -> None: ...


# TODO: Introduce a common pathway for message deserialization
class TranscriptionServer:
  def __init__(self):
    self.client_manager: ClientManagerProtocol = cast(
      ClientManagerProtocol, WebSocketClientManager()
    )
    self.no_voice_activity_chunks: int = 0
    self.logger: BoundLogger = get_logger("svr")
    self.transcription_config: TranscriptionConfig | None = None

    # RTSP coordinator for stream management
    self._rtsp_coordinator: RTSPStreamCoordinator | None = None
    # Connection handler for WebSocket routing
    self._connection_handler: WebSocketConnectionHandler | None = None

  async def initialize_client(
    self, websocket: ServerConnection, msg: TranscriptionSetupMessage
  ) -> WebSocketStreamingClient:
    if self.transcription_config is None:
      raise RuntimeError("Transcription config not loaded")

    self.logger.debug(f"initialize_client: Starting initialization for client {msg.stream}")

    # Use config transcription settings as defaults, allow client overrides
    # Parse and validate client options using wire protocol data structure
    user_options = msg.options
    client_overrides = user_options.model_dump(exclude_none=True)
    source_mode = user_options.source_mode
    client_overrides.pop("source_mode", None)

    # Create configuration for the streaming client with client overrides
    transcription_config = self.transcription_config.model_copy(update=client_overrides)

    # Log the effective transcription configuration
    self.logger.info(
      "Client transcription config",
      stream=msg.stream,
      model=transcription_config.model,
      custom_model=str(transcription_config.custom_model)
      if transcription_config.custom_model
      else None,
      language=transcription_config.language,
      use_vad=transcription_config.use_vad,
      num_workers=transcription_config.num_workers,
      send_last_n_segments=transcription_config.send_last_n_segments,
      source_mode=source_mode,
      same_output_threshold=transcription_config.same_output_threshold,
      initial_prompt=transcription_config.initial_prompt,
      hotwords=transcription_config.hotwords,
      gpu_name=transcription_config.gpu_name,
    )

    client = WebSocketStreamingClient(
      websocket=websocket,
      stream_name=msg.stream,
      get_audio_func=self.get_audio_from_websocket,
      transcription_config=transcription_config,
      source_mode=source_mode,
    )

    # Start the client and get the completion task
    completion_task = await client.start()
    setattr(client, "_completion_task", completion_task)
    self.client_manager.add_client(websocket, client)

    return client

  async def get_audio_from_websocket(self, websocket: ServerConnection) -> Float32Audio | bool:
    """
    Receives audio buffer from websocket and creates a numpy array out of it.
    """
    frame_data = await websocket.recv()

    if frame_data == b"END_OF_AUDIO":
      return False

    # Ensure frame_data is bytes for numpy processing
    if isinstance(frame_data, str):
      frame_data = frame_data.encode()
    return np.frombuffer(frame_data, dtype=np.float32)

  async def recv_audio(self, websocket: ServerConnection) -> None:
    """
    Handle WebSocket client connection (transcriber or subscriber).
    """
    if not self._connection_handler:
      self.logger.error("Connection handler not initialized")
      return

    connection_res = await self._connection_handler.handle_connection(websocket)
    if not connection_res:
      return

    try:
      match connection_res:
        case SubscriberConnection():
          await self._handle_subscriber_lifecycle(websocket)

        case TranscriberConnection(client, source_mode):
          self.logger.info(
            "Handling transcriber lifecycle",
            stream=client.stream_name,
            source_mode=source_mode,
          )
          completion_task = cast(
            asyncio.Task[None] | None, getattr(client, "_completion_task", None)
          )
          if completion_task is None:
            raise RuntimeError("TranscriberConnection missing completion task — client not started")
          await completion_task
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
        if isinstance(message, str):
          await self._reject_subscriber_control_frame(websocket, message)
          continue

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
      subscriber_manager = (
        self._connection_handler.subscriber_manager if self._connection_handler else None
      )
      if subscriber_manager:
        await subscriber_manager.unsubscribe_client(websocket)
      self.logger.debug("_handle_subscriber_lifecycle: Subscriber cleanup complete")

  async def _reject_subscriber_control_frame(
    self, websocket: ServerConnection, message_json: str
  ) -> None:
    """Reject live-control commands from RTSP subscriber sessions."""
    try:
      message = deserialize_message(message_json)
    except Exception:
      self.logger.warning("Received invalid text frame from subscriber")
      return

    if isinstance(message, FlushControlMessage):
      await websocket.send(
        serialize_message(
          ErrorMessage(
            stream=message.stream,
            message=RTSP_FLUSH_REJECTION,
          )
        )
      )
      return

    if isinstance(message, RecordingStartedMessage):
      await websocket.send(
        serialize_message(
          ErrorMessage(
            stream=message.stream,
            message=RTSP_RECORDING_START_REJECTION,
          )
        )
      )
      return

    if isinstance(message, UtteranceCancelledMessage):
      await websocket.send(
        serialize_message(
          ErrorMessage(
            stream=message.stream,
            message=RTSP_UTTERANCE_CANCEL_REJECTION,
          )
        )
      )
      return

    self.logger.warning(
      "Received unexpected control frame from subscriber",
      message_type=message.type,
    )

  async def run(
    self,
    host: str,
    config_path: str,
    port: int = 9090,
  ) -> None:
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
        + "See documentation for valid configuration structure."
      )
      raise
    except Exception as e:
      self.logger.error("Failed to load configuration file", error=str(e), config_path=config_path)
      self.logger.error("Ensure the configuration file exists and is readable.")
      raise

    self.client_manager = WebSocketClientManager()

    # Initialize RTSP streams if configured
    if rtsp_config.streams:
      self._rtsp_coordinator = RTSPStreamCoordinator(rtsp_config, self.transcription_config)
      await self._rtsp_coordinator.initialize()

    # Create connection handler with subscriber manager from coordinator (if available)
    subscriber_manager = (
      self._rtsp_coordinator.subscriber_manager if self._rtsp_coordinator else None
    )
    self._connection_handler = WebSocketConnectionHandler(
      client_initializer=self.initialize_client,
      subscriber_manager=subscriber_manager,
    )

    async def connection_handler(websocket: ServerConnection) -> None:
      await self.recv_audio(websocket)

    websocket_server = WebSocketServer(connection_handler, host, port)

    # Run WebSocket server, with RTSP shutdown on exit if configured
    try:
      await websocket_server.start()
    finally:
      if self._rtsp_coordinator:
        await self._rtsp_coordinator.shutdown()

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
