from pathlib import Path
from typing import Protocol, cast

from structlog.stdlib import BoundLogger
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed, InvalidMessage

from eavesdrop.common import get_logger
from eavesdrop.server.config import TranscriptionConfig, load_config_from_file
from eavesdrop.server.connection_handler import WebSocketConnectionHandler
from eavesdrop.server.langfuse_tracing import flush_langfuse
from eavesdrop.server.streaming import WebSocketStreamingClient
from eavesdrop.server.websocket import WebSocketClientManager, WebSocketServer
from eavesdrop.wire import TranscriptionSetupMessage


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

    # Create configuration for the streaming client with client overrides
    transcription_config = TranscriptionConfig.model_validate(
      self.transcription_config.model_dump() | client_overrides
    )

    # Log the effective transcription configuration
    self.logger.info(
      "Client transcription config",
      stream=msg.stream,
      language=transcription_config.language,
      use_vad=transcription_config.use_vad,
      send_last_n_segments=transcription_config.send_last_n_segments,
    )

    client = WebSocketStreamingClient(
      websocket=websocket,
      stream_name=msg.stream,
      transcription_config=transcription_config,
    )

    # Start the client; it records its own completion task internally
    _ = await client.start()
    self.client_manager.add_client(websocket, client)

    return client

  async def recv_audio(self, websocket: ServerConnection) -> None:
    """
    Handle a live WebSocket transcription connection.
    """
    if not self._connection_handler:
      self.logger.error("Connection handler not initialized")
      return

    connection_res = await self._connection_handler.handle_connection(websocket)
    if not connection_res:
      return

    try:
      client = connection_res.client
      self.logger.info("Handling transcriber lifecycle", stream=client.stream_name)
      completion_task = client.completion_task
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
      self.transcription_config = eavesdrop_config.transcription

    except ValueError as e:
      self.logger.exception(
        "Configuration validation failed", error=str(e), config_path=config_path
      )
      self.logger.error(
        "Please check your configuration file format and values. "
        + "See documentation for valid configuration structure."
      )
      raise
    except Exception as e:
      self.logger.exception(
        "Failed to load configuration file", error=str(e), config_path=config_path
      )
      self.logger.error("Ensure the configuration file exists and is readable.")
      raise

    self.client_manager = WebSocketClientManager()

    self._connection_handler = WebSocketConnectionHandler(
      client_initializer=self.initialize_client,
    )

    async def connection_handler(websocket: ServerConnection) -> None:
      await self.recv_audio(websocket)

    websocket_server = WebSocketServer(connection_handler, host, port)

    # Flush tracing when the WebSocket server exits.
    try:
      await websocket_server.start()
    finally:
      flush_langfuse()

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
