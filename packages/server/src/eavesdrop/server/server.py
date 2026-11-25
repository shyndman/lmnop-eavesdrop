from pathlib import Path
from typing import Protocol, cast

from structlog.stdlib import BoundLogger
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed, InvalidMessage

from eavesdrop.common import get_logger
from eavesdrop.server.config import TranscriptionConfig, load_config_from_file
from eavesdrop.server.connection_handler import (
  TranscriberConnection,
  WebSocketConnectionHandler,
)
from eavesdrop.server.langfuse_tracing import flush_langfuse
from eavesdrop.server.streaming import WebSocketStreamingClient
from eavesdrop.server.websocket import WebSocketClientManager, WebSocketServer
from eavesdrop.wire import TranscriptionSetupMessage


class ClientManagerProtocol(Protocol):
  def add_client(self, websocket: ServerConnection, client: WebSocketStreamingClient) -> None: ...

  def get_client(self, websocket: ServerConnection) -> WebSocketStreamingClient | None: ...

  def remove_client(self, websocket: ServerConnection) -> None: ...


class TranscriptionServer:
  """WebSocket transcription server managing live Orukeet client sessions."""

  def __init__(self) -> None:
    self.client_manager: ClientManagerProtocol = cast(
      ClientManagerProtocol, WebSocketClientManager()
    )
    self.logger: BoundLogger = get_logger("svr")
    self.transcription_config: TranscriptionConfig | None = None
    self._connection_handler: WebSocketConnectionHandler | None = None

  async def initialize_client(
    self, websocket: ServerConnection, msg: TranscriptionSetupMessage
  ) -> WebSocketStreamingClient:
    """Create and start a live streaming client after setup validation."""
    if self.transcription_config is None:
      raise RuntimeError("Transcription config not loaded")

    self.logger.debug(f"initialize_client: Starting initialization for client {msg.stream}")
    client_overrides = msg.options.model_dump(exclude_none=True)
    transcription_config = TranscriptionConfig.model_validate(
      self.transcription_config.model_dump() | client_overrides
    )

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
    _ = await client.start()
    self.client_manager.add_client(websocket, client)
    return client

  async def recv_audio(self, websocket: ServerConnection) -> None:
    """Handle one WebSocket client connection lifecycle."""
    if not self._connection_handler:
      self.logger.error("Connection handler not initialized")
      return

    connection_res = await self._connection_handler.handle_connection(websocket)
    if not connection_res:
      return

    try:
      match connection_res:
        case TranscriberConnection(client):
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
    """Run the transcription server."""
    try:
      eavesdrop_config = load_config_from_file(Path(config_path))
      self.transcription_config = eavesdrop_config.transcription
    except ValueError as e:
      self.logger.error("Configuration validation failed", error=str(e), config_path=config_path)
      raise
    except Exception as e:
      self.logger.error("Failed to load configuration file", error=str(e), config_path=config_path)
      raise

    self.client_manager = WebSocketClientManager()
    self._connection_handler = WebSocketConnectionHandler(client_initializer=self.initialize_client)

    async def connection_handler(websocket: ServerConnection) -> None:
      await self.recv_audio(websocket)

    websocket_server = WebSocketServer(connection_handler, host, port)
    try:
      await websocket_server.start()
    finally:
      flush_langfuse()

  def cleanup(self, websocket: ServerConnection) -> None:
    """Clean up resources associated with a given client's websocket."""
    self.logger.debug("cleanup: Starting cleanup process")
    client = self.client_manager.get_client(websocket)
    if client:
      self.client_manager.remove_client(websocket)
