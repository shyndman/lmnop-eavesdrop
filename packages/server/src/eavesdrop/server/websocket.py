import asyncio
import time
from collections.abc import Awaitable, Callable

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed, InvalidMessage

from eavesdrop.server.logs import get_logger


class WebSocketClientManager:
  def __init__(self):
    """
    Initializes the WebSocketClientManager for tracking client connections.
    """
    self.clients = {}
    self.start_times = {}
    self.logger = get_logger("ws/clientmgr")

  def add_client(self, websocket, client):
    """
    Adds a client and their connection start time to the tracking dictionaries.

    Args:
        websocket: The websocket associated with the client to add.
        client: The client object to be added and tracked.
    """
    self.logger.debug(f"Adding client {client.stream_name} to client manager")
    self.clients[websocket] = client
    self.start_times[websocket] = time.time()
    self.logger.debug(f"Client {client.stream_name} added. Total clients: {len(self.clients)}")

  def get_client(self, websocket):
    """
    Retrieves a client associated with the given websocket.

    Args:
        websocket: The websocket associated with the client to retrieve.

    Returns:
        The client object if found, False otherwise.
    """
    if websocket in self.clients:
      return self.clients[websocket]
    return False

  def remove_client(self, websocket):
    """
    Removes a client and their connection start time from the tracking dictionaries. Performs
    cleanup on the client if necessary.

    websocket: The websocket associated with the client to be removed.
    """
    client = self.clients.pop(websocket, None)
    if client:
      self.logger.debug(f"Removing client {client.stream_name} from client manager")
      client.cleanup()
    else:
      self.logger.debug("No client found for websocket during removal")
    self.start_times.pop(websocket, None)
    self.logger.debug(f"Client removed. Remaining clients: {len(self.clients)}")


class WebSocketServer:
  """Wrapper around WebSocket server that handles connection errors gracefully"""

  def __init__(
    self,
    handler: Callable[[ServerConnection], Awaitable[None]],
    host,
    port,
    **kwargs,
  ):
    self.handler = handler
    self.host = host
    self.port = port
    self.kwargs = kwargs
    self.logger = get_logger("ws/server")

  async def start(self):
    self.logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
    async with serve(
      self.handler,
      self.host,
      self.port,
      **self.kwargs,
    ):
      await asyncio.Future()  # run forever

  async def error_handling_wrapper(self, websocket: ServerConnection):
    """Wrapper that catches and logs connection errors without crashing"""
    addr = websocket.remote_address

    try:
      self.logger.info("Connection begin", address=addr, websocket_id=websocket.id)
      await self.handler(websocket)
    except (EOFError, InvalidMessage):
      self.logger.debug(
        "Connection from failed handshake (likely port scan/health check)",
        websocket_id=websocket.id,
      )
      # Don't re-raise - this is expected for non-WebSocket clients
    except ConnectionClosed as e:
      self.logger.debug("Connection closed", error=e, websocket_id=websocket.id)
    except (KeyboardInterrupt, SystemExit):
      raise
    except Exception:
      self.logger.exception("Connection unexpected error", websocket_id=websocket.id)

  def __exit__(self, *args):
    pass
