import asyncio
import json
import time
from collections.abc import Awaitable, Callable

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed, InvalidMessage

from .logs import get_logger


class ClientManager:
  def __init__(self, max_clients=4, max_connection_time=600):
    """
    Initializes the ClientManager with specified limits on client connections and connection
    durations.

    max_clients: The maximum number of simultaneous client connections allowed. Set to 0 or None for
      no limit.
    max_connection_time: The maximum duration (in seconds) a client can stay connected. Set to 0 or
      None for no limit.
    """
    self.clients = {}
    self.start_times = {}
    self.max_clients = max_clients
    self.max_connection_time = max_connection_time
    self.logger = get_logger("client_manager")

  def add_client(self, websocket, client):
    """
    Adds a client and their connection start time to the tracking dictionaries.

    Args:
        websocket: The websocket associated with the client to add.
        client: The client object to be added and tracked.
    """
    self.logger.debug(f"Adding client {client.client_uid} to client manager")
    self.clients[websocket] = client
    self.start_times[websocket] = time.time()
    self.logger.debug(f"Client {client.client_uid} added. Total clients: {len(self.clients)}")

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
      self.logger.debug(f"Removing client {client.client_uid} from client manager")
      client.cleanup()
    else:
      self.logger.debug("No client found for websocket during removal")
    self.start_times.pop(websocket, None)
    self.logger.debug(f"Client removed. Remaining clients: {len(self.clients)}")

  def get_wait_time(self):
    """
    Calculates the estimated wait time for new clients based on the remaining connection times of
    current clients.

    Returns 0 if there are available slots.
    """
    wait_time = None
    if not self.max_connection_time:
      return 0

    for start_time in self.start_times.values():
      current_client_time_remaining = self.max_connection_time - (time.time() - start_time)
      if wait_time is None or current_client_time_remaining < wait_time:
        wait_time = current_client_time_remaining
    return wait_time / 60 if wait_time is not None else 0

  async def is_server_full(self, websocket, options):
    """
    Checks if the server is at its maximum client capacity and sends a wait message to the client if
    necessary.

    Args:
        websocket: The websocket of the client attempting to connect.
        options: A dictionary of options that may include the client's unique identifier.

    Returns:
        True if the server is full, False otherwise.
    """
    if not self.max_clients:
      return False

    self.logger.debug(f"Checking server capacity: {len(self.clients)}/{self.max_clients} clients")
    if len(self.clients) >= self.max_clients:
      wait_time = self.get_wait_time()
      self.logger.debug(
        f"Server full, sending wait message to client {options['uid']}. Wait time: {wait_time:.1f} "
        "minutes"
      )
      response = {"uid": options["uid"], "status": "WAIT", "message": wait_time}
      await websocket.send(json.dumps(response))
      return True
    self.logger.debug(f"Server has capacity, allowing client {options['uid']} to connect")
    return False

  async def is_client_timeout(self, websocket):
    """
    Checks if a client has exceeded the maximum allowed connection time and disconnects them if so,
    issuing a warning.

    Args:
        websocket: The websocket associated with the client to check.

    Returns:
        True if the client's connection time has exceeded the maximum limit, False otherwise.
    """
    if not self.max_connection_time:
      return False

    elapsed_time = time.time() - self.start_times[websocket]
    if elapsed_time >= self.max_connection_time:
      await self.clients[websocket].disconnect()
      self.logger.warning(
        f"Client with uid '{self.clients[websocket].client_uid}' disconnected due to overtime."
      )
      return True
    return False


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
    self.connection_count = 0
    self.failed_connections = 0
    self.logger = get_logger("websocket_server")

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
    self.connection_count += 1
    client_addr = getattr(websocket, "remote_address", ("unknown", 0))
    conn_id = self.connection_count

    try:
      self.logger.info(f"Connection #{conn_id} from {client_addr[0]}:{client_addr[1]}")
      await self.handler(websocket)

    except (EOFError, InvalidMessage) as e:
      self.failed_connections += 1
      if "did not receive a valid HTTP request" in str(
        e
      ) or "connection closed while reading" in str(e):
        self.logger.debug(
          f"Connection #{conn_id} from {client_addr[0]} failed handshake "
          "(likely port scan/health check)"
        )
      else:
        self.logger.warning(f"Connection #{conn_id} from {client_addr[0]} handshake error: {e}")
      # Don't re-raise - this is expected for non-WebSocket clients

    except ConnectionClosed as e:
      self.logger.debug(f"Connection #{conn_id} from {client_addr[0]} closed: {e}")
    except (KeyboardInterrupt, SystemExit):
      raise
    except Exception:
      self.logger.exception(f"Connection #{conn_id} from {client_addr[0]} unexpected error")

  def __exit__(self, *args):
    self.logger.info(
      f"Server stats: {self.connection_count} total connections, {self.failed_connections} "
      "failed handshakes"
    )
