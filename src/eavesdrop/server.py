import json
import os
import socket
import time
from collections.abc import Callable
from enum import Enum

import numpy as np
import psutil
from websockets.exceptions import ConnectionClosed, InvalidMessage
from websockets.sync.server import ServerConnection, serve

from .backend import ServeClientFasterWhisper
from .base import ServeClientBase
from .logs import get_logger


class RobustWebSocketServer:
  """Wrapper around WebSocket server that handles connection errors gracefully"""

  def __init__(
    self,
    handler: Callable[[ServerConnection, str | None], None],
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

  def __enter__(self):
    def error_handling_wrapper(websocket: ServerConnection, path=None):
      """Wrapper that catches and logs connection errors without crashing"""
      self.connection_count += 1
      client_addr = getattr(websocket, "remote_address", ("unknown", 0))
      conn_id = self.connection_count

      try:
        self.logger.info(f"Connection #{conn_id} from {client_addr[0]}:{client_addr[1]}")
        return self.handler(websocket, path)

      except (EOFError, InvalidMessage) as e:
        self.failed_connections += 1
        if "did not receive a valid HTTP request" in str(
          e
        ) or "connection closed while reading" in str(e):
          self.logger.debug(
            f"Connection #{conn_id} from {client_addr[0]} failed handshake (likely port scan/health check)"
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

    self._server = serve(
      error_handling_wrapper,  # type: ignore
      self.host,
      self.port,
      **self.kwargs,
    )
    return self._server.__enter__()

  def __exit__(self, *args):
    self.logger.info(
      f"Server stats: {self.connection_count} total connections, {self.failed_connections} failed handshakes"
    )
    return self._server.__exit__(*args)


class ClientManager:
  def __init__(self, max_clients=4, max_connection_time=600):
    """
    Initializes the ClientManager with specified limits on client connections and connection durations.

    Args:
        max_clients (int, optional): The maximum number of simultaneous client connections allowed. Defaults to 4.
        max_connection_time (int, optional): The maximum duration (in seconds) a client can stay connected. Defaults
                                             to 600 seconds (10 minutes).
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
    Removes a client and their connection start time from the tracking dictionaries. Performs cleanup on the
    client if necessary.

    Args:
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
    Calculates the estimated wait time for new clients based on the remaining connection times of current clients.

    Returns:
        The estimated wait time in minutes for new clients to connect. Returns 0 if there are available slots.
    """
    wait_time = None
    for start_time in self.start_times.values():
      current_client_time_remaining = self.max_connection_time - (time.time() - start_time)
      if wait_time is None or current_client_time_remaining < wait_time:
        wait_time = current_client_time_remaining
    return wait_time / 60 if wait_time is not None else 0

  def is_server_full(self, websocket, options):
    """
    Checks if the server is at its maximum client capacity and sends a wait message to the client if necessary.

    Args:
        websocket: The websocket of the client attempting to connect.
        options: A dictionary of options that may include the client's unique identifier.

    Returns:
        True if the server is full, False otherwise.
    """
    self.logger.debug(f"Checking server capacity: {len(self.clients)}/{self.max_clients} clients")
    if len(self.clients) >= self.max_clients:
      wait_time = self.get_wait_time()
      self.logger.debug(
        f"Server full, sending wait message to client {options['uid']}. Wait time: {wait_time:.1f} minutes"
      )
      response = {"uid": options["uid"], "status": "WAIT", "message": wait_time}
      websocket.send(json.dumps(response))
      return True
    self.logger.debug(f"Server has capacity, allowing client {options['uid']} to connect")
    return False

  def is_client_timeout(self, websocket):
    """
    Checks if a client has exceeded the maximum allowed connection time and disconnects them if so, issuing a warning.

    Args:
        websocket: The websocket associated with the client to check.

    Returns:
        True if the client's connection time has exceeded the maximum limit, False otherwise.
    """
    elapsed_time = time.time() - self.start_times[websocket]
    if elapsed_time >= self.max_connection_time:
      self.clients[websocket].disconnect()
      self.logger.warning(
        f"Client with uid '{self.clients[websocket].client_uid}' disconnected due to overtime."
      )
      return True
    return False


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

  def initialize_client(
    self,
    websocket,
    options,
    faster_whisper_custom_model_path,
  ):
    self.logger.debug(
      f"initialize_client: Starting initialization for client {options['uid']} with backend: {self.backend.value}"
    )
    client: ServeClientBase | None = None

    try:
      if self.backend.is_faster_whisper():
        self.logger.debug("initialize_client: Initializing faster_whisper backend")
        self.logger.debug("initialize_client: Importing ServeClientFasterWhisper")

        # model is of the form namespace/repo_name and not a filesystem path
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
        )

        self.logger.info("initialize_client: Running faster_whisper backend.")
        self.logger.debug("initialize_client: faster_whisper client created successfully")
    except Exception:
      self.logger.exception("initialize_client: Error creating faster_whisper client")
      return

    if client is None:
      self.logger.error(f"initialize_client: Client is None for backend {self.backend.value}")
      raise ValueError(f"Backend type {self.backend.value} not recognised or not handled.")

    self.logger.debug("initialize_client: Client created successfully")

    self.logger.debug("initialize_client: Adding client to client manager")
    self.client_manager.add_client(websocket, client)
    self.logger.debug("initialize_client: Client initialization completed")

  def get_audio_from_websocket(self, websocket):
    """
    Receives audio buffer from websocket and creates a numpy array out of it.

    Args:
        websocket: The websocket to receive audio from.

    Returns:
        A numpy array containing the audio.
    """
    self.logger.debug("get_audio_from_websocket: About to receive data")
    frame_data = websocket.recv()
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

    Args:
        websocket: The websocket connection
        audio_array: numpy array of audio samples
    """
    import os
    import time

    import soundfile as sf

    # Get or create debug file for this websocket
    if websocket not in self.debug_audio_files:
      client = self.client_manager.get_client(websocket) if self.client_manager else None
      client_uid = client.client_uid if client else "unknown"
      timestamp = int(time.time())

      # Create filename: {debug_audio_path}_{client_uid}_{timestamp}.wav
      filename = f"{self.debug_audio_path}_{client_uid}_{timestamp}.wav"

      # Create directory if it doesn't exist
      os.makedirs(
        os.path.dirname(filename) if os.path.dirname(filename) else ".",
        exist_ok=True,
      )

      # Open file for writing
      file_handle = sf.SoundFile(filename, mode="w", samplerate=self.RATE, channels=1, format="WAV")
      self.debug_audio_files[websocket] = (file_handle, filename)

      self.logger.info(f"Debug audio capture started: {filename}")

    # Write audio data to file
    file_handle, filename = self.debug_audio_files[websocket]
    try:
      file_handle.write(audio_array)
      self.logger.debug(f"Debug audio: wrote {len(audio_array)} samples to {filename}")
    except Exception:
      self.logger.exception(f"Error writing debug audio to {filename}")

  def handle_new_connection(
    self,
    websocket,
    faster_whisper_custom_model_path,
  ):
    try:
      self.logger.info("handle_new_connection: New client connected")
      self.logger.debug("handle_new_connection: Waiting for client options...")

      self.logger.debug("handle_new_connection: About to receive options from websocket")
      options = websocket.recv()
      self.logger.debug(f"handle_new_connection: Raw options received: {options}")

      self.logger.debug("handle_new_connection: Parsing JSON options")
      options = json.loads(options)
      self.logger.debug(f"handle_new_connection: Parsed client options: {options}")

      if options.get("type") == "health_check":
        self.logger.info("handle_new_connection: Health check successful")
        websocket.close()
        return False

      self.use_vad = options.get("use_vad")
      self.logger.debug(f"handle_new_connection: VAD enabled: {self.use_vad}")

      self.logger.debug("handle_new_connection: Checking if server is full")
      if self.client_manager.is_server_full(websocket, options):
        self.logger.debug(
          f"handle_new_connection: Closing connection for client {options['uid']} - server full"
        )
        websocket.close()
        return False  # Indicates that the connection should not continue

      self.logger.debug("handle_new_connection: About to initialize client")
      self.initialize_client(
        websocket,
        options,
        faster_whisper_custom_model_path,
      )
      self.logger.debug(f"handle_new_connection: Client {options['uid']} initialized successfully")
      return True
    except json.JSONDecodeError:
      self.logger.exception("handle_new_connection: Failed to decode JSON from client")
      return False
    except ConnectionClosed:
      self.logger.info("handle_new_connection: Connection closed by client")
      return False
    except (KeyboardInterrupt, SystemExit):
      raise
    except Exception:
      self.logger.exception("handle_new_connection: Error during new connection initialization")
      return False

  def process_audio_frames(self, websocket):
    self.logger.debug("process_audio_frames: Starting")

    self.logger.debug("process_audio_frames: Getting audio from websocket")
    frame_np = self.get_audio_from_websocket(websocket)
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

  def recv_audio(
    self,
    websocket,
    backend: BackendType = BackendType.FASTER_WHISPER,
    faster_whisper_custom_model_path=None,
  ):
    """
    Receive audio chunks from a client in an infinite loop.

    Continuously receives audio frames from a connected client
    over a WebSocket connection. It processes the audio frames using a
    voice activity detection (VAD) model to determine if they contain speech
    or not. If the audio frame contains speech, it is added to the client's
    audio data for ASR.
    If the maximum number of clients is reached, the method sends a
    "WAIT" status to the client, indicating that they should wait
    until a slot is available.
    If a client's connection exceeds the maximum allowed time, it will
    be disconnected, and the client's resources will be cleaned up.

    Args:
        websocket (WebSocket): The WebSocket connection for the client.
        backend (str): The backend to run the server with.
        faster_whisper_custom_model_path (str): path to custom faster whisper model.

    Raises:
        Exception: If there is an error during the audio frame processing.
    """
    self.logger.debug(f"recv_audio: Starting for backend {backend.value}")
    self.backend = backend

    self.logger.debug("recv_audio: About to handle new connection")
    if not self.handle_new_connection(
      websocket,
      faster_whisper_custom_model_path,
    ):
      self.logger.debug("recv_audio: handle_new_connection returned False, exiting")
      return

    self.logger.debug("recv_audio: Entering main audio processing loop")
    assert self.client_manager is not None
    try:
      loop_count = 0
      while not self.client_manager.is_client_timeout(websocket):
        loop_count += 1
        if loop_count % 100 == 0:  # Log every 100 iterations to avoid spam
          self.logger.debug(f"recv_audio: Main loop iteration {loop_count}")

        self.logger.debug("recv_audio: About to process audio frames")
        if not self.process_audio_frames(websocket):
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
        websocket.close()
      self.logger.debug("recv_audio: Deleting websocket reference")
      del websocket

  def run(
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
  ):
    """
    Run the transcription server.

    Args:
        host (str): The host address to bind the server.
        port (int): The port number to bind the server.
    """
    self.cache_path = cache_path
    self.debug_audio_path = debug_audio_path
    self.debug_audio_files = {}  # websocket -> (file_handle, filename)

    # Log server configuration at startup
    import platform

    self.logger.info("=" * 50)
    self.logger.info("Eavesdrop Server Configuration:")
    self.logger.info("=" * 50)
    self.logger.info(f"Host: {host}")
    self.logger.info(f"Port: {port}")
    self.logger.info(f"Backend: {backend}")
    self.logger.info(f"Single Model Mode: {single_model}")
    self.logger.info(f"Max Clients: {max_clients}")
    self.logger.info(f"Max Connection Time: {max_connection_time}s")
    self.logger.info(f"Cache Path: {cache_path}")
    if debug_audio_path:
      self.logger.info(f"Debug Audio Path: {debug_audio_path}")
    if faster_whisper_custom_model_path:
      self.logger.info(f"Custom Faster Whisper Model: {faster_whisper_custom_model_path}")

    # Network diagnostics
    self.logger.info("=" * 50)
    self.logger.info("Network Diagnostics:")
    self.logger.info(f"Platform: {platform.system()} {platform.release()}")
    self.logger.info(f"Python version: {platform.python_version()}")

    # Check network interfaces
    try:
      import netifaces
      interfaces = netifaces.interfaces()
      self.logger.info(f"Available network interfaces: {interfaces}")
    except ImportError:
      self.logger.debug("netifaces not available for interface checking")

    # Check if host resolves

    try:
      if host == "0.0.0.0":
        self.logger.info("Host 0.0.0.0 - will bind to all interfaces")
      else:
        resolved_ip = socket.gethostbyname(host)
        self.logger.info(f"Host {host} resolves to: {resolved_ip}")
    except socket.gaierror:
      self.logger.exception(f"Host resolution failed for {host}")

    # Check for processes using the port
    try:
      connections = psutil.net_connections()
      port_users = [conn for conn in connections if conn.laddr.port == port]
      if port_users:
        self.logger.warning(f"Port {port} is already in use by: {port_users}")
      else:
        self.logger.info(f"Port {port} appears to be free")
    except Exception:
      self.logger.exception("Could not check port usage")

    # Check network connectivity
    if host == "0.0.0.0":
      self._bind_to_all_interfaces(port)
    else:
      self.logger.info(f"Server will bind specifically to {host}:{port}")

    self.logger.info("=" * 50)

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
        # TODO: load model initially
      else:
        self.logger.info("Single model mode currently only works with custom models.")
    if not BackendType.is_valid(backend):
      raise ValueError(
        f"{backend} is not a valid backend type. Choose backend from {BackendType.valid_types()}"
      )

    self.logger.debug(f"run: Testing port {port} availability on {host}")

    # Test if port is available before starting server
    try:
      test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      test_socket.bind((host, port))
      test_socket.close()
      self.logger.debug(f"run: Port {port} is available")
    except socket.error:
      self.logger.exception(f"run: Port {port} is not available")
      raise

    self.logger.debug("run: Creating WebSocket server handler")

    def connection_wrapper(websocket, path=None):
      """Wrapper to log connection attempts and handle WebSocket errors"""
      client_address = getattr(websocket, "remote_address", "unknown")
      self.logger.info(f"run: New WebSocket connection established from {client_address}")
      self.logger.debug(f"run: Connection path: {path}")
      if path == "/health":
        self.logger.info("run: Health check request received")
        # Send a simple HTTP 200 OK response
        websocket.send(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
        websocket.close()
        return

      self.logger.debug(f"run: WebSocket state: {getattr(websocket, 'state', 'unknown')}")

      try:
        # Log WebSocket headers for debugging
        if hasattr(websocket, "request_headers"):
          headers = dict(websocket.request_headers)
          self.logger.debug(f"run: WebSocket headers: {headers}")

          # Check for common client types
          user_agent = headers.get("user-agent", "").lower()
          if "curl" in user_agent:
            self.logger.info(f"run: Client appears to be curl: {user_agent}")
          elif "python" in user_agent:
            self.logger.info(f"run: Client appears to be Python client: {user_agent}")
          elif "browser" in user_agent or "mozilla" in user_agent:
            self.logger.info(f"run: Client appears to be browser: {user_agent}")
          else:
            self.logger.info(f"run: Unknown client type: {user_agent}")

        if hasattr(websocket, "subprotocol"):
          self.logger.debug(f"run: WebSocket subprotocol: {websocket.subprotocol}")

        # Validate WebSocket is properly established
        if hasattr(websocket, "state"):
          if websocket.state != 1:  # 1 = OPEN state
            self.logger.warning(f"run: WebSocket not in OPEN state: {websocket.state}")
            return

        self.logger.debug(f"run: Starting recv_audio for client {client_address}")
        result = self.recv_audio(
          websocket,
          backend=BackendType(backend),
          faster_whisper_custom_model_path=faster_whisper_custom_model_path,
        )
        self.logger.debug(f"run: recv_audio completed for client {client_address}")
        return result

      except EOFError as e:
        self.logger.warning(f"run: Client {client_address} disconnected during handshake: {e}")
        self.logger.debug(
          "run: This usually means the client sent incomplete HTTP/WebSocket headers"
        )
        return
      except ConnectionClosed as e:
        self.logger.info(f"run: Client {client_address} closed connection: {e}")
        return
      except Exception:
        self.logger.exception(f"run: Connection from {client_address} failed")
        # Don't re-raise - let the server continue handling other connections
        return

    handler = connection_wrapper
    self.logger.debug("run: Handler created successfully")

    self.logger.debug(f"run: About to create WebSocket server on {host}:{port}")

    # Configure WebSocket server options for better error handling
    server_kwargs = {
      "ping_interval": 20,  # Send ping every 20 seconds
      "ping_timeout": 10,  # Wait 10 seconds for pong
      "close_timeout": 10,  # Wait 10 seconds for close handshake
      "max_size": 2**20,  # 1MB max message size
      "max_queue": 32,  # Queue up to 32 messages
    }
    self.logger.debug(f"run: WebSocket server config: {server_kwargs}")

    try:
      with RobustWebSocketServer(handler, host, port, **server_kwargs) as server:
        self.logger.info(f"run: WebSocket server successfully bound to {host}:{port}")
        self.logger.debug(f"run: Server object: {server}")
        self.logger.debug(f"run: Server socket: {server.socket}")
        self.logger.debug(f"run: Server listening on: {server.socket.getsockname()}")

        # Test server socket is actually listening
        if hasattr(server.socket, "getsockname"):
          actual_address = server.socket.getsockname()
          self.logger.info(f"run: Server actually listening on {actual_address}")

        self.logger.debug("run: About to call server.serve_forever()")

        # Final verification that server is ready
        import threading
        import time

        def connection_test():
          """Test if server is actually accepting connections"""
          time.sleep(1)  # Give server time to start
          try:
            import websocket

            ws = websocket.create_connection(
              f"ws://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}"
            )
            ws.send(json.dumps({"type": "health_check"}))
            ws.close()
            self.logger.info(
              f"run: Server connection test PASSED - port {port} is accepting connections"
            )
          except Exception:
            self.logger.exception("run: Server connection test ERROR")

        def periodic_stats():
          """Log server statistics periodically"""
          while True:
            time.sleep(30)  # Log stats every 30 seconds
            if hasattr(server, "_server") and hasattr(server._server, "connection_count"):
              total = server._server.connection_count
              failed = server._server.failed_connections
              successful = total - failed
              self.logger.info(
                f"Server stats: {total} connections ({successful} successful, {failed} failed handshakes)"
              )
            else:
              self.logger.debug("Server stats not available")

        # Start connection test and stats monitoring in background
        test_thread = threading.Thread(target=connection_test, daemon=True)
        test_thread.start()

        stats_thread = threading.Thread(target=periodic_stats, daemon=True)
        stats_thread.start()

        self.logger.info(
          f"run: Starting WebSocket server on {host}:{port} - ready for connections!"
        )
        self.logger.info("run: Server configuration complete, entering serve_forever() loop")

        # Enhanced connection monitoring
        original_serve_forever = server.serve_forever
        connection_count = [0]  # Use list for closure modification

        def monitored_serve_forever():
          self.logger.info("run: WebSocket server now accepting connections")
          try:
            return original_serve_forever()
          except KeyboardInterrupt:
            self.logger.info("run: Server shutdown requested via keyboard interrupt")
            raise
          except Exception:
            self.logger.exception("run: Server serve_forever() failed")
            raise

        server.serve_forever = monitored_serve_forever
        server.serve_forever()
    except Exception:
      self.logger.exception("run: Failed to create or start WebSocket server")
      raise

  def _bind_to_all_interfaces(self, port):
    self.logger.info("Server will bind to all network interfaces")
    try:
      import netifaces

      for interface in netifaces.interfaces():
        addrs = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in addrs:
          for addr in addrs[netifaces.AF_INET]:
            if addr.get("addr"):
              self.logger.info(f"Available on interface {interface}: {addr['addr']}:{port}")
    except ImportError:
      self.logger.debug("netifaces not available - cannot list network interfaces")
    except Exception:
      self.logger.exception("Could not enumerate network interfaces")

  def cleanup(self, websocket):
    """
    Cleans up resources associated with a given client's websocket.

    Args:
        websocket: The websocket associated with the client to be cleaned up.
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
      file_handle, filename = self.debug_audio_files[websocket]
      try:
        file_handle.close()
        self.logger.info(f"Debug audio capture finished: {filename}")
      except Exception:
        self.logger.exception(f"Error closing debug audio file {filename}")
      finally:
        del self.debug_audio_files[websocket]
    else:
      self.logger.debug("cleanup: No client found for websocket during cleanup")
