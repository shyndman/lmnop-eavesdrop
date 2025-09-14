"""CLI interface for eavesdrop active listener.

Provides command-line interface using Clypi framework with server argument parsing,
async execution, and graceful shutdown handling.
"""

import asyncio
import signal
from typing import NamedTuple, TypedDict

import sounddevice as sd
import structlog
from clypi import Command, arg

from eavesdrop.active_listener.client import EavesdropClientWrapper
from eavesdrop.active_listener.text_manager import TextState, TypingOperation
from eavesdrop.active_listener.typer import DesktopTyper


class AudioDevice(TypedDict):
  """Type definition for sounddevice query_devices() return value."""

  name: str
  hostapi: int
  max_input_channels: int
  max_output_channels: int
  default_low_input_latency: float
  default_low_output_latency: float
  default_high_input_latency: float
  default_high_output_latency: float
  default_samplerate: float


class ServerHostPort(NamedTuple):
  """Named tuple for server host and port."""

  host: str
  port: int


def parse_server(value: str | list[str]) -> ServerHostPort:
  assert isinstance(value, str), "Server must be a string in hostname:port format"

  """Parse server argument in hostname:port format."""
  if not isinstance(value, str):
    raise ValueError("Server must be a string")

  if not value:
    raise ValueError("Invalid server format: cannot be empty")

  if ":" not in value:
    raise ValueError("Invalid server format: must be hostname:port")

  host, port_str = value.rsplit(":", 1)

  if not host:
    raise ValueError("Invalid server format: missing hostname")

  if not port_str:
    raise ValueError("Invalid server format: missing port after colon")

  try:
    port = int(port_str)
  except ValueError:
    raise ValueError("Invalid port: must be numeric")

  if port < 1 or port > 65535:
    raise ValueError("Invalid port: must be between 1 and 65535")

  return ServerHostPort(host=host, port=port)


class ListDevices(Command):
  """List available audio devices.

  Shows all available audio input and output devices with their IDs,
  names, channel counts, and host API information.
  """

  async def run(self) -> None:
    """List all available audio input devices."""
    try:
      all_devices = sd.query_devices()
      print("Audio Input Devices (for use with --audio-device):")
      print("=" * 55)

      # Get default device info
      default_input = sd.default.device[0] if isinstance(sd.default.device, tuple) else None

      input_devices = []
      for i, device in enumerate(all_devices):
        device_typed: AudioDevice = device  # Type annotation for the device
        if device_typed["max_input_channels"] > 0:
          input_devices.append((i, device_typed))

      if not input_devices:
        print("No input devices found.")
        return

      for device_id, device in input_devices:
        # Check if this is the default input device
        is_default = device_id == default_input

        # Show device name prominently for easy copying
        device_name = device["name"]
        default_marker = " [DEFAULT INPUT]" if is_default else ""

        print(f"Device: {device_name}{default_marker}")
        print(f"  ID: {device_id}")
        print(f"  Input channels: {device['max_input_channels']}")
        print(f"  Low input latency: {device['default_low_input_latency']:.4f}s")
        print(f"  High input latency: {device['default_high_input_latency']:.4f}s")
        print()

    except Exception as e:
      print(f"Error querying audio devices: {e}")
      raise


class ActiveListener(Command):
  """Eavesdrop active listener - real-time voice-to-text dictation.

  Connects to an eavesdrop transcription server, captures audio from the
  specified input device, and automatically types transcription results
  into the currently focused desktop application.
  """

  subcommand: ListDevices | None

  server: ServerHostPort = arg(default=ServerHostPort("localhost", 9090), parser=parse_server)
  audio_device: str = arg(default="default")

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.logger = structlog.get_logger("🤸👂")
    self._client: EavesdropClientWrapper | None = None
    self._typer = DesktopTyper()
    self._text_state = TextState()
    self._shutdown_event = asyncio.Event()

  async def run(self) -> None:
    """Main entry point for the command."""
    await self._initialize_components()
    await self._setup_signal_handlers()
    await self._start_transcription_loop()

  async def _initialize_components(self) -> None:
    """Initialize all application components."""
    self.logger.info("Initializing components", server=self.server, audio_device=self.audio_device)

    # Initialize desktop typer
    self._typer.initialize()

    if not self._typer.is_available():
      raise Exception("ydotool is not available - check permissions and installation")

    # Initialize eavesdrop client
    self._client = EavesdropClientWrapper(
      host=self.server.host, port=self.server.port, audio_device=self.audio_device
    )
    await self._client.initialize()

    # Assert client is initialized for type checker
    assert self._client is not None

  async def _setup_signal_handlers(self) -> None:
    """Setup graceful shutdown signal handlers."""

    def signal_handler(signum, _frame):
      self.logger.info("Received shutdown signal", signal=signum)
      self._shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

  async def _start_transcription_loop(self) -> None:
    """Start the main transcription processing loop."""
    self.logger.info("Starting transcription loop")
    assert self._client is not None  # Should be initialized before this is called

    # Start audio streaming
    await self._client.start_streaming()

    # Main message processing loop
    try:
      async for message in self._client:
        if self._shutdown_event.is_set():
          break

        await self._handle_transcription_message(message)

    except Exception:
      self.logger.exception("Error in transcription loop")
      raise

  async def _handle_transcription_message(self, message) -> None:
    """Handle incoming transcription messages from the server."""
    try:
      # Process segments in the message
      for segment in message.segments:
        if segment.completed:
          # Handle completed segment
          if (
            self._text_state.current_segment_id == segment.id
            and self._text_state.current_in_progress_text
          ):
            self._text_state.apply_segment_completion(segment.text)
        else:
          # Handle in-progress segment
          text_update = self._text_state.calculate_update(segment)
          await self._execute_text_update(text_update)

    except Exception:
      self.logger.exception("Error handling transcription message")
      raise

  async def _execute_text_update(self, text_update) -> None:
    """Execute a text update by creating and running a typing operation."""
    operation = TypingOperation(
      operation_id=f"op-{asyncio.get_event_loop().time()}",
      chars_to_delete=text_update.chars_to_delete,
      text_to_type=text_update.text_to_type,
      timestamp=asyncio.get_event_loop().time(),
      completed=False,
    )

    # Execute with retry
    success = self._typer.execute_with_retry(operation)
    if not success:
      self.logger.error("Failed to execute typing operation", operation_id=operation.operation_id)

  async def _handle_shutdown(self) -> None:
    """Handle graceful shutdown sequence."""
    self.logger.info("Starting graceful shutdown")

    if self._client:
      await self._client.shutdown()

    self.logger.info("Shutdown complete")

  async def _monitor_connection_health(self) -> None:
    """Monitor connection health and attempt recovery if needed."""
    if not self._client:
      return

    if not self._client.check_connection_health():
      await self._client.attempt_reconnection()
