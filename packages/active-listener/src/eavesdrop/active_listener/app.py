"""Core application controller for eavesdrop active listener.

Manages the application lifecycle, component coordination, and UI integration
for the voice-driven text workspace system.
"""

import asyncio
import signal
from pathlib import Path

from eavesdrop.active_listener.client import EavesdropClientWrapper
from eavesdrop.active_listener.typist import YdoToolTypist
from eavesdrop.active_listener.ui_channel import UIChannel
from eavesdrop.active_listener.workspace import TextTranscriptionWorkspace
from eavesdrop.common import get_logger
from eavesdrop.wire import TranscriptionMessage


class App:
  """Application controller for voice-driven text workspace.

  Manages the complete application lifecycle including component creation,
  UI subprocess management, signal handling, and transcription processing.

  :param client: Eavesdrop server client wrapper
  :type client: EavesdropClientWrapper
  :param ui_channel: UI subprocess communication channel
  :type ui_channel: UIChannel
  :param workspace: Text transcription workspace manager
  :type workspace: TextTranscriptionWorkspace
  :param typist: Desktop typing automation component
  :type typist: YdoToolTypist
  """

  def __init__(
    self,
    client: EavesdropClientWrapper,
    ui_channel: UIChannel,
    workspace: TextTranscriptionWorkspace,
    typist: YdoToolTypist,
  ) -> None:
    """Initialize application with pre-configured components.

    :param client: Connected eavesdrop client for server communication
    :type client: EavesdropClientWrapper
    :param ui_channel: Active UI communication channel
    :type ui_channel: UIChannel
    :param workspace: Configured text workspace manager
    :type workspace: TextTranscriptionWorkspace
    :param typist: Desktop typing automation component
    :type typist: YdoToolTypist
    """
    self._client = client
    self._ui_channel = ui_channel
    self._workspace = workspace
    self._typist = typist
    self._shutdown_event = asyncio.Event()
    self.logger = get_logger("app")

  @classmethod
  def create(
    cls, server_host: str, server_port: int, audio_device: str, ui_bin_path: Path
  ) -> "App":
    """Create and configure complete application with all components.

    Factory method that creates all necessary components and wires them together
    for a fully functional application instance.

    :param server_host: Eavesdrop server hostname
    :type server_host: str
    :param server_port: Eavesdrop server port number
    :type server_port: int
    :param audio_device: Audio input device name or identifier
    :type audio_device: str
    :param ui_bin_path: Path to UI executable binary
    :type ui_bin_path: Path
    :return: Fully configured application instance
    :rtype: App
    :raises RuntimeError: If component creation or wiring fails
    """
    try:
      # Create eavesdrop client wrapper for server communication
      client = EavesdropClientWrapper(host=server_host, port=server_port, audio_device=audio_device)

      # Create UI communication channel with current working directory
      ui_channel = UIChannel(ui_bin_path=ui_bin_path, working_dir=Path.cwd())

      # Create text transcription workspace with UI channel dependency
      workspace = TextTranscriptionWorkspace(ui_channel=ui_channel)

      # Create desktop typing component
      typist = YdoToolTypist()

      # Wire all components together in the App instance
      return cls(client=client, ui_channel=ui_channel, workspace=workspace, typist=typist)

    except Exception as e:
      raise RuntimeError(f"Failed to create application components: {e}")

  async def run(self) -> None:
    """Main application entry point with complete lifecycle management.

    Initializes all components, sets up signal handlers, starts the UI subprocess,
    and runs the main transcription processing loop until shutdown.

    :raises RuntimeError: If any critical component fails during startup or operation
    """
    self.logger.info("Starting Active Listener application")

    try:
      # Initialize all components in sequence
      await asyncio.gather(
        self._client.initialize(),
        self._ui_channel.start(),
        self._setup_signal_handlers(),
      )

      # Start the main transcription processing loop
      await self._client.start_streaming()
      self.logger.info("Audio streaming started, entering main loop")

      # Main message processing loop
      async for message in self._client:
        if self._shutdown_event.is_set():
          self.logger.info("Shutdown event received, breaking main loop")
          break

        await self._handle_transcription_message(message)

    except Exception as e:
      self.logger.exception("Critical error during application startup or operation")
      # Ensure cleanup happens even on startup failures
      await self._cleanup()
      raise RuntimeError(f"Application failed: {e}")

    finally:
      # Always perform cleanup
      await self._cleanup()

  def shutdown(self) -> None:
    """Signal the application to shutdown gracefully.

    Sets shutdown event to stop the main processing loop and initiate cleanup.
    """
    self.logger.info("Shutdown requested")
    self._shutdown_event.set()

  async def _setup_signal_handlers(self) -> None:
    """Setup signal handlers for graceful shutdown.

    Configures SIGINT and SIGTERM handlers to trigger application shutdown
    when the process receives termination signals.
    """

    def signal_handler(signum, _frame):
      self.logger.info("Received shutdown signal", signal=signum)
      self.shutdown()
      # Create a task to force client shutdown asynchronously
      asyncio.create_task(self._force_shutdown())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    self.logger.info("Signal handlers configured")

  async def _force_shutdown(self) -> None:
    """Force shutdown of client to break blocking operations."""
    try:
      # Give the app a moment to shutdown gracefully
      await asyncio.sleep(0.1)
      # If still running, force client disconnect to break async iteration
      await self._client.shutdown()
    except Exception:
      self.logger.exception("Error during forced shutdown")

  async def _cleanup(self) -> None:
    """Perform comprehensive cleanup of all components.

    Shuts down UI subprocess, client connection, and other resources in
    proper order to ensure clean application termination.

    :raises RuntimeError: If cleanup operations fail
    """
    self.logger.info("Performing application cleanup")

    try:
      # Shutdown client connection
      await self._client.shutdown()
      self.logger.info("Client shutdown complete")

      # Shutdown UI subprocess first
      await self._ui_channel.shutdown()
      self.logger.info("UI channel shutdown complete")

      self.logger.info("Application cleanup complete")

    except Exception as e:
      self.logger.exception("Error during cleanup")
      raise RuntimeError(f"Failed to cleanup application: {e}")

  async def _handle_transcription_message(self, message: TranscriptionMessage) -> None:
    """Handle incoming transcription messages from the server.

    Delegates message processing to the text workspace and logs activity
    for debugging and monitoring purposes.

    :param message: Transcription message from eavesdrop server
    :type message: TranscriptionMessage
    :raises RuntimeError: If message processing fails
    """
    try:
      # Delegate message processing to the text workspace
      self._workspace.process_transcription_message(message)

    except Exception as e:
      self.logger.exception("Error handling transcription message")
      raise RuntimeError(f"Failed to handle transcription message: {e}")
