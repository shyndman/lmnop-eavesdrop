#!/usr/bin/env python3
"""
Main microphone client application for Eavesdrop transcription.
"""

import argparse
import asyncio
import threading
import time
import uuid

from .audio import AudioCapture
from .connection import WebSocketConnection
from .interface import TerminalInterface


class MicrophoneClient:
  """Main microphone client that orchestrates audio, connection, and interface."""

  def __init__(self, host: str = "home-brainbox", port: int = 9090):
    self.host = host
    self.port = port
    self.stream_name = str(uuid.uuid4())
    self.session_active = False
    self.running = True

    # Initialize components
    self.audio = AudioCapture(on_error=self._on_error)
    self.connection = WebSocketConnection(
      host=host,
      port=port,
      stream_name=self.stream_name,
      on_ready=self._on_ready,
      on_transcription=self._on_transcription,
      on_error=self._on_error,
    )
    self.interface = TerminalInterface(
      on_start_session=self.start_session,
      on_stop_session=self.stop_session,
      on_exit=self._on_exit,
    )

  def _on_ready(self, backend: str):
    """Handle server ready message."""
    self.interface.safe_print(f"[INFO]: Server ready! Backend: {backend}")

  def _on_transcription(self, text: str):
    """Handle transcription result."""
    self.interface.safe_print(f'Transcription: "{text}"')

  def _on_error(self, message: str):
    """Handle error messages."""
    self.interface.safe_print(f"[ERROR]: {message}")

  def _on_exit(self):
    """Handle exit request."""
    self.running = False
    if self.session_active:
      self.stop_session()

  def start_session(self):
    """Start a new recording session."""
    if self.session_active:
      return

    try:
      self.interface.safe_print("\n[Starting Recording Session...]")
      self.session_active = True
      self.interface.set_session_active(True)

      # Reset connection tracking
      self.connection.reset_session_tracking()

      # Start session in thread
      session_thread = threading.Thread(target=self._run_session_sync)
      session_thread.daemon = True
      session_thread.start()

    except Exception as e:
      self.interface.safe_print(f"Error starting session: {e}")
      self.session_active = False
      self.interface.set_session_active(False)

  def _run_session_sync(self):
    """Synchronous wrapper for async session."""
    try:
      asyncio.run(self._run_session())
    except Exception as e:
      self.interface.safe_print(f"Session error: {e}")
      self.session_active = False
      self.interface.set_session_active(False)

  async def _run_session(self):
    """Run complete recording session."""
    try:
      # Create WebSocket connection
      self.interface.safe_print("Connecting to Eavesdrop server...")
      await self.connection.connect()
      self.interface.safe_print("[INFO]: WebSocket opened, sending configuration...")
      self.interface.safe_print("[INFO]: Configuration sent, waiting for server ready...")

      # Start message handling task
      message_task = asyncio.create_task(self.connection.handle_messages())

      # Wait for connection to be established
      await asyncio.sleep(2)

      # Start audio capture
      self.interface.safe_print("Starting audio capture...")
      self.audio.start_recording()

      # Start audio streaming task
      audio_task = asyncio.create_task(self._stream_audio())

      self.interface.safe_print("[Recording...] (Press SPACEBAR to stop)")

      # Keep session alive until stopped
      try:
        while self.session_active:
          await asyncio.sleep(0.1)
      finally:
        # Clean up tasks
        message_task.cancel()
        audio_task.cancel()
        try:
          await message_task
        except asyncio.CancelledError:
          pass
        try:
          await audio_task
        except asyncio.CancelledError:
          pass

    except Exception as e:
      self.interface.safe_print(f"Session error: {e}")
    finally:
      await self.connection.disconnect()

  async def _stream_audio(self):
    """Stream audio data to server."""
    try:
      while self.audio.is_recording() and self.session_active:
        audio_data = await self.audio.get_audio_data()
        if audio_data:
          await self.connection.send_audio_data(audio_data)

      # Send END_OF_AUDIO signal
      if self.connection.is_connected():
        self.interface.safe_print("[DEBUG] Sending END_OF_AUDIO")
        await self.connection.send_end_of_audio()

    except Exception as e:
      self.interface.safe_print(f"Audio streaming error: {e}")

  def stop_session(self):
    """Stop current recording session."""
    if not self.session_active:
      return

    self.interface.safe_print("\n[Stopping Session...]")
    self.session_active = False
    self.interface.set_session_active(False)

    # Stop audio recording
    self.audio.stop_recording()
    self.interface.safe_print("[DEBUG] Audio stream stopped")

    self.interface.safe_print("\nPress SPACEBAR to start new recording session...")

  def run(self):
    """Main execution loop."""
    try:
      self.interface.show_welcome_message(self.host, self.port)

      # Setup terminal
      self.interface.setup_terminal()

      # Start keyboard listener
      self.interface.start_keyboard_listener()

      # Main loop
      try:
        while self.running:
          time.sleep(0.1)
      except KeyboardInterrupt:
        self.running = False

    except Exception as e:
      print(f"Error: {e}")
    finally:
      self.cleanup()

  def cleanup(self):
    """Clean up resources."""
    self.running = False

    if self.session_active:
      self.stop_session()

    self.interface.cleanup()


def parse_host_port(host_port: str) -> tuple[str, int]:
  """Parse host:port string into separate components."""
  if ":" in host_port:
    host, port_str = host_port.rsplit(":", 1)
    try:
      port = int(port_str)
      return host, port
    except ValueError:
      raise ValueError(f"Invalid port number: {port_str}")
  else:
    # Just a host, use default port
    return host_port, 9090


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Async WebSocket microphone client for Eavesdrop transcription server"
  )
  parser.add_argument(
    "host_port",
    nargs="?",
    default="home-brainbox:9090",
    help="Server host:port (default: home-brainbox:9090)",
  )

  args = parser.parse_args()
  host, port = parse_host_port(args.host_port)

  client = MicrophoneClient(host, port)
  client.run()
