#!/usr/bin/env python3
"""
Async WebSocket microphone client for Eavesdrop transcription server.
Connects via websockets, records audio via sounddevice, uses typed wire protocol.
Use spacebar to toggle recording sessions.
"""

import argparse
import asyncio
import queue
import select
import sys
import termios
import threading
import time
import tty
import uuid

import numpy as np
import sounddevice as sd
import websockets
from websockets.asyncio.client import ClientConnection

from eavesdrop.wire import (
  ClientType,
  ErrorMessage,
  ServerReadyMessage,
  TranscriptionMessage,
  TranscriptionSetupMessage,
  UserTranscriptionOptions,
  WebSocketHeaders,
  deserialize_message,
  serialize_message,
)

# Audio configuration constants (WhisperLive compatible)
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.float32
BLOCKSIZE = 4096


class MicrophoneClient:
  def __init__(self, host: str = "home-brainbox", port: int = 9090):
    self.host = host
    self.port = port
    self.ws: ClientConnection | None = None
    self.session_active = False
    self.running = True
    self.stream_name = str(uuid.uuid4())

    # Audio streaming
    self.audio_queue: queue.Queue[bytes] = queue.Queue()
    self.audio_stream: sd.InputStream | None = None
    self.recording = False

    # Latency tracking
    self.first_audio_sent_time: float | None = None
    self.session_end_sent_time: float | None = None
    self.first_response_received = False
    self.session_end_received = False
    self.audio_sending_started = False

    # Terminal control
    self.old_settings: list | None = None

  def setup_terminal(self):
    """Setup terminal for raw input to detect spacebar presses."""
    try:
      self.old_settings = termios.tcgetattr(sys.stdin)
      tty.setraw(sys.stdin.fileno())
    except Exception:
      pass

  def restore_terminal(self):
    """Restore normal terminal settings."""
    try:
      if self.old_settings:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    except Exception:
      pass

  def safe_print(self, message: str):
    """Print with proper terminal handling."""
    self.restore_terminal()
    print(message)
    sys.stdout.flush()
    self.setup_terminal()

  async def on_message(self, message_json: str):
    """Handle WebSocket messages from server."""
    try:
      message = deserialize_message(message_json)

      match message:
        case ServerReadyMessage() as ready_msg:
          if ready_msg.stream == self.stream_name:
            self.safe_print(f"[INFO]: Server ready! Backend: {ready_msg.backend}")

        case TranscriptionMessage() as transcription:
          if transcription.stream == self.stream_name and transcription.segments:
            # Get the text from segments
            text_parts = [seg.text.strip() for seg in transcription.segments if seg.text.strip()]
            if text_parts:
              text = " ".join(text_parts)
              self.safe_print(f'Transcription: "{text}"')

        case ErrorMessage() as error:
          if error.stream == self.stream_name or error.stream is None:
            self.safe_print(f"[ERROR]: {error.message}")

        case _:
          # Handle unexpected message types
          self.safe_print(f"[DEBUG]: Received unexpected message: {type(message)}")

    except Exception as e:
      self.safe_print(f"Error processing message: {e}")

  def audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
    """Sounddevice audio callback."""
    if status:
      self.safe_print(f"[WARN] Audio status: {status}")

    if self.recording:
      # Convert to float32 and queue for transmission
      audio_data = indata.copy().astype(DTYPE)
      try:
        self.audio_queue.put_nowait(audio_data.tobytes())
      except queue.Full:
        pass  # Drop frame if queue is full

  def start_session(self):
    """Start a new recording session."""
    if self.session_active:
      return

    try:
      self.safe_print("\n[Starting Recording Session...]")
      self.session_active = True

      # Reset tracking
      self.first_response_received = False
      self.session_end_received = False
      self.audio_sending_started = False

      # Start session in thread
      session_thread = threading.Thread(target=self._run_session_sync)
      session_thread.daemon = True
      session_thread.start()

    except Exception as e:
      self.safe_print(f"Error starting session: {e}")
      self.session_active = False

  def _run_session_sync(self):
    """Synchronous wrapper for async session."""
    try:
      asyncio.run(self._run_session())
    except Exception as e:
      self.safe_print(f"Session error: {e}")
      self.session_active = False

  async def _run_session(self):
    """Run complete recording session."""
    try:
      # Create WebSocket connection
      self.safe_print("Connecting to Eavesdrop server...")
      socket_url = f"ws://{self.host}:{self.port}"

      headers = {WebSocketHeaders.CLIENT_TYPE.value: ClientType.TRANSCRIBER.value}

      async with websockets.connect(socket_url, additional_headers=headers) as websocket:
        self.ws = websocket
        self.safe_print("[INFO]: WebSocket opened, sending configuration...")

        # Send client configuration
        config_message = TranscriptionSetupMessage(
          stream=self.stream_name,
          options=UserTranscriptionOptions(
            initial_prompt=None,
            hotwords=None,
            beam_size=5,
            word_timestamps=False,
          ),
        )

        await websocket.send(serialize_message(config_message))
        self.safe_print("[INFO]: Configuration sent, waiting for server ready...")

        # Start message handling task
        message_task = asyncio.create_task(self._handle_messages(websocket))

        # Wait for connection to be established
        await asyncio.sleep(2)

        # Start audio capture
        self.safe_print("Starting audio capture...")
        self._start_audio_stream()

        # Start audio streaming task
        audio_task = asyncio.create_task(self._stream_audio(websocket))

        self.safe_print("[Recording...] (Press SPACEBAR to stop)")

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
      self.safe_print(f"Session error: {e}")
    finally:
      self.ws = None

  async def _handle_messages(self, websocket: ClientConnection):
    """Handle incoming WebSocket messages."""
    try:
      async for message in websocket:
        # Convert bytes to string if necessary
        message_str = message if isinstance(message, str) else message.decode("utf-8")
        await self.on_message(message_str)
    except Exception as e:
      self.safe_print(f"Message handling error: {e}")

  def stop_session(self):
    """Stop current recording session."""
    if not self.session_active:
      return

    self.safe_print("\n[Stopping Session...]")
    self.session_active = False
    self.recording = False
    self.session_end_sent_time = time.time()

    # Stop audio stream
    if self.audio_stream:
      try:
        self.audio_stream.stop()
        self.audio_stream.close()
        self.audio_stream = None
        self.safe_print("[DEBUG] Audio stream stopped")
      except Exception as e:
        self.safe_print(f"Error stopping audio: {e}")

    self.safe_print("\nPress SPACEBAR to start new recording session...")

  def _start_audio_stream(self):
    """Start sounddevice audio stream."""
    try:
      self.safe_print("[DEBUG] Starting audio stream...")
      self.audio_stream = sd.InputStream(
        device=None,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype=DTYPE,
        latency="low",
        blocksize=BLOCKSIZE,
        callback=self.audio_callback,
      )
      self.recording = True
      self.audio_stream.start()
      self.safe_print("[DEBUG] Audio stream started")

    except Exception as e:
      self.safe_print(f"Error starting audio: {e}")
      raise

  async def _stream_audio(self, websocket: ClientConnection):
    """Stream audio data to server."""
    try:
      while self.recording and self.session_active:
        try:
          # Get audio data
          audio_data = self.audio_queue.get(timeout=0.1)

          if not self.audio_sending_started:
            self.audio_sending_started = True
            self.first_audio_sent_time = time.time()
            self.safe_print("[DEBUG] Started sending audio data")

          # Send binary audio data
          await websocket.send(audio_data)

        except queue.Empty:
          continue
        except Exception as e:
          self.safe_print(f"Error streaming audio: {e}")
          break

      # Send END_OF_AUDIO signal
      if self.ws:
        try:
          self.safe_print("[DEBUG] Sending END_OF_AUDIO")
          await websocket.send(b"END_OF_AUDIO")
        except Exception as e:
          self.safe_print(f"Error sending END_OF_AUDIO: {e}")

    except Exception as e:
      self.safe_print(f"Audio streaming error: {e}")

  def keyboard_listener(self):
    """Listen for spacebar presses."""
    try:
      while self.running:
        if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
          char = sys.stdin.read(1)
          if char == " ":  # Spacebar
            if self.session_active:
              self.stop_session()
            else:
              self.start_session()
          elif char == "\x03":  # Ctrl+C
            self.running = False
            break
    except KeyboardInterrupt:
      self.running = False

  def run(self):
    """Main execution loop."""
    try:
      print("Eavesdrop Microphone Client")
      print(f"Server: {self.host}:{self.port}")
      print("\nPress SPACEBAR to start recording...")
      print("Press Ctrl+C to exit\n")

      # Setup terminal
      self.setup_terminal()

      # Start keyboard listener
      keyboard_thread = threading.Thread(target=self.keyboard_listener)
      keyboard_thread.daemon = True
      keyboard_thread.start()

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

    self.restore_terminal()


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
