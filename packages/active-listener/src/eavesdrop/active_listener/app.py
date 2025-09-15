"""Core application logic for eavesdrop active listener.

Handles the main transcription processing loop, coordinating between
client, text state management, and desktop typing operations.
"""

import asyncio

from eavesdrop.active_listener.client import EavesdropClientWrapper
from eavesdrop.active_listener.text_manager import TextState, TextUpdate, TypingOperation
from eavesdrop.active_listener.typist import YdoToolTypist
from eavesdrop.common import get_logger
from eavesdrop.wire import TranscriptionMessage


class App:
  """Core application that manages the transcription processing loop."""

  def __init__(self, client: EavesdropClientWrapper, typist: YdoToolTypist):
    self._client = client
    self._typist = typist
    self._text_state = TextState()
    self._shutdown_event = asyncio.Event()
    self.logger = get_logger("app")

  async def start(self) -> None:
    """Start the main transcription processing loop."""
    self.logger.info("Starting transcription loop")

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

  def shutdown(self) -> None:
    """Signal the application to shutdown gracefully."""
    self.logger.info("Shutdown requested")
    self._shutdown_event.set()

  async def _handle_transcription_message(self, message: TranscriptionMessage) -> None:
    """Handle incoming transcription messages from the server."""
    try:
      # self.logger.debug("Received transcription message", segments=message.segments)
      # Process segments in the message
      for segment in message.segments:
        self.logger.debug("Processing segment", segment=segment)
        if update := self._text_state.process_segment(segment):
          # self.logger.debug("Text update generated", update=update)
          await self._execute_text_update(update)

    except Exception:
      self.logger.exception("Error handling transcription message")
      raise

  async def _execute_text_update(self, text_update: TextUpdate) -> None:
    """Execute a text update by creating and running a typing operation."""
    operation = TypingOperation(
      operation_id=f"op-{asyncio.get_event_loop().time()}",
      chars_to_delete=text_update.chars_to_delete,
      text_to_type=text_update.text_to_type,
      timestamp=asyncio.get_event_loop().time(),
      completed=False,
    )

    # Execute with retry
    success = self._typist.execute_with_retry(operation)
    if not success:
      self.logger.error("Failed to execute typing operation", operation_id=operation.operation_id)
