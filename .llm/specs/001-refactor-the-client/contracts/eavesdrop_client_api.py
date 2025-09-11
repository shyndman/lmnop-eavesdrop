"""
API Contract Definition for EavesdropClient Library

This file defines the complete public API surface that the refactored client library must implement.
Used for contract testing to ensure API compliance.
"""

from collections.abc import AsyncIterator, Callable
from eavesdrop.wire import TranscriptionMessage


class EavesdropClient:
    """Unified client for eavesdrop transcription services with mode-specific factory methods."""

    # Factory Methods
    @classmethod
    def transcriber(
        cls,
        host: str = "localhost",
        port: int = 9090,
        audio_device: int | str | None = None,
        beam_size: int = 5,
        word_timestamps: bool = False,
        initial_prompt: str | None = None,
        hotwords: list[str] | None = None,
        on_ready: Callable[[str], None] | None = None,
        on_transcription: Callable[[TranscriptionMessage], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        on_streaming_started: Callable[[], None] | None = None,
        on_streaming_stopped: Callable[[], None] | None = None,
    ) -> "EavesdropClient":
        """
        Create transcriber mode client for sending audio to server for transcription.

        Args:
            host: WebSocket server hostname/IP
            port: WebSocket server port
            audio_device: Audio device selection (index, name, substring, or None for default)
            beam_size: Whisper beam search size
            word_timestamps: Include word-level timestamps
            initial_prompt: Initial prompt for transcription context
            hotwords: Priority words for recognition
            on_ready: Callback for server ready notification (backend_name)
            on_transcription: Callback for transcription results
            on_error: Callback for error messages
            on_streaming_started: Callback for streaming start events
            on_streaming_stopped: Callback for streaming stop events

        Returns:
            EavesdropClient instance configured for transcriber mode

        Raises:
            ValueError: Invalid audio device specification
            OSError: Audio device not accessible
        """
        ...

    @classmethod
    def subscriber(
        cls,
        host: str,
        port: int,
        stream_names: list[str],
        on_ready: Callable[[str], None] | None = None,
        on_transcription: Callable[[TranscriptionMessage], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        on_stream_status: Callable[[str, str, str | None], None] | None = None,
    ) -> "EavesdropClient":
        """
        Create subscriber mode client for receiving transcriptions from RTSP streams.

        Args:
            host: WebSocket server hostname/IP
            port: WebSocket server port
            stream_names: List of RTSP stream names to subscribe to
            on_ready: Callback for server ready notification (backend_name)
            on_transcription: Callback for transcription results
            on_error: Callback for error messages
            on_stream_status: Callback for stream status changes (stream, status, message)

        Returns:
            EavesdropClient instance configured for subscriber mode

        Raises:
            ValueError: Empty or invalid stream names list
        """
        ...

    # Async Iterator Protocol
    def __aiter__(self) -> AsyncIterator[TranscriptionMessage]:
        """Return async iterator for transcription messages."""
        ...

    async def __anext__(self) -> TranscriptionMessage:
        """Return next transcription message."""
        ...

    # Async Context Manager Protocol
    async def __aenter__(self) -> "EavesdropClient":
        """
        Enter async context manager, automatically connecting to server.

        Returns:
            Self for use in async with statements

        Raises:
            ConnectionError: Failed to establish WebSocket connection
        """
        ...

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit async context manager, automatically disconnecting from server.

        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)
        """
        ...

    # Connection Management
    async def connect(self) -> None:
        """
        Establish WebSocket connection to server.

        For transcriber mode: Starts audio capture but not transmission.
        For subscriber mode: Immediately begins receiving transcriptions.

        Raises:
            ConnectionError: Failed to establish WebSocket connection
            OSError: Audio device initialization failed (transcriber mode)
        """
        ...

    async def disconnect(self) -> None:
        """
        Close WebSocket connection and clean up resources.

        For transcriber mode: Stops audio capture and transmission.
        For subscriber mode: Stops receiving transcriptions.
        """
        ...

    # Transcriber Mode Control (raises RuntimeError if called in subscriber mode)
    async def start_streaming(self) -> None:
        """
        Begin sending captured audio data to server.

        Must be called after connect() in transcriber mode.
        Audio capture continues during streaming.

        Raises:
            RuntimeError: Not in transcriber mode or not connected
        """
        ...

    async def stop_streaming(self) -> None:
        """
        Stop sending audio data to server while maintaining capture and connection.

        Allows restart via start_streaming() without reconnection.

        Raises:
            RuntimeError: Not in transcriber mode or not currently streaming
        """
        ...

    # Status Properties
    def is_connected(self) -> bool:
        """Return True if WebSocket connection is established."""
        ...

    def is_streaming(self) -> bool:
        """
        Return True if actively streaming audio (transcriber mode only).

        Returns:
            True if streaming audio, False otherwise

        Raises:
            RuntimeError: Called in subscriber mode
        """
        ...


# Expected Callback Signatures
OnReadyCallback = Callable[[str], None]  # backend_name
OnTranscriptionCallback = Callable[[TranscriptionMessage], None]  # message
OnErrorCallback = Callable[[str], None]  # error_message
OnStreamingStartedCallback = Callable[[], None]  # no args
OnStreamingStoppedCallback = Callable[[], None]  # no args
OnStreamStatusCallback = Callable[
    [str, str, str | None], None
]  # stream, status, message


# Expected Exception Types
class EavesdropClientError(Exception):
    """Base exception for EavesdropClient errors."""

    pass


class InvalidAudioDeviceError(EavesdropClientError, ValueError):
    """Raised when specified audio device is invalid or inaccessible."""

    pass


class InvalidStreamNameError(EavesdropClientError, ValueError):
    """Raised when stream name list is empty or contains invalid names."""

    pass


class ConnectionError(EavesdropClientError):
    """Raised when WebSocket connection fails."""

    pass


class ModeError(EavesdropClientError, RuntimeError):
    """Raised when operation is invalid for current client mode."""

    pass
