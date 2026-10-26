"""Live WebSocket audio transcription abstractions."""

from typing import TYPE_CHECKING

from eavesdrop.server.config import TranscriptionConfig
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.interfaces import (
  TranscriptionResult,
  TranscriptionSink,
)

if TYPE_CHECKING:
  from eavesdrop.server.streaming.audio_flow import (
    WebSocketTranscriptionSink,
  )
  from eavesdrop.server.streaming.client import WebSocketStreamingClient
  from eavesdrop.server.streaming.processor import StreamingTranscriptionProcessor


def __getattr__(name: str) -> object:
  if name == "WebSocketTranscriptionSink":
    from eavesdrop.server.streaming.audio_flow import WebSocketTranscriptionSink

    return WebSocketTranscriptionSink

  if name == "WebSocketStreamingClient":
    from eavesdrop.server.streaming.client import WebSocketStreamingClient

    return WebSocketStreamingClient

  if name == "StreamingTranscriptionProcessor":
    from eavesdrop.server.streaming.processor import StreamingTranscriptionProcessor

    return StreamingTranscriptionProcessor

  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
  "AudioStreamBuffer",
  "TranscriptionResult",
  "TranscriptionSink",
  "StreamingTranscriptionProcessor",
  "TranscriptionConfig",
  "WebSocketTranscriptionSink",
  "WebSocketStreamingClient",
]
