"""
Streaming audio transcription abstractions.

This module provides reusable abstractions for streaming audio transcription
that can be used by various input sources (WebSocket, RTSP, etc.).
"""

from typing import TYPE_CHECKING

from eavesdrop.server.config import TranscriptionConfig
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.interfaces import (
  AudioSource,
  TranscriptionResult,
  TranscriptionSink,
)

if TYPE_CHECKING:
  from eavesdrop.server.streaming.audio_flow import (
    WebSocketAudioSource,
    WebSocketTranscriptionSink,
  )
  from eavesdrop.server.streaming.client import WebSocketStreamingClient
  from eavesdrop.server.streaming.debug_capture import AudioDebugCapture
  from eavesdrop.server.streaming.processor import StreamingTranscriptionProcessor


def __getattr__(name: str) -> object:
  if name in {"WebSocketAudioSource", "WebSocketTranscriptionSink"}:
    from eavesdrop.server.streaming.audio_flow import (
      WebSocketAudioSource,
      WebSocketTranscriptionSink,
    )

    exports = {
      "WebSocketAudioSource": WebSocketAudioSource,
      "WebSocketTranscriptionSink": WebSocketTranscriptionSink,
    }
    return exports[name]

  if name == "WebSocketStreamingClient":
    from eavesdrop.server.streaming.client import WebSocketStreamingClient

    return WebSocketStreamingClient

  if name == "AudioDebugCapture":
    from eavesdrop.server.streaming.debug_capture import AudioDebugCapture

    return AudioDebugCapture

  if name == "StreamingTranscriptionProcessor":
    from eavesdrop.server.streaming.processor import StreamingTranscriptionProcessor

    return StreamingTranscriptionProcessor

  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
  "AudioDebugCapture",
  "AudioStreamBuffer",
  "AudioSource",
  "TranscriptionResult",
  "TranscriptionSink",
  "StreamingTranscriptionProcessor",
  "TranscriptionConfig",
  "WebSocketAudioSource",
  "WebSocketTranscriptionSink",
  "WebSocketStreamingClient",
]
