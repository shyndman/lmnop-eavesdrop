"""
Streaming audio transcription abstractions.

This module provides reusable abstractions for streaming audio transcription
that can be used by various input sources (WebSocket, RTSP, etc.).
"""

from eavesdrop.server.config import TranscriptionConfig
from eavesdrop.server.streaming.buffer import AudioStreamBuffer
from eavesdrop.server.streaming.interfaces import (
  AudioSource,
  TranscriptionResult,
  TranscriptionSink,
)
from eavesdrop.server.streaming.processor import StreamingTranscriptionProcessor
from eavesdrop.server.streaming.websocket_adapters import (
  WebSocketAudioSource,
  WebSocketTranscriptionSink,
)
from eavesdrop.server.streaming.websocket_client import WebSocketStreamingClient

__all__ = [
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
