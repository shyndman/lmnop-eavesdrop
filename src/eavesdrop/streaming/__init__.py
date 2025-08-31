"""
Streaming audio transcription abstractions.

This module provides reusable abstractions for streaming audio transcription
that can be used by various input sources (WebSocket, RTSP, etc.).
"""

from .buffer import AudioStreamBuffer, BufferConfig
from .interfaces import AudioSource, TranscriptionResult, TranscriptionSink
from .processor import StreamingTranscriptionProcessor, TranscriptionConfig
from .websocket_adapters import WebSocketAudioSource, WebSocketTranscriptionSink
from .websocket_client import WebSocketStreamingClient

__all__ = [
  "AudioStreamBuffer",
  "BufferConfig",
  "AudioSource",
  "TranscriptionResult",
  "TranscriptionSink",
  "StreamingTranscriptionProcessor",
  "TranscriptionConfig",
  "WebSocketAudioSource",
  "WebSocketTranscriptionSink",
  "WebSocketStreamingClient",
]
