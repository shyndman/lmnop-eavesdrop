"""
Streaming audio transcription abstractions.

This module provides reusable abstractions for streaming audio transcription
that can be used by various input sources (WebSocket, RTSP, etc.).
"""

from ..config import TranscriptionConfig
from .buffer import AudioStreamBuffer
from .interfaces import AudioSource, TranscriptionResult, TranscriptionSink
from .processor import StreamingTranscriptionProcessor
from .websocket_adapters import WebSocketAudioSource, WebSocketTranscriptionSink
from .websocket_client import WebSocketStreamingClient

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
