"""
Eavesdrop client library.

Python client library for connecting to eavesdrop transcription services.
"""

from eavesdrop.client.audio import AudioCapture
from eavesdrop.client.connection import WebSocketConnection
from eavesdrop.client.core import EavesdropClient, FileTranscriptionResult
from eavesdrop.client.events import (
  ConnectedEvent,
  DisconnectedEvent,
  LiveClientEvent,
  ReconnectedEvent,
  ReconnectingEvent,
  TranscriptionEvent,
)

__all__ = [
  "EavesdropClient",
  "FileTranscriptionResult",
  "AudioCapture",
  "WebSocketConnection",
  "ConnectedEvent",
  "DisconnectedEvent",
  "ReconnectingEvent",
  "ReconnectedEvent",
  "TranscriptionEvent",
  "LiveClientEvent",
]
