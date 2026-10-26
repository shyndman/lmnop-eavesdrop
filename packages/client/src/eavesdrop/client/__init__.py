"""
Eavesdrop client library.

Python client library for connecting to eavesdrop transcription services.
"""

from eavesdrop.client.audio import AudioCapture
from eavesdrop.client.connection import WebSocketConnection
from eavesdrop.client.core import EavesdropClient

__all__ = [
  "EavesdropClient",
  "AudioCapture",
  "WebSocketConnection",
]
