"""Smoke tests for the client package test harness.

These tests document the core API goal for the client package: factory methods
must provide a stable, typed interface for creating mode-specific clients.
"""

from eavesdrop.client.core import EavesdropClient
from eavesdrop.wire import ClientType


def test_transcriber_factory_sets_transcriber_mode() -> None:
  client = EavesdropClient.transcriber(audio_device="default")

  assert client._client_type == ClientType.TRANSCRIBER


def test_subscriber_factory_sets_subscriber_mode() -> None:
  client = EavesdropClient.subscriber(stream_names=["office"])

  assert client._client_type == ClientType.RTSP_SUBSCRIBER
