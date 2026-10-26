#!/usr/bin/env python3
"""
Test script for EavesdropClient transcriber mode.

Usage:
    python simple_transcriber.py --server localhost:8080 [--audio-device default]
"""

import argparse
import asyncio
import os
import sys
import uuid
from typing import cast

from eavesdrop.client import EavesdropClient
from eavesdrop.common import get_logger, setup_logging
from eavesdrop.wire import TranscriptionMessage


def _format_segment_time_range(start: float, end: float) -> str:
  return f"({start:.2f}s - {end:.2f}s)"


def print_transcription(message: TranscriptionMessage) -> None:
  """Print each timed segment received from the server.

  :param message: Current transcript update for the recording.
  """
  print(f"\n[Recording {message.recording_id}]")
  for segment in message.segments:
    time_range = _format_segment_time_range(segment.start, segment.end)
    print(f"  {time_range} {segment.text}")


async def test_transcriber(host: str, port: int, audio_device: str = "default") -> int:
  """Capture microphone audio and print pushed transcript updates until interrupted.

  :param host: Server hostname.
  :param port: Server port.
  :param audio_device: Microphone device to capture.
  :returns: Zero after normal shutdown.
  """
  client = EavesdropClient.transcriber(
    host=host,
    port=port,
    audio_device=audio_device,
    on_transcription=print_transcription,
  )
  recording_id = uuid.uuid4().hex
  try:
    await client.connect()
    await client.start_streaming(recording_id)
    print("Listening for transcriptions (press Ctrl+C to stop)...")
    await asyncio.Event().wait()
  finally:
    try:
      if client.is_streaming():
        await client.stop_streaming()
        await client.flush(recording_id)
    finally:
      await client.disconnect()

  return 0


def configure_logging(
  *, json_logs: bool, correlation_id: str | None, log_namespace: str | None
) -> None:
  """Configure shared structured logging for the microphone client example."""
  setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    json_output=json_logs,
    correlation_id=correlation_id,
    filter_to_logger=log_namespace,
  )


def main() -> int:
  """Main entry point."""
  parser = argparse.ArgumentParser(description="Test EavesdropClient transcriber mode")
  _ = parser.add_argument(
    "--server",
    default="localhost:8080",
    help="Server address as host:port (default: localhost:8080)",
  )
  _ = parser.add_argument(
    "--audio-device", default="default", help="Audio device to use (default: default)"
  )
  _ = parser.add_argument(
    "--json_logs",
    action="store_true",
    default=os.getenv("JSON_LOGS", "false").lower() in ("true", "1", "yes", "on"),
    help="Output logs in JSON format. (Env: JSON_LOGS)",
  )
  _ = parser.add_argument(
    "--correlation_id",
    type=str,
    default=os.getenv("CORRELATION_ID"),
    help="Correlation ID for log tracing. (Env: CORRELATION_ID)",
  )
  _ = parser.add_argument(
    "--log_namespace",
    type=str,
    default=os.getenv("LOG_NAMESPACE"),
    help=(
      "Restrict output to a logger namespace. "
      "Provide prefixes like 'client' to debug specific subsystems. "
      "(Env: LOG_NAMESPACE)"
    ),
  )

  args = parser.parse_args()
  server = cast(str, args.server)
  audio_device = cast(str, args.audio_device)
  json_logs = cast(bool, args.json_logs)
  correlation_id = cast(str | None, args.correlation_id)
  log_namespace = cast(str | None, args.log_namespace)
  configure_logging(
    json_logs=json_logs,
    correlation_id=correlation_id,
    log_namespace=log_namespace,
  )
  logger = get_logger("client/mic")

  # Parse server address
  try:
    if ":" in server:
      host, port_str = server.rsplit(":", 1)
      port = int(port_str)
    else:
      host = server
      port = 8080
  except ValueError:
    print(f"Error: Invalid server address '{server}'. Use format 'host:port'")
    return 1

  try:
    logger.info(
      "starting microphone transcription client",
      server=server,
      audio_device=audio_device,
    )
    return asyncio.run(test_transcriber(host, port, audio_device))
  except KeyboardInterrupt:
    print("\nExiting...")
    return 0


if __name__ == "__main__":
  sys.exit(main())
