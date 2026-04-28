#!/usr/bin/env python3
"""
Test script for EavesdropClient transcriber mode.

Usage:
    python test_transcriber.py --host localhost --port 8080 [--audio-device default]
"""

import argparse
import asyncio
import os
import sys
import uuid
from typing import cast

from eavesdrop.client import EavesdropClient
from eavesdrop.common import get_logger, setup_logging


def _format_segment_time_range(start: float, end: float) -> str:
  return f"({start:.2f}s - {end:.2f}s)"


async def test_transcriber(
  host: str,
  port: int,
  audio_device: str = "default",
  model: str = "distil-small.en",
) -> int:
  """Test the transcriber client with a real server."""
  print("Creating transcriber client...")
  print(f"  Host: {host}")
  print(f"  Port: {port}")
  print(f"  Audio device: {audio_device}")
  print(f"  Model: {model}")

  client: EavesdropClient | None = None

  try:
    # Create transcriber client
    client = EavesdropClient.transcriber(
      host=host,
      port=port,
      audio_device=audio_device,
      model=model,
      beam_size=5,
      word_timestamps=True,
      initial_prompt="Test transcription session",
    )
    print("✅ Client created successfully")

    # Test connection
    print("\nConnecting to server...")
    await client.connect()
    print("✅ Connected successfully")
    print(f"   Connection status: {client.is_connected()}")

    # Start streaming
    recording_id = uuid.uuid4().hex
    print("\nStarting audio streaming...")
    await client.start_streaming(recording_id)
    print("✅ Streaming started")
    print(f"   Streaming status: {client.is_streaming()}")

    # Listen for transcriptions
    print("\nListening for transcriptions (press Ctrl+C to stop)...")
    message_count = 0

    async for event in client:
      if event.family != "transcription":
        print(f"\n[Client event] {event.family}")
        continue

      message_count += 1
      message = event.message
      print(f"\n[Message {message_count}]")
      print(f"  Stream: {message.stream}")
      print(f"  Language: {message.language}")
      print(f"  Segments: {len(message.segments)}")

      if message.segments:
        for i, segment in enumerate(message.segments):
          time_range = _format_segment_time_range(
            segment.absolute_start_time,
            segment.absolute_end_time,
          )
          print(
            f"    Segment {i}: '{segment.text}' ",
            f"{time_range} [{segment.avg_logprob:.2f}]",
            sep="",
          )

  except KeyboardInterrupt:
    print("\n\nInterrupted by user")
  except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()
    return 1
  finally:
    # Cleanup
    if client is not None:
      print("\nCleaning up...")
      if client.is_streaming():
        await client.stop_streaming()
        print("✅ Stopped streaming")

      if client.is_connected():
        await client.disconnect()
        print("✅ Disconnected")

  print("\n🎉 Test completed successfully!")
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
    "--model",
    default="distil-small.en",
    help="Whisper model alias to use (default: distil-small.en)",
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
  model = cast(str, args.model)
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
      model=model,
    )
    return asyncio.run(test_transcriber(host, port, audio_device, model))
  except KeyboardInterrupt:
    print("\nExiting...")
    return 0


if __name__ == "__main__":
  sys.exit(main())
