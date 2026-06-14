#!/usr/bin/env python3
"""
Stream audio to Eavesdrop and print finalized English translations.

Usage:
    python simple_translator.py --server localhost:8080 [--audio-device default] \
        [--model large-v3] [--source-language ja]
"""

import argparse
import asyncio
import os
import sys
import uuid
from typing import cast

from eavesdrop.client import EavesdropClient
from eavesdrop.common import get_logger, setup_logging
from eavesdrop.wire import TranscriptionTask, UserTranscriptionOptions


def _format_segment_time_range(start: float, end: float) -> str:
  return f"({start:.2f}s - {end:.2f}s)"


async def run_translator(
  host: str,
  port: int,
  audio_device: str = "default",
  model: str = "large-v3",
  source_language: str | None = None,
) -> int:
  """Stream audio and print finalized English translations.

  :param host: Eavesdrop server host.
  :type host: str
  :param port: Eavesdrop server port.
  :type port: int
  :param audio_device: Audio device to capture (mic or output monitor).
  :type audio_device: str
  :param model: Multilingual Whisper model alias capable of translation.
  :type model: str
  :param source_language: Optional source-language hint; ``None`` auto-detects.
  :type source_language: str | None
  :return: Process exit code.
  :rtype: int
  """
  print("Creating translator client...")
  print(f"  Host: {host}")
  print(f"  Port: {port}")
  print(f"  Audio device: {audio_device}")
  print(f"  Model: {model}")
  print(f"  Source language: {source_language or 'auto-detect'}")

  client: EavesdropClient | None = None

  try:
    # Model/task/language live in the session override to keep one source of truth.
    client = EavesdropClient.transcriber(
      host=host,
      port=port,
      audio_device=audio_device,
    )
    print("✅ Client created successfully")

    # Request translation. word_timestamps stays unset: translate mode rejects it.
    setup_options = UserTranscriptionOptions(
      model=model,
      task=TranscriptionTask.TRANSLATE,
      language=source_language,
    )

    print("\nConnecting to server...")
    await client.connect(setup_options)
    print("✅ Connected successfully")

    # Start streaming
    recording_id = uuid.uuid4().hex
    print("\nStarting audio streaming...")
    await client.start_streaming(recording_id)
    print("✅ Streaming started")

    print("\nListening for translations (press Ctrl+C to stop)...\n")
    printed_ids: set[int] = set()

    async for event in client:
      if event.family == "language_detection":
        print(f"[detected: {event.language} ({event.probability:.2f})]")
        continue

      if event.family != "transcription":
        continue

      for segment in event.message.segments:
        if not segment.completed or segment.id in printed_ids:
          continue
        printed_ids.add(segment.id)
        time_range = _format_segment_time_range(
          segment.absolute_start_time,
          segment.absolute_end_time,
        )
        print(f"{time_range} {segment.text.strip()}")

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

  return 0


def configure_logging(
  *, json_logs: bool, correlation_id: str | None, log_namespace: str | None
) -> None:
  """Configure shared structured logging for the translator client example."""
  setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    json_output=json_logs,
    correlation_id=correlation_id,
    filter_to_logger=log_namespace,
  )


def main() -> int:
  """Main entry point."""
  parser = argparse.ArgumentParser(
    description="Stream audio to Eavesdrop and print English translation"
  )
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
    default="large-v3",
    help="Multilingual Whisper model alias (default: large-v3). Must support translation.",
  )
  _ = parser.add_argument(
    "--source-language",
    default=None,
    help="Source language code hint (e.g. 'ja'); omit to auto-detect.",
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
  source_language = cast(str | None, args.source_language)
  json_logs = cast(bool, args.json_logs)
  correlation_id = cast(str | None, args.correlation_id)
  log_namespace = cast(str | None, args.log_namespace)
  configure_logging(
    json_logs=json_logs,
    correlation_id=correlation_id,
    log_namespace=log_namespace,
  )
  logger = get_logger("client/translate")

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
      "starting translation client",
      server=server,
      audio_device=audio_device,
      model=model,
      source_language=source_language,
    )
    return asyncio.run(run_translator(host, port, audio_device, model, source_language))
  except KeyboardInterrupt:
    print("\nExiting...")
    return 0


if __name__ == "__main__":
  sys.exit(main())
