"""CLI wrapper for one-shot local audio file transcription."""

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from eavesdrop.common import get_logger, setup_logging

from .core import EavesdropClient


@dataclass(frozen=True)
class TranscribeFileConfig:
  """CLI configuration for one-shot file transcription.

  :param audio_file: Existing local audio file path.
  :type audio_file: Path
  :param host: Transcription server hostname.
  :type host: str
  :param port: Transcription server port.
  :type port: int
  :param audio_device: Placeholder device value required by transcriber mode.
  :type audio_device: str
  :param timeout_s: Optional per-call timeout in seconds.
  :type timeout_s: float | None
  :param json_logs: Whether to emit structured JSON logs.
  :type json_logs: bool
  :param correlation_id: Optional correlation identifier bound to logs.
  :type correlation_id: str | None
  :param log_namespace: Optional logger namespace filter.
  :type log_namespace: str | None
  """

  audio_file: Path
  host: str
  port: int
  audio_device: str
  timeout_s: float | None
  json_logs: bool
  correlation_id: str | None
  log_namespace: str | None


def parse_audio_file(value: str) -> Path:
  """Validate that audio file argument points to an existing file.

  :param value: CLI argument value.
  :type value: str
  :returns: Resolved existing file path.
  :rtype: Path
  :raises argparse.ArgumentTypeError: If the path does not exist or is not a file.
  """
  path = Path(value).expanduser().resolve()

  if not path.exists():
    raise argparse.ArgumentTypeError(f"Audio file does not exist: {path}")

  if not path.is_file():
    raise argparse.ArgumentTypeError(f"Audio path is not a file: {path}")

  return path


def parse_port(value: str) -> int:
  """Parse and validate TCP port number.

  :param value: CLI port string.
  :type value: str
  :returns: Parsed port number.
  :rtype: int
  :raises argparse.ArgumentTypeError: If the value is not an integer in [1, 65535].
  """
  try:
    port = int(value)
  except ValueError as exc:
    raise argparse.ArgumentTypeError(f"Port must be numeric: {value}") from exc

  if port < 1 or port > 65535:
    raise argparse.ArgumentTypeError(f"Port must be between 1 and 65535: {port}")

  return port


def parse_timeout(value: str) -> float:
  """Parse and validate positive timeout seconds.

  :param value: CLI timeout string.
  :type value: str
  :returns: Positive timeout in seconds.
  :rtype: float
  :raises argparse.ArgumentTypeError: If timeout is not a positive number.
  """
  try:
    timeout_s = float(value)
  except ValueError as exc:
    raise argparse.ArgumentTypeError(f"Timeout must be numeric: {value}") from exc

  if timeout_s <= 0:
    raise argparse.ArgumentTypeError("Timeout must be greater than zero")

  return timeout_s


def parse_config() -> TranscribeFileConfig:
  """Parse command-line args into typed transcription config.

  :returns: Parsed transcription configuration.
  :rtype: TranscribeFileConfig
  """
  parser = argparse.ArgumentParser(
    description="Transcribe a local audio file with eavesdrop-client"
  )
  _ = parser.add_argument("--audio-file", required=True, type=parse_audio_file)
  _ = parser.add_argument("--host", default="localhost")
  _ = parser.add_argument("--port", default=9090, type=parse_port)
  _ = parser.add_argument("--audio-device", default="default")
  _ = parser.add_argument("--timeout-s", type=parse_timeout, default=None)
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

  return TranscribeFileConfig(
    audio_file=cast(Path, args.audio_file),
    host=cast(str, args.host),
    port=cast(int, args.port),
    audio_device=cast(str, args.audio_device),
    timeout_s=cast(float | None, args.timeout_s),
    json_logs=cast(bool, args.json_logs),
    correlation_id=cast(str | None, args.correlation_id),
    log_namespace=cast(str | None, args.log_namespace),
  )


def configure_logging(config: TranscribeFileConfig) -> None:
  """Configure client CLI logging using the shared common setup."""
  setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    json_output=config.json_logs,
    correlation_id=config.correlation_id,
    filter_to_logger=config.log_namespace,
  )


async def transcribe_file(config: TranscribeFileConfig) -> int:
  """Run one-shot transcription and print the final result text.

  :param config: Parsed CLI configuration.
  :type config: TranscribeFileConfig
  :returns: Process exit code.
  :rtype: int
  """
  client = EavesdropClient.transcriber(
    host=config.host,
    port=config.port,
    audio_device=config.audio_device,
  )
  result = await client.transcribe_file(
    file_path=str(config.audio_file),
    timeout_s=config.timeout_s,
  )

  for warning in result.warnings:
    print(f"Warning: {warning}", file=sys.stderr)

  print(result.text)
  return 0


def main() -> int:
  """CLI entrypoint for one-shot file transcription.

  :returns: Process exit code.
  :rtype: int
  """
  config = parse_config()
  configure_logging(config)
  logger = get_logger("client/cli")
  logger.info(
    "starting file transcription",
    audio_file=str(config.audio_file),
    host=config.host,
    port=config.port,
  )

  try:
    return asyncio.run(transcribe_file(config))
  except KeyboardInterrupt:
    return 130
  except Exception as exc:
    print(f"Transcription failed: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
