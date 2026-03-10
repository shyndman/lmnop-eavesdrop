"""CLI wrapper for one-shot local audio file transcription."""

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from eavesdrop.common import setup_logging_from_env

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
  """

  audio_file: Path
  host: str
  port: int
  audio_device: str
  timeout_s: float | None


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

  args = parser.parse_args()

  return TranscribeFileConfig(
    audio_file=cast(Path, args.audio_file),
    host=cast(str, args.host),
    port=cast(int, args.port),
    audio_device=cast(str, args.audio_device),
    timeout_s=cast(float | None, args.timeout_s),
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
  setup_logging_from_env()
  config = parse_config()

  try:
    return asyncio.run(transcribe_file(config))
  except KeyboardInterrupt:
    return 130
  except Exception as exc:
    print(f"Transcription failed: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
