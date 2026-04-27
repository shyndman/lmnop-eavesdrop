# basedpyright: reportUnusedCallResult=false

import argparse
import asyncio
import os


class CLIArgs(argparse.Namespace):
  port: int = 0
  config: str | None = None
  json_logs: bool = False
  correlation_id: str | None = None
  log_namespace: str | None = None


from eavesdrop.common import get_logger, setup_logging


def get_env_str(env_var: str, default: str | None = None) -> str | None:
  """Get string environment variable with default fallback."""
  value = os.getenv(env_var)
  return default if value is None else value


def get_env_int(env_var: str, default: int) -> int:
  """Get integer environment variable with default fallback."""
  value = os.getenv(env_var)
  if value is None:
    return default

  try:
    return int(value)
  except ValueError:
    return default


def get_env_bool(env_var: str, default: bool) -> bool:
  """Get boolean environment variable with default fallback."""
  value = os.getenv(env_var)
  if value is None:
    return default

  return value.lower() in ("true", "1", "yes", "on")


async def _async_main() -> None:
  parser = argparse.ArgumentParser()
  _ = parser.add_argument(
    "--port",
    "-p",
    type=int,
    default=get_env_int("EAVESDROP_PORT", 9090),
    help="Websocket port to run the server on. (Env: EAVESDROP_PORT)",
  )
  _ = parser.add_argument(
    "--config",
    type=str,
    default=get_env_str("EAVESDROP_CONFIG"),
    help="Path to the configuration file. (Env: EAVESDROP_CONFIG)",
  )

  _ = parser.add_argument(
    "--json_logs",
    action="store_true",
    default=get_env_bool("JSON_LOGS", False),
    help="Output logs in JSON format. (Env: JSON_LOGS)",
  )
  _ = parser.add_argument(
    "--correlation_id",
    type=str,
    default=get_env_str("CORRELATION_ID"),
    help="Correlation ID for log tracing. (Env: CORRELATION_ID)",
  )
  _ = parser.add_argument(
    "--log_namespace",
    type=str,
    default=get_env_str("LOG_NAMESPACE"),
    help=(
      "Restrict output to a logger namespace. "
      "Provide prefixes like 'tracing' to debug specific subsystems. "
      "(Env: LOG_NAMESPACE)"
    ),
  )
  args: CLIArgs = parser.parse_args(namespace=CLIArgs())

  # Validate required config path
  if not args.config:
    parser.error(
      "Configuration file path is required. Set --config or EAVESDROP_CONFIG environment variable."
    )

  # Setup structured logging
  log_level = os.getenv("LOG_LEVEL", "INFO").upper()
  setup_logging(
    level=log_level,
    json_output=args.json_logs,
    correlation_id=args.correlation_id,
    filter_to_logger=args.log_namespace,
  )
  logger = get_logger("main")

  logger.info(
    "Starting Eavesdrop Server",
    port=args.port,
    config_path=args.config,
  )

  from eavesdrop.server.server import TranscriptionServer

  server = TranscriptionServer()
  await server.run(
    "0.0.0.0",
    args.config,
    port=args.port,
  )


def main() -> None:
  try:
    asyncio.run(_async_main())
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  main()
