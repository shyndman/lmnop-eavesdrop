import argparse
import asyncio
import os

from eavesdrop.server.logs import get_logger, setup_logging


def get_env_or_default(env_var, default, var_type: type = str):
  """Get environment variable with type conversion and default fallback."""
  value = os.getenv(env_var)
  if value is None:
    return default

  if var_type is bool:
    # Handle boolean environment variables
    return value.lower() in ("true", "1", "yes", "on")
  elif var_type is int:
    try:
      return int(value)
    except ValueError:
      return default
  else:
    return value


async def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--port",
    "-p",
    type=int,
    default=get_env_or_default("EAVESDROP_PORT", 9090, int),
    help="Websocket port to run the server on. (Env: EAVESDROP_PORT)",
  )
  parser.add_argument(
    "--config",
    type=str,
    default=get_env_or_default("EAVESDROP_CONFIG", None),
    help="Path to the configuration file. (Env: EAVESDROP_CONFIG)",
  )
  parser.add_argument(
    "--debug_audio_path",
    type=str,
    default=None,
    help="Path prefix for debug audio files. When set, audio received from clients will be "
    "saved as .wav files for debugging.",
  )
  parser.add_argument(
    "--json_logs",
    action="store_true",
    default=get_env_or_default("JSON_LOGS", False, bool),
    help="Output logs in JSON format. (Env: JSON_LOGS)",
  )
  parser.add_argument(
    "--correlation_id",
    type=str,
    default=get_env_or_default("CORRELATION_ID", None),
    help="Correlation ID for log tracing. (Env: CORRELATION_ID)",
  )
  args = parser.parse_args()

  # Validate required config path
  if not args.config:
    parser.error(
      "Configuration file path is required. Set --config or EAVESDROP_CONFIG environment variable."
    )

  # Setup structured logging
  log_level = os.getenv("LOG_LEVEL", "INFO").upper()
  setup_logging(level=log_level, json_output=args.json_logs, correlation_id=args.correlation_id)
  logger = get_logger("main")

  logger.info(
    "Starting Eavesdrop Server",
    port=args.port,
    config_path=args.config,
    debug_audio_enabled=bool(args.debug_audio_path),
  )

  from eavesdrop.server.server import TranscriptionServer

  server = TranscriptionServer()
  await server.run(
    "0.0.0.0",
    args.config,
    port=args.port,
    debug_audio_path=args.debug_audio_path,
  )


if __name__ == "__main__":
  try:
    asyncio.run(main())
  except KeyboardInterrupt:
    pass
