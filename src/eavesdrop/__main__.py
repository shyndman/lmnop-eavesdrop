import argparse
import asyncio
import os

from .logs import get_logger, setup_logging
from .server import TranscriptionServer


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
    help="Path to the RTSP streams config file. (Env: EAVESDROP_CONFIG)",
  )
  parser.add_argument(
    "--backend",
    "-b",
    type=str,
    default=get_env_or_default("EAVESDROP_BACKEND", "faster_whisper"),
    help='Backend: "faster_whisper" (Env: EAVESDROP_BACKEND)',
  )
  parser.add_argument(
    "--faster_whisper_custom_model_path",
    "-fw",
    type=str,
    default=get_env_or_default("EAVESDROP_FW_MODEL_PATH", None),
    help="Custom Faster Whisper Model (Env: EAVESDROP_FW_MODEL_PATH)",
  )
  parser.add_argument(
    "--omp_num_threads",
    "-omp",
    type=int,
    default=get_env_or_default("EAVESDROP_OMP_NUM_THREADS", 1, int),
    help="Number of threads to use for OpenMP (Env: EAVESDROP_OMP_NUM_THREADS)",
  )
  parser.add_argument(
    "--no_single_model",
    "-nsm",
    action="store_true",
    default=get_env_or_default("EAVESDROP_NO_SINGLE_MODEL", False, bool),
    help="Set this if every connection should instantiate its own model. Only relevant for "
    "custom model, passed using -fw. (Env: EAVESDROP_NO_SINGLE_MODEL)",
  )
  parser.add_argument(
    "--max_clients",
    type=int,
    default=get_env_or_default("EAVESDROP_MAX_CLIENTS", 4, int),
    help="Maximum clients supported by the server. (Env: EAVESDROP_MAX_CLIENTS)",
  )
  parser.add_argument(
    "--max_connection_time",
    type=int,
    default=get_env_or_default("EAVESDROP_MAX_CONNECTION_TIME", 300, int),
    help="Maximum connection time in seconds. (Env: EAVESDROP_MAX_CONNECTION_TIME)",
  )
  parser.add_argument(
    "--cache_path",
    "-c",
    type=str,
    default=get_env_or_default("EAVESDROP_CACHE_PATH", "/app/.cache/eavesdrop/"),
    help="Path to cache the converted ctranslate2 models. (Env: EAVESDROP_CACHE_PATH)",
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
  parser.add_argument(
    "--gpu-name",
    "-g",
    type=str,
    default=get_env_or_default("EAVESDROP_GPU_NAME", None),
    help="GPU device name to use for inference. Run 'python -c \"import torch; "
    "[print(f'Device {i}: {torch.cuda.get_device_name(i)}') for i in "
    "range(torch.cuda.device_count())]\"' to see available GPUs. (Env: EAVESDROP_GPU_NAME)",
  )
  args = parser.parse_args()

  # Setup structured logging
  log_level = os.getenv("LOG_LEVEL", "INFO").upper()
  setup_logging(level=log_level, json_output=args.json_logs, correlation_id=args.correlation_id)
  logger = get_logger("run_server")

  if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = str(args.omp_num_threads)
    logger.info("Set OMP_NUM_THREADS", threads=args.omp_num_threads)

  logger.info(
    "Starting Eavesdrop Server",
    port=args.port,
    backend=args.backend,
    max_clients=args.max_clients,
    cache_path=args.cache_path,
    debug_audio_enabled=bool(args.debug_audio_path),
  )

  server = TranscriptionServer()
  await server.run(
    "0.0.0.0",
    port=args.port,
    backend=args.backend,
    faster_whisper_custom_model_path=args.faster_whisper_custom_model_path,
    single_model=not args.no_single_model,
    max_clients=args.max_clients,
    max_connection_time=args.max_connection_time,
    cache_path=args.cache_path,
    debug_audio_path=args.debug_audio_path,
    gpu_name=args.gpu_name,
  )


if __name__ == "__main__":
  try:
    asyncio.run(main())
  except KeyboardInterrupt:
    pass
