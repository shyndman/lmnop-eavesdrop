import json
import subprocess
import sys

from .logs import get_logger


def discover_gpus() -> dict[str, int]:
  """
  Discover AMD GPUs using amd-smi command.

  Returns:
      Dict mapping asic_serial -> gpu index

  Raises:
      SystemExit: If amd-smi command fails or is not available
  """
  logger = get_logger("gpu_discovery")
  logger.debug("Discovering AMD GPUs using amd-smi")

  try:
    result = subprocess.run(
      ["amd-smi", "static", "--asic", "--json"], capture_output=True, text=True, check=True
    )
  except FileNotFoundError:
    logger.error("amd-smi command not found. Please install AMD SMI tools.")
    sys.exit(1)
  except subprocess.CalledProcessError as e:
    logger.error("amd-smi command failed", return_code=e.returncode, stderr=e.stderr)
    sys.exit(1)

  try:
    gpu_data = json.loads(result.stdout)
  except json.JSONDecodeError as e:
    logger.error("Failed to parse amd-smi JSON output", error=str(e))
    sys.exit(1)

  if not isinstance(gpu_data, list):
    logger.error("Unexpected amd-smi output format - expected list of GPUs")
    sys.exit(1)

  gpu_map = {}
  for gpu_info in gpu_data:
    if "gpu" not in gpu_info or "asic" not in gpu_info:
      logger.warning("Skipping malformed GPU entry", gpu_info=gpu_info)
      continue

    gpu_index = gpu_info["gpu"]
    asic_serial = gpu_info["asic"].get("asic_serial", "")
    market_name = gpu_info["asic"].get("market_name", "Unknown")

    if not asic_serial or asic_serial == "0x0000000000000000":
      logger.warning(
        "Skipping GPU with missing or null serial", gpu_index=gpu_index, market_name=market_name
      )
      continue

    gpu_map[asic_serial] = gpu_index
    logger.debug(
      "Discovered GPU", gpu_index=gpu_index, asic_serial=asic_serial, market_name=market_name
    )

  logger.info("GPU discovery completed", discovered_gpus=len(gpu_map))
  return gpu_map


def resolve_gpu_index(gpu_serial: str | None) -> int:
  """
  Resolve GPU serial to device index.

  Args:
      gpu_serial: Optional GPU serial number. If None, returns 0 (default GPU).

  Returns:
      GPU device index

  Raises:
      SystemExit: If specified GPU serial is not found
  """
  logger = get_logger("gpu_resolution")

  if gpu_serial is None:
    logger.debug("No GPU serial specified, using default device index 0")
    return 0

  logger.debug("Resolving GPU serial to device index", gpu_serial=gpu_serial)

  gpu_map = discover_gpus()

  if gpu_serial not in gpu_map:
    available_serials = list(gpu_map.keys())
    logger.error(
      "GPU with specified serial not found",
      requested_serial=gpu_serial,
      available_serials=available_serials,
    )
    logger.error("Run 'amd-smi static --asic --json' to see available GPUs")
    sys.exit(1)

  device_index = gpu_map[gpu_serial]
  logger.info(
    "Resolved GPU serial to device index", gpu_serial=gpu_serial, device_index=device_index
  )

  return device_index
