import sys

try:
  import torch
except ImportError:
  torch = None

from .logs import get_logger


def discover_gpus() -> dict[str, int]:
  """
  Discover GPUs using PyTorch.

  Returns:
      Dict mapping device_name -> gpu index

  Raises:
      SystemExit: If PyTorch is not available or no GPUs found
  """
  logger = get_logger("gpu_discovery")
  logger.debug("Discovering GPUs using PyTorch")

  if torch is None:
    logger.error("PyTorch not available. Cannot discover GPUs.")
    sys.exit(1)

  if not torch.cuda.is_available():
    logger.error("CUDA not available. No GPUs found.")
    sys.exit(1)

  device_count = torch.cuda.device_count()
  if device_count == 0:
    logger.error("No CUDA devices found")
    sys.exit(1)

  gpu_map = {}
  for i in range(device_count):
    device_name = torch.cuda.get_device_name(i)
    gpu_map[device_name] = i
    logger.debug("Discovered GPU", gpu_index=i, device_name=device_name)

  logger.info("GPU discovery completed", discovered_gpus=len(gpu_map))
  return gpu_map


def resolve_gpu_index(gpu_name: str | None) -> int:
  """
  Resolve GPU name to device index.

  Args:
      gpu_name: Optional GPU device name. If None, returns 0 (default GPU).

  Returns:
      GPU device index

  Raises:
      SystemExit: If specified GPU name is not found
  """
  logger = get_logger("gpu_resolution")

  if gpu_name is None:
    logger.debug("No GPU name specified, using default device index 0")
    return 0

  logger.debug("Resolving GPU name to device index", gpu_name=gpu_name)

  gpu_map = discover_gpus()

  if gpu_name not in gpu_map:
    available_names = list(gpu_map.keys())
    logger.error(
      "GPU with specified name not found",
      requested_name=gpu_name,
      available_names=available_names,
    )
    logger.error(
      "Run 'python -c \"import torch; [print(f'Device {i}: "
      "{torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]\"' to see "
      "available GPUs"
    )
    sys.exit(1)

  device_index = gpu_map[gpu_name]
  logger.info("Resolved GPU name to device index", gpu_name=gpu_name, device_index=device_index)

  return device_index
