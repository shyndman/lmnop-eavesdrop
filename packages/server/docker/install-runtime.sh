#!/bin/bash
set -euo pipefail
export ORT_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_TELEMETRY=1

# Snapshot the wheel identity, not GPU availability (builds do not need a GPU).
python - <<'PY' > /tmp/eavesdrop-torch-before.json
import json
import torch

print(json.dumps({"version": torch.__version__, "cuda": torch.version.cuda, "hip": torch.version.hip}))
PY

# Preserve matching vendor libraries as well as Torch itself. Excluding just
# torch could still replace torchaudio or NVIDIA/ROCm libraries with incompatible
# wheels pulled from the portable host lockfile.
python - <<'PY' > /tmp/eavesdrop-vendor-packages.txt
from importlib.metadata import distributions

protected = {"torch", "torchaudio", "torchvision", "triton", "pytorch-triton", "pytorch-triton-rocm"}
for distribution in distributions():
    name = distribution.metadata["Name"].lower().replace("_", "-")
    if name in protected or name.startswith(("nvidia-", "rocm-", "pytorch-triton-")):
        protected.add(name)
print("\n".join(sorted(protected)))
PY
mapfile -t vendor_packages < /tmp/eavesdrop-vendor-packages.txt
skip_packages=()
for package in "${vendor_packages[@]}"; do
  skip_packages+=(--no-install-package "$package")
done

uv sync --locked --active --inexact --no-dev \
  --package eavesdrop-server --extra transcription "${skip_packages[@]}"

python - <<'PY'
import json
import os
from pathlib import Path

import torch
import torchaudio
import torchvision
import nemo.collections.asr
import silero_vad

before = json.loads(Path("/tmp/eavesdrop-torch-before.json").read_text())
after = {"version": torch.__version__, "cuda": torch.version.cuda, "hip": torch.version.hip}
if before != after:
    raise RuntimeError(f"Dependency installation replaced the base Torch build: {before} -> {after}")
accelerator = os.environ["EAVESDROP_ACCELERATOR"]
if accelerator == "cuda" and torch.version.cuda is None:
    raise RuntimeError("CUDA image does not contain a CUDA Torch build")
if accelerator == "rocm" and torch.version.hip is None:
    raise RuntimeError("ROCm image does not contain a HIP Torch build")
if accelerator == "cpu" and (torch.version.cuda is not None or torch.version.hip is not None):
    raise RuntimeError("CPU image unexpectedly contains a GPU Torch build")
print(f"Parakeet dependencies import successfully; preserved {accelerator} Torch {torch.__version__}")
PY
rm /tmp/eavesdrop-torch-before.json /tmp/eavesdrop-vendor-packages.txt
