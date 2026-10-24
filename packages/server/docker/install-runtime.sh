#!/bin/bash
set -euo pipefail
export ORT_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_TELEMETRY=1

# Keep the CPU-only Torch/audio wheels installed for standalone Silero VAD.
uv sync --locked --active --inexact --no-dev \
  --package eavesdrop-server --extra transcription \
  --no-install-package torch --no-install-package torchaudio

python - <<'PY'
import huggingface_hub
import sherpa_onnx
import silero_vad
import torch
import torchaudio

if torch.version.cuda is not None or torch.version.hip is not None:
    raise RuntimeError("CPU image unexpectedly contains a GPU Torch build")
print(f"Orukeet CPU dependencies import successfully; Silero uses Torch {torch.__version__}")
PY
