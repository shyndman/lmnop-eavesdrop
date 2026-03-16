#!/bin/bash

set -euo pipefail

if [ -f /etc/profile.d/eavesdrop-rocm.sh ]; then
  # Preserve the runtime ROCm override chosen at build time.
  source /etc/profile.d/eavesdrop-rocm.sh
fi

echo "ENVIRONMENT POST ACTIVATION"
echo "~~~ Variables ~~~"
env

exec python "$@"
