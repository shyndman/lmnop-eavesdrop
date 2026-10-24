#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
cd "$REPO_ROOT"

CPU_BASE_IMAGE=${CPU_BASE_IMAGE:-eavesdrop-cpu-base}
SERVER_IMAGE=${SERVER_IMAGE:-ghcr.io/shyndman/lmnop-eavesdrop:latest}

if [ "$#" -ne 0 ]; then
    echo "Usage: $0 (CPU only; configure CPU_BASE_IMAGE and SERVER_IMAGE via environment)" >&2
    exit 1
fi

docker build --progress=plain \
    -f packages/server/docker/Dockerfile.cpu-base -t "$CPU_BASE_IMAGE" .
docker build --progress=plain \
    --build-arg BASE_IMAGE="$CPU_BASE_IMAGE" \
    -f packages/server/docker/Dockerfile -t "$SERVER_IMAGE" .
