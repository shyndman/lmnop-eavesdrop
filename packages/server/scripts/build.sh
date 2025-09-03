#!/bin/bash

set -e

# Get script directory and change to server package root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGES_DIR="$(realpath "$SCRIPT_DIR/../..")"

cd "$PACKAGES_DIR"

BACKEND=${1:-rocm}
GFX_ARCH=${GFX_ARCH:-'gfx1030;gfx1100'}
HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-11.0.0}

case "$BACKEND" in
    rocm)
        echo "Building ROCm base image..."
        docker build \
            --build-arg GFX_ARCH="$GFX_ARCH" \
            --build-arg HSA_OVERRIDE_GFX_VERSION="$HSA_OVERRIDE_GFX_VERSION" \
            --progress=plain \
            -f server/docker/Dockerfile.rocm-base \
            -t eavesdrop-rocm-base .

        echo "Building eavesdrop with ROCm backend..."
        docker build \
            --build-arg BASE_IMAGE=eavesdrop-rocm-base \
            --progress=plain \
            -f server/docker/Dockerfile \
            -t ghcr.io/shyndman/lmnop-eavesdrop:latest .
        ;;
    cuda)
        echo "Building CUDA base image..."
        docker build \
            --progress=plain \
            -f server/docker/Dockerfile.cuda-base \
            -t eavesdrop-cuda-base .

        echo "Building eavesdrop with CUDA backend..."
        docker build \
            --build-arg BASE_IMAGE=eavesdrop-cuda-base \
            --progress=plain \
            -f server/docker/Dockerfile \
            -t ghcr.io/shyndman/lmnop-eavesdrop:latest .
        ;;
    *)
        echo "Usage: $0 {rocm|cuda}"
        echo "Environment variables:"
        echo "  GFX_ARCH (ROCm only): GPU architectures (default: 'gfx1030;gfx1100')"
        echo "  HSA_OVERRIDE_GFX_VERSION (ROCm only): HSA version override (default: 11.0.0)"
        exit 1
        ;;
esac

echo "Build completed successfully!"
