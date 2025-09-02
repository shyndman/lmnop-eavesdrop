# lmnop:eavesdrop

## Docker Build

The Docker image supports customization for different AMD GPU architectures:

### Build Arguments

- `GFX_ARCH` (default: `gfx1100`) - Target AMD GPU architecture
- `HSA_OVERRIDE_GFX_VERSION` (optional) - HSA version override

### Usage Examples

```bash
# Default build (gfx1100 + auto HSA override for iGPUs)
docker build -f docker/Dockerfile .

# Custom GPU architecture (discrete cards)
docker build -f docker/Dockerfile --build-arg GFX_ARCH=gfx1030 .

# Custom GPU with explicit HSA override
docker build -f docker/Dockerfile --build-arg GFX_ARCH=gfx1100 --build-arg HSA_OVERRIDE_GFX_VERSION=10.3.0 .
```

**Note:** When using `GFX_ARCH=gfx1100`, `HSA_OVERRIDE_GFX_VERSION=11.0.3` is automatically set unless explicitly overridden. This is required for AMD iGPUs and some discrete cards.

## Docker Run

### Build and Run Commands

```bash
# Build the image with progress output
docker build --progress=plain -f docker/Dockerfile -t ghcr.io/shyndman/lmnop-eavesdrop:latest .

# Run the container with GPU access and port mapping
docker run --interactive --tty \
  --device /dev/kfd \
  --device /dev/dri \
  --security-opt seccomp=unconfined \
  --rm \
  --publish '9090:9090' \
  --name eavesdrop \
  --volume ~/.cache/lmnop/eavesdrop:/app/.cache/eavesdrop/ \
  ghcr.io/shyndman/lmnop-eavesdrop:latest

# Combined build and run command
docker build --progress=plain -f docker/Dockerfile -t ghcr.io/shyndman/lmnop-eavesdrop:latest . && \
docker run --interactive --tty \
  --device /dev/kfd \
  --device /dev/dri \
  --security-opt seccomp=unconfined \
  --rm \
  --publish '9090:9090' \
  --name eavesdrop \
  --volume ~/.cache/lmnop/eavesdrop:/app/.cache/eavesdrop/ \
  ghcr.io/shyndman/lmnop-eavesdrop:latest
```

### Run Options Explained

- `--device /dev/kfd` - AMD GPU compute device access
- `--device /dev/dri` - Direct Rendering Infrastructure access
- `--security-opt seccomp=unconfined` - Relaxed security for GPU operations
- `--publish '9090:9090'` - Maps container port 9090 to host port 9090
- `--volume ~/.cache/lmnop/eavesdrop:/app/.cache/eavesdrop/` - Persists cache data
- `--rm` - Automatically removes container when stopped
