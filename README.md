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
