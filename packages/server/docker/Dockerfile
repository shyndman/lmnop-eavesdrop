# syntax=docker/dockerfile:1
ARG BASE_IMAGE=eavesdrop-cpu-base
FROM $BASE_IMAGE

SHELL ["/bin/bash", "-c"]
ENV ORT_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    HF_HOME=/root/.cache/eavesdrop/huggingface
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
# Keep the workspace layout and authoritative root lockfile intact. The other
# manifests are needed to validate the lock; their packages are not installed.
COPY pyproject.toml uv.lock ./
COPY packages/active-listener/pyproject.toml ./packages/active-listener/pyproject.toml
COPY packages/client/pyproject.toml ./packages/client/pyproject.toml
COPY packages/common/pyproject.toml ./packages/common/pyproject.toml
COPY packages/server/pyproject.toml ./packages/server/pyproject.toml
COPY packages/wire/pyproject.toml ./packages/wire/pyproject.toml
COPY packages/common/src/ ./packages/common/src/
COPY packages/wire/src/ ./packages/wire/src/
COPY packages/server/src/ ./packages/server/src/
COPY packages/server/docker/install-runtime.sh /app/install-runtime.sh

# Retain the base's CPU Torch/audio wheels for standalone Silero VAD.
RUN --mount=type=cache,target=/root/.cache/uv \
    bash /app/install-runtime.sh

COPY --chmod=755 packages/server/docker/entrypoint.sh /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["-m", "eavesdrop.server"]
