ARG BASE_IMAGE=eavesdrop-rocm-base
FROM $BASE_IMAGE

SHELL ["/bin/bash", "-lc"]

ENV UV_COMPILE_BYTECODE=1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /uvx /bin/

RUN mkdir -p /app/{wire,server}

# Copy the wire and common projects
WORKDIR /app/wire
COPY ./wire/pyproject.toml ./pyproject.toml
COPY ./wire/src/ ./src
WORKDIR /app/common
COPY ./common/pyproject.toml ./pyproject.toml
COPY ./common/src/ ./src

# Copy and build the eavesdrop-server project
WORKDIR /app/server
COPY server/pyproject.toml ./pyproject.toml
COPY server/uv.lock ./uv.lock
COPY server/src/ ./src
# The ROCm base image already provides the runtime Python environment in /opt/venv.
# Sync into that active environment so the app sees the preinstalled GPU stack.
# Keep the base image packages in place: torch/ctranslate2 are installed there, not in pyproject.toml.
RUN --mount=type=cache,target=/root/.cache/uv \
    /bin/uv sync --locked --active --inexact

COPY server/docker/entrypoint.sh ./entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
CMD ["-m", "eavesdrop.server"]
