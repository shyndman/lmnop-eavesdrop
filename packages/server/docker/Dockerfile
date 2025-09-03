ARG BASE_IMAGE=eavesdrop-rocm-base
FROM $BASE_IMAGE

SHELL ["conda", "run", "-n", "py_3.12", "/bin/bash", "-c"]

ENV UV_COMPILE_BYTECODE=1

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /uvx /bin/

RUN mkdir -p /app/{wire,server}

# Copy the eavesdrop-wire project
WORKDIR /app/wire
COPY ./wire/pyproject.toml ./pyproject.toml
COPY ./wire/src/ ./src

# Copy and build the eavesdrop-server project
WORKDIR /app/server
COPY server/pyproject.toml ./pyproject.toml
COPY server/uv.lock ./uv.lock
RUN /bin/uv venv --system-site-packages
RUN --mount=type=cache,target=/root/.cache/uv \
    /bin/uv sync --locked

# Copy eavesdrop source directly from project
COPY server/src/ ./src
COPY server/docker/entrypoint.sh ./entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
CMD ["-m", "eavesdrop.server"]
