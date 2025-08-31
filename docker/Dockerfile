ARG BASE_IMAGE=eavesdrop-rocm-base
FROM $BASE_IMAGE

SHELL ["conda", "run", "-n", "py_3.12", "/bin/bash", "-c"]

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /uvx /bin/

# Create app directory and copy Eavesdrop source
RUN mkdir /app
WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock
RUN /bin/uv venv --system-site-packages
RUN /bin/uv sync --locked

# Copy eavesdrop source directly from project
COPY src/ /app/src
COPY docker/entrypoint.sh /app/entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
CMD ["-m", "eavesdrop"]
