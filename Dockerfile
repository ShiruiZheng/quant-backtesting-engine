# Build/run the FastAPI service with uv, pinned to the same locked deps as CI.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Faster, more reproducible installs inside the image.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies against the committed lockfile (no dev extras in the image).
# README.md is copied because pyproject.toml references it as the package readme.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev

EXPOSE 8000

CMD ["uv", "run", "--no-dev", "uvicorn", "backtester.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
