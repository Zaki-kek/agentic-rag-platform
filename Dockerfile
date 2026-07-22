# syntax=docker/dockerfile:1

# --- Stage 1: builder -------------------------------------------------------
# Installs the project and its dependencies into an isolated virtualenv so the
# runtime image can copy only the built artifacts (no build toolchain, no caches).
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Create the virtualenv up front so every pip install lands inside it.
RUN python -m venv "$VIRTUAL_ENV"

# Copy the metadata needed to build the wheel first for better layer caching.
COPY pyproject.toml README.md ./
COPY app ./app

RUN pip install ".[llm]"

# --- Stage 2: runtime -------------------------------------------------------
# Slim final image: only the venv and app sources, run as a non-root user.
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Create an unprivileged user to run the service.
RUN groupadd --system app && useradd --system --gid app --home-dir /app app

# Copy the pre-built virtualenv and application code from the builder stage.
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/app ./app

USER app

EXPOSE 8000

# Liveness probe hits the FastAPI /health endpoint using the stdlib only
# (no extra packages such as curl need to be installed in the runtime image).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
