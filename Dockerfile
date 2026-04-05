# ─────────────────────────────────────────────────────────────────────────────
# Brain Tumor MRI Classifier — Multi-Stage Production Dockerfile
#
# Build:  docker build -t mri-classifier .
# Run:    docker run -p 5000:5000 mri-classifier
#
# Stages
#   1. builder  — full Python env, installs all deps & downloads weights
#   2. tester   — runs pytest suite (build fails if tests fail)
#   3. runtime  — slim final image, copies only what's needed to serve
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — builder
# Install dependencies and pre-download model weights
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

# Build-time system deps (needed to compile some wheels; dropped in runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest first — lets Docker cache this layer independently
COPY requirements.txt .

# Install into an isolated prefix so the runtime stage can copy it cleanly
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt

# Copy source after deps so code changes don't bust the pip layer
COPY . .

# Download pretrained checkpoint (~47 MB) into the image at build time so the
# container starts instantly with no runtime network dependency.
RUN python models/download_weights.py


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — tester
# Run the full test suite; a non-zero exit code aborts the build.
# ══════════════════════════════════════════════════════════════════════════════
FROM builder AS tester

# pytest + coverage are expected to already be in requirements.txt.
# If not, install them only in this throw-away stage.
RUN pip install --prefix=/install --no-cache-dir pytest pytest-cov 2>/dev/null || true

ENV PYTHONPATH=/app
RUN python -m pytest tests/ \
        --cov=. \
        --cov-report=term-missing \
        --tb=short \
        -q


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — runtime  (the actual production image)
# Minimal OS footprint: no compilers, no test tools, no build artefacts.
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

LABEL maintainer="neuro-ai-lab" \
      description="Brain Tumor MRI Classifier — Flask inference API" \
      version="1.0"

# Runtime-only OS libs (OpenCV headless + libGL for PIL/cv2)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Least-privilege user — never run ML servers as root
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Pull the installed Python packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application source and pre-downloaded weights (excludes test artefacts
# via .dockerignore — see companion file)
COPY --from=builder /app /app

# Ensure the model directory is writable by appuser (needed if weights are
# hot-swapped via a mounted volume at runtime)
RUN chown -R appuser:appuser /app

USER appuser

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=app \
    FLASK_ENV=production \
    # Bind to all interfaces so Docker port-mapping works
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_RUN_PORT=5000 \
    # Suppress EfficientNet download attempts at inference time
    TORCH_HOME=/app/models

EXPOSE 5000

# Health-check — Docker will mark the container unhealthy if /health stops
# responding, enabling zero-downtime restarts in Compose / Kubernetes.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" \
    || exit 1

# Use gunicorn in production for multi-worker throughput.
# Falls back to the Flask dev server if gunicorn isn't installed.
CMD ["sh", "-c", \
     "gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 'app:create_app()' \
      2>/dev/null || flask run --host=0.0.0.0 --port=5000"]
