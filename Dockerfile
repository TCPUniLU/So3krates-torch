# syntax=docker/dockerfile:1
# Build with GPU support (default):  docker build -t torchkrates .
# Build CPU-only:                    docker build --build-arg CUDA=false -t torchkrates-cpu .

ARG CUDA=true

# ── builder base selection ─────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder-true
FROM python:3.11-slim                            AS builder-false
FROM builder-${CUDA}                             AS builder

ARG CUDA=true
ENV DEBIAN_FRONTEND=noninteractive

# On CUDA base, Python 3.11 is not pre-installed
RUN if [ "$CUDA" = "true" ]; then \
      apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip && \
      rm -rf /var/lib/apt/lists/* ; \
    fi

# Create an isolated virtualenv so the runtime copy is predictable
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements.txt pyproject.toml README.md ./
COPY src/ src/

# Install torch ecosystem together so pip resolves compatible versions
# in one pass. For CPU builds use the CPU-only index to avoid the large
# CUDA wheels; for GPU builds the default PyPI index provides CUDA wheels.
RUN if [ "$CUDA" = "false" ]; then \
      pip install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu ; \
    else \
      pip install --no-cache-dir torch torchvision torchaudio ; \
    fi && \
    pip install --no-cache-dir .

# ── runtime base selection ─────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime-true
FROM python:3.11-slim                              AS runtime-false
FROM runtime-${CUDA}                               AS runtime

ARG CUDA=true
ENV DEBIAN_FRONTEND=noninteractive

# On CUDA runtime base, install a minimal Python 3.11 + venv support
RUN if [ "$CUDA" = "true" ]; then \
      apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv && \
      rm -rf /var/lib/apt/lists/* ; \
    fi

# Copy the full virtualenv from the builder stage
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4

WORKDIR /workspace

# Default: print help for the training CLI.
# Override at runtime, e.g.:
#   docker run --gpus all -v $(pwd)/data:/workspace/data torchkrates \
#     torchkrates-train --config /workspace/data/config.yaml
CMD ["torchkrates-train", "--help"]
