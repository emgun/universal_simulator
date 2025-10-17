# Production Dockerfile - Optimized multi-stage build
# Based on VastAI's PyTorch image for reliable CUDA/Triton setup
# 
# Target size: <1GB compressed
# Benefits: 2-3 min faster startup vs git clone

# ============================================================
# Stage 1: Builder - Install dependencies
# ============================================================
FROM vastai/pytorch:latest AS builder

WORKDIR /build

# Copy only dependency files first (better caching)
COPY requirements.txt pyproject.toml ./
COPY src/ ./src/

# Activate venv and install dependencies
# Use --no-deps for our package to avoid reinstalling torch
RUN . /venv/main/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps -e . && \
    # Aggressive cleanup
    find /venv/main -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true && \
    find /venv/main -type f -name '*.pyc' -delete && \
    find /venv/main -type f -name '*.pyo' -delete && \
    find /venv/main -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true && \
    rm -rf /root/.cache /tmp/* /var/tmp/*

# ============================================================
# Stage 2: Runtime - Minimal final image
# ============================================================
FROM vastai/pytorch:latest

WORKDIR /app

# Copy only the built venv from builder
COPY --from=builder /venv/main /venv/main

# Copy only essential application files
COPY configs/ ./configs/
COPY scripts/train.py scripts/evaluate.py scripts/precompute_latent_cache.py ./scripts/
COPY src/ ./src/

# Install minimal system dependencies (build-essential for torch.compile)
# git and rclone already in vastai/pytorch
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create data directory
RUN mkdir -p /app/data/pdebench

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PATH="/venv/main/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Default command
CMD ["python", "scripts/train.py", "--config", "configs/train_burgers_32dim.yaml", "--stage", "all"]
