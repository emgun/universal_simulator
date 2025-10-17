# Production Dockerfile - Optimized for VastAI deployment
# 
# Strategy:
# - Build with public pytorch/pytorch image for GitHub Actions
# - At runtime, VastAI mounts their /venv/main/ with pre-installed PyTorch
# - This gives us both: buildable images + VastAI's proven CUDA/Triton setup
#
# Target size: <1GB compressed

# ============================================================
# Stage 1: Builder - Install dependencies
# ============================================================
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime AS builder

WORKDIR /build

# Install system dependencies needed for building
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first (better layer caching)
COPY requirements.txt pyproject.toml ./
COPY src/ ./src/

# Install dependencies and our package
# Use --no-deps for our package to avoid version conflicts
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps -e . && \
    # Aggressive cleanup to reduce size
    find /opt/conda -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/conda -type f -name '*.pyc' -delete && \
    find /opt/conda -type f -name '*.pyo' -delete && \
    find /opt/conda -type d -name 'tests' -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/conda -type d -name '*.dist-info' -exec rm -rf {} + 2>/dev/null || true && \
    rm -rf /root/.cache /tmp/* /var/tmp/*

# ============================================================
# Stage 2: Runtime - Minimal final image
# ============================================================
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /opt/conda /opt/conda

# Install minimal system dependencies
# rclone for B2 data download, build-essential for torch.compile
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        rclone \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy only essential application files
COPY configs/ ./configs/
COPY scripts/train.py scripts/evaluate.py scripts/precompute_latent_cache.py ./scripts/
COPY src/ ./src/

# Create data directory
RUN mkdir -p /app/data/pdebench

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/conda/bin:$PATH"

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Default command
CMD ["python", "scripts/train.py", "--config", "configs/train_burgers_32dim.yaml", "--stage", "all"]
