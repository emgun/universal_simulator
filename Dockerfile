# Multi-stage production Dockerfile for Universal Physics Stack
# Optimized for fast instance startup and reliable training

# ============================================================================
# Stage 1: Builder - Install dependencies and compile PyTorch kernels
# ============================================================================
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    build-essential \
    gcc \
    g++ \
    rclone \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better caching
WORKDIR /build
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional tools
RUN pip install --no-cache-dir \
    wandb \
    pydantic \
    python-dotenv

# Pre-compile PyTorch (trigger kernel compilation)
RUN python -c "import torch; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')" \
    && python -c "import torch; torch.randn(10, 10, device='cuda' if torch.cuda.is_available() else 'cpu')" || true

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_MODULE_LOADING=LAZY

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    rclone \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 trainer \
    && mkdir -p /workspace \
    && chown -R trainer:trainer /workspace

# Switch to non-root user
USER trainer
WORKDIR /workspace

# Set up directories
RUN mkdir -p /workspace/checkpoints \
    && mkdir -p /workspace/data \
    && mkdir -p /workspace/logs

# Copy application code (will be overridden by git clone in practice)
# This is mainly for local testing
COPY --chown=trainer:trainer . /workspace/universal_simulator

# Verify installation
RUN python -c "import torch; import wandb; import numpy; print('✅ All packages installed')" \
    && python -c "import torch; print(f'PyTorch {torch.__version__} CUDA available: {torch.cuda.is_available()}')"

# Default working directory
WORKDIR /workspace

# Expose ports (if needed for monitoring)
EXPOSE 8888 6006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch" || exit 1

# Default command (can be overridden)
CMD ["/bin/bash"]

