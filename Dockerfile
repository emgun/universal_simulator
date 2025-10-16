# Production Dockerfile - Optimized for fast VastAI launches
# All code and dependencies pre-installed
# Only needs to download data and run

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    rclone \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt pyproject.toml ./

# Install Python dependencies (exclude torch - already in base image)
# Filter out torch to avoid duplicate installation
RUN grep -v "^torch" requirements.txt > requirements-docker.txt && \
    pip install --no-cache-dir -r requirements-docker.txt && \
    rm requirements-docker.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Install package in editable mode and clean up
RUN pip install --no-cache-dir -e . && \
    # Remove pip cache
    rm -rf /root/.cache/pip && \
    # Remove __pycache__ and .pyc files
    find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /app -type f -name "*.pyc" -delete && \
    # Remove .egg-info build artifacts
    find /app -type d -name "*.egg-info" -path "*/src/*" -prune -o -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Create data directory
RUN mkdir -p /app/data/pdebench

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command (can be overridden)
CMD ["python", "scripts/train.py", "--config", "configs/train_burgers_32dim.yaml", "--stage", "all"]
