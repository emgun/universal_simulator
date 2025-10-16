# Production Dockerfile - Optimized for fast VastAI launches
# All code and dependencies pre-installed
# Only needs to download data and run

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    rclone \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt pyproject.toml ./

# Install Python dependencies (not editable yet, need src/)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Install package in editable mode (now that src/ exists)
RUN pip install --no-cache-dir -e .

# Create data directory
RUN mkdir -p /app/data/pdebench

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command (can be overridden)
CMD ["python", "scripts/train.py", "--config", "configs/train_burgers_32dim.yaml", "--stage", "all"]
