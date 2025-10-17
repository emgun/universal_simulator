# Production Dockerfile - Based on VastAI's PyTorch image
# 
# Benefits:
# - VastAI's proven PyTorch/CUDA/Triton setup
# - All our dependencies pre-installed
# - No git clone or pip install needed
# - 2-3 min faster startup vs git clone method

FROM vastai/pytorch:latest

WORKDIR /app

# Install system dependencies (git, rclone already in vastai/pytorch)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Activate VastAI's PyTorch venv and install our dependencies
COPY requirements.txt pyproject.toml ./
COPY src/ ./src/

# Use VastAI's preinstalled PyTorch venv
RUN . /venv/main/bin/activate && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e . && \
    rm -rf /root/.cache/pip && \
    find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /app -type f -name "*.pyc" -delete

# Copy configs and scripts
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Create data directory
RUN mkdir -p /app/data/pdebench

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH="/venv/main/bin:$PATH"

# Default command (can be overridden)
CMD ["python", "scripts/train.py", "--config", "configs/train_burgers_32dim.yaml", "--stage", "all"]
