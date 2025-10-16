# Docker Usage Guide

This document explains how to use the containerized Universal Physics Stack for reproducible training.

## Prerequisites

- Docker 20.10+ with GPU support (`nvidia-docker2`)
- NVIDIA GPU with CUDA 12.1+ drivers
- Docker Compose 1.28+ (optional, for easier management)

## Quick Start

### Option 1: Docker Compose (Recommended for Local Development)

```bash
# 1. Copy environment template
cp .env.example .env  # If you have one
# Edit .env with your credentials

# 2. Build and start container
docker-compose up --build -d

# 3. Attach to container
docker-compose exec universal-simulator bash

# 4. Run training
cd universal_simulator
python scripts/train.py configs/train_pdebench.yaml
```

### Option 2: Direct Docker Commands

```bash
# 1. Build image
docker build -t universal-simulator:latest .

# 2. Run container with GPU
docker run --gpus all \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/configs:/workspace/configs:ro \
  --env-file .env \
  -it universal-simulator:latest bash

# 3. Inside container
cd universal_simulator
python scripts/train.py configs/train_pdebench.yaml
```

## Image Details

### Multi-Stage Build

The Dockerfile uses a multi-stage build for optimization:

1. **Builder Stage** (`nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`)
   - Installs build tools (gcc, g++, build-essential)
   - Compiles Python packages with C extensions
   - Pre-compiles PyTorch CUDA kernels
   - ~3GB intermediate image

2. **Runtime Stage** (`nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`)
   - Minimal runtime-only CUDA libraries
   - Copies compiled packages from builder
   - Non-root user for security
   - ~1.5GB final image

### Image Size

- **Full image:** ~1.5 GB
- **With dependencies:** ~3.5 GB (includes PyTorch, NumPy, etc.)
- **Compressed:** ~900 MB when pushed to registry

## Volume Mounts

The container expects these directories:

- `/workspace/checkpoints` - Model checkpoints (read/write)
- `/workspace/data` - Training data (read/write)
- `/workspace/logs` - Training logs (read/write)
- `/workspace/configs` - Configuration files (read-only)

## Environment Variables

Required:
- `WANDB_API_KEY` - WandB authentication
- `WANDB_PROJECT` - WandB project name
- `WANDB_ENTITY` - WandB entity/username

Optional:
- `B2_KEY_ID` - Backblaze B2 key ID (for data download)
- `B2_APP_KEY` - Backblaze B2 app key
- `B2_BUCKET` - B2 bucket name (default: `pdebench`)
- `B2_S3_ENDPOINT` - B2 S3 endpoint
- `B2_S3_REGION` - B2 S3 region
- `CUDA_VISIBLE_DEVICES` - GPU selection (default: `0`)

## Container Registry

### Push to Docker Hub

```bash
# Tag image
docker tag universal-simulator:latest <username>/universal-simulator:latest
docker tag universal-simulator:latest <username>/universal-simulator:v1.0.0

# Push
docker push <username>/universal-simulator:latest
docker push <username>/universal-simulator:v1.0.0
```

### Push to GitHub Container Registry (GHCR)

```bash
# Login
echo $GITHUB_TOKEN | docker login ghcr.io -u <username> --password-stdin

# Tag
docker tag universal-simulator:latest ghcr.io/<username>/universal-simulator:latest

# Push
docker push ghcr.io/<username>/universal-simulator:latest
```

## VastAI Integration

To use the Docker image with VastAI:

### Option 1: Pre-built Image (Recommended)

```bash
# In vast_launch.py or onstart script
docker pull <username>/universal-simulator:latest
docker run --gpus all \
  -v /workspace/checkpoints:/workspace/checkpoints \
  -v /workspace/data:/workspace/data \
  --env-file /workspace/.env \
  <username>/universal-simulator:latest \
  bash -c "cd universal_simulator && python scripts/train.py $TRAIN_CONFIG"
```

### Option 2: Build on Instance

```bash
# In onstart.sh
git clone https://github.com/emgun/universal_simulator.git
cd universal_simulator
docker build -t universal-simulator:latest .
docker run --gpus all ... (same as above)
```

## Testing Locally

```bash
# 1. Build
docker build -t universal-simulator:latest .

# 2. Test dry-run
docker run --gpus all -v $(pwd)/configs:/workspace/configs:ro \
  universal-simulator:latest \
  python universal_simulator/scripts/dry_run.py /workspace/configs/train_pdebench.yaml --estimate-only

# 3. Test training (short run)
docker run --gpus all \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/configs:/workspace/configs:ro \
  --env-file .env \
  universal-simulator:latest \
  python universal_simulator/scripts/train.py /workspace/configs/train_pdebench.yaml
```

## Troubleshooting

### GPU Not Available

```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If fails, install nvidia-docker2:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Permission Issues

```bash
# The container runs as user 'trainer' (UID 1000)
# Ensure mounted volumes are accessible:
sudo chown -R 1000:1000 checkpoints/ data/ logs/
```

### Out of Memory

```bash
# Reduce batch size in config
# Or allocate more shared memory:
docker run --gpus all --shm-size=8g ...
```

### Slow Build

```bash
# Use BuildKit for parallel builds:
DOCKER_BUILDKIT=1 docker build -t universal-simulator:latest .

# Or use docker-compose with BuildKit:
DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker-compose build
```

## Performance Comparison

### Without Container
- Instance startup: ~5 min (install dependencies)
- First training: ~15 min
- Compilation errors: Common

### With Container
- Image pull: ~2 min (compressed)
- Instance startup: ~30 sec
- First training: ~15 min (same)
- Compilation errors: None (pre-compiled)

**Savings:** ~4.5 min per instance launch, 100% reliability

## Security

- Container runs as non-root user (`trainer`, UID 1000)
- No unnecessary packages in runtime image
- Read-only config mounts recommended
- Environment variables for secrets (not in image)

## Next Steps

1. **Push to Registry:** Make image available for fast instance pulls
2. **CI/CD:** Auto-build on push
3. **Versioning:** Tag releases (e.g., `v1.0.0`)
4. **Monitoring:** Add health checks and monitoring

## Resources

- [NVIDIA Docker Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Docker Multi-Stage Builds](https://docs.docker.com/develop/develop-images/multistage-build/)
- [VastAI Docker](https://vast.ai/docs/gpu-instances/docker)

