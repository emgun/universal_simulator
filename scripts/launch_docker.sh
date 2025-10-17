#!/bin/bash
# Launch training with pre-built Docker image (fastest method)
# 
# Prerequisites:
#   - VastAI env-vars configured (run once): python scripts/vast_launch.py setup-env
#   - Docker image built: https://github.com/emgun/universal_simulator/actions
#
# Usage:
#   ./scripts/launch_docker.sh [config_name]
#
# Example:
#   ./scripts/launch_docker.sh train_burgers_32dim
#
# Benefits:
#   - 1-2 min startup (vs 3-4 min git clone)
#   - All code/deps pre-installed
#   - VastAI's proven PyTorch/CUDA/Triton setup

set -e

CONFIG=${1:-train_burgers_32dim}

# Docker image from GitHub Container Registry
# Use branch-specific tag for latest build
IMAGE="ghcr.io/emgun/universal_simulator:feature-sota_burgers_upgrades"

echo "═══════════════════════════════════════════════════════════"
echo "Docker Launch: ${CONFIG}.yaml"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Image: $IMAGE"
echo ""

# Find best available instance
echo "🔍 Finding best RTX 4090 instance..."
OFFER=$(vastai search offers 'reliability > 0.98 num_gpus=1 gpu_name=RTX_4090 dph < 0.5' -o 'dph' --raw | jq -r '.[0]')
INSTANCE_ID=$(echo "$OFFER" | jq -r '.id')
DPH=$(echo "$OFFER" | jq -r '.dph_total')

echo "✅ Selected offer $INSTANCE_ID (\$$DPH/hr)"
echo ""

# Simple onstart: download data + run training
# All code/deps already in image, VastAI env-vars injected automatically
ONSTART_CMD="
cd /app && \
mkdir -p data/pdebench && \
rclone copy --config <(echo '[B2TRAIN]
type = s3
provider = B2
access_key_id = '\$B2_KEY_ID'
secret_access_key = '\$B2_APP_KEY'
endpoint = '\$B2_S3_ENDPOINT'
region = '\$B2_S3_REGION'
') B2TRAIN:pdebench/full/burgers1d/burgers1d_train_000.h5 data/pdebench/ && \
ln -sf burgers1d_train_000.h5 data/pdebench/burgers1d_train.h5 && \
export TRAIN_CONFIG=configs/${CONFIG}.yaml && \
export TRAIN_STAGE=all && \
export RESET_CACHE=1 && \
/venv/main/bin/python scripts/train.py --config configs/${CONFIG}.yaml --stage all
"

echo "🚀 Launching instance..."
vastai create instance "$INSTANCE_ID" \
    --image "$IMAGE" \
    --disk 64 \
    --ssh \
    --onstart-cmd "$ONSTART_CMD"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Instance Launched!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Monitor:"
echo "  vastai show instance <ID>"
echo "  vastai logs <ID>"
echo ""
echo "Timeline:"
echo "  ~30 sec: Image pull (compressed ~300-500MB)"
echo "  ~30 sec: Data download (1.57GB)"
echo "  ~30 sec: Latent cache precompute"
echo "  Then: Training begins"
echo ""
echo "Expected startup: 1-2 min to training (vs 3-4 min git clone)"
echo ""
