#!/bin/bash
# Launch production training run on VastAI with pre-built Docker image
# 
# Usage:
#   ./scripts/launch_production.sh [config_name] [instance_id]
#
# Example:
#   ./scripts/launch_production.sh train_burgers_32dim 12345678

set -e

CONFIG=${1:-train_burgers_32dim}
INSTANCE=${2:-}

# Docker image from GitHub Container Registry
IMAGE="ghcr.io/emgun/universal_simulator:latest"

# Find best available instance if not specified
if [ -z "$INSTANCE" ]; then
    echo "🔍 Finding best available RTX 4090 instance..."
    INSTANCE=$(vastai search offers 'reliability > 0.98 num_gpus=1 gpu_name=RTX_4090 dph < 0.5' -o 'dph' --raw | jq -r '.[0].id')
    echo "✅ Selected instance: $INSTANCE"
fi

echo "🚀 Launching training on instance $INSTANCE"
echo "   Config: configs/${CONFIG}.yaml"
echo "   Image: $IMAGE"
echo ""

vastai create instance $INSTANCE \
    --image "$IMAGE" \
    --disk 50 \
    --ssh \
    --env "-e WANDB_API_KEY=${WANDB_API_KEY} -e WANDB_PROJECT=universal-simulator -e WANDB_ENTITY=${WANDB_ENTITY} -e B2_KEY_ID=${B2_KEY_ID} -e B2_APP_KEY=${B2_APP_KEY} -e B2_S3_ENDPOINT=${B2_S3_ENDPOINT} -e B2_S3_REGION=${B2_S3_REGION}" \
    --onstart-cmd "mkdir -p ~/.config/rclone && echo '[B2TRAIN]' > ~/.config/rclone/rclone.conf && echo 'type = s3' >> ~/.config/rclone/rclone.conf && echo 'provider = Other' >> ~/.config/rclone/rclone.conf && echo \"access_key_id = \${B2_KEY_ID}\" >> ~/.config/rclone/rclone.conf && echo \"secret_access_key = \${B2_APP_KEY}\" >> ~/.config/rclone/rclone.conf && echo \"endpoint = \${B2_S3_ENDPOINT}\" >> ~/.config/rclone/rclone.conf && echo \"region = \${B2_S3_REGION}\" >> ~/.config/rclone/rclone.conf && cd /app && mkdir -p data/pdebench && rclone copy B2TRAIN:pdebench/full/burgers1d/burgers1d_train_000.h5 data/pdebench/ && ln -sf data/pdebench/burgers1d_train_000.h5 data/pdebench/burgers1d_train.h5 && python scripts/train.py --config configs/${CONFIG}.yaml --stage all"

echo ""
echo "✅ Instance launched!"
echo "   Monitor: vastai show instance \$INSTANCE_ID"
echo "   Logs: vastai logs \$INSTANCE_ID"
