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
    --env "WANDB_API_KEY=${WANDB_API_KEY}" \
    --env "WANDB_PROJECT=${WANDB_PROJECT:-universal-simulator}" \
    --env "WANDB_ENTITY=${WANDB_ENTITY}" \
    --env "B2_KEY_ID=${B2_KEY_ID}" \
    --env "B2_APP_KEY=${B2_APP_KEY}" \
    --env "B2_BUCKET=${B2_BUCKET:-pdebench}" \
    --env "B2_S3_ENDPOINT=${B2_S3_ENDPOINT}" \
    --env "B2_S3_REGION=${B2_S3_REGION}" \
    --onstart-cmd "
        cd /app && \
        export RCLONE_CONFIG_B2TRAIN_TYPE=s3 && \
        export RCLONE_CONFIG_B2TRAIN_PROVIDER=Other && \
        export RCLONE_CONFIG_B2TRAIN_ACCESS_KEY_ID=\$B2_KEY_ID && \
        export RCLONE_CONFIG_B2TRAIN_SECRET_ACCESS_KEY=\$B2_APP_KEY && \
        export RCLONE_CONFIG_B2TRAIN_ENDPOINT=\$B2_S3_ENDPOINT && \
        export RCLONE_CONFIG_B2TRAIN_REGION=\$B2_S3_REGION && \
        mkdir -p data/pdebench && \
        echo '📦 Downloading training data...' && \
        rclone copy B2TRAIN:pdebench/full/burgers1d/burgers1d_train_000.h5 data/pdebench/ --progress && \
        cd data/pdebench && ln -sf burgers1d_train_000.h5 burgers1d_train.h5 && cd ../.. && \
        echo '🚀 Starting training...' && \
        python scripts/train.py --config configs/${CONFIG}.yaml --stage all
    "

echo ""
echo "✅ Instance launched!"
echo "   Monitor: vastai show instance \$INSTANCE_ID"
echo "   Logs: vastai logs \$INSTANCE_ID"
