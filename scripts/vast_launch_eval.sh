#!/usr/bin/env bash
set -euo pipefail

# Vast.ai launcher for TTC evaluation - auto-selects fastest available GPU
# Usage: ENV_FILE=.env ./scripts/vast_launch_eval.sh

ROOT_DIR=${ROOT_DIR:-$PWD}
ENV_FILE=${ENV_FILE:-.env}

# GPU preferences - prioritize speed
GPU_PREFERENCES=${GPU_PREFERENCES:-"H200 H100 A100 RTX_4090"}
IMAGE=${IMAGE:-pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime}
DISK_GB=${DISK_GB:-120}

# Evaluation configs
VAL_CONFIG=${VAL_CONFIG:-configs/eval_burgers_512dim_ttc_val.yaml}
TEST_CONFIG=${TEST_CONFIG:-configs/eval_burgers_512dim_ttc_test.yaml}

# Dataset files to upload
DATA_DIR=${DATA_DIR:-artifacts/wandb_datasets/burgers1d_full_v1}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints}

# Load environment
echo "[eval] Loading env from $ENV_FILE (if present)"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

WANDB_PROJECT=${WANDB_PROJECT:-universal-simulator}
WANDB_ENTITY=${WANDB_ENTITY:-emgun-morpheus-space}

# Check required tools
for cmd in vastai jq ssh scp; do
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "[eval] Error: $cmd not found" >&2
    exit 1
  }
done

# Function to find best available GPU
pick_fastest_gpu() {
  local gpu_prefs="$1"

  for gpu in $gpu_prefs; do
    echo "[eval] Searching for $gpu..." >&2

    # Find cheapest reliable instance of this GPU type
    offer_id=$(vastai search offers --raw | jq -r \
      --arg GPU "$gpu" \
      'map(select(
        (.dph_total!=null) and
        (.num_gpus==1) and
        (.gpu_name|test($GPU; "i")) and
        (.reliability2>0.9) and
        (.inet_down>200) and
        (.disk_space >= 100)
      )) | sort_by(.dph_total) | .[0].id // empty')

    if [ -n "$offer_id" ]; then
      echo "[eval] Found $gpu offer: $offer_id" >&2
      echo "$offer_id"
      return 0
    fi
  done

  echo "[eval] No suitable GPU found from: $gpu_prefs" >&2
  return 1
}

echo "[eval] === Vast.ai TTC Evaluation Launcher ==="
echo "[eval] Selecting fastest available GPU..."

OFFER_ID=$(pick_fastest_gpu "$GPU_PREFERENCES")
if [ -z "$OFFER_ID" ]; then
  echo "[eval] Error: No suitable GPU offers found" >&2
  exit 1
fi

GPU_INFO=$(vastai search offers --raw | jq -r ".[] | select(.id==$OFFER_ID) | {gpu:.gpu_name, price:.dph_total} | @json")
GPU_NAME=$(echo "$GPU_INFO" | jq -r .gpu)
GPU_PRICE=$(echo "$GPU_INFO" | jq -r .price)

echo "[eval] Selected: $GPU_NAME (\$$GPU_PRICE/hr, offer $OFFER_ID)"

# Create onstart script
ONSTART_SCRIPT=$(mktemp)
cat > "$ONSTART_SCRIPT" << 'ONSTART_EOF'
#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "=== TTC Eval Instance Setup ==="

# Install dependencies
command -v git >/dev/null 2>&1 || (apt-get update && apt-get install -y git)
command -v pip >/dev/null 2>&1 || (apt-get update && apt-get install -y python3-pip)

# Setup workspace
mkdir -p /workspace
cd /workspace

# Clone/update repo
if [ ! -d universal_simulator ]; then
  git clone https://github.com/emgun/universal_simulator.git universal_simulator
fi
cd universal_simulator
git pull origin main

# Install Python packages
python3 -m pip install --upgrade pip
python3 -m pip install -e .[dev]

echo "=== Downloading Checkpoints from W&B ==="
export WANDB_API_KEY="__WANDB_API_KEY__"
export WANDB_PROJECT="__WANDB_PROJECT__"
export WANDB_ENTITY="__WANDB_ENTITY__"

mkdir -p checkpoints/scale
PYTHONPATH=src python scripts/download_checkpoints_from_wandb.py \
  --dest checkpoints/scale \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT"

echo "=== Running TTC Evaluation (Validation) ==="
export PDEBENCH_ROOT=/workspace/universal_simulator/data/pdebench

PYTHONPATH=src python scripts/evaluate.py \
  --config __VAL_CONFIG__ \
  --operator checkpoints/scale/operator.pt \
  --diffusion checkpoints/scale/diffusion_residual.pt \
  --device cuda \
  --output-prefix reports/ttc_eval_val \
  --print-json

echo "=== Running TTC Evaluation (Test) ==="
PYTHONPATH=src python scripts/evaluate.py \
  --config __TEST_CONFIG__ \
  --operator checkpoints/scale/operator.pt \
  --diffusion checkpoints/scale/diffusion_residual.pt \
  --device cuda \
  --output-prefix reports/ttc_eval_test \
  --print-json

echo "=== Evaluation Complete! ==="
ls -lh reports/ttc_eval_*

echo "Shutting down in 60 seconds..."
sleep 60
sync
command -v poweroff >/dev/null 2>&1 && poweroff || true
ONSTART_EOF

# Substitute environment variables
sed -i.bak "s|__WANDB_API_KEY__|${WANDB_API_KEY:-}|g" "$ONSTART_SCRIPT"
sed -i.bak "s|__WANDB_PROJECT__|${WANDB_PROJECT}|g" "$ONSTART_SCRIPT"
sed -i.bak "s|__WANDB_ENTITY__|${WANDB_ENTITY}|g" "$ONSTART_SCRIPT"
sed -i.bak "s|__VAL_CONFIG__|${VAL_CONFIG}|g" "$ONSTART_SCRIPT"
sed -i.bak "s|__TEST_CONFIG__|${TEST_CONFIG}|g" "$ONSTART_SCRIPT"

echo "[eval] Launching instance (offer $OFFER_ID)..."
INSTANCE_ID=$(vastai create instance "$OFFER_ID" \
  --image "$IMAGE" \
  --disk "$DISK_GB" \
  --onstart-cmd "$(cat "$ONSTART_SCRIPT")" \
  --ssh \
  --raw | jq -r '.new_contract // empty')

rm -f "$ONSTART_SCRIPT" "$ONSTART_SCRIPT.bak"

if [ -z "$INSTANCE_ID" ]; then
  echo "[eval] Error: Failed to launch instance" >&2
  exit 1
fi

echo "[eval] ✓ Instance launched: $INSTANCE_ID"
echo "[eval] Waiting for SSH (30s)..."
sleep 30

# Get SSH info
SSH_INFO=$(vastai show instances --raw | jq -r ".[] | select(.id==$INSTANCE_ID) | {host:.ssh_host, port:.ssh_port} | @json")
SSH_HOST=$(echo "$SSH_INFO" | jq -r .host)
SSH_PORT=$(echo "$SSH_INFO" | jq -r .port)

echo "[eval] SSH: ssh -p $SSH_PORT root@$SSH_HOST"

# Upload datasets
echo "[eval] Uploading datasets ($DATA_DIR)..."
scp -P "$SSH_PORT" -o StrictHostKeyChecking=no -r \
  "$DATA_DIR"/*.h5 \
  "root@$SSH_HOST:/workspace/universal_simulator/data/pdebench/" || {
  echo "[eval] Warning: Dataset upload failed, will rely on W&B download"
}

echo ""
echo "[eval] === Instance Details ==="
echo "[eval] ID: $INSTANCE_ID"
echo "[eval] GPU: $GPU_NAME"
echo "[eval] Cost: \$$GPU_PRICE/hr"
echo "[eval] SSH: ssh -p $SSH_PORT root@$SSH_HOST"
echo ""
echo "[eval] Monitor logs: vastai logs $INSTANCE_ID --tail 100"
echo "[eval] Instance will auto-shutdown after completion"
echo ""
