#!/usr/bin/env bash
set -euo pipefail

# Launch Vast.ai instance and run evaluation with downloaded checkpoints
cd "$(dirname "$0")/.." || exit 1

source .env

OFFER_ID=${1:-23013704}  # Default to RTX A2000

echo "Launching instance $OFFER_ID..."
INSTANCE_ID=$(vastai create instance "$OFFER_ID" \
  --image pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime \
  --disk 50 \
  --ssh \
  --env WANDB_API_KEY="$WANDB_API_KEY" \
  --env WANDB_PROJECT=universal-simulator \
  --env WANDB_ENTITY=emgun-morpheus-space \
  --onstart-cmd "apt-get update && apt-get install -y git && git clone https://github.com/emgun-morpheus-space/universal_simulator.git /workspace/repo || true" \
  | jq -r '.new_contract')

echo "Instance ID: $INSTANCE_ID"
echo "Waiting for instance to be running..."

for i in $(seq 1 60); do
  STATUS=$(vastai show instance "$INSTANCE_ID" --raw | jq -r '.actual_status')
  if [ "$STATUS" = "running" ]; then
    echo "Instance is running!"
    break
  fi
  echo "Status: $STATUS (attempt $i/60)"
  sleep 5
done

echo "Getting SSH URL..."
URL=$(vastai ssh-url "$INSTANCE_ID" | tail -n1)
HOST=${URL#ssh://root@}
HOST=${HOST%:*}
PORT=${URL##*:}

echo "SSH: $HOST:$PORT"
echo "Waiting for SSH to be ready..."
for i in $(seq 1 30); do
  if nc -z -w 5 "$HOST" "$PORT" 2>/dev/null; then
    echo "SSH is ready!"
    break
  fi
  sleep 5
done

echo "Uploading checkpoints and configs..."
scp -P "$PORT" -o StrictHostKeyChecking=no \
  remote_consistency_run/operator.pt \
  remote_consistency_run/diffusion_residual.pt \
  root@"$HOST":/workspace/

scp -P "$PORT" -o StrictHostKeyChecking=no \
  configs/eval_pdebench_scale_ttc.yaml \
  configs/eval_pdebench_scale_test_ttc.yaml \
  root@"$HOST":/workspace/

echo "Setting up and running evaluation..."
ssh -p "$PORT" -o StrictHostKeyChecking=no root@"$HOST" 'bash -s' << 'ENDSSH'
cd /workspace/repo || cd /workspace/universal_simulator || exit 1

# Install requirements
pip install -q wandb omegaconf hydra-core pydantic numpy scipy matplotlib h5py tqdm

# Download minimal dataset (just for evaluation structure)
mkdir -p data/pdebench
wget -q -O data/pdebench/burgers1d_test.h5 https://huggingface.co/datasets/pdebench/burgers1d/resolve/main/burgers1d_test.h5 || true

# Copy checkpoints
mkdir -p checkpoints/scale
cp /workspace/operator.pt checkpoints/scale/
cp /workspace/diffusion_residual.pt checkpoints/scale/
cp /workspace/*.yaml configs/ 2>/dev/null || true

# Run evaluation
export PYTHONPATH=src
export PDEBENCH_ROOT=data/pdebench

echo "Running validation evaluation..."
python scripts/evaluate.py \
  --config configs/eval_pdebench_scale_ttc.yaml \
  --operator checkpoints/scale/operator.pt \
  --diffusion checkpoints/scale/diffusion_residual.pt

echo "Running test evaluation..."
python scripts/evaluate.py \
  --config configs/eval_pdebench_scale_test_ttc.yaml \
  --operator checkpoints/scale/operator.pt \
  --diffusion checkpoints/scale/diffusion_residual.pt

echo "Syncing to W&B..."
wandb sync --sync-all

echo "✓ Evaluation complete!"
ENDSSH

echo ""
echo "Instance ID: $INSTANCE_ID"
echo "To destroy: vastai destroy instance $INSTANCE_ID"





