#!/usr/bin/env bash
set -eo pipefail

# One-shot launcher: pick a cheap offer, create instance, hydrate, run, and tail logs
# Prereqs: vastai CLI logged in; .env in repo root with WANDB_*, B2_* vars

REPO_DIR=${REPO_DIR:-$PWD}
cd "$REPO_DIR"

# Load .env if present
if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

GPU_MODEL=${GPU_MODEL:-RTX_4090}
DISK_GB=${DISK_GB:-200}
IMAGE=${IMAGE:-pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime}
WANDB_DATASETS=${WANDB_DATASETS:-burgers1d_full_v1}
WANDB_PROJECT=${WANDB_PROJECT:-universal-simulator}
WANDB_ENTITY=${WANDB_ENTITY:-${WANDB_USERNAME:-}}

echo "Searching cheapest 1x $GPU_MODEL…"
CHEAP_ID=$(vastai search offers "gpu_name=${GPU_MODEL} num_gpus=1 reliability>0.95" | awk 'NR>1{print $1, $10}' | sort -k2,2n | awk 'NR==1{print $1}')
if [ -z "$CHEAP_ID" ]; then
  echo "Falling back to H200 search…"
  CHEAP_ID=$(vastai search offers "gpu_name=H200 num_gpus=1 reliability>0.95" | awk 'NR>1{print $1, $10}' | sort -k2,2n | awk 'NR==1{print $1}')
fi
[ -n "$CHEAP_ID" ] || { echo "No suitable offer found"; exit 1; }
echo "Selected offer: $CHEAP_ID"

OVERRIDES="TRAIN_CONFIG=configs/train_burgers_quality_v2.yaml,EVAL_CONFIG=configs/eval_pdebench_scale_ttc.yaml,EVAL_TEST_CONFIG=configs/eval_pdebench_scale_test_ttc.yaml,PDEBENCH_ROOT=/workspace/universal_simulator/data/pdebench/burgers1d_full_v1,PRECOMPUTE_LATENT=1"

echo "Creating instance…"
python scripts/vast_launch.py launch \
  --offer-id "$CHEAP_ID" \
  --image "$IMAGE" \
  --disk "$DISK_GB" \
  --datasets "$WANDB_DATASETS" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY" \
  ${WANDB_API_KEY:+--wandb-api-key "$WANDB_API_KEY"} \
  ${B2_KEY_ID:+--b2-key-id "$B2_KEY_ID"} \
  ${B2_APP_KEY:+--b2-app-key "$B2_APP_KEY"} \
  ${B2_BUCKET:+--b2-bucket "$B2_BUCKET"} \
  ${B2_PREFIX:+--b2-prefix "$B2_PREFIX"} \
  ${B2_S3_ENDPOINT:+--b2-s3-endpoint "$B2_S3_ENDPOINT"} \
  ${B2_S3_REGION:+--b2-s3-region "$B2_S3_REGION"} \
  --overrides "$OVERRIDES"

echo "Resolving new instance id…"
INST_ID=$(vastai show instances --raw | jq -r ".[] | select(.image_uuid==\"${IMAGE}\" and .onstart!=null) | .id" | sort -n | tail -n1)
[ -n "$INST_ID" ] || { echo "Could not resolve instance id"; exit 1; }
echo "Instance id: $INST_ID"

echo "Waiting for SSH…"
SSH_URL=$(vastai ssh-url "$INST_ID" | tail -n1)
HOST=$(printf '%s' "$SSH_URL" | sed -E 's#ssh://root@([^:]+):([0-9]+)#\1#')
PORT=$(printf '%s' "$SSH_URL" | sed -E 's#ssh://root@([^:]+):([0-9]+)#\2#')
for _ in $(seq 1 120); do nc -z "$HOST" "$PORT" && break || sleep 5; done

echo "Uploading .env and helper scripts…"
scp -P "$PORT" -o StrictHostKeyChecking=no .env root@"$HOST":/workspace/universal_simulator/.env || true
scp -P "$PORT" -o StrictHostKeyChecking=no scripts/remote_fix_and_run.sh root@"$HOST":/workspace/universal_simulator/scripts/remote_fix_and_run.sh
scp -P "$PORT" -o StrictHostKeyChecking=no scripts/run_remote_scale.sh root@"$HOST":/workspace/universal_simulator/scripts/run_remote_scale.sh

echo "Launching remote pipeline…"
ssh -o StrictHostKeyChecking=no root@"$HOST" -p "$PORT" "bash -lc 'cd /workspace/universal_simulator && chmod +x scripts/*.sh && nohup bash scripts/remote_fix_and_run.sh > run_full.log 2>&1 & disown'"

echo "Tailing logs… (Ctrl-C to stop)"
vastai logs "$INST_ID" --tail 200






