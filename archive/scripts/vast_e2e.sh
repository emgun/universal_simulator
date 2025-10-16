#!/usr/bin/env bash
set -euo pipefail

# End-to-end Vast.ai pipeline: provision → hydrate (B2) → precompute → train → eval → tail logs
# Usage:
#   ENV_FILE=.env GPU_PREF=RTX_4090 ./scripts/vast_e2e.sh
#
# Required tools: vastai, jq, ssh, scp, nc

ROOT_DIR=${ROOT_DIR:-$PWD}
ENV_FILE=${ENV_FILE:-.env}
GPU_PREF=${GPU_PREF:-RTX_4090}
FALLBACK_GPU=${FALLBACK_GPU:-H200}
IMAGE=${IMAGE:-pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime}
DISK_GB=${DISK_GB:-200}

DATASET_KEY=${DATASET_KEY:-burgers1d_full_v1}
TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_burgers_quality_v2.yaml}
EVAL_CONFIG=${EVAL_CONFIG:-configs/eval_pdebench_scale_ttc.yaml}
EVAL_TEST_CONFIG=${EVAL_TEST_CONFIG:-configs/eval_pdebench_scale_test_ttc.yaml}
PDEBENCH_ROOT_REMOTE=${PDEBENCH_ROOT_REMOTE:-/workspace/universal_simulator/data/pdebench/burgers1d_full_v1}

# Optional training stability tweaks (help avoid OOM / compile stalls)
TORCHINDUCTOR_DISABLE=${TORCHINDUCTOR_DISABLE:-1}
RESET_CACHE=${RESET_CACHE:-0}
PRECOMPUTE_LATENT=${PRECOMPUTE_LATENT:-1}
TRAIN_STAGE=${TRAIN_STAGE:-all}
# Pass any config overrides through to train/eval (Hydra-style if supported)
TRAIN_EXTRA_ARGS=${TRAIN_EXTRA_ARGS:-}

need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1" >&2; exit 1; }; }
need vastai; need jq; need ssh; need scp; need nc

echo "[e2e] Loading env from $ENV_FILE (if present)"
if [ -f "$ENV_FILE" ]; then set -a; # shellcheck disable=SC1090
  . "$ENV_FILE"; set +a; fi
WANDB_PROJECT=${WANDB_PROJECT:-universal-simulator}
WANDB_ENTITY=${WANDB_ENTITY:-emgun-morpheus-space}

pick_offer() {
  local gpu="$1"
  vastai search offers --raw | jq -r \
    --arg GPU "$gpu" \
    'map(select((.dph_total!=null) and (.num_gpus==1) and (.gpu_name|test($GPU; "i")) and (.reliability2>0.9) and (.inet_down>200)))
     | sort_by(.dph_total) | .[0].id // empty'
}

echo "[e2e] Selecting offer (pref=$GPU_PREF, fallback=$FALLBACK_GPU)"
OFFER_ID=$(pick_offer "$GPU_PREF")
if [ -z "$OFFER_ID" ]; then
  echo "[e2e] No $GPU_PREF found; trying $FALLBACK_GPU"
  OFFER_ID=$(pick_offer "$FALLBACK_GPU")
fi
[ -n "$OFFER_ID" ] || { echo "[e2e] No suitable offer found" >&2; exit 1; }
echo "[e2e] Using offer $OFFER_ID"

echo "[e2e] Launching instance"
python scripts/vast_launch.py launch \
  --offer-id "$OFFER_ID" \
  --image "$IMAGE" \
  --disk "$DISK_GB" \
  --datasets "$DATASET_KEY" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY" \
  ${WANDB_API_KEY:+--wandb-api-key "$WANDB_API_KEY"} \
  ${B2_KEY_ID:+--b2-key-id "$B2_KEY_ID"} \
  ${B2_APP_KEY:+--b2-app-key "$B2_APP_KEY"} \
  ${B2_BUCKET:+--b2-bucket "$B2_BUCKET"} \
  ${B2_PREFIX:+--b2-prefix "$B2_PREFIX"} \
  ${B2_S3_ENDPOINT:+--b2-s3-endpoint "$B2_S3_ENDPOINT"} \
  ${B2_S3_REGION:+--b2-s3-region "$B2_S3_REGION"} \
  --overrides "TRAIN_CONFIG=$TRAIN_CONFIG,EVAL_CONFIG=$EVAL_CONFIG,EVAL_TEST_CONFIG=$EVAL_TEST_CONFIG,PDEBENCH_ROOT=$PDEBENCH_ROOT_REMOTE,PRECOMPUTE_LATENT=$PRECOMPUTE_LATENT"

echo "[e2e] Finding instance and SSH URL"
sleep 3
ID=$(vastai show instances --raw | jq -r '.[] | select(.image_uuid=="'"$IMAGE"'" and (.cur_state=="running" or .actual_status=="running")) | .id' | sort -n | tail -n1)
[ -n "$ID" ] || { echo "[e2e] Could not find running instance for $IMAGE" >&2; exit 1; }
URL=$(vastai ssh-url "$ID" | tail -n1)
HOST=${URL#ssh://root@}; HOST=${HOST%:*}; PORT=${URL##*:}
echo "[e2e] Instance $ID at $HOST:$PORT"

echo "[e2e] Waiting for SSH to come up"
for i in $(seq 1 60); do nc -z "$HOST" "$PORT" && break || sleep 5; done

echo "[e2e] Uploading .env and helper scripts"
scp -P "$PORT" -o StrictHostKeyChecking=no "$ROOT_DIR/.env" root@"$HOST":/workspace/universal_simulator/.env || true
scp -P "$PORT" -o StrictHostKeyChecking=no "$ROOT_DIR/scripts/remote_hydrate_b2_once.sh" root@"$HOST":/workspace/universal_simulator/scripts/remote_hydrate_b2_once.sh
scp -P "$PORT" -o StrictHostKeyChecking=no "$ROOT_DIR/scripts/remote_launch_once.sh" root@"$HOST":/workspace/universal_simulator/scripts/remote_launch_once.sh
scp -P "$PORT" -o StrictHostKeyChecking=no "$ROOT_DIR/scripts/run_remote_scale.sh" root@"$HOST":/workspace/universal_simulator/scripts/run_remote_scale.sh

echo "[e2e] Prepare remote and hydrate from B2 (if creds present)"
ssh -o StrictHostKeyChecking=no -p "$PORT" root@"$HOST" "bash -lc 'cd /workspace/universal_simulator && chmod +x scripts/*.sh && scripts/remote_hydrate_b2_once.sh'" || true

echo "[e2e] Launch training+eval (precompute=$PRECOMPUTE_LATENT)"
ssh -o StrictHostKeyChecking=no -p "$PORT" root@"$HOST" "bash -lc 'cd /workspace/universal_simulator && export TORCHINDUCTOR_DISABLE=$TORCHINDUCTOR_DISABLE RESET_CACHE=$RESET_CACHE PRECOMPUTE_LATENT=$PRECOMPUTE_LATENT TRAIN_CONFIG=$TRAIN_CONFIG EVAL_CONFIG=$EVAL_CONFIG EVAL_TEST_CONFIG=$EVAL_TEST_CONFIG TRAIN_STAGE=$TRAIN_STAGE PDEBENCH_ROOT=$PDEBENCH_ROOT_REMOTE TRAIN_EXTRA_ARGS=\"$TRAIN_EXTRA_ARGS\" && scripts/remote_launch_once.sh'"

echo "[e2e] Tailing logs (Ctrl-C to stop)"
while true; do
  vastai logs "$ID" --tail 200 || true
  sleep 20
done






