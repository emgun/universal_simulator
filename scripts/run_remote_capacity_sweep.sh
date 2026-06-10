#!/usr/bin/env bash
set -euo pipefail

# Validation-only operator capacity sweep on medium-v1 (north-star roadmap P1.2).
#
# Hydrates medium-v1 train/val shards only (the test split is never fetched),
# measures a persistence validation baseline, then trains and evaluates one
# candidate per capacity tier with no roll-shift estimators and the default
# decoded_persistence_residual_alpha=1.0 (pure model prediction), so the
# score reflects the learned operator alone. Selection is validation-only.
#
# Safe default: DRY_RUN=1 prints every command without running anything.
#
# Typical remote invocation (via scripts/launch_remote_medium_vast.sh):
#   REMOTE_SCRIPT=scripts/run_remote_capacity_sweep.sh \
#   EXTRA_PIPELINE_ARGS="RUN_SWEEP=1 PUBLISH_SWEEP_ARTIFACTS=1" \
#   DRY_RUN=0 bash scripts/launch_remote_medium_vast.sh

apply_cli_assignments() {
  local assignment
  for assignment in "$@"; do
    case "$assignment" in
      *=*) export "$assignment" ;;
      "") ;;
      *)
        echo "Unexpected argument '${assignment}'. Pass options as KEY=VALUE assignments." >&2
        exit 2
        ;;
    esac
  done
}

apply_cli_assignments "$@"

read_env_key() {
  local file="$1"; shift
  local key="$1"; shift || true
  if [ ! -f "$file" ]; then
    return 1
  fi
  local line
  while IFS= read -r line; do
    line="${line#${line%%[![:space:]]*}}"
    [ -z "$line" ] && continue
    [ "${line:0:1}" = "#" ] && continue
    if [[ "$line" =~ ^[[:space:]]*$key[[:space:]]*[:=][[:space:]]*(.*)$ ]]; then
      local val="${BASH_REMATCH[1]}"
      if [[ "$val" =~ ^"(.*)"$ ]]; then
        echo "${BASH_REMATCH[1]}"
      elif [[ "$val" =~ ^'(.*)'$ ]]; then
        echo "${BASH_REMATCH[1]}"
      else
        echo "$val"
      fi
      return 0
    fi
  done < "$file"
  return 1
}

load_optional_env() {
  local env_file="$1"
  [ -f "$env_file" ] || return 0
  : "${WANDB_API_KEY:=$(read_env_key "$env_file" WANDB_API_KEY || true)}"
  : "${WANDB_PROJECT:=$(read_env_key "$env_file" WANDB_PROJECT || true)}"
  : "${WANDB_ENTITY:=$(read_env_key "$env_file" WANDB_ENTITY || true)}"
  : "${B2_KEY_ID:=$(read_env_key "$env_file" B2_KEY_ID || read_env_key "$env_file" B2_ACCOUNT_ID || true)}"
  : "${B2_APP_KEY:=$(read_env_key "$env_file" B2_APP_KEY || read_env_key "$env_file" B2_APPLICATION_KEY || true)}"
  : "${B2_BUCKET:=$(read_env_key "$env_file" B2_BUCKET || read_env_key "$env_file" B2_BUCKET_NAME || true)}"
  : "${B2_S3_ENDPOINT:=$(read_env_key "$env_file" B2_S3_ENDPOINT || true)}"
  : "${B2_S3_REGION:=$(read_env_key "$env_file" B2_S3_REGION || true)}"
  export WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY B2_KEY_ID B2_APP_KEY B2_BUCKET B2_S3_ENDPOINT B2_S3_REGION
}

normalize_list() {
  echo "$1" | tr ',' ' '
}

print_command() {
  printf ' %q' "$@"
  echo
}

run_or_echo() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY_RUN command:"
    print_command "$@"
  else
    "$@"
  fi
}

ensure_rclone() {
  if command -v rclone >/dev/null 2>&1; then
    return 0
  fi
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y rclone
    return 0
  fi
  echo "rclone is required for B2 hydration and publishing." >&2
  exit 1
}

configure_b2_rclone() {
  local purpose="$1"; shift || true
  : "${B2_KEY_ID:?Set B2_KEY_ID to ${purpose}}"
  : "${B2_APP_KEY:?Set B2_APP_KEY to ${purpose}}"
  : "${B2_BUCKET:?Set B2_BUCKET to ${purpose}}"
  ensure_rclone
  if [ -n "${B2_S3_ENDPOINT:-}" ] || [ -n "${B2_S3_REGION:-}" ]; then
    export RCLONE_CONFIG_UPSB2_TYPE=s3
    export RCLONE_CONFIG_UPSB2_PROVIDER=Other
    export RCLONE_CONFIG_UPSB2_ACCESS_KEY_ID="${B2_KEY_ID}"
    export RCLONE_CONFIG_UPSB2_SECRET_ACCESS_KEY="${B2_APP_KEY}"
    [ -n "${B2_S3_ENDPOINT:-}" ] && export RCLONE_CONFIG_UPSB2_ENDPOINT="${B2_S3_ENDPOINT}"
    [ -n "${B2_S3_REGION:-}" ] && export RCLONE_CONFIG_UPSB2_REGION="${B2_S3_REGION}"
  else
    export RCLONE_CONFIG_UPSB2_TYPE=b2
    export RCLONE_CONFIG_UPSB2_ACCOUNT="${B2_KEY_ID}"
    export RCLONE_CONFIG_UPSB2_KEY="${B2_APP_KEY}"
  fi
}

ENV_FILE=${ENV_FILE:-.env}
load_optional_env "$ENV_FILE"

VERSION=${VERSION:-medium-v1}
REMOTE_PREFIX=${REMOTE_PREFIX:-$VERSION}
PIPELINE_ROOT=${PIPELINE_ROOT:-reports/demo/remote_capacity_sweep}
OUTPUT_ROOT=${OUTPUT_ROOT:-reports/capacity_sweep_remote}
DATA_ROOT=${DATA_ROOT:-data/pdebench_medium_v1_runtime}
TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_multitask_heterogeneous_light_best.yaml}
TASKS=${TASKS:-burgers1d,advection1d,darcy2d}
STAGES=${STAGES:-operator,decoder,operator_decoded,joint_codec_operator}
DEVICE=${DEVICE:-cuda}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
TRAIN_COUNT=${TRAIN_COUNT:-512}
EVAL_COUNT=${EVAL_COUNT:-128}
EVAL_SPLIT=${EVAL_SPLIT:-val}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-16}
BATCH_SIZE=${BATCH_SIZE:-16}
PATIENCE=${PATIENCE:-3}
OPERATOR_EPOCHS=${OPERATOR_EPOCHS:-12}
DECODER_EPOCHS=${DECODER_EPOCHS:-6}
OPERATOR_DECODED_EPOCHS=${OPERATOR_DECODED_EPOCHS:-6}
JOINT_EPOCHS=${JOINT_EPOCHS:-4}
TIERS=${TIERS:-current,tier_a,tier_b,tier_c,tier_d}
RUN_SWEEP=${RUN_SWEEP:-0}
RUN_PERSISTENCE=${RUN_PERSISTENCE:-1}
FETCH_DATA=${FETCH_DATA:-1}
DRY_RUN=${DRY_RUN:-1}
ALLOW_WANDB=${ALLOW_WANDB:-1}
PUBLISH_SWEEP_ARTIFACTS=${PUBLISH_SWEEP_ARTIFACTS:-0}
SWEEP_ARTIFACT_PREFIX=${SWEEP_ARTIFACT_PREFIX:-remote-runs/capacity-sweep}
RUN_NAME_PREFIX=${RUN_NAME_PREFIX:-ups_medium_capacity}
PERSISTENCE_RUN_NAME=${PERSISTENCE_RUN_NAME:-persistence_medium_v1_val}

if [ "$EVAL_SPLIT" = "test" ]; then
  echo "Refusing EVAL_SPLIT=test: the capacity sweep is validation-only by contract." >&2
  exit 1
fi

mkdir -p "$PIPELINE_ROOT" "$OUTPUT_ROOT" "$DATA_ROOT"

# Tier definitions: latent_dim hidden_dim depths tokens decoder_hidden batch.
# Larger tiers use smaller batches: decoded-rollout training graphs at 128
# tokens OOM a 24GB card at batch 16.
tier_overrides() {
  case "$1" in
    current) echo "16 32 [1,1,1] 16 32 $BATCH_SIZE" ;;
    tier_a) echo "32 64 [2,2,2] 32 64 $BATCH_SIZE" ;;
    tier_b) echo "64 128 [2,2,2] 64 128 $BATCH_SIZE" ;;
    tier_c) echo "96 256 [3,3,3] 128 256 4" ;;
    tier_d) echo "128 384 [4,4,4] 128 384 2" ;;
    *) return 1 ;;
  esac
}

FETCH_KEYS=""
for task in $(normalize_list "$TASKS"); do
  FETCH_KEYS="$FETCH_KEYS ${task}/${task}_train.h5 ${task}/${task}_val.h5"
done

if [ "$FETCH_DATA" -eq 1 ]; then
  echo "Hydrating ${VERSION} train/val shards into ${DATA_ROOT} (test split intentionally not fetched)"
  # shellcheck disable=SC2086
  run_or_echo env \
    B2_ENV_FILE="$ENV_FILE" \
    B2_PREFIX="$REMOTE_PREFIX" \
    DATA_ROOT="$DATA_ROOT" \
    CLEAN_OLD_SPLITS=0 \
    DRY_RUN="$DRY_RUN" \
    bash scripts/fetch_datasets_b2.sh $FETCH_KEYS
fi

if [ "$RUN_PERSISTENCE" -eq 1 ]; then
  persistence_cmd=(
    python scripts/run_persistence_baseline.py
    --config "$TRAIN_CONFIG"
    --name "$PERSISTENCE_RUN_NAME"
    --output-root "$OUTPUT_ROOT"
    --data-root "$DATA_ROOT"
    --split "$EVAL_SPLIT"
    --max-samples "$EVAL_COUNT"
    --rollout-steps "$ROLLOUT_STEPS"
  )
  for task in $(normalize_list "$TASKS"); do
    persistence_cmd+=(--task "$task")
  done
  echo "Persistence ${EVAL_SPLIT} baseline command:"
  run_or_echo "${persistence_cmd[@]}"
fi

if [ "$RUN_SWEEP" -ne 1 ]; then
  echo "RUN_SWEEP=0: skipping capacity tier runs."
  echo "Remote capacity sweep pipeline complete."
  exit 0
fi

for tier in $(normalize_list "$TIERS"); do
  spec=$(tier_overrides "$tier") || {
    echo "Unknown capacity tier '${tier}'." >&2
    exit 2
  }
  read -r latent_dim hidden_dim depths tokens decoder_hidden tier_batch <<< "$spec"
  run_name="${RUN_NAME_PREFIX}_${tier}"
  tier_cmd=(
    python scripts/run_light_experiment.py
    --config "$TRAIN_CONFIG"
    --name "$run_name"
    --output-root "$OUTPUT_ROOT"
    --device "$DEVICE"
    --decoded
    --decoded-rollout-steps "$ROLLOUT_STEPS"
    --override "data.root=$DATA_ROOT"
    --override "data.max_samples=$TRAIN_COUNT"
    --override "latent.dim=$latent_dim"
    --override "latent.tokens=$tokens"
    --override "operator.pdet.input_dim=$latent_dim"
    --override "operator.pdet.hidden_dim=$hidden_dim"
    --override "operator.pdet.depths=$depths"
    --override "decoder.hidden_dim=$decoder_hidden"
    --override "training.batch_size=$tier_batch"
    --override "training.patience=$PATIENCE"
    --override "stages.operator.epochs=$OPERATOR_EPOCHS"
    --override "stages.decoder.epochs=$DECODER_EPOCHS"
    --override "stages.operator_decoded.epochs=$OPERATOR_DECODED_EPOCHS"
    --override "stages.joint_codec_operator.epochs=$JOINT_EPOCHS"
    --override 'operator.conditioning.sources={"task_id":3,"equation_signature":15}'
    --eval-override "data.root=$DATA_ROOT"
    --eval-override "data.split=$EVAL_SPLIT"
    --eval-override "data.max_samples=$EVAL_COUNT"
    # decoded_persistence_residual_alpha is intentionally left at its 1.0
    # default (pure model prediction); 0.0 would score persistence itself.
    --promotion-rule "decoded_rollout_nrmse<=1.0"
  )
  for stage in $(normalize_list "$STAGES"); do
    tier_cmd+=(--stage "$stage")
  done
  if [ "$ALLOW_WANDB" -eq 1 ]; then
    tier_cmd+=(--allow-wandb)
  fi
  echo "Capacity tier '${tier}' command (latent_dim=${latent_dim}, hidden=${hidden_dim}, depths=${depths}, tokens=${tokens}, batch=${tier_batch}):"
  if ! run_or_echo "${tier_cmd[@]}"; then
    echo "Capacity tier '${tier}' FAILED; continuing with remaining tiers." >&2
    failed_tiers="${failed_tiers:-} ${tier}"
  fi
done

if [ -n "${failed_tiers:-}" ]; then
  echo "Failed tiers:${failed_tiers}" >&2
fi

if [ "$PUBLISH_SWEEP_ARTIFACTS" -eq 1 ]; then
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  artifact_name=${SWEEP_ARTIFACT_NAME:-capacity_sweep_${VERSION}_${stamp}.tar.gz}
  artifact_path="/tmp/${artifact_name}"
  remote_key="${SWEEP_ARTIFACT_PREFIX%/}/${artifact_name}"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY_RUN=1: would publish sweep artifacts to b2://${B2_BUCKET:-<bucket>}/${remote_key}"
  else
    tar -czf "$artifact_path" "$PIPELINE_ROOT" "$OUTPUT_ROOT"
    configure_b2_rclone "publish sweep artifacts"
    rclone copyto "$artifact_path" "UPSB2:${B2_BUCKET}/${remote_key}"
    echo "Published sweep artifacts: b2://${B2_BUCKET}/${remote_key}"
  fi
fi

echo "Remote capacity sweep pipeline complete."
