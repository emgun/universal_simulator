#!/usr/bin/env bash
set -euo pipefail

# Validation-only rollout-stability recipe sweep at fixed tier_b capacity
# (north-star roadmap P1.3 / explore bet E1).
#
# The P1.2 capacity sweep showed single-step competence with 2-3x error
# collapse by horizon 16 and capacity saturation at ~750K params. This sweep
# holds tier_b capacity fixed (latent 64, hidden 128, depths [2,2,2], tokens
# 64) and varies the training recipe: decoded rollout training pressure,
# horizon-weighted rollout loss, semigroup consistency, and training budget.
#
# Same data contract as the capacity sweep: medium-v1 train/val only, the
# test split is never fetched, no roll-shift estimators, pure model decoded
# prediction, selection on validation only.

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
PIPELINE_ROOT=${PIPELINE_ROOT:-reports/demo/remote_recipe_sweep}
OUTPUT_ROOT=${OUTPUT_ROOT:-reports/recipe_sweep_remote}
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
RECIPES=${RECIPES:-r_rollout8,r_rollout16,r_hpower,r_semigroup,r_long,r_combo}
RUN_SWEEP=${RUN_SWEEP:-0}
FETCH_DATA=${FETCH_DATA:-1}
DRY_RUN=${DRY_RUN:-1}
ALLOW_WANDB=${ALLOW_WANDB:-1}
PUBLISH_SWEEP_ARTIFACTS=${PUBLISH_SWEEP_ARTIFACTS:-0}
SWEEP_ARTIFACT_PREFIX=${SWEEP_ARTIFACT_PREFIX:-remote-runs/recipe-sweep}
RUN_NAME_PREFIX=${RUN_NAME_PREFIX:-ups_medium_recipe}
SUMMARIZE_SWEEP=${SUMMARIZE_SWEEP:-1}
SWEEP_SUMMARY_JSON=${SWEEP_SUMMARY_JSON:-$PIPELINE_ROOT/recipe_sweep_summary.json}
SWEEP_BASELINE_JSON=${SWEEP_BASELINE_JSON:-docs/research/artifacts/p1_capacity_sweep_medium_v1_val.json}
SWEEP_CONTRACT_JSON=${SWEEP_CONTRACT_JSON:-docs/research/p1_rollout_stability_recipe_sweep_contract.json}

if [ "$EVAL_SPLIT" = "test" ]; then
  echo "Refusing EVAL_SPLIT=test: the recipe sweep is validation-only by contract." >&2
  exit 1
fi

mkdir -p "$PIPELINE_ROOT" "$OUTPUT_ROOT" "$DATA_ROOT"

# Recipe-specific overrides, newline separated. The base epoch budget matches
# the P1.2 capacity sweep (12/6/6/4) unless a recipe overrides it.
recipe_overrides() {
  case "$1" in
    r_rollout8) printf '%s\n' \
      "stages.operator_decoded.rollout_steps=8" \
      "stages.joint_codec_operator.rollout_steps=8" ;;
    r_rollout16) printf '%s\n' \
      "stages.operator_decoded.rollout_steps=16" \
      "stages.joint_codec_operator.rollout_steps=16" ;;
    r_hpower) printf '%s\n' \
      "stages.operator_decoded.rollout_steps=16" \
      "stages.joint_codec_operator.rollout_steps=16" \
      "stages.operator_decoded.rollout_loss_horizon_power=2.0" \
      "stages.joint_codec_operator.rollout_loss_horizon_power=2.0" ;;
    r_semigroup) printf '%s\n' \
      "training.lambda_semigroup=0.3" ;;
    r_long) printf '%s\n' \
      "stages.operator.epochs=36" \
      "stages.decoder.epochs=12" \
      "stages.operator_decoded.epochs=18" \
      "stages.joint_codec_operator.epochs=12" \
      "training.patience=5" ;;
    r_combo) printf '%s\n' \
      "stages.operator_decoded.rollout_steps=16" \
      "stages.joint_codec_operator.rollout_steps=16" \
      "stages.operator_decoded.rollout_loss_horizon_power=2.0" \
      "stages.joint_codec_operator.rollout_loss_horizon_power=2.0" \
      "training.lambda_semigroup=0.3" \
      "stages.operator.epochs=36" \
      "stages.decoder.epochs=12" \
      "stages.operator_decoded.epochs=18" \
      "stages.joint_codec_operator.epochs=12" \
      "training.patience=5" ;;
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

if [ "$RUN_SWEEP" -ne 1 ]; then
  echo "RUN_SWEEP=0: skipping recipe runs."
  echo "Remote recipe sweep pipeline complete."
  exit 0
fi

for recipe in $(normalize_list "$RECIPES"); do
  extra=$(recipe_overrides "$recipe") || {
    echo "Unknown recipe '${recipe}'." >&2
    exit 2
  }
  run_name="${RUN_NAME_PREFIX}_${recipe}"
  recipe_cmd=(
    python scripts/run_light_experiment.py
    --config "$TRAIN_CONFIG"
    --name "$run_name"
    --output-root "$OUTPUT_ROOT"
    --device "$DEVICE"
    --decoded
    --decoded-rollout-steps "$ROLLOUT_STEPS"
    --override "data.root=$DATA_ROOT"
    --override "data.max_samples=$TRAIN_COUNT"
    --override "latent.dim=64"
    --override "latent.tokens=64"
    --override "operator.pdet.input_dim=64"
    --override "operator.pdet.hidden_dim=128"
    --override "operator.pdet.depths=[2,2,2]"
    --override "decoder.hidden_dim=128"
    --override "training.batch_size=$BATCH_SIZE"
    --override "training.patience=$PATIENCE"
    --override "stages.operator.epochs=12"
    --override "stages.decoder.epochs=6"
    --override "stages.operator_decoded.epochs=6"
    --override "stages.joint_codec_operator.epochs=4"
    --override 'operator.conditioning.sources={"task_id":3,"equation_signature":15}'
    --eval-override "data.root=$DATA_ROOT"
    --eval-override "data.split=$EVAL_SPLIT"
    --eval-override "data.max_samples=$EVAL_COUNT"
    --promotion-rule "decoded_rollout_nrmse<=1.0"
  )
  while IFS= read -r override; do
    [ -n "$override" ] && recipe_cmd+=(--override "$override")
  done <<< "$extra"
  for stage in $(normalize_list "$STAGES"); do
    recipe_cmd+=(--stage "$stage")
  done
  if [ "$ALLOW_WANDB" -eq 1 ]; then
    recipe_cmd+=(--allow-wandb)
  fi
  echo "Recipe '${recipe}' command:"
  if ! run_or_echo "${recipe_cmd[@]}"; then
    echo "Recipe '${recipe}' FAILED; continuing with remaining recipes." >&2
    failed_recipes="${failed_recipes:-} ${recipe}"
  fi
done

if [ -n "${failed_recipes:-}" ]; then
  echo "Failed recipes:${failed_recipes}" >&2
fi

artifact_name=""
artifact_path=""
remote_key=""
artifact_handle=""
if [ "$PUBLISH_SWEEP_ARTIFACTS" -eq 1 ]; then
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  artifact_name=${SWEEP_ARTIFACT_NAME:-recipe_sweep_${VERSION}_${stamp}.tar.gz}
  artifact_path="/tmp/${artifact_name}"
  remote_key="${SWEEP_ARTIFACT_PREFIX%/}/${artifact_name}"
  artifact_handle="b2://${B2_BUCKET:-<bucket>}/${remote_key}"
fi

if [ "$SUMMARIZE_SWEEP" -eq 1 ]; then
  summary_cmd=(
    python scripts/summarize_recipe_sweep.py
    --output-root "$OUTPUT_ROOT"
    --baseline-json "$SWEEP_BASELINE_JSON"
    --contract-json "$SWEEP_CONTRACT_JSON"
    --output-json "$SWEEP_SUMMARY_JSON"
  )
  [ -n "$artifact_handle" ] && summary_cmd+=(--artifact "$artifact_handle")
  echo "Recipe sweep summary command:"
  run_or_echo "${summary_cmd[@]}"
fi

if [ "$PUBLISH_SWEEP_ARTIFACTS" -eq 1 ]; then
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY_RUN=1: would publish sweep artifacts to ${artifact_handle}"
  else
    tar -czf "$artifact_path" "$PIPELINE_ROOT" "$OUTPUT_ROOT"
    configure_b2_rclone "publish sweep artifacts"
    rclone copyto "$artifact_path" "UPSB2:${B2_BUCKET}/${remote_key}"
    echo "Published sweep artifacts: b2://${B2_BUCKET}/${remote_key}"
  fi
fi

echo "Remote recipe sweep pipeline complete."
