#!/usr/bin/env bash
set -euo pipefail

# Remote validation-only model-side beta transport-head runner.
#
# Contract:
# - Hydrate standard light-v1 validation shards from B2 for Burgers/Darcy.
# - Sequentially hydrate official Advection train files and build train/val
#   beta-provenance shards without downloading or building held-out test data.
# - Build a validation-only full-task root that replaces advection with the
#   beta-provenance shard.
# - Restore the ignored UPS checkpoint source from a small B2 checkpoint archive.
# - Run CPU-only validation with model_side_transport_head enabled, no evaluator
#   roll-shift sidecar, no held-out access, and the model-side summary validator.

apply_cli_assignments() {
  local assignment
  for assignment in "$@"; do
    case "$assignment" in
      *=*) export "$assignment" ;;
      "") ;;
      *)
        echo "Unexpected argument '${assignment}'. Pass KEY=VALUE assignments." >&2
        exit 2
        ;;
    esac
  done
}

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
  : "${B2_KEY_ID:=$(read_env_key "$env_file" B2_KEY_ID || read_env_key "$env_file" B2_ACCOUNT_ID || true)}"
  : "${B2_APP_KEY:=$(read_env_key "$env_file" B2_APP_KEY || read_env_key "$env_file" B2_APPLICATION_KEY || true)}"
  : "${B2_BUCKET:=$(read_env_key "$env_file" B2_BUCKET || read_env_key "$env_file" B2_BUCKET_NAME || true)}"
  : "${B2_S3_ENDPOINT:=$(read_env_key "$env_file" B2_S3_ENDPOINT || true)}"
  : "${B2_S3_REGION:=$(read_env_key "$env_file" B2_S3_REGION || true)}"
  export B2_KEY_ID B2_APP_KEY B2_BUCKET B2_S3_ENDPOINT B2_S3_REGION
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
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY_RUN=1: would install rclone"
    return 0
  fi
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y rclone
    return 0
  fi
  echo "rclone is required for B2 hydration/publishing." >&2
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
    export RCLONE_CONFIG_UPSB2_PROVIDER=B2
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

hydrate_checkpoint_source() {
  if [ -e "$CHECKPOINT_SOURCE" ]; then
    echo "Checkpoint source already exists: ${CHECKPOINT_SOURCE}"
    return 0
  fi
  : "${CHECKPOINT_SOURCE_B2_KEY:?Set CHECKPOINT_SOURCE_B2_KEY for the UPS checkpoint archive}"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY_RUN=1: would hydrate checkpoint source from b2://${B2_BUCKET:-<bucket>}/${CHECKPOINT_SOURCE_B2_KEY}"
    return 0
  fi

  configure_b2_rclone "hydrate checkpoint source"
  local archive_path extract_root extracted_dir
  archive_path="${CHECKPOINT_SOURCE_ARCHIVE_PATH:-/tmp/$(basename "$CHECKPOINT_SOURCE_B2_KEY")}"
  extract_root="${CHECKPOINT_EXTRACT_ROOT:-/tmp/ups_checkpoint_source}"
  rm -rf "$extract_root"
  mkdir -p "$extract_root" "$(dirname "$CHECKPOINT_SOURCE")"
  rclone copyto "UPSB2:${B2_BUCKET}/${CHECKPOINT_SOURCE_B2_KEY}" "$archive_path"
  tar -xzf "$archive_path" -C "$extract_root"
  if [ -e "$extract_root/$CHECKPOINT_SOURCE" ]; then
    mkdir -p "$(dirname "$CHECKPOINT_SOURCE")"
    cp -a "$extract_root/$CHECKPOINT_SOURCE" "$CHECKPOINT_SOURCE"
  elif [ -d "$extract_root/ups_light_task_signature_trained_residual" ]; then
    cp -a "$extract_root/ups_light_task_signature_trained_residual" "$CHECKPOINT_SOURCE"
  else
    extracted_dir=$(find "$extract_root" -mindepth 1 -maxdepth 1 -type d | head -1 || true)
    if [ -n "$extracted_dir" ]; then
      cp -a "$extracted_dir" "$CHECKPOINT_SOURCE"
    fi
  fi
  if [ ! -e "$CHECKPOINT_SOURCE" ]; then
    echo "Checkpoint archive did not produce expected source path: ${CHECKPOINT_SOURCE}" >&2
    exit 1
  fi
}

publish_artifacts() {
  local stamp artifact_name artifact_path remote_key
  local -a paths=()
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  artifact_name=${ARTIFACT_NAME:-model_side_transport_head_real_shard_${stamp}.tar.gz}
  artifact_path="/tmp/${artifact_name}"
  remote_key="${ARTIFACT_PREFIX%/}/${artifact_name}"

  for path in \
    "$OUTPUT_ROOT/$RUN_NAME" \
    "$FULL_ROOT_MANIFEST_JSON" \
    "$SEQUENTIAL_HYDRATION_JSON" \
    "$HYDRATION_VALIDATION_JSON" \
    "$HYDRATION_RUN_JSON"; do
    [ -e "$path" ] && paths+=("$path")
  done
  if [ "${#paths[@]}" -eq 0 ]; then
    echo "No model-side transport-head artifacts found to publish." >&2
    exit 1
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY_RUN=1: would publish artifacts to b2://${B2_BUCKET:-<bucket>}/${remote_key}"
    return 0
  fi
  tar -czf "$artifact_path" "${paths[@]}"
  configure_b2_rclone "publish model-side transport-head artifacts"
  rclone copyto "$artifact_path" "UPSB2:${B2_BUCKET}/${remote_key}"
  echo "Published model-side transport-head artifacts: b2://${B2_BUCKET}/${remote_key}"
}

apply_cli_assignments "$@"

ENV_FILE=${ENV_FILE:-.env}
load_optional_env "$ENV_FILE"

WORKDIR=${WORKDIR:-$PWD}
cd "$WORKDIR"

DRY_RUN=${DRY_RUN:-1}
DATA_ROOT=${DATA_ROOT:-data/pdebench}
B2_PREFIX=${B2_PREFIX:-light-v1}
STANDARD_DATA_KEYS=${STANDARD_DATA_KEYS:-burgers1d/burgers1d_val.h5,darcy2d/darcy2d_val.h5}
HYDRATION_PLAN_JSON=${HYDRATION_PLAN_JSON:-reports/research/sota_loop/official_advection_hydration_plan.json}
SEQUENTIAL_HYDRATION_JSON=${SEQUENTIAL_HYDRATION_JSON:-reports/research/sota_loop/model_side_transport_head_real_shard/official_advection_sequential_hydration_run.json}
HYDRATION_VALIDATION_JSON=${HYDRATION_VALIDATION_JSON:-reports/research/sota_loop/model_side_transport_head_real_shard/official_advection_hydration_plan_validation.json}
HYDRATION_RUN_JSON=${HYDRATION_RUN_JSON:-reports/research/sota_loop/model_side_transport_head_real_shard/official_advection_hydration_plan_run.json}
FULL_TASK_ROOT=${FULL_TASK_ROOT:-reports/research/sota_loop/model_side_transport_head_real_shard/full_task_beta_val_root}
FULL_ROOT_MANIFEST_JSON=${FULL_ROOT_MANIFEST_JSON:-reports/research/sota_loop/model_side_transport_head_real_shard/full_task_beta_val_root_manifest.json}
CHECKPOINT_SOURCE=${CHECKPOINT_SOURCE:-reports/research/sota_loop/learned_capacity_gate/ups_light_local_joint_rollout4_residual_ft_val}
CHECKPOINT_SOURCE_B2_KEY=${CHECKPOINT_SOURCE_B2_KEY:-remote-runs/checkpoints/ups_light_task_signature_trained_residual_20260526T1928Z.tar.gz}
TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_multitask_heterogeneous_light_best.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-reports/research/sota_loop/model_side_transport_head_real_shard}
RUN_NAME=${RUN_NAME:-ups_light_p2_model_side_beta_transport_head_val}
DEVICE=${DEVICE:-cpu}
PUBLISH_ARTIFACTS=${PUBLISH_ARTIFACTS:-1}
ARTIFACT_PREFIX=${ARTIFACT_PREFIX:-remote-runs/model-side-transport-head}

mkdir -p "$DATA_ROOT" "$OUTPUT_ROOT" "$(dirname "$SEQUENTIAL_HYDRATION_JSON")"

echo "Hydrating standard light-v1 validation shards into ${DATA_ROOT}"
# shellcheck disable=SC2086
run_or_echo env \
  B2_ENV_FILE="$ENV_FILE" \
  B2_PREFIX="$B2_PREFIX" \
  DATA_ROOT="$DATA_ROOT" \
  CLEAN_OLD_SPLITS=0 \
  DRY_RUN="$DRY_RUN" \
  bash scripts/fetch_datasets_b2.sh $(normalize_list "$STANDARD_DATA_KEYS")

hydrate_checkpoint_source

echo "Sequentially hydrating official Advection train files for beta-provenance validation shards"
sequential_args=(
  python scripts/hydrate_official_advection_source_sequential.py
  --plan-json "$HYDRATION_PLAN_JSON"
  --output-json "$SEQUENTIAL_HYDRATION_JSON"
  --overwrite
  --cleanup-raw
)
if [ "$DRY_RUN" -eq 0 ]; then
  sequential_args+=(--execute --execute-downloads)
fi
run_or_echo "${sequential_args[@]}"

shard_validate_args=(
  python scripts/run_transport_official_hydration_plan.py
  --plan-json "$HYDRATION_PLAN_JSON"
  --validation-json "$HYDRATION_VALIDATION_JSON"
  --min-download-bytes 60000000000
  --output-json "$HYDRATION_RUN_JSON"
  --stage shard
  --stage validate
  --stage audit
)
if [ "$DRY_RUN" -eq 0 ]; then
  shard_validate_args+=(--execute)
fi
run_or_echo "${shard_validate_args[@]}"

build_root_cmd=(
  python scripts/build_p2_parameter_full_task_root.py
  --base-root "$DATA_ROOT"
  --advection-root data/pdebench_official_advection_light
  --out-root "$FULL_TASK_ROOT"
  --manifest-json "$FULL_ROOT_MANIFEST_JSON"
  --split val
  --overwrite
)
run_or_echo "${build_root_cmd[@]}"

validation_cmd=(
  python scripts/run_light_experiment.py
  --config "$TRAIN_CONFIG"
  --name "$RUN_NAME"
  --output-root "$OUTPUT_ROOT"
  --stage operator_decoded
  --skip-training
  --checkpoint-source "$CHECKPOINT_SOURCE"
  --decoded
  --decoded-rollout-steps 16
  --device "$DEVICE"
  --override "data.root=$FULL_TASK_ROOT"
  --override data.split=val
  --override data.max_samples=32
  --override "data.param_keys=[beta]"
  --override "operator.conditioning.sources={task_id: 3, equation_signature: 15}"
  --eval-override evaluation.skip_missing_tasks=false
  --eval-override evaluation.decoded_persistence_residual_alpha=0.0
  --eval-override evaluation.report_all_horizon_metrics=true
  --eval-override "model_side_transport_head={enabled: true, tasks: [advection1d], required_params: [beta], features: [\"param:beta\", bias], init: {\"param:beta\": 10.236877359639507, bias: -0.08098891730605368}, mode: periodic_roll, apply_at: decoded_rollout, missing_param_policy: skip}"
)
run_or_echo "${validation_cmd[@]}"

summary_path="$OUTPUT_ROOT/$RUN_NAME/summary.json"
validator_cmd=(
  python scripts/validate_model_side_transport_head_summary.py
  "$summary_path"
)
run_or_echo "${validator_cmd[@]}"

if [ "$PUBLISH_ARTIFACTS" -eq 1 ]; then
  publish_artifacts
fi

echo "Remote model-side transport-head real-shard validation complete."
