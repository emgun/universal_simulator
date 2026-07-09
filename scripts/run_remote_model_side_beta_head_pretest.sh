#!/usr/bin/env bash
set -euo pipefail

# Dry-run-first remote wrapper for the scoped model-side beta-head held-out
# pretest. Defaults preview commands only. Actual execution requires both
# DRY_RUN=0 and ALLOW_HELDOUT_PRETEST=1.

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

run_shell_or_echo() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY_RUN shell command:"
    print_command bash -lc "$1"
  else
    bash -lc "$1"
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

ensure_hydration_plan() {
  if [ -f "$HYDRATION_PLAN_JSON" ]; then
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY_RUN=1: would generate missing hydration plan ${HYDRATION_PLAN_JSON}"
    return 0
  fi

  echo "Hydration plan missing; generating ${HYDRATION_PLAN_JSON}"
  python scripts/plan_transport_official_hydration.py \
    --output-json "$HYDRATION_PLAN_JSON"
}

validate_pretest_plan_shape() {
  if [ ! -f "$HYDRATION_PLAN_JSON" ] && [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY_RUN=1: would validate pretest hydration plan shape"
    return 0
  fi

  python - "$HYDRATION_PLAN_JSON" "$ADVECTION_SPLIT_BLOCK_SIZE" "$ADVECTION_TRAIN_COUNT" "$ADVECTION_VAL_COUNT" "$ADVECTION_TEST_COUNT" "$ADVECTION_VAL_BLOCK_OFFSET" "$ADVECTION_TEST_BLOCK_OFFSET" <<'PY'
import json
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "samples_per_file": int(sys.argv[2]),
    "train_count": int(sys.argv[3]),
    "val_count": int(sys.argv[4]),
    "reserved_test_count": int(sys.argv[5]),
}
policy = plan.get("stratified_split_policy") or {}
expected_policy = {
    "val_block_offset": int(sys.argv[6]),
    "test_block_offset": int(sys.argv[7]),
}
blockers = []
for key, value in expected.items():
    if int(plan.get(key) or 0) != value:
        blockers.append(f"{key} expected {value}, got {plan.get(key)}")
for key, value in expected_policy.items():
    if int(policy.get(key) or 0) != value:
        blockers.append(f"stratified_split_policy.{key} expected {value}, got {policy.get(key)}")
held_out = plan.get("held_out_test_policy") or {}
if held_out.get("test_split_downloaded") is not False:
    blockers.append("hydration plan must not download official test split")
if held_out.get("test_split_sharded") is not False:
    blockers.append("hydration plan must not shard official test split")
if blockers:
    raise SystemExit("; ".join(blockers))
print("Pretest hydration plan shape validated.")
PY
}

publish_artifacts() {
  local stamp artifact_name artifact_path remote_key
  local -a paths=()
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  artifact_name=${ARTIFACT_NAME:-model_side_beta_head_pretest_${stamp}.tar.gz}
  artifact_path="/tmp/${artifact_name}"
  remote_key="${ARTIFACT_PREFIX%/}/${artifact_name}"

  for path in \
    "$OUTPUT_ROOT/$RUN_NAME" \
    "$FULL_ROOT_MANIFEST_JSON" \
    "$SEQUENTIAL_HYDRATION_JSON" \
    "$HYDRATION_VALIDATION_JSON" \
    "$OFFICIAL_SHARD_MANIFEST" \
    "$CONTRACT_JSON"; do
    [ -e "$path" ] && paths+=("$path")
  done
  if [ "${#paths[@]}" -eq 0 ]; then
    echo "No model-side beta-head pretest artifacts found to publish." >&2
    exit 1
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY_RUN=1: would publish artifacts to b2://${B2_BUCKET:-<bucket>}/${remote_key}"
    return 0
  fi
  tar -czf "$artifact_path" "${paths[@]}"
  configure_b2_rclone "publish model-side beta-head pretest artifacts"
  rclone copyto "$artifact_path" "UPSB2:${B2_BUCKET}/${remote_key}"
  echo "Published model-side beta-head pretest artifacts: b2://${B2_BUCKET}/${remote_key}"
}

heldout_command_from_contract() {
  python - "$CONTRACT_JSON" <<'PY'
import json
import sys
from pathlib import Path

contract = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
command = (contract.get("intended_held_out") or {}).get("command")
if not command:
    raise SystemExit("contract missing intended_held_out.command")
print(command)
PY
}

apply_cli_assignments "$@"

ENV_FILE=${ENV_FILE:-.env}
load_optional_env "$ENV_FILE"

WORKDIR=${WORKDIR:-$PWD}
cd "$WORKDIR"

DRY_RUN=${DRY_RUN:-1}
ALLOW_HELDOUT_PRETEST=${ALLOW_HELDOUT_PRETEST:-0}
DATA_ROOT=${DATA_ROOT:-data/pdebench}
B2_PREFIX=${B2_PREFIX:-light-v1}
STANDARD_DATA_KEYS=${STANDARD_DATA_KEYS:-burgers1d/burgers1d_val.h5,burgers1d/burgers1d_test.h5,darcy2d/darcy2d_val.h5,darcy2d/darcy2d_test.h5}
HYDRATION_PLAN_JSON=${HYDRATION_PLAN_JSON:-reports/research/sota_loop/official_advection_hydration_plan.json}
SEQUENTIAL_HYDRATION_JSON=${SEQUENTIAL_HYDRATION_JSON:-reports/research/sota_loop/model_side_transport_head_heldout_pretest/official_advection_sequential_hydration_run.json}
HYDRATION_VALIDATION_JSON=${HYDRATION_VALIDATION_JSON:-reports/research/sota_loop/model_side_transport_head_heldout_pretest/official_advection_hydration_plan_validation.json}
OFFICIAL_SOURCE_ROOT=${OFFICIAL_SOURCE_ROOT:-data/pdebench_official_advection_hydrated}
OFFICIAL_LIGHT_ROOT=${OFFICIAL_LIGHT_ROOT:-data/pdebench_official_advection_light}
OFFICIAL_SHARD_MANIFEST=${OFFICIAL_SHARD_MANIFEST:-reports/research/sota_loop/model_side_transport_head_heldout_pretest/official_advection_pretest_shards_manifest.yaml}
FULL_TASK_ROOT=${FULL_TASK_ROOT:-reports/research/sota_loop/model_side_transport_head_heldout_pretest/full_task_beta_pretest_root}
FULL_ROOT_MANIFEST_JSON=${FULL_ROOT_MANIFEST_JSON:-reports/research/sota_loop/model_side_transport_head_heldout_pretest/full_task_beta_pretest_root_manifest.json}
CHECKPOINT_SOURCE=${CHECKPOINT_SOURCE:-reports/research/sota_loop/learned_capacity_gate/ups_light_local_joint_rollout4_residual_ft_val}
CHECKPOINT_SOURCE_B2_KEY=${CHECKPOINT_SOURCE_B2_KEY:-remote-runs/checkpoints/ups_light_task_signature_trained_residual_20260526T1928Z.tar.gz}
CONTRACT_JSON=${CONTRACT_JSON:-docs/research/2026-06-25-p2-model-side-beta-head-heldout-pretest-contract.json}
MEASUREMENT_KEY=${MEASUREMENT_KEY:-9c028afbfb85328fd21fc7de4cffb277fbde274aa042ad63e6499abc562addc3}
OUTPUT_ROOT=${OUTPUT_ROOT:-reports/research/sota_loop/model_side_transport_head_heldout_pretest}
RUN_NAME=${RUN_NAME:-ups_light_p2_model_side_beta_transport_head_scoped_pretest}
PUBLISH_ARTIFACTS=${PUBLISH_ARTIFACTS:-1}
ARTIFACT_PREFIX=${ARTIFACT_PREFIX:-remote-runs/model-side-beta-head-pretest}
ADVECTION_SPLIT_BLOCK_SIZE=${ADVECTION_SPLIT_BLOCK_SIZE:-48}
ADVECTION_TRAIN_COUNT=${ADVECTION_TRAIN_COUNT:-256}
ADVECTION_VAL_COUNT=${ADVECTION_VAL_COUNT:-64}
ADVECTION_TEST_COUNT=${ADVECTION_TEST_COUNT:-64}
ADVECTION_VAL_BLOCK_OFFSET=${ADVECTION_VAL_BLOCK_OFFSET:-32}
ADVECTION_TEST_BLOCK_OFFSET=${ADVECTION_TEST_BLOCK_OFFSET:-40}

if [ "$DRY_RUN" -eq 0 ] && [ "$ALLOW_HELDOUT_PRETEST" -ne 1 ]; then
  echo "Refusing held-out pretest execution without ALLOW_HELDOUT_PRETEST=1." >&2
  exit 2
fi

mkdir -p "$DATA_ROOT" "$OUTPUT_ROOT" "$(dirname "$SEQUENTIAL_HYDRATION_JSON")"

python scripts/validate_p2_model_side_beta_head_pretest_contract.py \
  --contract-json "$CONTRACT_JSON"

echo "Hydrating standard light-v1 val/test shards into ${DATA_ROOT}"
# shellcheck disable=SC2086
run_or_echo env \
  B2_ENV_FILE="$ENV_FILE" \
  B2_PREFIX="$B2_PREFIX" \
  DATA_ROOT="$DATA_ROOT" \
  CLEAN_OLD_SPLITS=0 \
  DRY_RUN="$DRY_RUN" \
  bash scripts/fetch_datasets_b2.sh $(normalize_list "$STANDARD_DATA_KEYS")

hydrate_checkpoint_source

ensure_hydration_plan
validate_pretest_plan_shape

validation_args=(
  python scripts/validate_transport_hydration_plan.py
  --plan-json "$HYDRATION_PLAN_JSON"
  --min-download-bytes 60000000000
  --output-json "$HYDRATION_VALIDATION_JSON"
)
run_or_echo "${validation_args[@]}"

echo "Sequentially hydrating official Advection train files for reserved val/test beta-provenance shards"
sequential_args=(
  python scripts/hydrate_official_advection_source_sequential.py
  --plan-json "$HYDRATION_PLAN_JSON"
  --hydrated-source-root "$OFFICIAL_SOURCE_ROOT"
  --output-json "$SEQUENTIAL_HYDRATION_JSON"
  --overwrite
  --cleanup-raw
)
if [ "$DRY_RUN" -eq 0 ]; then
  sequential_args+=(--execute --execute-downloads)
fi
run_or_echo "${sequential_args[@]}"

advection_shard_args=(
  python scripts/make_light_hdf5_shards.py
  --root "$OFFICIAL_SOURCE_ROOT"
  --out-root "$OFFICIAL_LIGHT_ROOT"
  --tasks advection1d
  --source-split train
  --split-source val=train
  --split-source test=train
  --split-block-size "$ADVECTION_SPLIT_BLOCK_SIZE"
  --split-block-offset train=0
  --split-block-offset val="$ADVECTION_VAL_BLOCK_OFFSET"
  --split-block-offset test="$ADVECTION_TEST_BLOCK_OFFSET"
  --train-count 0
  --val-count "$ADVECTION_VAL_COUNT"
  --test-count "$ADVECTION_TEST_COUNT"
  --manifest "$OFFICIAL_SHARD_MANIFEST"
  --overwrite
)
run_or_echo "${advection_shard_args[@]}"

root_args=(
  python scripts/build_p2_model_side_beta_head_pretest_root.py
  --base-root "$DATA_ROOT"
  --advection-root "$OFFICIAL_LIGHT_ROOT"
  --out-root "$FULL_TASK_ROOT"
  --manifest-json "$FULL_ROOT_MANIFEST_JSON"
  --contract-json "$CONTRACT_JSON"
  --measurement-key "$MEASUREMENT_KEY"
  --allow-heldout-pretest-root
  --overwrite
)
run_or_echo "${root_args[@]}"

heldout_cmd="$(heldout_command_from_contract)"
echo "Running pre-registered scoped held-out command from ${CONTRACT_JSON}"
run_shell_or_echo "$heldout_cmd"

validation_summary="$OUTPUT_ROOT/$RUN_NAME/summary.json"
validation_summary_args=(
  python scripts/validate_model_side_transport_head_summary.py
  "$validation_summary"
)
run_or_echo "${validation_summary_args[@]}"

if [ "$PUBLISH_ARTIFACTS" -eq 1 ]; then
  publish_artifacts
fi

echo "Remote model-side beta-head scoped pretest route complete."
