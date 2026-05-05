#!/usr/bin/env bash
set -euo pipefail

# Run a matched physical-space persistence baseline against B2-hosted smoke shards.

apply_cli_assignments() {
  local assignment
  for assignment in "$@"; do
    case "$assignment" in
      *=*) export "$assignment" ;;
      "")
        ;;
      *)
        echo "Unexpected argument '${assignment}'. Pass KEY=VALUE assignments." >&2
        exit 2
        ;;
    esac
  done
}

normalize_list() {
  echo "$1" | tr ',' ' '
}

configure_artifact_rclone() {
  : "${B2_KEY_ID:?Set B2_KEY_ID to publish baseline artifacts}"
  : "${B2_APP_KEY:?Set B2_APP_KEY to publish baseline artifacts}"
  : "${B2_BUCKET:?Set B2_BUCKET to publish baseline artifacts}"
  if ! command -v rclone >/dev/null 2>&1; then
    echo "rclone is required to publish baseline artifacts." >&2
    exit 1
  fi
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

publish_artifacts() {
  local stamp artifact_name artifact_path remote_key
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  artifact_name=${BASELINE_ARTIFACT_NAME:-remote_smoke_baseline_${stamp}.tar.gz}
  artifact_path="/tmp/${artifact_name}"
  remote_key="${BASELINE_ARTIFACT_PREFIX%/}/${artifact_name}"

  tar -czf "$artifact_path" "$OUTPUT_ROOT/$RUN_NAME"
  configure_artifact_rclone
  rclone copyto "$artifact_path" "UPSB2:${B2_BUCKET}/${remote_key}"
  echo "Published baseline artifacts: b2://${B2_BUCKET}/${remote_key}"
}

apply_cli_assignments "$@"

ENV_FILE=${ENV_FILE:-.env}
DATA_ROOT=${DATA_ROOT:-data/pdebench}
OUTPUT_ROOT=${OUTPUT_ROOT:-reports/light_experiments_remote}
CONFIG=${CONFIG:-configs/train_multitask_heterogeneous_light_best.yaml}
RUN_NAME=${RUN_NAME:-persistence_smoke_v1_test}
TASKS=${TASKS:-burgers1d,advection1d,darcy2d}
SPLIT=${SPLIT:-test}
REMOTE_B2_PREFIX=${REMOTE_B2_PREFIX:-smoke-v1}
MAX_SAMPLES=${MAX_SAMPLES:-4}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-4}
PROMOTION_RULE=${PROMOTION_RULE:-decoded_rollout_nrmse<=1.0}
FETCH_DATA=${FETCH_DATA:-1}
DRY_RUN=${DRY_RUN:-0}
PUBLISH_BASELINE_ARTIFACTS=${PUBLISH_BASELINE_ARTIFACTS:-0}
BASELINE_ARTIFACT_PREFIX=${BASELINE_ARTIFACT_PREFIX:-remote-runs/smoke}

mkdir -p "$DATA_ROOT" "$OUTPUT_ROOT"

dataset_keys=()
task_args=()
for task in $(normalize_list "$TASKS"); do
  dataset_keys+=("${task}/${task}_${SPLIT}.h5")
  task_args+=(--task "$task")
done

if [ "$FETCH_DATA" -eq 1 ]; then
  B2_ENV_FILE="$ENV_FILE" B2_PREFIX="$REMOTE_B2_PREFIX" DATA_ROOT="$DATA_ROOT" \
    CLEAN_OLD_SPLITS=0 bash scripts/fetch_datasets_b2.sh "${dataset_keys[@]}"
fi

cmd=(
  python scripts/run_persistence_baseline.py
  --config "$CONFIG"
  --name "$RUN_NAME"
  --output-root "$OUTPUT_ROOT"
  --data-root "$DATA_ROOT"
  --split "$SPLIT"
  --max-samples "$MAX_SAMPLES"
  --rollout-steps "$ROLLOUT_STEPS"
  --promotion-rule "$PROMOTION_RULE"
  "${task_args[@]}"
)

echo "Persistence baseline command:"
printf ' %q' "${cmd[@]}"
echo

if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY_RUN=1: skipping persistence baseline."
  exit 0
fi

PYTHONPATH=src "${cmd[@]}"

if [ "$PUBLISH_BASELINE_ARTIFACTS" -eq 1 ]; then
  publish_artifacts
fi

echo "Remote smoke persistence baseline complete."
