#!/usr/bin/env bash
set -euo pipefail

# Orchestrate the medium-v1 confirmation gate for the current UPS candidate.
#
# Defaults are dry-run and do not launch training or contact B2. Set DRY_RUN=0
# and RUN_CANDIDATE=1 / RUN_PERSISTENCE=1 on a remote box after reviewing the
# printed commands.

apply_cli_assignments() {
  local assignment
  for assignment in "$@"; do
    case "$assignment" in
      *=*) export "$assignment" ;;
      "")
        ;;
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
  : "${WANDB_GROUP:=$(read_env_key "$env_file" WANDB_GROUP || true)}"
  : "${WANDB_TAGS:=$(read_env_key "$env_file" WANDB_TAGS || true)}"
  : "${WANDB_JOB_TYPE:=$(read_env_key "$env_file" WANDB_JOB_TYPE || true)}"
  : "${B2_KEY_ID:=$(read_env_key "$env_file" B2_KEY_ID || read_env_key "$env_file" B2_ACCOUNT_ID || true)}"
  : "${B2_APP_KEY:=$(read_env_key "$env_file" B2_APP_KEY || read_env_key "$env_file" B2_APPLICATION_KEY || true)}"
  : "${B2_BUCKET:=$(read_env_key "$env_file" B2_BUCKET || read_env_key "$env_file" B2_BUCKET_NAME || true)}"
  : "${B2_S3_ENDPOINT:=$(read_env_key "$env_file" B2_S3_ENDPOINT || true)}"
  : "${B2_S3_REGION:=$(read_env_key "$env_file" B2_S3_REGION || true)}"
  export WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY WANDB_GROUP WANDB_TAGS WANDB_JOB_TYPE B2_KEY_ID B2_APP_KEY B2_BUCKET B2_S3_ENDPOINT B2_S3_REGION
}

normalize_list() {
  echo "$1" | tr ',' ' '
}

append_unique_word() {
  local var_name="$1"; shift
  local value="$1"; shift || true
  [ -n "$value" ] || return 0
  local current="${!var_name:-}"
  local existing
  for existing in $current; do
    [ "$existing" = "$value" ] && return 0
  done
  if [ -n "$current" ]; then
    printf -v "$var_name" "%s %s" "$current" "$value"
  else
    printf -v "$var_name" "%s" "$value"
  fi
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
  if [ "${INSTALL_RCLONE:-1}" -ne 1 ]; then
    echo "rclone is required for B2 hydration; set INSTALL_RCLONE=1 or preinstall rclone." >&2
    exit 1
  fi
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y rclone
    return 0
  fi
  echo "rclone is required for B2 hydration and automatic install is only implemented for apt-get hosts." >&2
  exit 1
}

configure_checkpoint_rclone() {
  : "${B2_KEY_ID:?Set B2_KEY_ID to hydrate checkpoint source}"
  : "${B2_APP_KEY:?Set B2_APP_KEY to hydrate checkpoint source}"
  : "${B2_BUCKET:?Set B2_BUCKET to hydrate checkpoint source}"
  if ! command -v rclone >/dev/null 2>&1; then
    echo "rclone is required to hydrate checkpoint source." >&2
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

hydrate_checkpoint_source() {
  [ -n "$CHECKPOINT_SOURCE" ] || return 0
  [ -n "$CHECKPOINT_SOURCE_B2_KEY" ] || return 0
  if [ -e "$CHECKPOINT_SOURCE" ]; then
    echo "Checkpoint source already exists: ${CHECKPOINT_SOURCE}"
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY_RUN=1: would hydrate checkpoint source from b2://${B2_BUCKET:-<bucket>}/${CHECKPOINT_SOURCE_B2_KEY}"
    return 0
  fi

  ensure_rclone
  configure_checkpoint_rclone
  local archive_path extract_root
  archive_path="${CHECKPOINT_SOURCE_ARCHIVE_PATH:-/tmp/$(basename "$CHECKPOINT_SOURCE_B2_KEY")}"
  extract_root="$(dirname "$CHECKPOINT_SOURCE")"
  mkdir -p "$extract_root"
  echo "Hydrating checkpoint source from b2://${B2_BUCKET}/${CHECKPOINT_SOURCE_B2_KEY}"
  rclone copyto "UPSB2:${B2_BUCKET}/${CHECKPOINT_SOURCE_B2_KEY}" "$archive_path"
  tar -xzf "$archive_path" -C "$extract_root"
  if [ ! -e "$CHECKPOINT_SOURCE" ]; then
    echo "Checkpoint archive did not produce expected source path: ${CHECKPOINT_SOURCE}" >&2
    exit 1
  fi
}

ENV_FILE=${ENV_FILE:-.env}
load_optional_env "$ENV_FILE"

VERSION=${VERSION:-medium-v1}
REMOTE_PREFIX=${REMOTE_PREFIX:-$VERSION}
MEDIUM_MANIFEST=${MEDIUM_MANIFEST:-docs/demo_medium_v1_data_manifest.yaml}
PIPELINE_ROOT=${PIPELINE_ROOT:-reports/demo/remote_medium_pipeline}
OUTPUT_ROOT=${OUTPUT_ROOT:-reports/medium_experiments_remote}
DATA_ROOT=${DATA_ROOT:-data/pdebench_medium_v1_runtime}
TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_multitask_heterogeneous_light_best.yaml}
TASKS=${TASKS:-burgers1d,advection1d,darcy2d}
STAGES=${STAGES:-operator,decoder,operator_decoded,joint_codec_operator}
DEVICE=${DEVICE:-cuda}
TRAIN_COUNT=${TRAIN_COUNT:-512}
EVAL_COUNT=${EVAL_COUNT:-128}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-32}
REQUIRED_GB=${REQUIRED_GB:-40}
SHARD_PREP_REQUIRED_GB=${SHARD_PREP_REQUIRED_GB:-0}
CHECK_B2=${CHECK_B2:-1}
PREP_SHARDS=${PREP_SHARDS:-1}
FETCH_DATA=${FETCH_DATA:-1}
RUN_CANDIDATE=${RUN_CANDIDATE:-0}
RUN_PERSISTENCE=${RUN_PERSISTENCE:-0}
DRY_RUN=${DRY_RUN:-1}
ALLOW_UNCHECKED_LIVE_RUNS=${ALLOW_UNCHECKED_LIVE_RUNS:-0}
ALLOW_WANDB=${ALLOW_WANDB:-0}
SKIP_TRAINING=${SKIP_TRAINING:-0}
CHECKPOINT_SOURCE=${CHECKPOINT_SOURCE:-}
CHECKPOINT_SOURCE_B2_KEY=${CHECKPOINT_SOURCE_B2_KEY:-}
CANDIDATE_RUN_NAME=${CANDIDATE_RUN_NAME:-ups_medium_shared_context_transport}
PERSISTENCE_RUN_NAME=${PERSISTENCE_RUN_NAME:-persistence_medium_v1_test}
CANDIDATE_PROMOTION_RULE=${CANDIDATE_PROMOTION_RULE:-decoded_rollout_nrmse<=1.0}

if [ -z "${CONTEXT_ESTIMATOR:-}" ]; then
  CONTEXT_ESTIMATOR='evaluation.decoded_context_roll_shift_estimator={candidate_shifts: [-4, -2, -1, 0, 1, 2, 4, 8, 16, 24, 32, 40, 48, 64], context_transitions: 8, coefficients: {slope: 0.9974352988185539, intercept: 0.0}, families: [transport, conservation], mode: roll_persistence, calibration_scope: shared_1d_transport}'
fi

mkdir -p "$PIPELINE_ROOT" "$OUTPUT_ROOT" "$DATA_ROOT"

readiness_json="$PIPELINE_ROOT/medium_readiness_before.json"
readiness_after_json="$PIPELINE_ROOT/medium_readiness_after.json"
prep_log="$PIPELINE_ROOT/medium_shard_prep.log"

shards_ready=0
echo "Checking ${VERSION} shard readiness..."
if [ "$CHECK_B2" -eq 1 ] && [ -f "$MEDIUM_MANIFEST" ]; then
  if python scripts/check_demo_b2_shards.py \
    --manifest "$MEDIUM_MANIFEST" \
    --env-file "$ENV_FILE" \
    --json "$readiness_json"; then
    shards_ready=1
    echo "${VERSION} shards are already ready."
  else
    echo "${VERSION} shards are not ready."
  fi
elif [ "$CHECK_B2" -eq 1 ]; then
  echo "Medium manifest missing: ${MEDIUM_MANIFEST}"
else
  echo "CHECK_B2=0: skipping B2 shard readiness check."
fi

if [ "$shards_ready" -ne 1 ] && [ "$PREP_SHARDS" -eq 1 ]; then
  echo "Running ${VERSION} shard prep. Log: ${prep_log}"
  echo "Medium shard prep assignments: VERSION=${VERSION} REMOTE_PREFIX=${REMOTE_PREFIX} REMOTE_B2_PREFIX=${REMOTE_PREFIX} TRAIN_COUNT=${TRAIN_COUNT} VAL_COUNT=${EVAL_COUNT} TEST_COUNT=${EVAL_COUNT}"
  (
    DRY_RUN="$DRY_RUN" \
      ENV_FILE="$ENV_FILE" \
      VERSION="$VERSION" \
      REMOTE_PREFIX="$REMOTE_PREFIX" \
      TRAIN_COUNT="$TRAIN_COUNT" \
      VAL_COUNT="$EVAL_COUNT" \
      TEST_COUNT="$EVAL_COUNT" \
      TASKS="$TASKS" \
      REQUIRED_GB="$SHARD_PREP_REQUIRED_GB" \
      bash scripts/run_remote_shard_prep_b2.sh
  ) 2>&1 | tee "$prep_log"
  if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_B2" -eq 1 ]; then
    echo "Re-checking ${VERSION} shard readiness after prep..."
    if python scripts/check_demo_b2_shards.py \
      --manifest "$MEDIUM_MANIFEST" \
      --env-file "$ENV_FILE" \
      --json "$readiness_after_json"; then
      shards_ready=1
    fi
  fi
fi

if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_B2" -eq 1 ] && [ "$shards_ready" -ne 1 ]; then
  echo "Refusing live medium runs because ${VERSION} shards are not ready." >&2
  exit 1
fi

if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_B2" -ne 1 ] && [ "$ALLOW_UNCHECKED_LIVE_RUNS" -ne 1 ]; then
  echo "Refusing live medium runs without CHECK_B2=1. Set ALLOW_UNCHECKED_LIVE_RUNS=1 only for controlled test environments." >&2
  exit 1
fi

FETCH_KEYS=""
append_unique_word FETCH_KEYS "burgers1d/burgers1d_train.h5"
append_unique_word FETCH_KEYS "advection1d/advection1d_train.h5"
append_unique_word FETCH_KEYS "darcy2d/darcy2d_train.h5"
append_unique_word FETCH_KEYS "burgers1d/burgers1d_test.h5"
append_unique_word FETCH_KEYS "advection1d/advection1d_test.h5"
append_unique_word FETCH_KEYS "darcy2d/darcy2d_test.h5"

if { [ "$RUN_CANDIDATE" -eq 1 ] || [ "$RUN_PERSISTENCE" -eq 1 ]; } && [ "$FETCH_DATA" -eq 1 ]; then
  echo "Hydrating ${VERSION} train/test shards into ${DATA_ROOT}"
  # shellcheck disable=SC2086
  run_or_echo env \
    B2_ENV_FILE="$ENV_FILE" \
    B2_PREFIX="$REMOTE_PREFIX" \
    DATA_ROOT="$DATA_ROOT" \
    CLEAN_OLD_SPLITS=0 \
    DRY_RUN="$DRY_RUN" \
    bash scripts/fetch_datasets_b2.sh $FETCH_KEYS
fi

if [ "$RUN_CANDIDATE" -eq 1 ]; then
  hydrate_checkpoint_source
fi

candidate_cmd=(
  python scripts/run_light_experiment.py
  --config "$TRAIN_CONFIG"
  --name "$CANDIDATE_RUN_NAME"
  --output-root "$OUTPUT_ROOT"
  --device "$DEVICE"
  --decoded
  --decoded-rollout-steps "$ROLLOUT_STEPS"
  --override "data.root=$DATA_ROOT"
  --override "data.max_samples=$TRAIN_COUNT"
  --override 'operator.conditioning.sources={"task_id":3,"equation_signature":15}'
  --eval-override "data.root=$DATA_ROOT"
  --eval-override "data.split=test"
  --eval-override "data.max_samples=$EVAL_COUNT"
  --eval-override "evaluation.decoded_persistence_residual_alpha=0.0"
  --eval-override "$CONTEXT_ESTIMATOR"
  --promotion-rule "$CANDIDATE_PROMOTION_RULE"
)

for stage in $(normalize_list "$STAGES"); do
  candidate_cmd+=(--stage "$stage")
done

if [ "$SKIP_TRAINING" -eq 1 ]; then
  candidate_cmd+=(--skip-training)
fi

if [ -n "$CHECKPOINT_SOURCE" ]; then
  candidate_cmd+=(--checkpoint-source "$CHECKPOINT_SOURCE")
fi

if [ "$ALLOW_WANDB" -eq 1 ]; then
  candidate_cmd+=(--allow-wandb)
fi

persistence_cmd=(
  python scripts/run_persistence_baseline.py
  --config "$TRAIN_CONFIG"
  --name "$PERSISTENCE_RUN_NAME"
  --output-root "$OUTPUT_ROOT"
  --data-root "$DATA_ROOT"
  --split test
  --max-samples "$EVAL_COUNT"
  --rollout-steps "$ROLLOUT_STEPS"
)

for task in $(normalize_list "$TASKS"); do
  persistence_cmd+=(--task "$task")
done

if [ "$RUN_CANDIDATE" -eq 1 ]; then
  echo "Medium candidate command:"
  echo "Medium candidate assignments: RUN_NAME=${CANDIDATE_RUN_NAME}"
  run_or_echo "${candidate_cmd[@]}"
else
  echo "RUN_CANDIDATE=0: skipping medium candidate command."
fi

if [ "$RUN_PERSISTENCE" -eq 1 ]; then
  echo "Medium persistence baseline command:"
  run_or_echo "${persistence_cmd[@]}"
else
  echo "RUN_PERSISTENCE=0: skipping medium persistence baseline command."
fi

echo "Remote medium confirmation pipeline complete."
