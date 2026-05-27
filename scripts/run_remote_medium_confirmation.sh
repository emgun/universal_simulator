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

ENV_FILE=${ENV_FILE:-.env}
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
