#!/usr/bin/env bash
set -euo pipefail

# Bounded P2.2 Poseidon ScOT channel-lift measurement.
#
# Contract:
# - Hydrate light-v1 train/eval shards.
# - Restore official Poseidon source at a pinned commit on the remote host.
# - Run Option A channel_lift validation by default.
# - Run held-out only when ALLOW_HELD_OUT_TEST=1 and a pre-registered ledger is
#   supplied.
# - Validate and optionally publish the resulting summary/artifacts to B2.
#
# Safe default: DRY_RUN=1 prints commands without running provider/data work.

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
  echo "rclone is required for B2 hydration/publishing." >&2
  exit 1
}

ensure_git() {
  if command -v git >/dev/null 2>&1; then
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY_RUN=1: would install git for pinned Poseidon source restore"
    return 0
  fi
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y git ca-certificates
    return 0
  fi
  echo "git is required to restore Poseidon source with commit provenance." >&2
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

restore_poseidon_source() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY_RUN=1: would restore Poseidon source at ${POSEIDON_REPO} commit ${POSEIDON_COMMIT}"
    return 0
  fi
  ensure_git
  if [ ! -d "$POSEIDON_REPO/.git" ]; then
    rm -rf "$POSEIDON_REPO"
    git clone "$POSEIDON_SOURCE_URL" "$POSEIDON_REPO"
  fi
  git -C "$POSEIDON_REPO" fetch --tags origin "$POSEIDON_COMMIT" || git -C "$POSEIDON_REPO" fetch --tags origin
  git -C "$POSEIDON_REPO" checkout --detach "$POSEIDON_COMMIT"
  actual_commit=$(git -C "$POSEIDON_REPO" rev-parse HEAD)
  if [ "$actual_commit" != "$POSEIDON_COMMIT" ]; then
    echo "Poseidon source commit mismatch: got ${actual_commit}, expected ${POSEIDON_COMMIT}" >&2
    exit 1
  fi
  PYTHONPATH="$POSEIDON_REPO:${PYTHONPATH:-}" python - <<'PY'
from scOT.model import ScOT, ScOTConfig

print(f"Poseidon import OK: {ScOT.__name__}, {ScOTConfig.__name__}")
PY
}

install_poseidon_runtime_deps() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY_RUN=1: would install Poseidon runtime dependencies without replacing torch"
    return 0
  fi
  python -m pip install \
    "transformers==4.29.2" \
    "accelerate==0.31.0" \
    "huggingface_hub" \
    "safetensors" \
    "psutil"
}

validate_summary() {
  local summary_path="$1"; shift
  local expected_split="$1"; shift
  local expected_held_out="$1"; shift
  local expected_measurement_key="$1"; shift
  local ledger_json="$1"; shift
  python - "$summary_path" "$G2A_NRMSE" "$TASK_COLLAPSE_NRMSE" "$TASKS" "$expected_split" "$expected_held_out" "$expected_measurement_key" "$ledger_json" <<'PY'
import json
import math
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
gate = float(sys.argv[2])
collapse = float(sys.argv[3])
expected_tasks = {part for part in sys.argv[4].replace(",", " ").split() if part}
expected_split = sys.argv[5]
expected_held_out = sys.argv[6] == "1"
expected_measurement_key = sys.argv[7]
ledger_json = Path(sys.argv[8]) if sys.argv[8] else None
if not summary_path.exists():
    raise SystemExit(f"missing summary: {summary_path}")
data = json.loads(summary_path.read_text())
errors = []
if data.get("status") != "validation_finetune_measurement_complete":
    errors.append(f"unexpected status: {data.get('status')!r}")
if data.get("split") != expected_split:
    errors.append(f"unexpected split: {data.get('split')!r}")
if data.get("held_out_test_used") is not expected_held_out:
    errors.append(f"held_out_test_used must be {expected_held_out}")
if data.get("held_out_test_data_read") is not expected_held_out:
    errors.append(f"held_out_test_data_read must be {expected_held_out}")
details = data.get("details") or {}
if details.get("adapter_mode") != "channel_lift":
    errors.append(f"unexpected adapter mode: {details.get('adapter_mode')!r}")
model = details.get("model") or {}
if model.get("embedding_recovery_replaced") is not False:
    errors.append("embedding_recovery_replaced must be false")
contract = details.get("contract") or {}
if contract.get("pretrained_embedding_recovery_intact") is not True:
    errors.append("pretrained_embedding_recovery_intact must be true")
trainable = details.get("trainable_parameters") or {}
if int(trainable.get("trainable_parameter_count", -1)) != 13:
    errors.append(f"trainable_parameter_count must be 13, got {trainable.get('trainable_parameter_count')!r}")
metrics = data.get("metrics") or {}
score = metrics.get("decoded_rollout_nrmse")
if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
    errors.append(f"decoded_rollout_nrmse must be finite, got {score!r}")
seen_tasks = {
    str(record.get("task"))
    for record in details.get("evaluation_records", [])
    if record.get("task")
}
if seen_tasks != expected_tasks:
    errors.append(f"evaluation tasks must be {sorted(expected_tasks)}, got {sorted(seen_tasks)}")
policy = details.get("held_out_test_policy") or {}
if expected_held_out:
    if policy.get("measurement_key") != expected_measurement_key:
        errors.append("held_out_test_policy.measurement_key must match contract")
    if policy.get("recorded") is not True:
        errors.append("held_out_test_policy.recorded must be true")
    if not ledger_json or not ledger_json.exists():
        errors.append(f"missing held-out ledger: {ledger_json}")
    else:
        ledger = json.loads(ledger_json.read_text())
        matches = [
            item
            for item in ledger.get("measurements", [])
            if isinstance(item, dict)
            and item.get("measurement_key") == expected_measurement_key
        ]
        if len(matches) != 1:
            errors.append("ledger must contain exactly one matching measurement")
for key, value in sorted(metrics.items()):
    if key.startswith("task_") and key.endswith("_decoded_rollout_nrmse"):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append(f"{key} must be finite, got {value!r}")
        elif float(value) >= collapse:
            errors.append(f"{key} indicates task collapse: {value} >= {collapse}")
if errors:
    for error in errors:
        print(f"SUMMARY_VALIDATION_ERROR: {error}", file=sys.stderr)
    raise SystemExit(1)
if expected_held_out:
    decision = "held_out_positive_transfer" if float(score) <= 0.4165820594268877 else "held_out_negative_or_mixed_transfer"
else:
    decision = "cleared_g2a" if float(score) <= gate else "did_not_clear_g2a"
print(f"SUMMARY_VALIDATION_OK split={expected_split} decoded_rollout_nrmse={score} gate={gate} decision={decision}")
PY
}

publish_artifacts() {
  local stamp artifact_name artifact_path remote_key
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  artifact_name=${ARTIFACT_NAME:-poseidon_channel_lift_${VERSION}_${stamp}.tar.gz}
  artifact_path="/tmp/${artifact_name}"
  remote_key="${ARTIFACT_PREFIX%/}/${artifact_name}"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY_RUN=1: would publish artifacts to b2://${B2_BUCKET:-<bucket>}/${remote_key}"
    return 0
  fi
  tar -czf "$artifact_path" "$OUTPUT_ROOT/$RUN_NAME"
  configure_b2_rclone "publish Poseidon channel-lift artifacts"
  rclone copyto "$artifact_path" "UPSB2:${B2_BUCKET}/${remote_key}"
  echo "Published Poseidon channel-lift artifacts: b2://${B2_BUCKET}/${remote_key}"
}

apply_cli_assignments "$@"

ENV_FILE=${ENV_FILE:-.env}
load_optional_env "$ENV_FILE"

WORKDIR=${WORKDIR:-$PWD}
cd "$WORKDIR"

VERSION=${VERSION:-light-v1}
REMOTE_PREFIX=${REMOTE_PREFIX:-$VERSION}
DATA_ROOT=${DATA_ROOT:-data/pdebench}
OUTPUT_ROOT=${OUTPUT_ROOT:-reports/research/sota_loop/external_baselines}
RUN_NAME=${RUN_NAME:-poseidon_scot_channel_lift_val_light_v1_e30_lr1e2_roll4}
TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_multitask_heterogeneous_light_best.yaml}
TASKS=${TASKS:-advection1d,burgers1d,darcy2d}
TRAIN_SPLIT=${TRAIN_SPLIT:-train}
EVAL_SPLIT=${EVAL_SPLIT:-val}
ALLOW_HELD_OUT_TEST=${ALLOW_HELD_OUT_TEST:-0}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-32}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-32}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-16}
ROLLOUT_LOSS_STEPS=${ROLLOUT_LOSS_STEPS:-4}
ROLLOUT_LOSS_WEIGHT=${ROLLOUT_LOSS_WEIGHT:-1.0}
EPOCHS=${EPOCHS:-30}
LEARNING_RATE=${LEARNING_RATE:-0.01}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
BATCH_SIZE=${BATCH_SIZE:-32}
GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-1.0}
SEED=${SEED:-17}
DEVICE=${DEVICE:-cuda}
POSEIDON_MODEL_SIZE=${POSEIDON_MODEL_SIZE:-T}
CHECKPOINT_FILE=${CHECKPOINT_FILE:-model.safetensors}
EXPECTED_CHECKPOINT_SHA256=${EXPECTED_CHECKPOINT_SHA256:-e97428c93a16cbb52a41bc4794eb71be3aed436fb9cc547d9eeebb20f3940fb2}
POSEIDON_REPO=${POSEIDON_REPO:-/tmp/poseidon-official}
POSEIDON_SOURCE_URL=${POSEIDON_SOURCE_URL:-https://github.com/camlab-ethz/poseidon.git}
POSEIDON_COMMIT=${POSEIDON_COMMIT:-b8fa28f59bd7f7673323f28d11a12c6f3a215c61}
TIME_VALUE=${TIME_VALUE:-1.0}
FETCH_DATA=${FETCH_DATA:-1}
CHECK_DATA=${CHECK_DATA:-1}
DRY_RUN=${DRY_RUN:-1}
PUBLISH_ARTIFACTS=${PUBLISH_ARTIFACTS:-1}
ARTIFACT_PREFIX=${ARTIFACT_PREFIX:-remote-runs/poseidon-channel-lift}
G2A_NRMSE=${G2A_NRMSE:-0.363424243629033}
TASK_COLLAPSE_NRMSE=${TASK_COLLAPSE_NRMSE:-0.95}
HELD_OUT_LEDGER_JSON=${HELD_OUT_LEDGER_JSON:-$OUTPUT_ROOT/$RUN_NAME/test_ledger.json}
HELD_OUT_MEASUREMENT_KEY=${HELD_OUT_MEASUREMENT_KEY:-}

if [ "$TRAIN_SPLIT" = "test" ]; then
  echo "Refusing test train split: Poseidon finetuning must train on train/val only." >&2
  exit 1
fi
if [ "$EVAL_SPLIT" = "test" ] && [ "$ALLOW_HELD_OUT_TEST" -ne 1 ]; then
  echo "Refusing test eval split without ALLOW_HELD_OUT_TEST=1." >&2
  exit 1
fi
if [ "$EVAL_SPLIT" = "test" ] && [ -z "$HELD_OUT_MEASUREMENT_KEY" ]; then
  echo "Refusing held-out eval without HELD_OUT_MEASUREMENT_KEY." >&2
  exit 1
fi

mkdir -p "$DATA_ROOT" "$OUTPUT_ROOT"

FETCH_KEYS=""
for task in $(normalize_list "$TASKS"); do
  FETCH_KEYS="$FETCH_KEYS ${task}/${task}_${TRAIN_SPLIT}.h5 ${task}/${task}_${EVAL_SPLIT}.h5"
done

if [ "$FETCH_DATA" -eq 1 ]; then
  echo "Hydrating ${VERSION} ${TRAIN_SPLIT}/${EVAL_SPLIT} shards into ${DATA_ROOT}"
  # shellcheck disable=SC2086
  run_or_echo env \
    B2_ENV_FILE="$ENV_FILE" \
    B2_PREFIX="$REMOTE_PREFIX" \
    DATA_ROOT="$DATA_ROOT" \
    CLEAN_OLD_SPLITS=0 \
    DRY_RUN="$DRY_RUN" \
    bash scripts/fetch_datasets_b2.sh $FETCH_KEYS
fi

if [ "$CHECK_DATA" -eq 1 ] && [ "$DRY_RUN" -eq 0 ]; then
  missing=0
  for task in $(normalize_list "$TASKS"); do
    for split in "$TRAIN_SPLIT" "$EVAL_SPLIT"; do
      file="${DATA_ROOT}/${task}_${split}.h5"
      if [ ! -f "$file" ] && ! compgen -G "${DATA_ROOT}/${task}_${split}_*.h5" >/dev/null; then
        echo "Missing required data file: ${file}" >&2
        missing=1
      fi
    done
  done
  [ "$missing" -eq 0 ] || exit 1
fi

install_poseidon_runtime_deps
restore_poseidon_source

cmd=(
  python scripts/run_external_poseidon_scot_finetune.py
  --config "$TRAIN_CONFIG"
  --name "$RUN_NAME"
  --output-root "$OUTPUT_ROOT"
  --train-split "$TRAIN_SPLIT"
  --eval-split "$EVAL_SPLIT"
  --max-train-samples "$MAX_TRAIN_SAMPLES"
  --max-eval-samples "$MAX_EVAL_SAMPLES"
  --rollout-steps "$ROLLOUT_STEPS"
  --poseidon-model-size "$POSEIDON_MODEL_SIZE"
  --checkpoint-file "$CHECKPOINT_FILE"
  --device "$DEVICE"
  --time-value "$TIME_VALUE"
  --data-root "$DATA_ROOT"
  --poseidon-repo "$POSEIDON_REPO"
  --expected-checkpoint-sha256 "$EXPECTED_CHECKPOINT_SHA256"
  --epochs "$EPOCHS"
  --learning-rate "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --batch-size "$BATCH_SIZE"
  --grad-clip-norm "$GRAD_CLIP_NORM"
  --adapter-mode channel_lift
  --rollout-loss-steps "$ROLLOUT_LOSS_STEPS"
  --rollout-loss-weight "$ROLLOUT_LOSS_WEIGHT"
  --seed "$SEED"
  --held-out-ledger-json "$HELD_OUT_LEDGER_JSON"
)
if [ "$EVAL_SPLIT" = "test" ]; then
  cmd+=(--allow-held-out-test-eval)
fi
cmd+=(--tasks)
for task in $(normalize_list "$TASKS"); do
  cmd+=("$task")
done

echo "Poseidon channel-lift command:"
print_command "${cmd[@]}"

summary_path="$OUTPUT_ROOT/$RUN_NAME/summary.json"
set +e
run_or_echo "${cmd[@]}"
run_status=$?
set -e
if [ "$run_status" -ne 0 ]; then
  echo "Poseidon channel-lift run failed with status ${run_status}" >&2
fi

validation_status=0
if [ "$DRY_RUN" -eq 0 ] && [ -f "$summary_path" ]; then
  set +e
  expected_held_out=0
  if [ "$EVAL_SPLIT" = "test" ]; then
    expected_held_out=1
  fi
  validate_summary "$summary_path" "$EVAL_SPLIT" "$expected_held_out" "$HELD_OUT_MEASUREMENT_KEY" "$HELD_OUT_LEDGER_JSON"
  validation_status=$?
  set -e
elif [ "$DRY_RUN" -eq 0 ]; then
  echo "No summary produced at ${summary_path}" >&2
  validation_status=1
fi

publish_status=0
if [ "$PUBLISH_ARTIFACTS" -eq 1 ]; then
  set +e
  publish_artifacts
  publish_status=$?
  set -e
fi

if [ "$run_status" -ne 0 ]; then
  exit "$run_status"
fi
if [ "$validation_status" -ne 0 ]; then
  exit "$validation_status"
fi
if [ "$publish_status" -ne 0 ]; then
  exit "$publish_status"
fi

echo "Remote Poseidon channel-lift validation complete."
