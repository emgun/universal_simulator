#!/usr/bin/env bash
set -euo pipefail

# Bounded P2 DPOT Tiny channel-lift validation.
#
# Contract:
# - Hydrate light-v1 train/validation shards.
# - Restore official DPOT source at a pinned commit on the remote host.
# - Download only the pinned Tiny checkpoint and verify its SHA256 in runner.
# - Run validation-only DPOT channel_lift; held-out test is forbidden.
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
    echo "DRY_RUN=1: would install git for pinned DPOT source restore"
    return 0
  fi
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y git ca-certificates
    return 0
  fi
  echo "git is required to restore DPOT source with commit provenance." >&2
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

install_dpot_runtime_deps() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY_RUN=1: would install DPOT runtime dependencies without replacing torch"
    return 0
  fi
  python -m pip install "einops>=0.7.0" "huggingface_hub" "psutil"
}

restore_dpot_source() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY_RUN=1: would restore DPOT source at ${DPOT_REPO} commit ${DPOT_COMMIT}"
    return 0
  fi
  ensure_git
  if [ ! -d "$DPOT_REPO/.git" ]; then
    rm -rf "$DPOT_REPO"
    git clone "$DPOT_SOURCE_URL" "$DPOT_REPO"
  fi
  git -C "$DPOT_REPO" fetch --tags origin "$DPOT_COMMIT" || git -C "$DPOT_REPO" fetch --tags origin
  git -C "$DPOT_REPO" checkout --detach "$DPOT_COMMIT"
  actual_commit=$(git -C "$DPOT_REPO" rev-parse HEAD)
  if [ "$actual_commit" != "$DPOT_COMMIT" ]; then
    echo "DPOT source commit mismatch: got ${actual_commit}, expected ${DPOT_COMMIT}" >&2
    exit 1
  fi
  PYTHONPATH="$DPOT_REPO:${PYTHONPATH:-}" python - <<'PY'
from models.dpot import DPOTNet

print(f"DPOT import OK: {DPOTNet.__name__}")
PY
}

download_dpot_checkpoint() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY_RUN=1: would download ${CHECKPOINT_FILE} from ${DPOT_CHECKPOINT_REPO}"
    return 0
  fi
  python - "$DPOT_REPO" "$DPOT_CHECKPOINT_REPO" "$CHECKPOINT_FILE" <<'PY'
import sys
from huggingface_hub import hf_hub_download

dpot_repo, checkpoint_repo, filename = sys.argv[1:4]
path = hf_hub_download(repo_id=checkpoint_repo, filename=filename, local_dir=dpot_repo)
print(path)
PY
}

validate_summary() {
  local summary_path="$1"; shift
  python - "$summary_path" "$GATE_AGGREGATE_NRMSE" "$GATE_ADVECTION_NRMSE" "$TASK_COLLAPSE_NRMSE" "$TASKS" <<'PY'
import json
import math
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
gate_aggregate = float(sys.argv[2])
gate_advection = float(sys.argv[3])
collapse = float(sys.argv[4])
expected_tasks = {part for part in sys.argv[5].replace(",", " ").split() if part}
if not summary_path.exists():
    raise SystemExit(f"missing summary: {summary_path}")
data = json.loads(summary_path.read_text())
errors = []
if data.get("status") != "validation_finetune_measurement_complete":
    errors.append(f"unexpected status: {data.get('status')!r}")
if data.get("measurement_type") != "dpot_finetune_validation_measurement":
    errors.append(f"unexpected measurement_type: {data.get('measurement_type')!r}")
if data.get("train_split") != "train":
    errors.append(f"unexpected train_split: {data.get('train_split')!r}")
if data.get("split") != "val":
    errors.append(f"unexpected split: {data.get('split')!r}")
if data.get("held_out_test_used") is not False:
    errors.append("held_out_test_used must be false")
if data.get("held_out_test_data_read") is not False:
    errors.append("held_out_test_data_read must be false")
if data.get("claim_comparable") is not False:
    errors.append("claim_comparable must be false")
details = data.get("details") or {}
if details.get("adapter_mode") != "channel_lift":
    errors.append(f"unexpected adapter mode: {details.get('adapter_mode')!r}")
if int((details.get("trainable_parameters") or {}).get("trainable_parameter_count", -1)) != 13:
    errors.append("trainable_parameter_count must be 13")
if details.get("history_steps") != 10:
    errors.append("history_steps must be 10")
if details.get("history_init") != "repeat_current":
    errors.append("history_init must be repeat_current")
seen_tasks = {
    str(record.get("task"))
    for record in details.get("evaluation_records", [])
    if record.get("task")
}
if seen_tasks != expected_tasks:
    errors.append(f"evaluation tasks must be {sorted(expected_tasks)}, got {sorted(seen_tasks)}")
metrics = data.get("metrics") or {}
score = metrics.get("decoded_rollout_nrmse")
if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
    errors.append(f"decoded_rollout_nrmse must be finite, got {score!r}")
advection = metrics.get("task_advection1d_decoded_rollout_nrmse")
if not isinstance(advection, (int, float)) or not math.isfinite(float(advection)):
    errors.append(f"advection decoded rollout NRMSE must be finite, got {advection!r}")
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
misses = []
if float(score) > gate_aggregate:
    misses.append("aggregate")
if float(advection) > gate_advection:
    misses.append("advection")
decision = "cleared_dpot_validation_gate" if not misses else "missed_" + "_and_".join(misses) + "_gate"
print(
    "SUMMARY_VALIDATION_OK "
    f"decoded_rollout_nrmse={score} "
    f"aggregate_gate={gate_aggregate} "
    f"advection={advection} "
    f"advection_gate={gate_advection} "
    f"decision={decision}"
)
PY
}

publish_artifacts() {
  local stamp artifact_name artifact_path remote_key
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  artifact_name=${ARTIFACT_NAME:-dpot_channel_lift_${VERSION}_${stamp}.tar.gz}
  artifact_path="/tmp/${artifact_name}"
  remote_key="${ARTIFACT_PREFIX%/}/${artifact_name}"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY_RUN=1: would publish artifacts to b2://${B2_BUCKET:-<bucket>}/${remote_key}"
    return 0
  fi
  if [ ! -d "$OUTPUT_ROOT/$RUN_NAME" ]; then
    echo "Refusing to publish missing artifact directory: $OUTPUT_ROOT/$RUN_NAME" >&2
    return 1
  fi
  tar -czf "$artifact_path" "$OUTPUT_ROOT/$RUN_NAME" || return 1
  configure_b2_rclone "publish DPOT channel-lift artifacts"
  rclone copyto "$artifact_path" "UPSB2:${B2_BUCKET}/${remote_key}" || return 1
  echo "Published DPOT channel-lift artifacts: b2://${B2_BUCKET}/${remote_key}"
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
RUN_NAME=${RUN_NAME:-dpot_tiny_channel_lift_val_light_v1_e30_lr1e2_roll4}
TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_multitask_heterogeneous_light_best.yaml}
TASKS=${TASKS:-advection1d,burgers1d,darcy2d}
TRAIN_SPLIT=${TRAIN_SPLIT:-train}
EVAL_SPLIT=${EVAL_SPLIT:-val}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-0}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-0}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-4}
EPOCHS=${EPOCHS:-30}
LEARNING_RATE=${LEARNING_RATE:-0.01}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-1.0}
SEED=${SEED:-17}
DEVICE=${DEVICE:-cuda}
IMAGE_SIZE=${IMAGE_SIZE:-128}
HISTORY_STEPS=${HISTORY_STEPS:-10}
HISTORY_INIT=${HISTORY_INIT:-repeat_current}
CHECKPOINT_FILE=${CHECKPOINT_FILE:-model_Ti.pth}
EXPECTED_CHECKPOINT_SHA256=${EXPECTED_CHECKPOINT_SHA256:-074c337f9b3a3c70253f8022ce6be7e7dfb809a91a7b00e46fbfedf9611d767f}
DPOT_REPO=${DPOT_REPO:-/tmp/dpot-official}
DPOT_SOURCE_URL=${DPOT_SOURCE_URL:-https://github.com/HaoZhongkai/DPOT.git}
DPOT_COMMIT=${DPOT_COMMIT:-dcd2f9a9359765e19ad63e2f3f879a2a8ce1aa17}
DPOT_CHECKPOINT_REPO=${DPOT_CHECKPOINT_REPO:-hzk17/DPOT}
FETCH_DATA=${FETCH_DATA:-1}
CHECK_DATA=${CHECK_DATA:-1}
DRY_RUN=${DRY_RUN:-1}
PUBLISH_ARTIFACTS=${PUBLISH_ARTIFACTS:-1}
ARTIFACT_PREFIX=${ARTIFACT_PREFIX:-remote-runs/dpot-channel-lift}
GATE_AGGREGATE_NRMSE=${GATE_AGGREGATE_NRMSE:-0.363424243629033}
GATE_ADVECTION_NRMSE=${GATE_ADVECTION_NRMSE:-0.4866576789288726}
TASK_COLLAPSE_NRMSE=${TASK_COLLAPSE_NRMSE:-0.95}

if [ "$TRAIN_SPLIT" = "test" ] || [ "$EVAL_SPLIT" = "test" ]; then
  echo "Refusing held-out test split for DPOT validation-only plan." >&2
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

install_dpot_runtime_deps
restore_dpot_source
download_dpot_checkpoint

cmd=(
  python scripts/run_external_dpot_finetune.py
  --config "$TRAIN_CONFIG"
  --name "$RUN_NAME"
  --output-root "$OUTPUT_ROOT"
  --train-split "$TRAIN_SPLIT"
  --eval-split "$EVAL_SPLIT"
  --max-train-samples "$MAX_TRAIN_SAMPLES"
  --max-eval-samples "$MAX_EVAL_SAMPLES"
  --rollout-steps "$ROLLOUT_STEPS"
  --dpot-repo "$DPOT_REPO"
  --dpot-source-commit "$DPOT_COMMIT"
  --checkpoint-file "$CHECKPOINT_FILE"
  --expected-checkpoint-sha256 "$EXPECTED_CHECKPOINT_SHA256"
  --device "$DEVICE"
  --data-root "$DATA_ROOT"
  --epochs "$EPOCHS"
  --learning-rate "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --batch-size "$BATCH_SIZE"
  --grad-clip-norm "$GRAD_CLIP_NORM"
  --adapter-mode channel_lift
  --history-steps "$HISTORY_STEPS"
  --history-init "$HISTORY_INIT"
  --image-size "$IMAGE_SIZE"
  --seed "$SEED"
)
cmd+=(--tasks)
for task in $(normalize_list "$TASKS"); do
  cmd+=("$task")
done

echo "DPOT channel-lift command:"
print_command "${cmd[@]}"

summary_path="$OUTPUT_ROOT/$RUN_NAME/summary.json"
set +e
run_or_echo "${cmd[@]}"
run_status=$?
set -e
if [ "$run_status" -ne 0 ]; then
  echo "DPOT channel-lift run failed with status ${run_status}" >&2
fi

validation_status=0
if [ "$DRY_RUN" -eq 0 ] && [ -f "$summary_path" ]; then
  set +e
  validate_summary "$summary_path"
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

echo "Remote DPOT channel-lift validation complete."
