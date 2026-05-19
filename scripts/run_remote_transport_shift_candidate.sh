#!/usr/bin/env bash
set -euo pipefail

# Remote/full-source pipeline for the benchmark-clean constant transport-shift gate.
#
# This is intended for a remote/data-prep box, not a local laptop. It hydrates
# full Advection train/val/test shards from B2, scans train windows for a target
# shift, builds a small candidate light shard with the selected train window and
# native held-out val/test starts, then runs scripts/run_transport_shift_gate.py.
#
# Safe local planning run:
#   DRY_RUN=1 ENV_FILE=/path/to/.env bash scripts/run_remote_transport_shift_candidate.sh
#
# Actual remote run:
#   DRY_RUN=0 ENV_FILE=.env bash scripts/run_remote_transport_shift_candidate.sh

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

apply_cli_assignments "$@"

ENV_FILE=${ENV_FILE:-.env}
load_optional_env "$ENV_FILE"

WORKDIR=${WORKDIR:-$PWD}
cd "$WORKDIR"

DRY_RUN=${DRY_RUN:-1}
TASK=${TASK:-advection1d}
DATA_ROOT=${DATA_ROOT:-/workspace/pdebench_full}
CANDIDATE_ROOT=${CANDIDATE_ROOT:-/workspace/pdebench_transport_candidate}
OUTPUT_ROOT=${OUTPUT_ROOT:-reports/research/sota_loop/remote_transport_shift_candidate}
REMOTE_B2_PREFIX=${REMOTE_B2_PREFIX:-full}
VERSION=${VERSION:-transport-shift-candidate-v1}
REMOTE_PREFIX=${REMOTE_PREFIX:-}
TRAIN_COUNT=${TRAIN_COUNT:-128}
VAL_COUNT=${VAL_COUNT:-32}
TEST_COUNT=${TEST_COUNT:-32}
VAL_START_INDEX=${VAL_START_INDEX:-0}
TEST_START_INDEX=${TEST_START_INDEX:-0}
WINDOW_SIZE=${WINDOW_SIZE:-32}
WINDOW_STRIDE=${WINDOW_STRIDE:-32}
SCAN_MAX_WINDOWS=${SCAN_MAX_WINDOWS:-}
TARGET_SHIFT=${TARGET_SHIFT:-40}
SCAN_ALL_SPLITS=${SCAN_ALL_SPLITS:-0}
REQUIRE_TEST_COMPATIBLE=${REQUIRE_TEST_COMPATIBLE:-0}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-16}
REFERENCE_METRIC_VALUE=${REFERENCE_METRIC_VALUE:-0.30780652221851373}
VAL_MIN_RELATIVE_IMPROVEMENT=${VAL_MIN_RELATIVE_IMPROVEMENT:-0.0}
REQUIRED_GB=${REQUIRED_GB:-80}
PUBLISH_CANDIDATE=${PUBLISH_CANDIDATE:-0}
AUDIT_REQUIRE_STATUS=${AUDIT_REQUIRE_STATUS:-achieved}

SHIFTS=${SHIFTS:--96,-88,-80,-72,-64,-56,-48,-40,-32,-24,-16,-8,0,8,16,24,32,40,48,56,64,72,80,88,96}

shift_args=()
IFS=',' read -r -a shift_values <<< "$SHIFTS"
for shift in "${shift_values[@]}"; do
  [ -n "$shift" ] && shift_args+=(--shift "$shift")
done

if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY_RUN: hydrate full ${TASK} train/val/test from B2 prefix ${REMOTE_B2_PREFIX} into ${DATA_ROOT}"
  if [ "$SCAN_ALL_SPLITS" -eq 1 ]; then
    echo "DRY_RUN: scan ${TASK}_{train,val,test}.h5 and select compatible native windows"
  else
    echo "DRY_RUN: scan ${TASK}_train.h5 with window_size=${WINDOW_SIZE}, stride=${WINDOW_STRIDE}, target_shift=${TARGET_SHIFT}"
  fi
  echo "DRY_RUN: build ${CANDIDATE_ROOT} with selected train start and held-out val/test starts"
  echo "DRY_RUN: run train/val gate and one held-out test only if validation passes"
  if [ "$SCAN_ALL_SPLITS" -eq 1 ]; then
    echo "DRY_RUN: audit final evidence with require_status=${AUDIT_REQUIRE_STATUS}"
  fi
  exit 0
fi

mkdir -p "$DATA_ROOT" "$CANDIDATE_ROOT" "$OUTPUT_ROOT"

if [ "$REQUIRED_GB" -gt 0 ]; then
  avail_gb=$(df -Pm "$DATA_ROOT" | awk 'NR==2{print int($4/1024)}')
  if [ "$avail_gb" -lt "$REQUIRED_GB" ]; then
    echo "Insufficient free disk: have ${avail_gb}GB, require ${REQUIRED_GB}GB at ${DATA_ROOT}." >&2
    exit 1
  fi
fi

ensure_rclone

B2_ENV_FILE="$ENV_FILE" \
  B2_PREFIX="$REMOTE_B2_PREFIX" \
  DATA_ROOT="$DATA_ROOT" \
  CLEAN_OLD_SPLITS=0 \
  DRY_RUN=0 \
  bash scripts/fetch_datasets_b2.sh \
    "${TASK}/${TASK}_train.h5" \
    "${TASK}/${TASK}_val.h5" \
    "${TASK}/${TASK}_test.h5"

run_scan() {
  local split="$1"
  local output_json="$2"
  local scan_args=(
    python scripts/scan_transport_train_windows.py
    --data-root "$DATA_ROOT"
    --task "$TASK"
    --split "$split"
    --window-size "$WINDOW_SIZE"
    --stride "$WINDOW_STRIDE"
    --rollout-steps "$ROLLOUT_STEPS"
    --output-json "$output_json"
  )
  if [ -n "$SCAN_MAX_WINDOWS" ]; then
    scan_args+=(--max-windows "$SCAN_MAX_WINDOWS")
  fi
  scan_args+=("${shift_args[@]}")
  "${scan_args[@]}"
}

train_scan_json="${OUTPUT_ROOT}/train_window_scan.json"
selection_json=""
run_scan train "$train_scan_json"

if [ "$SCAN_ALL_SPLITS" -eq 1 ]; then
  val_scan_json="${OUTPUT_ROOT}/val_window_scan.json"
  test_scan_json="${OUTPUT_ROOT}/test_window_scan.json"
  selection_json="${OUTPUT_ROOT}/compatible_window_selection.json"
  run_scan val "$val_scan_json"
  run_scan test "$test_scan_json"
  select_args=(
    python scripts/select_transport_compatible_windows.py
    --train-scan "$train_scan_json"
    --val-scan "$val_scan_json"
    --test-scan "$test_scan_json"
    --output-json "$selection_json"
  )
  if [ "$REQUIRE_TEST_COMPATIBLE" -eq 1 ]; then
    select_args+=(--require-test)
  fi
  "${select_args[@]}"
  read -r selected_train_start selected_val_start selected_test_start < <(
    python - "$selection_json" "$VAL_START_INDEX" "$TEST_START_INDEX" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not payload.get("compatible"):
    raise SystemExit("No compatible train/val transport window selection found")
selected = payload["selected"]["windows"]
train_start = int(selected["train"]["start_index"])
val_start = int(selected["val"]["start_index"])
test_start = int(selected.get("test", {}).get("start_index", int(sys.argv[3])))
print(train_start, val_start, test_start)
PY
  )
else
  selected_train_start=$(
    python - "$train_scan_json" "$TARGET_SHIFT" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
target = int(sys.argv[2])
for row in payload.get("windows", []):
    if int(row["best"]["shift"]) == target:
        print(int(row["start_index"]))
        raise SystemExit(0)
raise SystemExit(f"No train window with best shift {target} found")
PY
  )
  selected_val_start="$VAL_START_INDEX"
  selected_test_start="$TEST_START_INDEX"
fi

manifest="${OUTPUT_ROOT}/candidate_manifest.yaml"
rm -rf "$CANDIDATE_ROOT"
python scripts/make_light_hdf5_shards.py \
  --root "$DATA_ROOT" \
  --out-root "$CANDIDATE_ROOT" \
  --tasks "$TASK" \
  --source-split train \
  --split-start-index "train=${selected_train_start}" \
  --split-start-index "val=${selected_val_start}" \
  --split-start-index "test=${selected_test_start}" \
  --train-count "$TRAIN_COUNT" \
  --val-count "$VAL_COUNT" \
  --test-count "$TEST_COUNT" \
  --version "$VERSION" \
  --manifest "$manifest" \
  --overwrite

gate_json="${OUTPUT_ROOT}/transport_shift_gate.json"
python scripts/run_transport_shift_gate.py \
  --data-root "$CANDIDATE_ROOT" \
  --task "$TASK" \
  --train-split train \
  --val-split val \
  --test-split test \
  --max-samples "$TRAIN_COUNT" \
  --test-max-samples "$TEST_COUNT" \
  --rollout-steps "$ROLLOUT_STEPS" \
  --reference-metric-value "$REFERENCE_METRIC_VALUE" \
  --val-min-relative-improvement "$VAL_MIN_RELATIVE_IMPROVEMENT" \
  --output-json "$gate_json" \
  "${shift_args[@]}"

if [ "$SCAN_ALL_SPLITS" -eq 1 ]; then
  audit_json="${OUTPUT_ROOT}/transport_shift_goal_audit.json"
  python scripts/audit_transport_shift_goal.py \
    --official-gate-json "$gate_json" \
    --compatible-window-selection-json "$selection_json" \
    --data-root "$CANDIDATE_ROOT" \
    --task "$TASK" \
    --output-json "$audit_json" \
    --require-status "$AUDIT_REQUIRE_STATUS"
fi

if [ "$PUBLISH_CANDIDATE" -eq 1 ]; then
  if [ -z "$REMOTE_PREFIX" ]; then
    echo "PUBLISH_CANDIDATE=1 requires REMOTE_PREFIX." >&2
    exit 1
  fi
  DRY_RUN=0 \
    BUILD_SHARDS=0 \
    ENV_FILE="$ENV_FILE" \
    VERSION="$VERSION" \
    REMOTE_PREFIX="$REMOTE_PREFIX" \
    OUT_ROOT="$CANDIDATE_ROOT" \
    MANIFEST="$manifest" \
    bash scripts/publish_light_hdf5_shards_b2.sh
fi

echo "Remote transport-shift candidate complete."
echo "Selected train start: ${selected_train_start}"
echo "Selected val start: ${selected_val_start}"
echo "Selected test start: ${selected_test_start}"
echo "Train scan: ${train_scan_json}"
echo "Gate: ${gate_json}"
