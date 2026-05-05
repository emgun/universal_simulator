#!/usr/bin/env bash
set -euo pipefail

# Sequentially hydrate full B2 PDEBench files, cut small demo shards, publish
# those shards back to B2, and delete full sources between tasks.
#
# This is intended for a cheap remote/data-prep box with enough disk for the
# largest single task, not for local laptops and not for GPU training.
#
# Safe planning run:
#   DRY_RUN=1 ENV_FILE=/path/to/.env bash scripts/run_remote_shard_prep_b2.sh
#
# Actual run:
#   DRY_RUN=0 ENV_FILE=/path/to/.env DATA_ROOT=/workspace/pdebench_full \
#     OUT_ROOT=/workspace/pdebench_light bash scripts/run_remote_shard_prep_b2.sh

normalize_list() {
  echo "$1" | tr ',' ' '
}

source_splits_for_task() {
  local task="$1"
  local env_name
  env_name="$(echo "${task}_SOURCE_SPLITS" | tr '[:lower:]' '[:upper:]')"
  local configured="${!env_name:-}"
  if [ -n "$configured" ]; then
    normalize_list "$configured"
    return 0
  fi
  case "$task" in
    darcy2d) echo "train test" ;;
    *) echo "train val test" ;;
  esac
}

source_keys_for_task_split() {
  local task="$1"
  local split="$2"
  local env_name
  env_name="$(echo "${task}_${split}_SOURCE_KEYS" | tr '[:lower:]' '[:upper:]')"
  local configured="${!env_name:-}"
  if [ -n "$configured" ]; then
    normalize_list "$configured"
    return 0
  fi
  echo "${task}/${task}_${split}.h5"
}

append_manifest_records() {
  local aggregate="$1"
  local task_manifest="$2"
  python - "$aggregate" "$task_manifest" <<'PY'
from pathlib import Path
import sys
import yaml

aggregate = Path(sys.argv[1])
task_manifest = Path(sys.argv[2])
task_payload = yaml.safe_load(task_manifest.read_text(encoding="utf-8")) or {}
if aggregate.exists():
    payload = yaml.safe_load(aggregate.read_text(encoding="utf-8")) or {}
else:
    payload = {key: task_payload.get(key) for key in ("version", "source_root", "out_root", "remote_prefix", "tasks", "splits")}
    payload["tasks"] = []
    payload["records"] = []
for task in task_payload.get("tasks", []):
    if task not in payload["tasks"]:
        payload["tasks"].append(task)
payload.setdefault("records", []).extend(task_payload.get("records", []))
aggregate.parent.mkdir(parents=True, exist_ok=True)
aggregate.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
PY
}

ENV_FILE=${ENV_FILE:-.env}
VERSION=${VERSION:-light-v1}
REMOTE_PREFIX=${REMOTE_PREFIX:-$VERSION}
TASKS=${TASKS:-"burgers1d advection1d darcy2d"}
DATA_ROOT=${DATA_ROOT:-data/pdebench_full}
OUT_ROOT=${OUT_ROOT:-data/pdebench_light}
MANIFEST=${MANIFEST:-docs/demo_data_manifest.yaml}
TRAIN_COUNT=${TRAIN_COUNT:-128}
VAL_COUNT=${VAL_COUNT:-32}
TEST_COUNT=${TEST_COUNT:-32}
START_INDEX=${START_INDEX:-0}
DRY_RUN=${DRY_RUN:-1}
CLEAN_SOURCE=${CLEAN_SOURCE:-1}

if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY_RUN: would build ${VERSION} shards for tasks: ${TASKS}"
  for task in $(normalize_list "$TASKS"); do
    echo "DRY_RUN: task=${task} source_splits=$(source_splits_for_task "$task")"
    for split in $(source_splits_for_task "$task"); do
      for key in $(source_keys_for_task_split "$task" "$split"); do
        echo "DRY_RUN: fetch full/${key} -> ${DATA_ROOT}"
      done
    done
    echo "DRY_RUN: cut ${task} shards into ${OUT_ROOT}/${task}"
  done
  echo "DRY_RUN: publish ${OUT_ROOT}/*.h5 and ${MANIFEST} to prefix ${REMOTE_PREFIX}"
  exit 0
fi

rm -f "$MANIFEST"
mkdir -p "$DATA_ROOT" "$OUT_ROOT"

for task in $(normalize_list "$TASKS"); do
  echo "Preparing task ${task}"
  task_out="${OUT_ROOT}/${task}"
  task_manifest="${OUT_ROOT}/${task}.manifest.yaml"
  rm -rf "$task_out"
  mkdir -p "$task_out"

  for split in $(source_splits_for_task "$task"); do
    for key in $(source_keys_for_task_split "$task" "$split"); do
      B2_ENV_FILE="$ENV_FILE" \
        B2_PREFIX=full \
        DATA_ROOT="$DATA_ROOT" \
        CLEAN_OLD_SPLITS=0 \
        DRY_RUN=0 \
        bash scripts/fetch_datasets_b2.sh "$key"
    done
  done

  python scripts/make_light_hdf5_shards.py \
    --root "$DATA_ROOT" \
    --out-root "$task_out" \
    --tasks "$task" \
    --train-count "$TRAIN_COUNT" \
    --val-count "$VAL_COUNT" \
    --test-count "$TEST_COUNT" \
    --start-index "$START_INDEX" \
    --version "$VERSION" \
    --remote-prefix "$REMOTE_PREFIX" \
    --manifest "$task_manifest" \
    --overwrite

  append_manifest_records "$MANIFEST" "$task_manifest"

  if [ "$CLEAN_SOURCE" -eq 1 ]; then
    rm -f "${DATA_ROOT}/${task}_"*.h5
  fi
done

flat_out="${OUT_ROOT}/${VERSION}_flat"
rm -rf "$flat_out"
mkdir -p "$flat_out"
find "$OUT_ROOT" -mindepth 2 -maxdepth 2 -type f -name "*.h5" -exec cp {} "$flat_out" \;

DRY_RUN=0 \
BUILD_SHARDS=0 \
ENV_FILE="$ENV_FILE" \
VERSION="$VERSION" \
REMOTE_PREFIX="$REMOTE_PREFIX" \
OUT_ROOT="$flat_out" \
MANIFEST="$MANIFEST" \
bash scripts/publish_light_hdf5_shards_b2.sh

echo "Remote shard prep complete. Manifest: ${MANIFEST}"
