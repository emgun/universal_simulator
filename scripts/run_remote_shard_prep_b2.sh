#!/usr/bin/env bash
set -euo pipefail

# Sequentially hydrate provenance-bearing PDEBench sources, build universally
# gated shards, publish them, and delete full sources between tasks.
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

source_splits_for_task() {
  local task="$1"
  local env_name
  env_name="$(echo "${task}_SOURCE_SPLITS" | tr '[:lower:]' '[:upper:]')"
  local configured="${!env_name:-}"
  if [ -n "$configured" ]; then
    normalize_list "$configured"
    return 0
  fi
  echo "train"
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

protocol_setting_for_task() {
  local task="$1"
  local suffix="$2"
  local env_name="$(echo "${task}_${suffix}" | tr '[:lower:]' '[:upper:]')"
  echo "${!env_name:-}"
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
    payload = {key: task_payload.get(key) for key in ("version", "source_root", "out_root", "remote_prefix", "tasks", "splits", "protocol_mode")}
    payload["tasks"] = []
    payload["records"] = []
    payload["protocol_gates"] = {}
for task in task_payload.get("tasks", []):
    if task not in payload["tasks"]:
        payload["tasks"].append(task)
payload.setdefault("records", []).extend(task_payload.get("records", []))
payload.setdefault("protocol_gates", {}).update(task_payload.get("protocol_gates", {}))
aggregate.parent.mkdir(parents=True, exist_ok=True)
aggregate.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
PY
}

ENV_FILE=${ENV_FILE:-.env}
VERSION=${VERSION:-strat-v1}
REMOTE_PREFIX=${REMOTE_PREFIX:-$VERSION}
TASKS=${TASKS:-"burgers1d advection1d darcy2d"}
DATA_ROOT=${DATA_ROOT:-data/pdebench_full}
VERSION_SLUG="${VERSION//[^[:alnum:]]/_}"
DEFAULT_OUT_ROOT="data/pdebench_${VERSION_SLUG}"
DEFAULT_MANIFEST="docs/${VERSION_SLUG}_data_manifest.yaml"
OUT_ROOT=${OUT_ROOT:-$DEFAULT_OUT_ROOT}
MANIFEST=${MANIFEST:-$DEFAULT_MANIFEST}
TRAIN_COUNT=${TRAIN_COUNT:-128}
VAL_COUNT=${VAL_COUNT:-32}
TEST_COUNT=${TEST_COUNT:-32}
DRY_RUN=${DRY_RUN:-1}
CLEAN_SOURCE=${CLEAN_SOURCE:-1}
FETCH_DATA=${FETCH_DATA:-1}
PUBLISH_SHARDS=${PUBLISH_SHARDS:-1}
REQUIRED_GB=${REQUIRED_GB:-0}

if [ "$VERSION" = "smoke-v1" ] || [ "$VERSION" = "light-v1" ] || [ "$VERSION" = "medium-v1" ]; then
  echo "${VERSION} is reserved for immutable legacy artifacts; choose a strat-v1 version label" >&2
  exit 2
fi
if [ "$REMOTE_PREFIX" = "smoke-v1" ] || [ "$REMOTE_PREFIX" = "light-v1" ] || [ "$REMOTE_PREFIX" = "medium-v1" ]; then
  echo "${REMOTE_PREFIX} is a frozen legacy remote prefix and cannot receive new artifacts" >&2
  exit 2
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY_RUN: would build ${VERSION} shards for tasks: ${TASKS}"
  for task in $(normalize_list "$TASKS"); do
    echo "DRY_RUN: task=${task} source_splits=$(source_splits_for_task "$task")"
    echo "DRY_RUN: task=${task} provenance=$(protocol_setting_for_task "$task" PROVENANCE_DATASETS) regime=$(protocol_setting_for_task "$task" REGIME_DATASET) field_kind=$(protocol_setting_for_task "$task" FIELD_KIND)"
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

if [ "$REQUIRED_GB" -gt 0 ]; then
  avail_gb=$(df -Pm "$DATA_ROOT" | awk 'NR==2{print int($4/1024)}')
  if [ "$avail_gb" -lt "$REQUIRED_GB" ]; then
    echo "Insufficient free disk for shard prep: have ${avail_gb}GB, require ${REQUIRED_GB}GB at ${DATA_ROOT}." >&2
    exit 1
  fi
fi

for task in $(normalize_list "$TASKS"); do
  echo "Preparing task ${task}"
  task_out="${OUT_ROOT}/${task}"
  task_manifest="${OUT_ROOT}/${task}.manifest.yaml"
  rm -rf "$task_out"
  mkdir -p "$task_out"

  source_split="$(source_splits_for_task "$task")"
  if [[ "$source_split" == *" "* ]]; then
    echo "${task} must use one canonical provenance-bearing source split, got: ${source_split}" >&2
    exit 2
  fi
  provenance_datasets="$(protocol_setting_for_task "$task" PROVENANCE_DATASETS)"
  regime_dataset="$(protocol_setting_for_task "$task" REGIME_DATASET)"
  field_kind="$(protocol_setting_for_task "$task" FIELD_KIND)"
  time_axis="$(protocol_setting_for_task "$task" TIME_AXIS)"
  task_env_prefix="$(echo "$task" | tr '[:lower:]' '[:upper:]')"
  if [ -z "$provenance_datasets" ] || [ -z "$regime_dataset" ] || [ -z "$field_kind" ]; then
    echo "${task} requires ${task_env_prefix}_PROVENANCE_DATASETS, ${task_env_prefix}_REGIME_DATASET, and ${task_env_prefix}_FIELD_KIND" >&2
    exit 2
  fi
  provenance_args=()
  for dataset in $(normalize_list "$provenance_datasets"); do
    provenance_args+=(--provenance-dataset "$dataset")
  done
  field_args=(--field-kind "$field_kind")
  if [ "$field_kind" = "temporal" ]; then
    if [ -z "$time_axis" ]; then
      echo "${task} temporal construction requires ${task_env_prefix}_TIME_AXIS" >&2
      exit 2
    fi
    field_args+=(--time-axis "$time_axis")
  elif [ -n "$time_axis" ]; then
    echo "${task} steady construction must not set ${task_env_prefix}_TIME_AXIS" >&2
    exit 2
  fi

  for split in $(source_splits_for_task "$task"); do
    for key in $(source_keys_for_task_split "$task" "$split"); do
      if [ "$FETCH_DATA" -eq 1 ]; then
        B2_ENV_FILE="$ENV_FILE" \
          B2_PREFIX=full \
          DATA_ROOT="$DATA_ROOT" \
          CLEAN_OLD_SPLITS=0 \
          DRY_RUN=0 \
          bash scripts/fetch_datasets_b2.sh "$key"
      else
        echo "Skipping fetch for ${key}; expecting source under ${DATA_ROOT}"
      fi
    done
  done

  python scripts/make_light_hdf5_shards.py \
    --root "$DATA_ROOT" \
    --out-root "$task_out" \
    --tasks "$task" \
    --source-split "$source_split" \
    --train-count "$TRAIN_COUNT" \
    --val-count "$VAL_COUNT" \
    --test-count "$TEST_COUNT" \
    "${provenance_args[@]}" \
    --regime-dataset "$regime_dataset" \
    "${field_args[@]}" \
    --version "$VERSION" \
    --remote-prefix "$REMOTE_PREFIX" \
    --manifest "$task_manifest" \
    --overwrite

  append_manifest_records "$MANIFEST" "$task_manifest"

  if [ "$CLEAN_SOURCE" -eq 1 ]; then
    rm -f "${DATA_ROOT}/${task}_"*.h5
  fi
done

if [ "$PUBLISH_SHARDS" -eq 1 ]; then
  flat_out="${OUT_ROOT}/${VERSION}_flat"
  rm -rf "$flat_out"
  mkdir -p "$flat_out"
  find "$OUT_ROOT" -mindepth 2 -maxdepth 2 -type f -name "*.h5" ! -path "$flat_out/*" -exec cp {} "$flat_out" \;

  DRY_RUN=0 \
  ENV_FILE="$ENV_FILE" \
  VERSION="$VERSION" \
  REMOTE_PREFIX="$REMOTE_PREFIX" \
  OUT_ROOT="$flat_out" \
  MANIFEST="$MANIFEST" \
  bash scripts/publish_light_hdf5_shards_b2.sh
else
  echo "PUBLISH_SHARDS=0: skipping B2 publish."
fi

echo "Remote shard prep complete. Manifest: ${MANIFEST}"
