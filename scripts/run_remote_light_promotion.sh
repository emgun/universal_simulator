#!/usr/bin/env bash
set -euo pipefail

# Run the current lightweight heterogeneous candidate on real B2-hosted PDEBench files.
#
# Safe dry run:
#   ENV_FILE=/Users/emerygunselman/Code/universal_simulator/.env \
#   DRY_RUN=1 bash scripts/run_remote_light_promotion.sh
#
# Actual full-data remote run requires explicit opt-in:
#   ENV_FILE=.env ALLOW_FULL_DATA=1 bash scripts/run_remote_light_promotion.sh

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

append_unique_word() {
  local var_name="$1"; shift
  local value="$1"; shift
  [ -n "$value" ] || return 0
  local current="${!var_name:-}"
  for existing in $current; do
    [ "$existing" = "$value" ] && return 0
  done
  if [ -n "$current" ]; then
    printf -v "$var_name" "%s %s" "$current" "$value"
  else
    printf -v "$var_name" "%s" "$value"
  fi
}

configure_artifact_rclone() {
  : "${B2_KEY_ID:?Set B2_KEY_ID to publish promotion artifacts}"
  : "${B2_APP_KEY:?Set B2_APP_KEY to publish promotion artifacts}"
  : "${B2_BUCKET:?Set B2_BUCKET to publish promotion artifacts}"
  if ! command -v rclone >/dev/null 2>&1; then
    echo "rclone is required to publish promotion artifacts." >&2
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
  artifact_name=${PROMOTION_ARTIFACT_NAME:-remote_light_promotion_${stamp}.tar.gz}
  artifact_path="/tmp/${artifact_name}"
  remote_key="${PROMOTION_ARTIFACT_PREFIX%/}/${artifact_name}"

  tar -czf "$artifact_path" "$OUTPUT_ROOT/$RUN_NAME"
  configure_artifact_rclone
  rclone copyto "$artifact_path" "UPSB2:${B2_BUCKET}/${remote_key}"
  echo "Published promotion artifacts: b2://${B2_BUCKET}/${remote_key}"
}

apply_cli_assignments() {
  local assignment
  for assignment in "$@"; do
    case "$assignment" in
      *=*) export "$assignment" ;;
      "")
        ;;
      *)
        echo "Unexpected argument '${assignment}'. Pass remote options as KEY=VALUE assignments." >&2
        exit 2
        ;;
    esac
  done
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

DATA_ROOT=${DATA_ROOT:-$WORKDIR/data/pdebench}
OUTPUT_ROOT=${OUTPUT_ROOT:-reports/light_experiments_remote}
TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_multitask_heterogeneous_light_best.yaml}
RUN_NAME=${RUN_NAME:-remote_heterogeneous_task_signature_joint32}
DEVICE=${DEVICE:-cuda}
TASKS=${TASKS:-burgers1d,advection1d,darcy2d}
EVAL_SPLIT=${EVAL_SPLIT:-test}
EXTRA_EVAL_SPLITS=${EXTRA_EVAL_SPLITS:-}
STAGES=${STAGES:-operator,decoder,operator_decoded,joint_codec_operator}
REMOTE_B2_PREFIX=${REMOTE_B2_PREFIX:-full}
FETCH_DATA=${FETCH_DATA:-1}
CHECK_DATA=${CHECK_DATA:-1}
ALLOW_WANDB=${ALLOW_WANDB:-0}
ALLOW_FULL_DATA=${ALLOW_FULL_DATA:-0}
REQUIRED_GB=${REQUIRED_GB:-180}
DRY_RUN=${DRY_RUN:-0}
PUBLISH_PROMOTION_ARTIFACTS=${PUBLISH_PROMOTION_ARTIFACTS:-0}
PROMOTION_ARTIFACT_PREFIX=${PROMOTION_ARTIFACT_PREFIX:-remote-runs/light}

mkdir -p "$DATA_ROOT" "$OUTPUT_ROOT"

if [ -z "${WANDB_API_KEY:-}" ]; then
  export WANDB_MODE=${WANDB_MODE:-offline}
fi

FETCH_SPLITS=""
append_unique_word FETCH_SPLITS train
append_unique_word FETCH_SPLITS "$EVAL_SPLIT"
for split in $(normalize_list "$EXTRA_EVAL_SPLITS"); do
  append_unique_word FETCH_SPLITS "$split"
done

DATASET_KEYS=""
USING_GENERATED_FULL_KEYS=0
if [ -n "${REMOTE_DATASET_FILES:-}" ]; then
  DATASET_KEYS="$(normalize_list "$REMOTE_DATASET_FILES")"
else
  USING_GENERATED_FULL_KEYS=1
  for task in $(normalize_list "$TASKS"); do
    for split in $FETCH_SPLITS; do
      append_unique_word DATASET_KEYS "${task}/${task}_${split}.h5"
    done
  done
fi

if [ "$DRY_RUN" -eq 0 ] && [ "$USING_GENERATED_FULL_KEYS" -eq 1 ] && [ "$REMOTE_B2_PREFIX" = "full" ] && [ "$ALLOW_FULL_DATA" -ne 1 ]; then
  echo "Refusing default full-data hydration without ALLOW_FULL_DATA=1." >&2
  echo "The default B2 full train/test files are large and the current HDF5 loader reads files into memory." >&2
  echo "Use a bounded shard prefix such as smoke-v1/light-v1, set REMOTE_DATASET_FILES, or set ALLOW_FULL_DATA=1 explicitly." >&2
  exit 1
fi

if [ "$DRY_RUN" -eq 0 ] && [ "$FETCH_DATA" -eq 1 ]; then
  AVAIL_GB=$(df -Pm "$DATA_ROOT" | awk 'NR==2{print int($4/1024)}')
  if [ "$AVAIL_GB" -lt "$REQUIRED_GB" ]; then
    echo "Insufficient free disk for hydration: have ${AVAIL_GB}GB, require ${REQUIRED_GB}GB." >&2
    echo "Override REQUIRED_GB only if REMOTE_DATASET_FILES points to a smaller shard set." >&2
    exit 1
  fi
fi

if [ "$FETCH_DATA" -eq 1 ]; then
  if [ "$DRY_RUN" -eq 0 ]; then
    ensure_rclone
  fi
  echo "Hydrating B2 datasets from prefix '${REMOTE_B2_PREFIX}' into ${DATA_ROOT}"
  if [ "$DRY_RUN" -eq 1 ]; then
    DRY_RUN=1 B2_ENV_FILE="$ENV_FILE" B2_PREFIX="$REMOTE_B2_PREFIX" DATA_ROOT="$DATA_ROOT" \
      CLEAN_OLD_SPLITS=0 bash scripts/fetch_datasets_b2.sh $DATASET_KEYS
  else
    B2_ENV_FILE="$ENV_FILE" B2_PREFIX="$REMOTE_B2_PREFIX" DATA_ROOT="$DATA_ROOT" \
      CLEAN_OLD_SPLITS=0 bash scripts/fetch_datasets_b2.sh $DATASET_KEYS
  fi
else
  echo "Skipping dataset hydration; expecting files under ${DATA_ROOT}"
fi

if [ "$CHECK_DATA" -eq 1 ] && [ "$DRY_RUN" -eq 0 ]; then
  missing=0
  for task in $(normalize_list "$TASKS"); do
    for split in $FETCH_SPLITS; do
      file="${DATA_ROOT}/${task}_${split}.h5"
      if [ ! -f "$file" ] && ! compgen -G "${DATA_ROOT}/${task}_${split}_*.h5" >/dev/null; then
        echo "Missing required data file: $file" >&2
        missing=1
      fi
    done
  done
  if [ "$missing" -ne 0 ]; then
    exit 1
  fi
fi

cmd=(
  python scripts/run_light_experiment.py
  --config "$TRAIN_CONFIG"
  --name "$RUN_NAME"
  --output-root "$OUTPUT_ROOT"
  --device "$DEVICE"
  --decoded
  --override "data.root=$DATA_ROOT"
  --eval-override "data.root=$DATA_ROOT"
  --eval-override "data.split=$EVAL_SPLIT"
)

for stage in $(normalize_list "$STAGES"); do
  cmd+=(--stage "$stage")
done

for split in $(normalize_list "$EXTRA_EVAL_SPLITS"); do
  cmd+=(--extra-eval-split "$split")
done

if [ "$ALLOW_WANDB" -eq 1 ]; then
  cmd+=(--allow-wandb)
fi

if [ -n "${LIGHT_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  extra_args=($LIGHT_EXTRA_ARGS)
  cmd+=("${extra_args[@]}")
fi

echo "Promotion command:"
printf ' %q' "${cmd[@]}"
echo

if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY_RUN=1: skipping training/evaluation."
  exit 0
fi

PYTHONPATH=src "${cmd[@]}"

if [ "$PUBLISH_PROMOTION_ARTIFACTS" -eq 1 ]; then
  publish_artifacts
fi

echo "Remote light promotion complete."
