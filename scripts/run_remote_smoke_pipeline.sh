#!/usr/bin/env bash
set -euo pipefail

# Orchestrate the cheapest remote UPS demo loop:
# 1. Check smoke shard readiness in B2.
# 2. Prepare/publish smoke shards if they are missing and PREP_SHARDS=1.
# 3. Generate a bounded smoke experiment queue.
# 4. Optionally run that queue when RUN_EXPERIMENTS=1.
#
# Defaults are safe: dry-run shard prep, dry-run generated queue, and no
# training. Set DRY_RUN=0 to publish shards. Set RUN_EXPERIMENTS=1 and
# QUEUE_DRY_RUN=0 only on a remote box with enough scratch space and after
# reviewing the generated queue.

ENV_FILE=${ENV_FILE:-.env}
PIPELINE_ROOT=${PIPELINE_ROOT:-reports/demo/remote_smoke_pipeline}
SMOKE_MANIFEST=${SMOKE_MANIFEST:-docs/demo_smoke_data_manifest.yaml}
SUMMARY_GLOB=${SUMMARY_GLOB:-reports/light_experiments_remote/*/summary.json}
QUEUE_VARIANTS=${QUEUE_VARIANTS:-"current_best no_conditioning task_signature_only"}
QUEUE_DIR=${QUEUE_DIR:-$PIPELINE_ROOT/queue}
OUTPUT_ROOT=${OUTPUT_ROOT:-reports/light_experiments_remote}
PREP_SHARDS=${PREP_SHARDS:-1}
RUN_EXPERIMENTS=${RUN_EXPERIMENTS:-0}
DRY_RUN=${DRY_RUN:-1}
QUEUE_DRY_RUN=${QUEUE_DRY_RUN:-1}
RUN_NAME_PREFIX=${RUN_NAME_PREFIX:-ups}
CHECK_B2=${CHECK_B2:-1}

mkdir -p "$PIPELINE_ROOT" "$QUEUE_DIR"

readiness_json="$PIPELINE_ROOT/readiness_before.json"
readiness_after_json="$PIPELINE_ROOT/readiness_after.json"
prep_log="$PIPELINE_ROOT/smoke_shard_prep.log"
queue_log="$PIPELINE_ROOT/smoke_queue.log"

check_readiness() {
  local output_json="$1"
  local -a args=(
    python scripts/check_demo_readiness.py
    --manifest "$SMOKE_MANIFEST" \
    --summary-glob "$SUMMARY_GLOB" \
    --baseline-run "" \
    --candidate-run "" \
    --env-file "$ENV_FILE" \
    --json "$output_json"
  )
  if [ "$CHECK_B2" -eq 1 ]; then
    args+=(--check-b2)
  fi
  "${args[@]}"
}

shards_ready() {
  local path="$1"
  python - "$path" <<'PY'
import json
import sys

payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
b2 = payload.get("b2") or {}
sys.exit(0 if b2.get("ok") else 1)
PY
}

echo "Checking smoke shard readiness..."
if check_readiness "$readiness_json"; then
  readiness_status=0
else
  readiness_status=$?
fi

if shards_ready "$readiness_json"; then
  echo "Smoke shards are already ready."
else
  echo "Smoke shards are not ready."
  if [ "$PREP_SHARDS" -eq 1 ]; then
    echo "Running smoke shard prep. Log: ${prep_log}"
    (
      DRY_RUN="$DRY_RUN" \
      ENV_FILE="$ENV_FILE" \
      MANIFEST="$SMOKE_MANIFEST" \
      bash scripts/run_smoke_shard_prep_b2.sh
    ) 2>&1 | tee "$prep_log"
    if [ "$DRY_RUN" -eq 0 ]; then
      echo "Re-checking smoke shard readiness after prep..."
      if check_readiness "$readiness_after_json"; then
        readiness_status=0
      else
        readiness_status=$?
      fi
      if ! shards_ready "$readiness_after_json"; then
        readiness_status=1
      fi
    else
      echo "DRY_RUN=1: skipping post-prep readiness enforcement."
    fi
  else
    echo "PREP_SHARDS=0: leaving smoke shards missing."
  fi
fi

variant_args=()
for variant in $QUEUE_VARIANTS; do
  variant_args+=(--variant "$variant")
done

queue_jsonl="$QUEUE_DIR/smoke_queue.jsonl"
queue_tsv="$QUEUE_DIR/smoke_queue.tsv"
queue_sh="$QUEUE_DIR/run_smoke_queue.sh"

python scripts/plan_demo_experiments.py \
  --tier smoke \
  "${variant_args[@]}" \
  --env-file "$ENV_FILE" \
  --dry-run-value "$QUEUE_DRY_RUN" \
  --run-prefix "$RUN_NAME_PREFIX" \
  --output-root "$OUTPUT_ROOT" \
  --output-jsonl "$queue_jsonl" \
  --output-tsv "$queue_tsv" \
  --output-sh "$queue_sh"

echo "Smoke queue generated:"
echo "  ${queue_jsonl}"
echo "  ${queue_tsv}"
echo "  ${queue_sh}"

if [ "$RUN_EXPERIMENTS" -eq 1 ]; then
  echo "Running smoke experiment queue. Log: ${queue_log}"
  bash "$queue_sh" 2>&1 | tee "$queue_log"
else
  echo "RUN_EXPERIMENTS=0: generated queue only."
fi

if [ "$readiness_status" -ne 0 ] && [ "$DRY_RUN" -eq 0 ]; then
  echo "Smoke readiness is still failing after requested actions." >&2
  exit "$readiness_status"
fi

echo "Remote smoke pipeline complete."
