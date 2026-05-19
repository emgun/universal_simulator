#!/usr/bin/env bash
set -euo pipefail

# Re-run the official light-v1 lagged observed-transition transport gate.
#
# Safe planning run:
#   DRY_RUN=1 bash scripts/run_official_observed_transport_shift_gate.sh
#
# Official guarded run:
#   bash scripts/run_official_observed_transport_shift_gate.sh
#
# Explicit debugging repeat after the ledger already contains this test key:
#   ALLOW_REPEAT_TEST=1 bash scripts/run_official_observed_transport_shift_gate.sh

DRY_RUN=${DRY_RUN:-0}
DATA_ROOT=${DATA_ROOT:-data/pdebench}
TASK=${TASK:-advection1d}
TRAIN_SPLIT=${TRAIN_SPLIT:-train}
VAL_SPLIT=${VAL_SPLIT:-val}
TEST_SPLIT=${TEST_SPLIT:-test}
MAX_SAMPLES=${MAX_SAMPLES:-128}
TEST_MAX_SAMPLES=${TEST_MAX_SAMPLES:-32}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-16}
REFERENCE_METRIC_VALUE=${REFERENCE_METRIC_VALUE:-0.30780652221851373}
VAL_MIN_RELATIVE_IMPROVEMENT=${VAL_MIN_RELATIVE_IMPROVEMENT:-0.0}
OUTPUT_ROOT=${OUTPUT_ROOT:-reports/research/sota_loop}
GATE_JSON=${GATE_JSON:-${OUTPUT_ROOT}/observed_transport_shift_gate_real_light_v1.json}
TEST_LEDGER_JSON=${TEST_LEDGER_JSON:-${OUTPUT_ROOT}/observed_transport_shift_test_ledger.json}
ALLOW_REPEAT_TEST=${ALLOW_REPEAT_TEST:-0}
SHIFTS=${SHIFTS:--96,-88,-80,-72,-64,-56,-48,-40,-32,-24,-16,-8,0,8,16,24,32,40,48,56,64,72,80,88,96}

shift_args=()
IFS=',' read -r -a shift_values <<< "$SHIFTS"
for shift in "${shift_values[@]}"; do
  [ -n "$shift" ] && shift_args+=(--shift "$shift")
done

repeat_args=()
if [ "$ALLOW_REPEAT_TEST" -eq 1 ]; then
  repeat_args+=(--allow-repeat-test)
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY_RUN: run official observed ${TASK} train=${TRAIN_SPLIT} val=${VAL_SPLIT} gate from ${DATA_ROOT}"
  echo "DRY_RUN: pass test_split=${TEST_SPLIT}; gate measures held-out test only if validation passes"
  echo "DRY_RUN: write gate evidence to ${GATE_JSON}"
  echo "DRY_RUN: enforce exactly-once held-out test ledger at ${TEST_LEDGER_JSON}"
  echo "DRY_RUN: allow_repeat_test=${ALLOW_REPEAT_TEST}"
  exit 0
fi

python scripts/run_observed_transport_shift_gate.py \
  --data-root "$DATA_ROOT" \
  --task "$TASK" \
  --train-split "$TRAIN_SPLIT" \
  --val-split "$VAL_SPLIT" \
  --test-split "$TEST_SPLIT" \
  --max-samples "$MAX_SAMPLES" \
  --test-max-samples "$TEST_MAX_SAMPLES" \
  --rollout-steps "$ROLLOUT_STEPS" \
  --reference-metric-value "$REFERENCE_METRIC_VALUE" \
  --val-min-relative-improvement "$VAL_MIN_RELATIVE_IMPROVEMENT" \
  --test-ledger-json "$TEST_LEDGER_JSON" \
  --output-json "$GATE_JSON" \
  "${repeat_args[@]}" \
  "${shift_args[@]}"
