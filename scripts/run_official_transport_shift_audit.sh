#!/usr/bin/env bash
set -euo pipefail

# Re-run the official light-v1 constant transport-shift gate and audit its evidence.
#
# Safe planning run:
#   DRY_RUN=1 bash scripts/run_official_transport_shift_audit.sh
#
# Diagnostic refresh that records current blocked status without failing:
#   AUDIT_REQUIRE_STATUS=report bash scripts/run_official_transport_shift_audit.sh
#
# Promotion check, fail-closed unless validation and the authorized held-out test are recorded:
#   AUDIT_REQUIRE_STATUS=achieved bash scripts/run_official_transport_shift_audit.sh

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
GATE_JSON=${GATE_JSON:-${OUTPUT_ROOT}/transport_shift_gate.json}
AUDIT_JSON=${AUDIT_JSON:-${OUTPUT_ROOT}/transport_shift_goal_audit.json}
COMPATIBILITY_JSON=${COMPATIBILITY_JSON:-${OUTPUT_ROOT}/remote_transport_shift_candidate_all_splits/compatible_window_selection.json}
AUDIT_REQUIRE_STATUS=${AUDIT_REQUIRE_STATUS:-achieved}
SHIFTS=${SHIFTS:--96,-88,-80,-72,-64,-56,-48,-40,-32,-24,-16,-8,0,8,16,24,32,40,48,56,64,72,80,88,96}
EXPECTED_TRAIN_SHA256=${EXPECTED_TRAIN_SHA256:-67925f6765b64695818e36087bc69efaa9adee42253db6ef7c89b723118581d1}
EXPECTED_VAL_SHA256=${EXPECTED_VAL_SHA256:-9b6fcf88ae8d92b42107c840a9fef9c17eea1992c84024ed0dd61be0b0fe7329}
EXPECTED_TEST_SHA256=${EXPECTED_TEST_SHA256:-4930a14afefa062d2d3a56ddda98ad76ff1e33eb150ed6f02fc36004fe0cdf93}

shift_args=()
IFS=',' read -r -a shift_values <<< "$SHIFTS"
for shift in "${shift_values[@]}"; do
  [ -n "$shift" ] && shift_args+=(--shift "$shift")
done

if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY_RUN: run official ${TASK} train=${TRAIN_SPLIT} val=${VAL_SPLIT} gate from ${DATA_ROOT}"
  echo "DRY_RUN: pass test_split=${TEST_SPLIT}; gate measures held-out test only if validation passes"
  echo "DRY_RUN: write gate evidence to ${GATE_JSON}"
  echo "DRY_RUN: audit ${GATE_JSON} with compatibility ${COMPATIBILITY_JSON}, require_status=${AUDIT_REQUIRE_STATUS}"
  echo "DRY_RUN: require data SHA-256 train=${EXPECTED_TRAIN_SHA256}, val=${EXPECTED_VAL_SHA256}, test=${EXPECTED_TEST_SHA256}"
  exit 0
fi

python scripts/run_transport_shift_gate.py \
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
  --output-json "$GATE_JSON" \
  "${shift_args[@]}"

python scripts/audit_transport_shift_goal.py \
  --official-gate-json "$GATE_JSON" \
  --compatible-window-selection-json "$COMPATIBILITY_JSON" \
  --data-root "$DATA_ROOT" \
  --task "$TASK" \
  --expected-data-sha256 "train=${EXPECTED_TRAIN_SHA256}" \
  --expected-data-sha256 "val=${EXPECTED_VAL_SHA256}" \
  --expected-data-sha256 "test=${EXPECTED_TEST_SHA256}" \
  --output-json "$AUDIT_JSON" \
  --require-status "$AUDIT_REQUIRE_STATUS"
