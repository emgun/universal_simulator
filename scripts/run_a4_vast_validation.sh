#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/workspace/universal_simulator}
PYTHON=${PYTHON:-/venv/main/bin/python}

cd "$REPO_ROOT"
export PYTHONPATH=src

exec "$PYTHON" scripts/plan_a4_baseline_validation.py \
  --neuraloperator-version 2.0.0 \
  --data-root data/a4/strat_v1_training_run \
  --output-root reports/a4_strat_v1_baselines \
  --output-plan reports/a4_strat_v1_baseline_plan.json \
  --device cuda \
  --execute \
  --confirm-validation-only
