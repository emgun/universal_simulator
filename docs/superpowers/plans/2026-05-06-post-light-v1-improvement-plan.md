# Post-Light-v1 W&B-Tracked Improvement Plan

## Task Contract

Task: turn the completed `light-v1` experiment loop into a W&B-backed iteration system, then use it to find a candidate that beats the matched persistence baseline.

Done condition: future remote runs record local summaries, B2 artifact handles, and W&B run handles; the next model candidates are gated against `persistence_light_v1_test` before any medium or SOTA-style run.

Mutable files:

- `scripts/run_light_experiment.py`
- `scripts/run_remote_light_promotion.sh`
- `scripts/collect_wandb_runs.py`
- `src/ups/utils/monitoring.py`
- `src/ups/eval/demo_scorecard.py`
- `docs/demo_runbook.md`
- `docs/demo_completion_audit.md`
- `worklog.md`

Fixed files:

- B2 `light-v1` data layout
- existing completed run summaries
- old broad roadmap, except for references from this narrower plan

Validation:

- unit tests for monitoring, light runner summaries, scorecard tracking fields, and W&B registry export
- `bash -n scripts/run_remote_light_promotion.sh`
- dry-run remote command with `ALLOW_WANDB=1`
- no active Vast instances before stopping

Constraints:

- no local training beyond synthetic/unit smoke tests
- no medium/full data spend until a `light-v1` candidate beats persistence
- no SOTA-style claim until benchmark-compatible baselines and split contracts are frozen

## Current Evidence

The infrastructure gate is now passed:

- `light-v1` B2 readiness: `9/9` keys present.
- UPS candidate: `ups_light_v1_task_signature_only`, `decoded_rollout_nrmse = 0.8881691012411048`.
- Persistence baseline: `persistence_light_v1_test`, `decoded_rollout_nrmse = 0.5701633411507036`.
- Baseline gate: failed with `baseline_improvement_passed=false`.
- Failure signal: UPS decoded rollout spectral energy error is `74.20507275975494` versus persistence `0.06721624190029686`.

Conclusion: the next blocker is not remote plumbing. It is decoded rollout stability and baseline-aware modeling.

## Phase 1: Make Tracking First-Class

Done condition: every future remote run has a local `summary.json` with W&B metadata and scorecards expose W&B columns.

Steps:

- [x] Keep W&B disabled by default for cheap local runs.
- [x] When `--allow-wandb` or `ALLOW_WANDB=1` is set, configure project, entity, group, tags, and job type from args or env.
- [x] Write W&B run metadata to `logs/wandb_runs.jsonl`.
- [x] Copy W&B run IDs and URLs into `summary.json`.
- [x] Add W&B tracking fields to demo scorecard rows and TSV output.
- [x] Add `scripts/collect_wandb_runs.py` to import historical W&B run summaries into `reports/wandb/`.

Validation:

```bash
pytest tests/unit/test_monitoring.py tests/unit/test_light_experiment_runner.py tests/unit/test_demo_scorecard.py tests/unit/test_collect_wandb_runs.py -q
bash -n scripts/run_remote_light_promotion.sh
```

## Phase 2: Backfill W&B History

Done condition: historical W&B runs that matter for UPS are exported locally and linked to current local/B2 artifacts when names overlap.

Commands:

```bash
python scripts/collect_wandb_runs.py \
  --entity "$WANDB_ENTITY" \
  --project "${WANDB_PROJECT:-universal-simulator}" \
  --limit 200 \
  --metric-prefix decoded_ \
  --metric-prefix task_ \
  --out-json reports/wandb/runs.json \
  --out-tsv reports/wandb/runs.tsv
```

Then rebuild the local scorecard:

```bash
python scripts/build_demo_report.py \
  --glob "reports/light_experiments_remote/*/summary.json" \
  --output-dir reports/demo/light_latest \
  --title "UPS Light-v1 Demo Scorecard" \
  --data-manifest docs/demo_data_manifest.yaml \
  --baseline-run persistence_light_v1_test \
  --baseline-metric decoded_rollout_nrmse \
  --baseline-min-improvement 0.2 \
  --promotion-rule "decoded_rollout_nrmse<=1.0" \
  --copy-summaries
```

## Phase 3: Architecture Iteration Queue

Done condition: at least one new candidate beats `persistence_light_v1_test` on held-out `light-v1`, or the plan records enough failed evidence to change direction.

Candidate order:

1. Persistence-residual decoder/operator: predict delta from physical persistence instead of absolute next field.
2. Stability-aware decoded rollout objective: add spectral energy and horizon-weighted decoded rollout penalties.
3. Hybrid residual gate: learn a per-task blend between persistence and UPS decoded prediction.
4. Task-specific failure isolation: run Burgers-only and Darcy-only residual variants if multitask instability masks wins.

Implemented first cheap residual screen:

- `evaluation.decoded_persistence_residual_alpha=0.0` exactly matches physical persistence in decoded evaluation.
- `evaluation.decoded_persistence_residual_alpha=1.0` is the existing UPS decoded rollout.
- `task_signature_residual_alpha25` evaluates `persistence + 0.25 * (UPS - persistence)`.
- `task_signature_residual_alpha50` evaluates `persistence + 0.50 * (UPS - persistence)`.
- Generated local ignored queue artifacts:
  - `reports/demo/residual_light_queue.jsonl`
  - `reports/demo/residual_light_queue.tsv`
  - `reports/demo/run_residual_light_queue.sh`

Keep/discard rule:

- Keep only if held-out `decoded_rollout_nrmse` improves over persistence by at least 20% overall or wins two of three tasks without a catastrophic third-task regression.
- Discard if spectral energy error remains orders of magnitude worse than persistence.
- Do not scale a candidate that only improves smoke.

## Phase 4: Remote Execution Pattern

Use W&B for every paid remote candidate:

```bash
ENV_FILE=/Users/emerygunselman/Code/universal_simulator/.env \
DRY_RUN=1 \
ALLOW_WANDB=1 \
WANDB_GROUP=light-v1-residual \
WANDB_TAGS=light-v1,residual,baseline-gated \
TASKS=burgers1d,advection1d,darcy2d \
REMOTE_B2_PREFIX=light-v1 \
EVAL_SPLIT=test \
REQUIRED_GB=10 \
LIGHT_EXTRA_ARGS="--override data.max_samples=128 --eval-override data.max_samples=32 --decoded-rollout-steps 16" \
bash scripts/run_remote_light_promotion.sh
```

Switch `DRY_RUN=0` only after reviewing the generated command and Vast offer.
For W&B-backed runs, use Vast `--install-mode experiment` from
`codex/residual-light-candidate` or newer; that profile installs `wandb`.
The monitoring layer now fails fast if W&B is enabled but the `wandb` package is
missing, preventing silent untracked paid runs.

Queue generation:

```bash
python scripts/plan_demo_experiments.py \
  --tier light \
  --variant task_signature_residual_alpha25 \
  --variant task_signature_residual_alpha50 \
  --run-prefix ups \
  --env-file /workspace/.env \
  --output-jsonl reports/demo/residual_light_queue.jsonl \
  --output-tsv reports/demo/residual_light_queue.tsv \
  --output-sh reports/demo/run_residual_light_queue.sh
```

## Stop Conditions

- Stop if W&B credentials or the `wandb` package are absent and a paid remote run would be untracked.
- Stop if B2 readiness fails for `light-v1`.
- Stop if the next change requires full-data hydration or medium-scale spend before the light gate passes.
- Stop if branch cleanup would delete an unpushed or unmerged branch.
