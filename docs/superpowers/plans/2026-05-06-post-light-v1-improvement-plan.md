# Post-Light-v1 W&B-Tracked Improvement Plan

## Task Contract

Task: turn the completed `light-v1` experiment loop into a W&B-backed iteration system, then use it to find a candidate that beats the matched persistence baseline.

Done condition: future remote runs record local summaries, B2 artifact handles, and W&B run handles; the next model candidates are gated against `persistence_light_v1_test` before any medium or SOTA-style run.

Mutable files:

- `scripts/run_light_experiment.py`
- `scripts/run_remote_light_promotion.sh`
- `scripts/collect_wandb_runs.py`
- `scripts/train.py`
- `src/ups/utils/monitoring.py`
- `scripts/plan_demo_experiments.py`
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
Pass `--b2-s3-endpoint "$B2_S3_ENDPOINT"` and
`--b2-s3-region "$B2_S3_REGION"` from the private `.env`; native B2 rclone
hydration hung on one host, while the S3 endpoint path copied `light-v1`
quickly. With `--args-mode`, do not trust auto-shutdown alone: watch for
`Published promotion artifacts:` and destroy the instance if the container
restarts the entrypoint.

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

## Remote Result: `task_signature_residual_alpha25`

- B2 artifact: `remote-runs/light/ups_light_residual_alpha25_20260506T1528Z.tar.gz`
- Local summary: `reports/light_experiments_remote/ups_light_task_signature_residual_alpha25/summary.json`
- W&B runs: `00ud83aw`, `3ugaodok`, `i3ej1zp9`, `dm8y4ccc`
- Decoded rollout NRMSE: `0.5486869325531744`
- Persistence baseline decoded rollout NRMSE: `0.5701633411507036`
- Baseline ratio: `0.9623328841973854`
- Baseline improvement fraction: `0.03766711580261458`
- Gate result: absolute promotion passed, 20% baseline-improvement gate failed.
- Interpretation: residual alpha `0.25` is a real improvement over persistence and a large improvement over the previous UPS light candidate (`0.8881691012411048`), but not enough for demo promotion. The high decoded rollout spectral energy error (`4.828118220727542` vs persistence `0.06721624190029686`) means the next iterations should optimize stability/energy, not just NRMSE.

## Remote Result: `task_signature_residual_alpha50`

- B2 artifact: `remote-runs/light/ups_light_residual_alpha50_20260506T1548Z.tar.gz`
- Local summary: `reports/light_experiments_remote/ups_light_task_signature_residual_alpha50/summary.json`
- W&B runs: `dr5wpv23`, `tp1wbop8`, `e3v1o3ce`, `axcvkdcy`
- Decoded rollout NRMSE: `0.6084554326486734`
- Persistence baseline decoded rollout NRMSE: `0.5701633411507036`
- Baseline ratio: `1.0671598623311852`
- Baseline improvement fraction: `-0.06715986233118525`
- Gate result: absolute promotion passed, baseline-improvement gate failed.
- Interpretation: alpha `0.50` is worse than persistence and worse than alpha `0.25`, confirming the useful residual blend is small and that simply mixing more UPS prediction into persistence is not the path to a demo-quality model.

Current decision: do not spend on more scalar blend sweeps. An eval-only checkpoint reuse path is now available for cheap alpha/stability probes; use it before any future scalar probe, then move to a trained persistence-residual or stability-regularized decoded objective.

Eval-only reuse command shape:

```bash
python scripts/run_light_experiment.py \
  --config configs/train_multitask_heterogeneous_light_best.yaml \
  --name ups_light_eval_only_candidate \
  --output-root reports/light_experiments_remote \
  --checkpoint-source reports/light_experiments_remote/ups_light_task_signature_residual_alpha25 \
  --skip-training \
  --decoded \
  --override data.root=/path/to/light-v1 \
  --eval-override data.root=/path/to/light-v1 \
  --eval-override data.split=test
```

## Implemented Next Iteration Surface

The next remote candidate should be `task_signature_trained_residual`, not
another eval-only scalar alpha sweep. The implementation now adds:

- A W&B `benchmark-summary` run from `scripts/run_light_experiment.py` that logs final benchmark metrics under `summary/*`, including decoded rollout metrics when decoded evaluation is enabled.
- Decoded training loss knobs for `operator_decoded` and `joint_codec_operator`:
  - `lambda_persistence_residual`
  - `lambda_persistence_residual_spectral`
  - `lambda_spectral`
  - `lambda_relative`
- Planner variant `task_signature_trained_residual`, which combines task-signature conditioning, persistence-residual decoded losses, residual-spectral loss, and the current best eval blend `evaluation.decoded_persistence_residual_alpha=0.25`.

Queue generation:

```bash
python scripts/plan_demo_experiments.py \
  --tier light \
  --variant task_signature_trained_residual \
  --run-prefix ups \
  --env-file /workspace/.env \
  --output-jsonl reports/demo/trained_residual_light_queue.jsonl \
  --output-tsv reports/demo/trained_residual_light_queue.tsv \
  --output-sh reports/demo/run_trained_residual_light_queue.sh
```

Remote dry-run shape:

```bash
ENV_FILE=/Users/emerygunselman/Code/universal_simulator/.env \
DRY_RUN=1 \
ALLOW_WANDB=1 \
WANDB_GROUP=light-v1-trained-residual \
WANDB_TAGS=light-v1,trained-residual,baseline-gated \
TASKS=burgers1d,advection1d,darcy2d \
REMOTE_B2_PREFIX=light-v1 \
EVAL_SPLIT=test \
REQUIRED_GB=10 \
STAGES=operator,decoder,operator_decoded,joint_codec_operator \
RUN_NAME=ups_light_task_signature_trained_residual \
LIGHT_EXTRA_ARGS="--override data.max_samples=128 --eval-override data.max_samples=32 --decoded-rollout-steps 16 --override operator.conditioning.sources={\"task_id\":3,\"equation_signature\":15} --override stages.operator_decoded.lambda_persistence_residual=0.5 --override stages.operator_decoded.lambda_persistence_residual_spectral=0.05 --override stages.joint_codec_operator.lambda_persistence_residual=0.5 --override stages.joint_codec_operator.lambda_persistence_residual_spectral=0.05 --override evaluation.decoded_persistence_residual_alpha=0.25" \
bash scripts/run_remote_light_promotion.sh
```

Promotion remains unchanged: do not scale until held-out `decoded_rollout_nrmse`
beats `persistence_light_v1_test` by at least 20% or produces a clearly
documented task-level reason to revise the gate.

## Remote Result: `task_signature_trained_residual`

- Vast contract: `36250467`
- B2 artifact: `remote-runs/light/ups_light_trained_residual_20260506T1755Z.tar.gz`
- Local summary: `reports/light_experiments_remote/ups_light_task_signature_trained_residual/summary.json`
- W&B runs: `4wps03re`, `u76hpryu`, `kv2z579u`, `quw7vz35`, `3dr2jyfa`
- Benchmark-summary W&B run: `3dr2jyfa`
- Decoded rollout NRMSE: `0.530536668470072`
- Persistence baseline decoded rollout NRMSE: `0.5701633411507036`
- Baseline ratio: `0.9304994379318442`
- Baseline improvement fraction: `0.06950056206815583`
- Gate result: absolute promotion passed, 20% baseline-improvement gate failed.
- Task decoded rollout NRMSE:
  - Burgers: `0.21524346565356076`
  - Advection: `0.7362082121022959`
  - Darcy: `0.27036938921296805`

Interpretation: this is the new best held-out `light-v1` candidate and validates
the trained residual/stability direction, but it is not demo-ready. The failure
is now concentrated in the transport/advection family, so the next queue should
test transport-specific conditioning/loss scaling or a per-family residual gate
instead of increasing the global residual weight.

## Local Eval-Only Result: `transport_residual_gate_alpha0p42`

- Local summary: `reports/light_experiments_remote/ups_light_transport_residual_gate_alpha0p42_eval/summary.json`
- Checkpoint source: `reports/light_experiments_remote/ups_light_task_signature_trained_residual`
- Data: held-out `light-v1` test shards only
- Training: skipped
- Gate config:
  - global `evaluation.decoded_persistence_residual_alpha=0.0`
  - `evaluation.decoded_persistence_residual_alpha_by_family={"transport":0.42}`
- Decoded rollout NRMSE: `0.5126627282110727`
- Persistence baseline decoded rollout NRMSE: `0.5701633411507036`
- Baseline ratio: `0.8991506314250525`
- Baseline improvement fraction: `0.10084936857494749`
- Gate result: absolute promotion passed, 20% baseline-improvement gate failed.

Interpretation: per-family residual gating is the best current light-v1 path and
roughly halves the remaining gap to the 20% baseline-improvement gate without
retraining. The next implementation should make this gate learned or improve the
transport/advection dynamics directly.
