# UPS Demo Runbook

This is the execution runbook for turning the current light-experiment plumbing
into a working held-out demo.

For current completion status and missing gates, see
`docs/demo_completion_audit.md`.

## Current Branch Stack

Apply or merge in this order if working from `main`:

1. `codex/autowork-semigroup-foundation`
   - remote light promotion wrapper
   - `data.max_samples` cap
   - bounded Burgers smoke record
   - demo benchmark contract
   - shard manifest generation
   - B2 publishing wrapper
2. `codex/demo-scorecard-loop`
   - `src/ups/eval/demo_scorecard.py`
   - `scripts/collect_light_results.py`
   - `scripts/build_demo_report.py`
3. `codex/demo-persistence-baseline`
   - decoded physical-space persistence baseline
   - `scripts/run_persistence_baseline.py`
4. `codex/demo-b2-shard-check`
   - `scripts/check_demo_b2_shards.py`
5. `codex/remote-shard-prep`
   - `scripts/run_remote_shard_prep_b2.sh`
6. `codex/demo-report-plots`
   - static metric plots embedded into `index.html`
7. `codex/demo-runbook-handoff`
   - this execution runbook
   - README quickstart pointer
8. `codex/demo-cost-tracking`
   - optional `cost.json` ingestion
   - provider, instance, GPU-hour, and estimated-dollar report fields
9. `codex/demo-experiment-queue`
   - smoke/light/medium JSONL, TSV, and shell queue generation
   - dry-run remote commands by default
10. `codex/demo-baseline-delta`
    - baseline comparison columns for the demo keep/discard gate
    - absolute glob support for report input discovery
11. `codex/demo-readiness-check`
    - one-command JSON readiness report
    - optional live B2 shard presence check
12. `codex/smoke-source-key-shards`
    - smoke-only source key overrides for cheaper shard-prep dry runs
    - explicit warning that derived smoke slices are not benchmark evidence
13. `codex/smoke-manifest-readiness`
    - `docs/demo_smoke_data_manifest.yaml`
    - smoke-specific readiness commands
14. `codex/harden-b2-fetcher`
    - `scripts/fetch_datasets_b2.sh` rejects empty-success `rclone lsjson` results
15. `codex/smoke-shard-prep-wrapper`
    - one-command smoke shard prep wrapper
16. `codex/fix-smoke-output-root`
    - smoke wrapper writes under `data/pdebench_smoke`
17. `codex/cheap-smoke-split-sources`
    - default smoke prep uses Burgers train shard, Advection val, and Darcy test
    - reduces default smoke source hydration to roughly 10 GiB
18. `codex/local-shard-prep-test-mode`
    - `FETCH_DATA=0` and `PUBLISH_SHARDS=0` for already-hydrated/test-mode shard cutting

## Current Evidence

The only real-data run so far is a plumbing smoke, not a benchmark:

- run: `vast_burgers_shard_cap8`
- artifact: `reports/light_experiments_remote/vast_burgers_shard_cap8/summary.json`
- metric: `decoded_rollout_nrmse = 0.9488316819858322`
- split: tiny slice of a train shard
- status: useful for validating remote execution only

Do not use this as a benchmark claim.

## B2 State

External env file:

```bash
ENV_FILE=/Users/emerygunselman/Code/universal_simulator/.env
```

Known B2 layout:

- full source data is under top-level `full/`
- `smoke-v1` is not published
- `light-v1` is not published
- live readiness check still reports `0/9` `light-v1` keys present
- live readiness check also reports `0/9` `smoke-v1` keys present
- `pdebench/full` is not the active prefix
- local machine has only about `1.9 GiB` free, so do not run smoke prep here

Approximate source sizes:

- `full/burgers1d/burgers1d_train.h5`: `69.045 GiB`
- `full/burgers1d/burgers1d_val.h5`: `7.704 GiB`
- `full/burgers1d/burgers1d_test.h5`: `15.36 GiB`
- `full/burgers1d/burgers1d_train_000.h5`: `1.57 GiB`
- `full/advection1d/advection1d_train.h5`: `46.03 GiB`
- `full/advection1d/advection1d_val.h5`: `7.704 GiB`
- `full/advection1d/advection1d_test.h5`: `7.704 GiB`
- `full/darcy2d/darcy2d_train.h5`: `2.441 GiB`
- `full/darcy2d/darcy2d_test.h5`: `0.613 GiB`

Plan at least 120-150 GiB scratch for shard prep, because Burgers needs train,
val, and test source files at the same time if using native splits.

## Step 1: Preflight Expected Shards

Smoke readiness check for plumbing-only remote iteration:

```bash
python scripts/check_demo_readiness.py \
  --manifest docs/demo_smoke_data_manifest.yaml \
  --summary-glob "reports/light_experiments_remote/*/summary.json" \
  --baseline-run "" \
  --candidate-run "" \
  --check-b2 \
  --env-file /Users/emerygunselman/Code/universal_simulator/.env
```

Held-out light readiness check:

```bash
python scripts/check_demo_readiness.py \
  --manifest docs/demo_data_manifest.yaml \
  --summary-glob "reports/light_experiments_remote/*/summary.json" \
  --baseline-run persistence_light_v1_test \
  --candidate-run ups_light_v1_current_best
```

Add `--check-b2 --env-file /Users/emerygunselman/Code/universal_simulator/.env`
when you want the readiness report to contact B2.

```bash
ENV_FILE=/Users/emerygunselman/Code/universal_simulator/.env \
python scripts/check_demo_b2_shards.py \
  --manifest docs/demo_data_manifest.yaml \
  --env-file /Users/emerygunselman/Code/universal_simulator/.env
```

Expected before publishing:

- exit code nonzero
- all 9 `light-v1` keys missing

Expected after publishing:

- exit code zero
- all 9 `light-v1` keys present

## Step 2: Dry-Run Remote Shard Prep

Run this locally first:

```bash
DRY_RUN=1 \
ENV_FILE=/Users/emerygunselman/Code/universal_simulator/.env \
VERSION=light-v1 \
DATA_ROOT=/workspace/pdebench_full \
OUT_ROOT=/workspace/pdebench_light \
bash scripts/run_remote_shard_prep_b2.sh
```

Expected:

- fetch plan for Burgers train/val/test
- fetch plan for Advection train/val/test
- fetch plan for Darcy train/test
- no network writes

For a cheaper smoke-only data-prep pass, you can opt into known smaller source
keys and derive smoke validation/test from the same small source. Do not use
this shortcut for benchmark claims.

```bash
DRY_RUN=1 bash scripts/run_smoke_shard_prep_b2.sh
```

If the source files are already hydrated on the remote box, skip fetching and
only cut local shards:

```bash
DRY_RUN=0 \
FETCH_DATA=0 \
PUBLISH_SHARDS=0 \
DATA_ROOT=/workspace/pdebench_full \
OUT_ROOT=/workspace/pdebench_smoke \
MANIFEST=/workspace/demo_smoke_data_manifest.yaml \
bash scripts/run_smoke_shard_prep_b2.sh
```

Default smoke source set size from live B2 inspection after split-source
shortcuts:

- `full/burgers1d/burgers1d_train_000.h5`: `1.570 GiB`
- `full/advection1d/advection1d_val.h5`: `7.704 GiB`
- `full/darcy2d/darcy2d_test.h5`: `0.613 GiB`

Plan roughly 12 GiB scratch for default three-task smoke prep. The wrapper
derives Advection and Darcy smoke train/val/test slices from smaller non-train
sources by default; this is why the output is plumbing-only.
`scripts/run_smoke_shard_prep_b2.sh` enforces `REQUIRED_GB=12` by default when
`DRY_RUN=0`; override it only if you have checked the source set manually.

To run the whole remote smoke pipeline on a remote/data-prep or cheap GPU box:

```bash
DRY_RUN=0 \
ENV_FILE=/workspace/.env \
PIPELINE_ROOT=reports/demo/remote_smoke_pipeline \
bash scripts/run_remote_smoke_pipeline.sh
```

This prepares/publishes missing `smoke-v1` shards, writes readiness artifacts,
and generates a smoke queue. Add `RUN_EXPERIMENTS=1 QUEUE_DRY_RUN=0` only after
reviewing `reports/demo/remote_smoke_pipeline/queue/run_smoke_queue.sh`.
`QUEUE_DRY_RUN` defaults to `1` even when `DRY_RUN=0` is used for shard prep.
Live smoke queue execution requires `CHECK_B2=1` unless
`ALLOW_UNCHECKED_LIVE_QUEUE=1` is explicitly set for a controlled test
environment.

Before tearing down the remote box, package artifacts:

```bash
OUTPUT=reports/demo/demo_artifacts.tar.gz \
bash scripts/package_demo_artifacts.sh
```

## Step 3: Run Remote Shard Prep

Use a cheap remote/data-prep machine, not a training GPU, unless GPU rental is
the cheapest available option at the time.

Minimum requirements:

- 150 GiB scratch disk
- `git`
- `python`
- `rclone`
- access to `.env` values through environment variables or copied env file

Command on the remote box:

```bash
DRY_RUN=0 \
ENV_FILE=/workspace/.env \
VERSION=light-v1 \
DATA_ROOT=/workspace/pdebench_full \
OUT_ROOT=/workspace/pdebench_light \
MANIFEST=/workspace/demo_data_manifest.yaml \
TRAIN_COUNT=128 \
VAL_COUNT=32 \
TEST_COUNT=32 \
bash scripts/run_remote_shard_prep_b2.sh
```

After completion, copy the generated manifest back into the repo or store it
with the run artifacts.

## Step 4: Run UPS Light Held-Out Candidate

Generate a bounded experiment queue before launching a variant matrix:

```bash
python scripts/plan_demo_experiments.py \
  --tier smoke \
  --variant current_best \
  --variant no_conditioning \
  --variant task_signature_only \
  --output-jsonl reports/demo/smoke_queue.jsonl \
  --output-tsv reports/demo/smoke_queue.tsv \
  --output-sh reports/demo/run_smoke_queue.sh
```

Review the generated shell plan first. It defaults to `DRY_RUN=1`; regenerate
with `--dry-run-value 0` only after B2 shard preflight passes.

Preflight:

```bash
ENV_FILE=/Users/emerygunselman/Code/universal_simulator/.env \
DRY_RUN=1 \
TASKS=burgers1d,advection1d,darcy2d \
TRAIN_CONFIG=configs/train_multitask_heterogeneous_light_best.yaml \
REMOTE_B2_PREFIX=light-v1 \
EVAL_SPLIT=test \
REQUIRED_GB=10 \
STAGES=operator,decoder,operator_decoded,joint_codec_operator \
LIGHT_EXTRA_ARGS="--override data.max_samples=128 --eval-override data.max_samples=32 --decoded-rollout-steps 16" \
bash scripts/run_remote_light_promotion.sh
```

Actual:

```bash
ENV_FILE=/workspace/.env \
DRY_RUN=0 \
TASKS=burgers1d,advection1d,darcy2d \
TRAIN_CONFIG=configs/train_multitask_heterogeneous_light_best.yaml \
REMOTE_B2_PREFIX=light-v1 \
EVAL_SPLIT=test \
REQUIRED_GB=10 \
STAGES=operator,decoder,operator_decoded,joint_codec_operator \
RUN_NAME=ups_light_v1_current_best \
OUTPUT_ROOT=reports/light_experiments_remote \
LIGHT_EXTRA_ARGS="--override data.max_samples=128 --eval-override data.max_samples=32 --decoded-rollout-steps 16" \
bash scripts/run_remote_light_promotion.sh
```

Copy `reports/light_experiments_remote/ups_light_v1_current_best/summary.json`
back before destroying the instance.

## Step 5: Run Persistence Baseline

Run on the same hydrated `light-v1` data root:

```bash
python scripts/run_persistence_baseline.py \
  --config configs/train_multitask_heterogeneous_light_best.yaml \
  --name persistence_light_v1_test \
  --output-root reports/light_experiments_remote \
  --data-root data/pdebench_light \
  --split test \
  --max-samples 32 \
  --rollout-steps 16 \
  --promotion-rule "decoded_rollout_nrmse<=1.0"
```

## Step 6: Build Demo Report

Optional per-run cost files can be passed with `--cost-json`. The report will
match records by `run_name` or `summary_json` and compute `cost_gpu_hours` and
`cost_estimated_usd` when possible.

Example `cost.json`:

```json
{
  "run_name": "ups_light_v1_current_best",
  "provider": "vast",
  "instance_type": "rtx4090-spot",
  "gpu_type": "RTX 4090",
  "gpu_count": 1,
  "wall_clock_hours": 1.5,
  "hourly_usd": 0.35
}
```

```bash
python scripts/build_demo_report.py \
  --glob "reports/light_experiments_remote/*/summary.json" \
  --output-dir reports/demo/latest \
  --title "UPS Light-v1 Demo Scorecard" \
  --data-manifest docs/demo_data_manifest.yaml \
  --cost-json reports/light_experiments_remote/ups_light_v1_current_best/cost.json \
  --cost-json reports/light_experiments_remote/persistence_light_v1_test/cost.json \
  --baseline-run persistence_light_v1_test \
  --baseline-metric decoded_rollout_nrmse \
  --baseline-min-improvement 0.2 \
  --promotion-rule "decoded_rollout_nrmse<=1.0" \
  --copy-summaries
```

Expected artifacts:

- `reports/demo/latest/index.html`
- `reports/demo/latest/metrics.tsv`
- `reports/demo/latest/scorecard.json`
- `reports/demo/latest/plots/decoded_rollout_nrmse.png`
- `reports/demo/latest/plots/decoded_step1_nrmse.png`
- copied summaries under `reports/demo/latest/summaries/`

The baseline columns are lower-is-better. A strong demo candidate should show
`baseline_improvement_passed=true` for `ups_light_v1_current_best` against
`persistence_light_v1_test`.

## Step 7: Decide Keep/Discard

Keep current UPS candidate only if:

- it beats persistence on held-out `decoded_rollout_nrmse`, and
- no task-level decoded rollout metric regresses catastrophically, and
- visual/report artifacts are generated cleanly.

If current UPS does not beat persistence:

- do not scale it
- inspect split/data handling first
- then run the planned variant matrix from
  `docs/superpowers/plans/2026-05-04-working-demo-sota-roadmap.md`

## Stop Rules

Stop and ask before:

- hydrating the default full 3-task 141 GiB set at once
- running any full-data benchmark
- spending beyond light/medium cost tier
- making SOTA claims
