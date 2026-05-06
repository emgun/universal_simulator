# UPS Demo Runbook

This is the execution runbook for turning the current light-experiment plumbing
into a working held-out demo.

For current completion status and missing gates, see
`docs/demo_completion_audit.md`.

For the post-light-v1 iteration loop, use
`docs/superpowers/plans/2026-05-06-post-light-v1-improvement-plan.md`.

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
19. `codex/remote-smoke-pipeline`
    - one-command remote smoke readiness, prep, and queue generation
20. `codex/safe-smoke-queue-default`
    - generated queue stays dry-run even when shard prep is live
21. `codex/smoke-disk-guard`
    - `REQUIRED_GB` guard for shard prep
22. `codex/demo-completion-audit`
    - explicit prompt-to-artifact completion checklist
23. `codex/enforce-smoke-ready-before-queue`
    - refuses live smoke queue when B2 smoke shards are missing
24. `codex/package-demo-artifacts`
    - packages remote artifacts and records missing paths
25. `codex/require-b2-check-for-live-smoke`
    - refuses unchecked live smoke queue unless explicitly overridden
26. `codex/safe-vast-smoke-launcher`
    - dry-run-first Vast smoke launcher
    - redacts B2/W&B secrets from dry-run launch output
27. `codex/vast-offer-summary`
    - compact Vast offer JSON/TSV snapshots without launching compute
28. `codex/vast-cheap-launch-order`
    - orders implicit Vast launch searches by `dph_total`
    - caps candidate search with `LIMIT`
29. `codex/vast-offer-id-launch`
    - optional `OFFER_ID` pinning for a reviewed Vast search result
    - uses `vastai create instance <offer_id>` instead of implicit launch search
30. `codex/vast-no-apt-onstart`
    - avoids `apt-get` in Vast onstart after setup stalls on Ubuntu mirrors
    - downloads `rclone` directly and falls back to a GitHub branch zip when `git` is unavailable
    - supports `SSH=0` one-shot launches to skip Vast SSH runtime injection when that setup stalls
    - supports `ARGS_MODE=1` to run `bash -lc` directly and bypass Vast onstart bootstrap
    - defaults the smoke launcher to `INSTALL_MODE=smoke` so shard prep does not pull full Torch/CUDA dev deps
    - supports `INSTALL_MODE=experiment` plus `EXTRA_PIPELINE_ARGS` for one-off smoke experiment runs
31. `codex/smoke-focused-variants`
    - focused smoke variants around task-signature conditioning
    - B2 artifact publishing for remote light promotion runs
    - remote `light-v1` shard prep, candidate, persistence baseline, and light scorecard records

## Current Evidence

The only real-data run so far is a plumbing smoke, not a benchmark:

- run: `vast_burgers_shard_cap8`
- artifact: `reports/light_experiments_remote/vast_burgers_shard_cap8/summary.json`
- metric: `decoded_rollout_nrmse = 0.9488316819858322`
- split: tiny slice of a train shard
- status: useful for validating remote execution only

Do not use this as a benchmark claim.

The first three-task remote smoke-v1 run completed on 2026-05-05 UTC:

- run: `ups_smoke_current_best`
- B2 artifact: `remote-runs/smoke/smoke_current_best_20260505T0600Z.tar.gz`
- local summary: `reports/light_experiments_remote/ups_smoke_current_best/summary.json`
- local smoke scorecard: `reports/demo/smoke_latest/scorecard.json`
- metric: `decoded_rollout_nrmse = 0.6297059754071941`
- status: passes the smoke promotion rule `decoded_rollout_nrmse<=1.0`

This is useful evidence that the remote smoke loop works end-to-end, but still
not a benchmark claim because `smoke-v1` is tiny and uses shortcut split sources.

The matched smoke persistence baseline also completed:

- run: `persistence_smoke_v1_test`
- B2 artifact:
  `remote-runs/smoke/persistence_smoke_v1_test_20260505T0615Z.tar.gz`
- local summary:
  `reports/light_experiments_remote/persistence_smoke_v1_test/summary.json`
- metric: `decoded_rollout_nrmse = 0.1876487120420463`
- comparison: `ups_smoke_current_best` fails baseline improvement on smoke
  (`0.6297059754071941` vs persistence `0.1876487120420463`)

Interpretation: the pipeline is now ready for cheap iteration, but the current
UPS config should not be scaled until it beats persistence on bounded held-out
shards.

The first smoke-v1 variant matrix completed on 2026-05-05 UTC:

- B2 artifact: `remote-runs/smoke/smoke_variants_20260505T0625Z.tar.gz`
- local summaries:
  `reports/light_experiments_remote/ups_smoke_no_conditioning/summary.json`,
  `reports/light_experiments_remote/ups_smoke_task_signature_only/summary.json`,
  `reports/light_experiments_remote/ups_smoke_semigroup0/summary.json`, and
  `reports/light_experiments_remote/ups_smoke_semigroup10/summary.json`
- best UPS row: `ups_smoke_task_signature_only`
- metric: `decoded_rollout_nrmse = 0.4793234406026068`
- comparison: better than `ups_smoke_current_best`
  (`0.6297059754071941`) but still worse than persistence
  (`0.1876487120420463`)

Interpretation: the smoke harness is now useful for screening candidate
directions. The strongest observed UPS change is narrower task-signature
conditioning, but the demo still needs either a stronger candidate or a more
appropriate held-out gate before scale-up.

A focused task-signature smoke-v1 matrix also completed on 2026-05-05 UTC:

- B2 artifact: `remote-runs/smoke/smoke_focused_variants_20260505T0613Z.tar.gz`
- rows:
  `ups_smoke_task_signature_semigroup0`,
  `ups_smoke_task_signature_joint48`,
  `ups_smoke_task_signature_rollout4`, and
  `ups_smoke_task_signature_joint48_rollout4`
- best focused row: `ups_smoke_task_signature_joint48`
- metric: `decoded_rollout_nrmse = 0.4971677039442661`
- comparison: worse than `ups_smoke_task_signature_only`
  (`0.4793234406026068`) and still worse than persistence
  (`0.1876487120420463`)

Interpretation: do not spend more smoke budget on semigroup-off, longer joint
training, or longer rollout-loss variants until a stronger decoded objective or
baseline-aware architecture exists.

A decoded/reconstruction smoke-v1 matrix completed on 2026-05-05 UTC:

- B2 artifact: `remote-runs/smoke/smoke_decoded_variants_20260505T0621Z.tar.gz`
- rows:
  `ups_smoke_task_signature_joint16`,
  `ups_smoke_task_signature_opdecoded4`,
  `ups_smoke_task_signature_opdecoded4_joint16`, and
  `ups_smoke_task_signature_recon0`
- best decoded follow-up row: `ups_smoke_task_signature_joint16`
- metric: `decoded_rollout_nrmse = 0.5951420314812053`
- comparison: worse than `ups_smoke_task_signature_only`
  (`0.4793234406026068`) and still worse than persistence
  (`0.1876487120420463`)

Interpretation: more decoded fine-tuning and simple reconstruction-weight
tweaks are not enough on smoke-v1. The next useful step is held-out `light-v1`
shard prep, then rerun only the best observed UPS and persistence rows.

Held-out `light-v1` shard prep completed on 2026-05-05 UTC:

- B2 readiness: `9/9` expected `light-v1` keys present
- local readiness artifact: `reports/demo/light_readiness_after_prep.json`
- status: bounded held-out data is ready for cheap remote experiments

The first held-out `light-v1` UPS candidate completed:

- run: `ups_light_v1_task_signature_only`
- B2 artifact: `remote-runs/light/ups_light_task_signature_20260505T0731Z.tar.gz`
- local summary: `reports/light_experiments_remote/ups_light_v1_task_signature_only/summary.json`
- metric: `decoded_rollout_nrmse = 0.8881691012411048`
- absolute promotion rule: passed `decoded_rollout_nrmse<=1.0`

The matched held-out `light-v1` persistence baseline completed:

- run: `persistence_light_v1_test`
- B2 artifact:
  `remote-runs/light/persistence_light_v1_test_20260505T0740Z.tar.gz`
- local summary:
  `reports/light_experiments_remote/persistence_light_v1_test/summary.json`
- metric: `decoded_rollout_nrmse = 0.5701633411507036`

The local light scorecard is `reports/demo/light_latest/scorecard.json`.
`reports/demo/light_readiness_after_runs.json` reports `ready=true` with no
blockers, so the experiment loop is now operational. Performance is not yet
demo-good: `ups_light_v1_task_signature_only` fails the baseline improvement
gate with delta `0.31800576009040127`, ratio `1.5577450129441892`, and
`baseline_improvement_passed=false`.

Task-level failure pattern on `light-v1`:

- Burgers: UPS `0.801173912475701` vs persistence `0.17446879799698398`
- Advection: UPS `0.9816829335662135` vs persistence `0.8086701258529039`
- Darcy: UPS `0.7462278194548689` vs persistence `0.20909552146272067`
- Spectral stability: UPS decoded rollout spectral energy error
  `74.20507275975494` vs persistence `0.06721624190029686`

Interpretation: stop scaling this candidate. The next useful architecture
iteration should be baseline-aware, such as a persistence-residual
decoder/operator path or a stability-regularized decoded rollout objective,
then rerun the same light scorecard.

Future paid remote runs should use W&B tracking. Set `ALLOW_WANDB=1` plus
`WANDB_PROJECT`, `WANDB_ENTITY`, and optionally `WANDB_GROUP`/`WANDB_TAGS`.
The light runner records W&B run IDs and URLs into each `summary.json`; the demo
scorecard surfaces those fields as `tracking_wandb_*` columns.
Use the Vast `experiment` install profile from `codex/residual-light-candidate`
or newer for W&B-backed remote runs; it installs `wandb`, and the monitoring
layer fails fast instead of silently skipping tracking when `wandb` is missing.
Pass the private `.env` Backblaze S3 endpoint and region to `vast_launch.py`
(`--b2-s3-endpoint` and `--b2-s3-region`). The native B2 rclone path can hang on
some Vast hosts; the S3 endpoint path copied `light-v1` reliably in the
residual alpha25 run. In `--args-mode`, monitor for the B2 publish line and
destroy the instance manually if the container restarts the entrypoint after
completion.

## B2 State

External env file:

```bash
ENV_FILE=/Users/emerygunselman/Code/universal_simulator/.env
```

Known B2 layout:

- full source data is under top-level `full/`
- `smoke-v1` is published and live-checked as ready (`9/9` keys present on
  2026-05-05 UTC)
- `light-v1` is published and live-checked as ready (`9/9` keys present on
  2026-05-05 UTC)
- held-out `light-v1` candidate and persistence summaries are present locally
- `pdebench/full` is not the active prefix
- local machine has only about `3.0 GiB` free, so do not run shard prep here

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

`scripts/run_remote_shard_prep_b2.sh` accepts `KEY=VALUE` CLI assignments, so it
can be launched through `scripts/vast_launch.py --remote-script` without a
custom wrapper.

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
  --candidate-run ups_light_v1_task_signature_only
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

Current expected result:

- `reports/demo/light_readiness_after_runs.json` has `ready=true`
- B2 readiness shows all 9 `light-v1` keys present
- baseline summary is `persistence_light_v1_test`
- candidate summary is `ups_light_v1_task_signature_only`

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

Dry-run a Vast.ai launch plan for the same pipeline without printing B2 secret
values:

```bash
ENV_FILE=/Users/emerygunselman/Code/universal_simulator/.env \
DRY_RUN=1 \
GIT_REF=codex/vast-no-apt-onstart \
DISK_GB=32 \
ORDER=dph_total \
LIMIT=10 \
SSH=0 \
ARGS_MODE=1 \
INSTALL_MODE=smoke \
bash scripts/launch_remote_smoke_vast.sh
```

Summarize current cheap Vast offers without launching:

```bash
python scripts/search_vast_smoke_offers.py \
  --limit 10 \
  --output-json reports/demo/vast_smoke_offers.json \
  --output-tsv reports/demo/vast_smoke_offers.tsv
```

If you choose a specific reviewed offer from that snapshot, pin it explicitly
instead of letting `vastai launch instance` re-run the search at launch time:

```bash
ENV_FILE=/Users/emerygunselman/Code/universal_simulator/.env \
DRY_RUN=1 \
GIT_REF=codex/vast-no-apt-onstart \
DISK_GB=32 \
OFFER_ID=<offer_id_from_search> \
SSH=0 \
ARGS_MODE=1 \
INSTALL_MODE=smoke \
bash scripts/launch_remote_smoke_vast.sh
```

Only switch that command to `DRY_RUN=0` after reviewing the generated onstart
script. Vast offer IDs are time-sensitive and single-use.

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

To run one cheap smoke experiment after `smoke-v1` readiness passes, launch a
new one-shot Vast instance with the experiment install profile:

```bash
ENV_FILE=/Users/emerygunselman/Code/universal_simulator/.env \
DRY_RUN=1 \
GIT_REF=codex/vast-no-apt-onstart \
DISK_GB=32 \
OFFER_ID=<offer_id_from_search> \
SSH=0 \
ARGS_MODE=1 \
INSTALL_MODE=experiment \
EXTRA_PIPELINE_ARGS="PREP_SHARDS=0 RUN_EXPERIMENTS=1 QUEUE_DRY_RUN=0 QUEUE_VARIANTS=current_best" \
bash scripts/launch_remote_smoke_vast.sh
```

Use `DRY_RUN=0` only after reviewing the generated command. This runs just the
`current_best` smoke queue entry against `smoke-v1`.
For no-SSH runs where files cannot be copied back directly, add
`PUBLISH_PIPELINE_ARTIFACTS=1` and a stable `PIPELINE_ARTIFACT_NAME=...` inside
`EXTRA_PIPELINE_ARGS` so the remote uploads a tarball under
`remote-runs/smoke/` in B2 before exit.

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
ALLOW_WANDB=1 \
WANDB_GROUP=light-v1-residual \
WANDB_TAGS=light-v1,residual,baseline-gated \
TASKS=burgers1d,advection1d,darcy2d \
TRAIN_CONFIG=configs/train_multitask_heterogeneous_light_best.yaml \
REMOTE_B2_PREFIX=light-v1 \
EVAL_SPLIT=test \
REQUIRED_GB=10 \
STAGES=operator,decoder,operator_decoded,joint_codec_operator \
RUN_NAME=ups_light_v1_task_signature_only \
OUTPUT_ROOT=reports/light_experiments_remote \
LIGHT_EXTRA_ARGS="--override data.max_samples=128 --eval-override data.max_samples=32 --decoded-rollout-steps 16" \
bash scripts/run_remote_light_promotion.sh
```

Copy `reports/light_experiments_remote/ups_light_v1_task_signature_only/summary.json`
back before destroying the instance.

After W&B-backed runs complete, backfill the local W&B registry:

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

To queue the first post-light residual candidates:

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

Current residual alpha25 result:

- B2 artifact: `remote-runs/light/ups_light_residual_alpha25_20260506T1528Z.tar.gz`
- W&B runs: `00ud83aw`, `3ugaodok`, `i3ej1zp9`, `dm8y4ccc`
- `decoded_rollout_nrmse = 0.5486869325531744`
- Persistence baseline `decoded_rollout_nrmse = 0.5701633411507036`
- Baseline improvement fraction `0.03766711580261458`, so it beats persistence slightly but fails the 20% baseline gate.

Current residual alpha50 result:

- B2 artifact: `remote-runs/light/ups_light_residual_alpha50_20260506T1548Z.tar.gz`
- W&B runs: `dr5wpv23`, `tp1wbop8`, `e3v1o3ce`, `axcvkdcy`
- `decoded_rollout_nrmse = 0.6084554326486734`
- Baseline improvement fraction `-0.06715986233118525`, so it is worse than persistence and worse than alpha25.
- Decision: stop scalar alpha sweeps beyond this point; use `--skip-training --checkpoint-source <run-or-checkpoints-dir>` for any future alpha probes and prioritize trained residual/stability objectives.

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
  "run_name": "ups_light_v1_task_signature_only",
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
  --output-dir reports/demo/light_latest \
  --title "UPS Light-v1 Demo Scorecard" \
  --data-manifest docs/demo_data_manifest.yaml \
  --cost-json reports/light_experiments_remote/ups_light_v1_task_signature_only/cost.json \
  --cost-json reports/light_experiments_remote/persistence_light_v1_test/cost.json \
  --baseline-run persistence_light_v1_test \
  --baseline-metric decoded_rollout_nrmse \
  --baseline-min-improvement 0.2 \
  --promotion-rule "decoded_rollout_nrmse<=1.0" \
  --copy-summaries
```

Expected artifacts:

- `reports/demo/light_latest/index.html`
- `reports/demo/light_latest/metrics.tsv`
- `reports/demo/light_latest/scorecard.json`
- `reports/demo/light_latest/plots/decoded_rollout_nrmse.png`
- `reports/demo/light_latest/plots/decoded_step1_nrmse.png`
- copied summaries under `reports/demo/light_latest/summaries/`

The baseline columns are lower-is-better. A strong demo candidate should show
`baseline_improvement_passed=true` for `ups_light_v1_task_signature_only` against
`persistence_light_v1_test`.

## Step 7: Decide Keep/Discard

Keep current UPS candidate only if:

- it beats persistence on held-out `decoded_rollout_nrmse`, and
- no task-level decoded rollout metric regresses catastrophically, and
- visual/report artifacts are generated cleanly.

If current UPS does not beat persistence:

- do not scale it
- inspect the persistence task breakdown and decoded rollout stability first
- prioritize a persistence-residual architecture or stability-aware decoded
  rollout objective before increasing data scale
- reuse the existing `light-v1` scorecard gate before any medium/full run

## Stop Rules

Stop and ask before:

- hydrating the default full 3-task 141 GiB set at once
- running any full-data benchmark
- spending beyond light/medium cost tier
- making SOTA claims
