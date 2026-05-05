# Light Experiment Loop

Use this loop to screen ideas cheaply before spending on larger remote runs.

## What exists

- `scripts/run_light_experiment.py`
  - resolves `include:` configs
  - applies `key=value` overrides
  - can bootstrap tiny PDEBench-style HDF5 files for smoke runs
  - runs one or more training stages
  - runs latent or decoded evaluation
  - writes `summary.json`, resolved configs, checkpoints, and `results.tsv`

- Cheap presets
  - `configs/train_burgers_light_operator.yaml`
  - `configs/train_burgers_light_joint.yaml`
  - `configs/train_multitask_light_operator.yaml`
  - `configs/train_multitask_light_joint_best.yaml`
  - `configs/train_multitask_heterogeneous_light_best.yaml`
  - `configs/eval_burgers_light_proxy.yaml`
  - `configs/eval_multitask_light_proxy.yaml`

## Verified commands

Single-task operator screen:

```bash
python scripts/run_light_experiment.py \
  --config configs/train_burgers_light_operator.yaml \
  --eval-config configs/eval_burgers_light_proxy.yaml \
  --name burgers_operator_trial \
  --output-root reports/light_experiments \
  --bootstrap-synthetic \
  --device cpu
```

Single-task decoded shortlist run:

```bash
python scripts/run_light_experiment.py \
  --config configs/train_burgers_light_joint.yaml \
  --eval-config configs/eval_burgers_light_proxy.yaml \
  --name burgers_joint_trial \
  --output-root reports/light_experiments \
  --bootstrap-synthetic \
  --device cpu \
  --decoded \
  --stage operator \
  --stage decoder \
  --stage joint_codec_operator
```

Multitask operator screen:

```bash
python scripts/run_light_experiment.py \
  --config configs/train_multitask_light_operator.yaml \
  --eval-config configs/eval_multitask_light_proxy.yaml \
  --name multitask_operator_trial \
  --output-root reports/light_experiments \
  --bootstrap-synthetic \
  --device cpu
```

Current multitask decoded shortlist winner:

```bash
python scripts/run_light_experiment.py \
  --config configs/train_multitask_light_joint_best.yaml \
  --name multitask_joint_best_trial \
  --output-root reports/light_experiments \
  --bootstrap-synthetic \
  --device cpu \
  --decoded \
  --stage operator \
  --stage decoder \
  --stage operator_decoded \
  --stage joint_codec_operator
```

Current heterogeneous multitask shortlist winner:

```bash
python scripts/run_light_experiment.py \
  --config configs/train_multitask_heterogeneous_light_best.yaml \
  --name multitask_heterogeneous_best_trial \
  --output-root reports/light_experiments \
  --bootstrap-synthetic \
  --synthetic-samples 8 \
  --synthetic-steps 6 \
  --device cpu \
  --decoded \
  --stage operator \
  --stage decoder \
  --stage operator_decoded \
  --stage joint_codec_operator
```

## Useful overrides

Change one variable at a time:

```bash
--override latent.dim=24
--override latent.tokens=24
--override training.lambda_semigroup=0.1
--override stages.operator.epochs=2
```

Bound real-data smoke cost before loading a large HDF5 shard:

```bash
--override data.max_samples=8
--eval-override data.max_samples=4
--decoded-rollout-steps 2
```

For decoded gates on semantic buckets:

```bash
--promotion-rule "max:family_*_decoded_rollout_nrmse<=0.95"
```

For held-out transfer evaluation:

```bash
--transfer-task advection1d --transfer-split val
```

## Outputs

Each run writes:

- `reports/light_experiments/<name>/resolved_train.yaml`
- `reports/light_experiments/<name>/resolved_eval.yaml`
- `reports/light_experiments/<name>/checkpoints/*`
- `reports/light_experiments/<name>/summary.json`
- `reports/light_experiments/<name>/summary_<split>.json` when `--extra-eval-split` is used

The cross-run table is:

- `reports/light_experiments/results.tsv`

`results.tsv` is keyed by `run_name`. Re-running the same name updates that row instead of appending duplicates.

For demo-ready aggregation across arbitrary local or remote `summary.json` files:

```bash
python scripts/collect_light_results.py \
  reports/light_experiments_remote/*/summary.json \
  --output-tsv reports/demo/latest/metrics.tsv \
  --output-json reports/demo/latest/scorecard.json \
  --data-manifest docs/demo_data_manifest.yaml \
  --promotion-rule "decoded_rollout_nrmse<=1.0"
```

To render a static HTML scorecard:

```bash
python scripts/build_demo_report.py \
  --glob "reports/light_experiments_remote/*/summary.json" \
  --output-dir reports/demo/latest \
  --title "UPS Demo Scorecard" \
  --data-manifest docs/demo_data_manifest.yaml \
  --promotion-rule "decoded_rollout_nrmse<=1.0" \
  --copy-summaries
```

To add a non-learned decoded persistence baseline to the same scorecard:

```bash
python scripts/run_persistence_baseline.py \
  --config configs/train_multitask_heterogeneous_light_best.yaml \
  --name persistence_light_test \
  --output-root reports/light_experiments_remote \
  --data-root data/pdebench_light \
  --split test \
  --max-samples 32 \
  --rollout-steps 16 \
  --promotion-rule "decoded_rollout_nrmse<=1.0"
```

## Promotion guidance

Use this sequence:

1. cheap operator-only latent screening
2. cheap decoded shortlist runs for the survivors
3. remote held-out runs only after the cheap loop improves the right metric

Do not treat synthetic-bootstrap numbers as benchmark numbers. They validate the loop and the relative behavior of code changes, not real PDEBench performance.

## Remote promotion

The local worktree may not contain `.env`; the current local repo copy does:

```bash
ENV_FILE=/Users/emerygunselman/Code/universal_simulator/.env
```

Use a dry run before renting compute or starting training:

```bash
ENV_FILE=/Users/emerygunselman/Code/universal_simulator/.env \
DRY_RUN=1 \
bash scripts/run_remote_light_promotion.sh
```

The default remote promotion script:

- loads B2 credentials from `ENV_FILE` without shell-sourcing it
- dry-runs hydration for `burgers1d`, `advection1d`, and `darcy2d` HDF5 files from the B2 `full/` prefix
- trains `configs/train_multitask_heterogeneous_light_best.yaml`
- evaluates the trained checkpoint on the real held-out `test` split
- writes outputs under `reports/light_experiments_remote/<run_name>/`

Default held-out split is `test` because the current B2 `full/darcy2d/` prefix has `darcy2d_train.h5` and `darcy2d_test.h5`, but no `darcy2d_val.h5`. The default full 3-task train/test file set is about 141 GiB, and the current HDF5 loader reads files into memory, so actual full-data hydration is blocked unless `ALLOW_FULL_DATA=1` is set explicitly. Prefer `REMOTE_DATASET_FILES=...` for smaller shard experiments until small B2 shards exist for all target tasks.

For a cheap real-data smoke against the available Burgers shard:

```bash
ENV_FILE=/Users/emerygunselman/Code/universal_simulator/.env \
TASKS=burgers1d \
TRAIN_CONFIG=configs/train_burgers_light_joint.yaml \
REMOTE_DATASET_FILES=burgers1d/burgers1d_train_000.h5 \
EVAL_SPLIT=train \
REQUIRED_GB=5 \
STAGES=operator,decoder,joint_codec_operator \
RUN_NAME=vast_burgers_shard_cap8 \
LIGHT_EXTRA_ARGS="--override data.max_samples=8 --eval-override data.max_samples=4 --decoded-rollout-steps 2" \
bash scripts/run_remote_light_promotion.sh
```

Verified 2026-05-01 on a Vast.ai RTX 4090 after the shard was already hydrated:

- run: `vast_burgers_shard_cap8`
- summary copied locally to `reports/light_experiments_remote/vast_burgers_shard_cap8/summary.json`
- `decoded_rollout_nrmse = 0.9488316819858322`
- `decoded_step1_nrmse = 0.9453324050249666`
- `duration_sec = 3.8071365356445312`

This is a remote plumbing check, not a benchmark, because it evaluates on a tiny slice of the train shard. For a held-out cheap benchmark, publish small `train/val/test` shards for `burgers1d`, `advection1d`, and `darcy2d`, then pass those keys via `REMOTE_DATASET_FILES`.

Small local or remote HDF5 shards can be cut from already-hydrated source files with:

```bash
python scripts/make_light_hdf5_shards.py \
  --root data/pdebench \
  --out-root data/pdebench_light \
  --tasks burgers1d advection1d darcy2d \
  --train-count 16 \
  --val-count 8 \
  --test-count 8 \
  --manifest docs/demo_data_manifest.yaml \
  --version light-v1 \
  --remote-prefix light-v1 \
  --overwrite
```

The shard builder prefers native split files when they exist. If a native split is missing, it falls back to `train` and records `derived_from_source_split: true` in the manifest; this is expected for `darcy2d` validation until a real `val` split exists.

Dry-run shard publishing without exposing B2 secrets:

```bash
DRY_RUN=1 \
BUILD_SHARDS=1 \
VERSION=light-v1 \
SOURCE_ROOT=data/pdebench \
OUT_ROOT=data/pdebench_light \
MANIFEST=docs/demo_data_manifest.yaml \
bash scripts/publish_light_hdf5_shards_b2.sh
```

When the dry run is correct and the local source files are present, publish with:

```bash
ENV_FILE=/Users/emerygunselman/Code/universal_simulator/.env \
DRY_RUN=0 \
BUILD_SHARDS=1 \
VERSION=light-v1 \
SOURCE_ROOT=data/pdebench \
OUT_ROOT=data/pdebench_light \
MANIFEST=docs/demo_data_manifest.yaml \
bash scripts/publish_light_hdf5_shards_b2.sh
```

If source files are only available in B2 `full/`, use a remote/data-prep box and hydrate one task at a time:

```bash
DRY_RUN=1 \
ENV_FILE=/Users/emerygunselman/Code/universal_simulator/.env \
VERSION=light-v1 \
DATA_ROOT=/workspace/pdebench_full \
OUT_ROOT=/workspace/pdebench_light \
bash scripts/run_remote_shard_prep_b2.sh
```

For smoke-only preparation, source-key overrides can avoid hydrating the
largest native source files when a smaller shard already exists:

```bash
DRY_RUN=1 bash scripts/run_smoke_shard_prep_b2.sh
```

This is only a plumbing shortcut. It intentionally derives smoke validation and
test slices from the fetched train source and must not be used for held-out
benchmark claims.

The actual run needs enough disk for the largest single task source set. Current B2 source sizes are approximately:

- `full/burgers1d`: train 69.045 GiB, val 7.704 GiB, test 15.36 GiB, plus an existing train shard 1.57 GiB
- `full/advection1d`: train 46.03 GiB, val 7.704 GiB, test 7.704 GiB
- `full/darcy2d`: train 2.441 GiB, test 0.613 GiB

After uploading those outputs to B2, run the remote promotion wrapper with `REMOTE_B2_PREFIX=light-v1`. The default generated keys match the publish layout, e.g. `light-v1/burgers1d/burgers1d_train.h5`.

Check whether the manifest's expected B2 keys exist before launching compute.
Use `docs/demo_smoke_data_manifest.yaml` for smoke-tier plumbing checks and
`docs/demo_data_manifest.yaml` for held-out light-tier checks:

```bash
python scripts/check_demo_readiness.py \
  --manifest docs/demo_smoke_data_manifest.yaml \
  --summary-glob "reports/light_experiments_remote/*/summary.json" \
  --baseline-run "" \
  --candidate-run "" \
  --check-b2 \
  --env-file /Users/emerygunselman/Code/universal_simulator/.env
```

```bash
python scripts/check_demo_readiness.py \
  --manifest docs/demo_data_manifest.yaml \
  --summary-glob "reports/light_experiments_remote/*/summary.json" \
  --baseline-run persistence_light_v1_test \
  --candidate-run ups_light_v1_current_best
```

Use `--check-b2` when credentials are available and you need a live shard
presence check.

```bash
ENV_FILE=/Users/emerygunselman/Code/universal_simulator/.env \
python scripts/check_demo_b2_shards.py \
  --manifest docs/demo_data_manifest.yaml \
  --env-file /Users/emerygunselman/Code/universal_simulator/.env
```

After remote runs, record cheap resource metadata beside each `summary.json`:

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

Then include those files in the demo report:

```bash
python scripts/build_demo_report.py \
  --glob "reports/light_experiments_remote/*/summary.json" \
  --output-dir reports/demo/latest \
  --cost-json reports/light_experiments_remote/ups_light_v1_current_best/cost.json \
  --cost-json reports/light_experiments_remote/persistence_light_v1_test/cost.json \
  --baseline-run persistence_light_v1_test \
  --baseline-metric decoded_rollout_nrmse \
  --baseline-min-improvement 0.2
```

Generate bounded smoke/light variant queues from the current roadmap:

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

The generated shell queue defaults to `DRY_RUN=1`. Use `--dry-run-value 0` only
after `scripts/check_demo_b2_shards.py` confirms the requested tier is present.

Vast.ai dry-run launch for the light promotion path:

```bash
python scripts/vast_launch.py launch \
  --dry-run \
  --repo-url <repo-url> \
  --git-ref codex/autowork-semigroup-foundation \
  --remote-script scripts/run_remote_light_promotion.sh \
  --skip-prefetch \
  --disk 256
```

## Current local signal

- Best cheap decoded multitask run so far: `ar_mt_joint_best_joint5`
- Config shape: `operator -> decoder -> operator_decoded -> joint_codec_operator`
- Best observed metric on the synthetic multitask harness:
  - `decoded_rollout_nrmse = 0.9301626682281494`
  - `decoded_step1_nrmse = 0.862861692905426`
- Main losers on the same harness:
  - wider latent/operator capacity at the same tiny budget
  - removing conditioning entirely
  - flat-only or node-only reduced conditioning surfaces
  - lowering the joint-stage learning rate

The cheap harness is currently saying the most promising near-term direction is not a new backbone. It is decoded training depth: keep the full conditioning path, include `operator_decoded`, and spend more budget in the joint codec/operator stage.

## Heterogeneous multitask signal

- Harder local harness:
  - `burgers1d + advection1d + darcy2d`
- The synthetic harness had to be fixed to support:
  - scalar 2D channel-first synthetic files
  - heterogeneous decoded metric aggregation across different grid sizes
- Best heterogeneous run so far:
  - `ar_3task_val8_task_signature_joint32`
  - `decoded_rollout_nrmse = 0.7919775048580832`
  - `decoded_step1_nrmse = 0.6038905236219648`
- Matched larger-split no-conditioning control:
  - `ar_3task_val8_no_conditioning_joint32`
  - `decoded_rollout_nrmse = 0.9471399362116322`
- Main interpretation:
  - the AdaLN conditioner needed exact-neutral modulation; the previous gate formula shrank activations even with zero-initialized conditioning projections
  - after that fix, reduced flat semantic conditioning is the best local signal
  - on the larger synthetic split, `task_id + equation_signature` beats fuller flat bundles and simpler single-source variants
  - set-structured node conditioning still trails the flat semantic surface on the cheap harness
  - the most promising immediate research direction is flat, explicit PDE/task semantics plus decoded joint training, not larger node-set conditioning
