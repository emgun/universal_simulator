# Demo Benchmark Contract

This contract defines what counts as evidence for the UPS working demo and what
does not. It exists to keep the demo fast while avoiding accidental benchmark
leakage or inflated claims.

## Goal

Build a reproducible held-out PDEBench demo for:

- `burgers1d`
- `advection1d`
- `darcy2d`

The demo may later expand to larger PDEBench or real-world datasets, but the
first credible target is a small, versioned, held-out shard set with matched
baselines and inspectable rollout artifacts.

## Claim Tiers

- `smoke`: validates plumbing only. May use tiny sample caps and train-shard
  evaluation. Must not be described as benchmark performance.
- `light`: uses versioned train/val/test shards and held-out evaluation. Good
  enough to decide which ideas deserve more compute.
- `medium`: uses larger held-out shards and matched baselines. Good enough for
  demo claims.
- `benchmark`: uses public-compatible splits, metrics, horizons, and baselines.
  Required before any SOTA wording.

## Canonical Demo Data

Initial shard versions:

- `smoke-v1`: tiny remote sanity set.
- `light-v1`: first real held-out experiment set.

Expected sample counts:

| Version | Train | Val | Test | Purpose |
| --- | ---: | ---: | ---: | --- |
| `smoke-v1` | 8 | 4 | 4 | remote plumbing and report sanity |
| `light-v1` | 128 | 32 | 32 | cheap held-out model selection |

Split rules:

- Prefer native source splits when present.
- If a native validation split is missing, derive `val` from `train` and mark
  it as derived in `docs/demo_data_manifest.yaml`.
- Never tune on `test`.
- Never call train-shard metrics benchmark results.

Known current source state:

- `burgers1d`: `train`, `val`, and `test` are expected under the B2 `full/`
  prefix.
- `advection1d`: `train`, `val`, and `test` are expected under the B2 `full/`
  prefix.
- `darcy2d`: `train` and `test` are expected; `val` may need to be derived from
  `train`.

## Canonical Metrics

Primary promotion metric:

- `decoded_rollout_nrmse`

Required supporting metrics:

- `decoded_step1_nrmse`
- `decoded_h4_nrmse` when available
- `decoded_h16_nrmse` when available
- `mse`
- `mae`
- `rmse`
- `task_*_decoded_rollout_nrmse`
- `family_*_decoded_rollout_nrmse`
- wall-clock duration
- estimated GPU cost

Promotion rules should prefer worst-case task or family performance over a
single aggregate when deciding whether a model is demo-safe.

## Baseline Contract

Minimum baselines:

- persistence or identity-style baseline
- current UPS best config
- at least one ablation control

Stronger baselines:

- FNO
- U-Net
- DiT/CNO-style operator if implemented and verified

Do not compare against public SOTA numbers unless the split, horizon, metric,
and preprocessing match exactly. Otherwise label the comparison as
`matched-in-repo`.

## Cost Tiers

| Tier | Data Cap | Expected Use | Approval |
| --- | --- | --- | --- |
| local synthetic | synthetic only | code sanity | no approval |
| remote smoke | `data.max_samples<=8` train, `<=4` eval | remote plumbing | no approval after dry-run |
| light held-out | `128/32/32` shards | idea selection | no approval after dry-run |
| medium | `512-2048` train samples | demo candidate | summarize cost first |
| full | full public split | SOTA attempt | explicit approval |

Hard safety rules:

- Default full-data hydration remains blocked unless `ALLOW_FULL_DATA=1`.
- Real-data light runs must set `data.max_samples` or use small B2 shards.
- Copy `summary.json` and report artifacts before destroying an instance.
- Destroy paid instances after artifact capture and verify zero active instances.

## Initial Remote Harness

Use dry-run first:

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

The corresponding smoke settings are:

```bash
LIGHT_EXTRA_ARGS="--override data.max_samples=8 --eval-override data.max_samples=4 --decoded-rollout-steps 2"
```

## Keep/Discard Rules

Keep a variant if:

- it improves held-out `decoded_rollout_nrmse`, or
- it improves worst-task or worst-family error without a material aggregate
  regression, or
- it fixes a demo artifact or runtime issue without changing metrics.

Discard a variant if:

- it only improves synthetic-bootstrap runs,
- it regresses held-out `test` after tuning on `val`,
- it increases cost without a compensating accuracy or speed benefit,
- it requires full-data spend before beating light held-out controls.

Escalate before running if:

- the command may hydrate the full 141 GiB 3-task data set,
- the expected run cost exceeds the current tier,
- the result would change the public claim tier,
- secrets or remote credentials are missing.

