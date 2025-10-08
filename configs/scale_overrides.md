# Hydra Override Cheatsheet for Scale Runs

These are common overrides to tweak large-scale experiments without editing base configs. Use them on the CLI, e.g.

```
PYTHONPATH=src python scripts/train.py --config configs/train_pdebench_scale.yaml \
  training.batch_size=64 training.amp=true \
  latent.dim=256 latent.tokens=128 \
  stages.operator.epochs=50 stages.operator.optimizer.lr=2e-4
```

## Training knobs
- `training.batch_size`: increase/decrease to match GPU memory.
- `training.amp`: enable mixed precision (`true`) or disable (`false`). Now wired across all stages via autocast/GradScaler.
- `training.grad_clip`: adjust gradient clipping threshold (set to `null` to disable). Applied per-stage if stage override exists.
- `training.ema_decay`: set EMA decay (e.g., `0.999`). You can override per-stage with `stages.<stage>.ema_decay`.
- `training.accum_steps`: enable gradient accumulation to simulate larger batches (default `1`).
- `training.dt`: timestep used in latent operator training.
- `training.distill_micro_batch`: split consistency-distill batches into smaller chunks to reduce GPU memory.

## Latent space
- `latent.dim`: embed dimension per token.
- `latent.tokens`: number of latent tokens produced by encoders.

## Stage-specific overrides
Each stage inherits from the top-level optimizer unless overridden. Example overrides:

- `stages.operator.optimizer.lr`: per-stage learning rate.
- `stages.operator.scheduler.t_max`: adjust cosine schedule horizon.
- `stages.diff_residual.epochs`: change diffusion-residual training length.
- `stages.consistency_distill.optimizer.name=adamw` (switch optimizer).
- `stages.operator.ema_decay=0.9995` (stage-specific EMA).
- `stages.operator.grad_clip=0.8` (stage-specific grad clipping).

## Encoder configuration
When working with non-grid datasets, adjust dataset config fields:

- `data.patch_size`: grid encoder patch size.
- `data.hidden_dim`, `data.message_passing_steps`: mesh/particle encoder capacity.
- `data.supernodes`: particle encoder pooling size.

## Multi-dataset sampling
If mixing tasks, create task-specific configs and launch via Hydra multirun:

```
PYTHONPATH=src python scripts/train.py --config configs/train_pdebench_scale.yaml \
  --multirun data.task=b u r g e r s 1 d , a d v e c t i o n 1 d
```

(Or pass a list of tasks directly: `data.task=[burgers1d,advection1d]`; the loader mixes via `ConcatDataset`.)

## Environment variables
- `PDEBENCH_ROOT`: local path where artifacts are hydrated (set automatically by `scripts/run_remote_scale.sh`).
- `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_GROUP`: logging metadata.

Refer to `docs/scaling_plan.md` for the broader roadmap.
