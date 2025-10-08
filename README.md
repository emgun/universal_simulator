# Universal Physics Stack (UPS)

Unified latent simulator with discretization-agnostic I/O, transformer core, few-step diffusion residual, steady-state latent prior, physics guards, multiphysics coupling, particles/contacts, DA, and safe control.

Quickstart
- Create env and install: `pip install -e .[dev]`
- Prepare deterministic flags: `bash scripts/prepare_env.sh`
- See milestone checklist: `MILESTONES_TODO.md`

Repository Structure (namespaced under `ups`)
- `src/ups/core`: latent state, conditioning, PDE‑Transformer blocks
- `src/ups/io`: grid/mesh/particle encoders, any‑point decoder
- `src/ups/models`: latent operator, diffusion residual, steady prior, guards, factor graph, particles
- `src/ups/training`: losses, loops, EMA/clip, curricula
- `src/ups/data`: schemas, datasets, transforms, collate
- `src/ups/inference`: rollout, steady solve, DA, control
- `src/ups/eval`: metrics, calibration, gates, reports
- `src/ups/discovery`: nondim, symbolic discovery
- `src/ups/active`: acquisition, mf calibration

Runbook (PoC)
1) `bash scripts/prepare_env.sh && pre-commit run --all-files`
2) `python scripts/prepare_data.py +dataset=poc_trio out=data/poc_trio.zarr`
3) `python scripts/train.py +config=train_multi_pde.yaml stage=operator`
4) `python scripts/train.py +config=train_multi_pde.yaml stage=diff_residual`
5) `python scripts/train.py +config=train_multi_pde.yaml stage=consistency_distill`
6) `python scripts/train.py +config=train_multi_pde.yaml stage=steady_prior`
7) `python scripts/evaluate.py ckpt=checkpoints/op_latest.ckpt mode=transient`
8) `python scripts/evaluate.py ckpt=checkpoints/steady_latest.ckpt mode=steady`
9) `python scripts/infer.py mode=transient input=examples/state.nc --save`
10) `python scripts/infer.py mode=steady bc=examples/bc.json --save`

Artifact Workflow (scaling)
- Package datasets with conversion scripts and upload once: `python scripts/upload_artifact.py <name> dataset artifacts/<file>.tar.gz` (metadata resides in `docs/dataset_registry.yaml`).
- Convert raw PDEBench shards with `python scripts/convert_pdebench_multimodal.py burgers1d ...` (see `docs/data_artifacts.md`).
- Hydrate datasets on any machine via registry helper: `python scripts/fetch_datasets.py burgers1d_subset_v1 --root data/pdebench --cache artifacts/cache`.
- Remote scale run: `WANDB_DATASETS="burgers1d_subset_v1" bash scripts/run_remote_scale.sh` (downloads datasets, runs staged training/eval).
- Reproduce published checkpoints: `bash scripts/repro_pdebench.sh` with `DATASETS`, `CHECKPOINT`, and `DIFFUSION_CHECKPOINT` env vars pointing to W&B artifacts.
- Common Hydra overrides for tuning large runs live in `configs/scale_overrides.md`.
- Pull run metrics for offline review: `python scripts/fetch_wandb_metrics.py jz11ge11 --project universal-simulator --out reports/wandb_jz11ge11.csv`.
- Launch remote training via Vast.ai: `python scripts/vast_launch.py launch --datasets burgers1d_subset_v1 --wandb-project universal-simulator --wandb-entity <entity>` (set your Vast API key with `python scripts/vast_launch.py set-key`).

CI
- GitHub Actions runs lint and unit tests on Python 3.10.

Notes
- Namespaced under `ups` to avoid stdlib collisions (e.g., `io`).
- Prefer bf16 mixed precision; unit-aware training via nondim.
- W&B logging is enabled by default (see `configs/defaults.yaml`). Run `wandb login` once per environment or set `logging.wandb.enabled=false` to disable.
