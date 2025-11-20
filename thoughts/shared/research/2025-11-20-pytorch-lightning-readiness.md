---
date: 2025-11-20T10:01:43-08:00
researcher: Emery Gunselman
git_commit: 37e8e5d
branch: fix/multi-task-crash
repository: universal_simulator
topic: "PyTorch Lightning enablement across training, evaluation, and logging"
tags: [research, codebase, lightning, training, evaluation, logging, distributed]
status: complete
last_updated: 2025-11-20
last_updated_by: Emery Gunselman
---

# Research: PyTorch Lightning enablement across training, evaluation, and logging

**Date**: 2025-11-20T10:01:43-08:00  
**Researcher**: Emery Gunselman  
**Git Commit**: 37e8e5d  
**Branch**: fix/multi-task-crash  
**Repository**: universal_simulator

## Research Question
What has been implemented—and what remains—to fully and optimally enable PyTorch Lightning across training, evaluation, and logging, with emphasis on speed/robustness features plus sizing, hyperparameter, and config tuning for different models and datasets?

## Summary
- Lightning is wired as an alternate operator-stage trainer with DDP/FSDP selection, AMP/compile, Muon hybrid optimizers, CPU offload, and per-task logging; it reuses the native data pipeline via a LightningDataModule and supports trainer.fit/test flows (scripts/train_lightning.py:1-138, src/ups/training/lightning_modules.py:1-333, src/ups/data/lightning_datamodule.py:1-45).
- Automation hooks choose Lightning via `--use-lightning` in the fast-to-SOTA pipeline and VastAI launcher; run_lightning_pipeline provides a sequential runner but only the operator stage is implemented (scripts/run_fast_to_sota.py:751-814, scripts/run_lightning_pipeline.py:1-58, scripts/vast_launch.py:97-111,260-334).
- Evaluation under Lightning runs through `trainer.test` plus the existing evaluate.py stages in run_fast_to_sota; validation is disabled inside the trainer (limit_val_batches=0) while checkpoints/early-stops still monitor `val/nrmse`, so gating relies on downstream evaluate scripts rather than Lightning val loops (scripts/train_lightning.py:95-138, scripts/run_fast_to_sota.py:840-1058).
- Performance tooling is inherited from configs and loaders: multi-task DDP sampling, cached latents, parallel encoding, and configurable workers/prefetch/pin_memory; configs document batch/accum/compile/activation-checkpointing/CPU-offload sizing for DDP/FSDP (src/ups/data/latent_pairs.py:700-1139, configs/train_pdebench_2task_baseline_ddp.yaml:1-140).
- Robustness features include simulated OOM triggers, deterministic/benchmark flags, strategy fallback, and Lightning-native checkpoint compatibility (src/ups/training/distributed_utils.py:1-47, src/ups/training/lightning_modules.py:217-305, tests/integration/test_lightning_parity.py:1-360).
- Docs cover Lightning setup, performance expectations, and known gaps; current limitations note operator-only coverage and historical distributed instability on some stacks (docs/lightning_training.md:1-189, docs/phase6_completion_summary.md:1-170, docs/distributed_training_outstanding_issues.md:360-417).
- Remaining work to “fully” enable Lightning: implement diffusion/consistency/steady-prior Lightning stages and wire them into run_lightning_pipeline/run_fast_to_sota; reconcile validation/checkpoint monitoring with disabled val loaders; align compile/DDP/FSDP guidance with the distributed crash notes; extend eval/logging parity (W&B context/artifacts) and finalize parity benchmarks beyond the existing integration tests.

## Detailed Findings

### Training Entrypoints & Strategies
- `scripts/train_lightning.py` maps config knobs to Lightning Trainer: DDP/FSDP selection from `training.num_gpus`/`use_fsdp2`, precision from AMP settings, grad clip/accum/deterministic/benchmark, WandB logger, and ModelCheckpoint/EarlyStopping (monitor `val/nrmse`) (scripts/train_lightning.py:35-138). Only the operator stage is exposed on the CLI.
- `scripts/run_lightning_pipeline.py` sequences stages but only runs the operator stage; diffusion/steady-prior stages are placeholders (scripts/run_lightning_pipeline.py:32-58).
- `scripts/run_fast_to_sota.py` selects Lightning training when `--use-lightning` is set, otherwise reuses the native script; torchrun is invoked when `training.num_gpus>1`, carrying WANDB context/environment into the Lightning process (scripts/run_fast_to_sota.py:751-814).
- VastAI launcher threads the flag so remote runs can opt into Lightning, tagging WandB appropriately (scripts/vast_launch.py:97-111,260-334).

### Lightning Modules & Optimization Features
- `OperatorLightningModule` builds the LatentOperator from config (supports `pdet_stack`/`pdet_unet`), optionally compiles it with `torch.compile` (modes `default|reduce-overhead|max-autotune`), and passes through dt/spectral/rollout/inverse-loss weights (src/ups/training/lightning_modules.py:204-288).
- Optimizer factory mirrors native behavior, supporting Adam/AdamW/SGD/Muon hybrid with CPU offload and scheduler mapping (StepLR/Cosine/ReduceLROnPlateau) (src/ups/training/lightning_modules.py:80-198,319-333).
- Training step logs per-task NRMSE for multi-task batches and calls the simulated OOM guard for robustness (src/ups/training/lightning_modules.py:223-305; src/ups/training/distributed_utils.py:1-47).

### Data Pipeline & Throughput Controls
- Lightning uses `UPSDataModule`, which wraps `build_latent_pair_loader` for train/val/test splits while preserving DDP-aware samplers (`replace_sampler_ddp=False`) (src/ups/data/lightning_datamodule.py:14-45).
- The underlying loader supports multi-task ConcatDataset, task-balanced DistributedSampler, latent cache preload (mmap), parallel encoding, worker/prefetch/pin_memory tuning, rollout horizon, and inverse-loss toggles (src/ups/data/latent_pairs.py:700-1139).

### Evaluation & Logging Path
- Inside Lightning, validation is disabled (`limit_val_batches=0`) but checkpoints/early-stops still point to `val/nrmse`; `trainer.test` is executed after fit using the test split from the datamodule (scripts/train_lightning.py:95-138).
- In the fast-to-SOTA flow, post-training evaluation still runs via the native `scripts/evaluate.py` stages for small/full eval and leaderboard promotion, independent of Lightning (scripts/run_fast_to_sota.py:873-1139).
- Logging uses Lightning’s WandbLogger with project/entity/run_name/tags from config; run metadata/artifacts are otherwise captured by the outer pipeline (scripts/train_lightning.py:83-138).

### Configs, Sizing, and Hyperparameters
- DDP reference config `configs/train_pdebench_2task_baseline_ddp.yaml` documents capacity and throughput settings: 2×A100, batch_size 30 with accum_steps 2, compile enabled (`compile_mode: default`), CPU optimizer offload, activation checkpointing, BF16 AMP, and task-balanced sampling; notes FSDP2 optional and physics priors/inverse losses disabled for memory (configs/train_pdebench_2task_baseline_ddp.yaml:1-140).
- Additional DDP configs (e.g., train_2task_advection_darcy_ddp.yaml, train_burgers_upt_full_ddp.yaml) provide latent dims/tokens, optimizer/scheduler choices, and TTC/logging defaults for multi-task and single-task runs; these configs are shared between native and Lightning since both read the same YAMLs.

### Testing & Validation
- Integration suite exercises Lightning on single-GPU, 2/4-GPU DDP, FSDP, compile toggle, checkpoint compatibility with native, and the multi-stage pipeline runner; tests assert successful completion rather than metric parity (tests/integration/test_lightning_parity.py:1-360).
- Historical distributed debug logs captured Lightning FSDP crashes on specific PyTorch/CUDA stacks (docs/distributed_training_outstanding_issues.md:360-417), while `docs/distributed_training_SIGSEGV_solution.md` lists mitigations (disable compile, tune NCCL) for multi-GPU stability.

### Documentation & Status
- `docs/lightning_training.md` provides user guidance (CLI, strategy selection, logging, performance tips, limitations) and notes operator-only coverage plus planned extensions (docs/lightning_training.md:1-189,400-472).
- `docs/phase6_completion_summary.md` marks the Lightning migration “complete” for the operator stage, enumerating components, tests, and remaining enhancements such as additional stages, DeepSpeed support, and advanced profiling (docs/phase6_completion_summary.md:1-170,200-250).
- The Phase 6 plan in thoughts outlines success criteria (parity vs native, task-balanced sampling under DDP, FSDP stability, multi-stage resume) and highlights pending manual checks (thoughts/shared/plans/2025-11-12-distributed-training-multi-task.md:2100-2260).

## Code References
- `scripts/train_lightning.py:35-138` – Lightning trainer CLI, strategy/precision mapping, WandB logger, checkpoint/early-stop, val disabled.
- `src/ups/training/lightning_modules.py:80-333` – Operator LightningModule with compile, Muon hybrid optimizer, scheduler mapping, per-task metrics, simulated OOM, spectral/rollout/inverse-loss wiring.
- `src/ups/data/lightning_datamodule.py:14-45` – Datamodule wrapping latent pair loaders with preserved samplers.
- `src/ups/data/latent_pairs.py:700-1139` – Multi-task loader, latent cache preload, DDP samplers, parallel encoding hooks.
- `scripts/run_fast_to_sota.py:751-814` – Selection of Lightning vs native training inside the orchestration pipeline.
- `scripts/run_lightning_pipeline.py:32-58` – Sequential Lightning runner (operator only, placeholders for later stages).
- `configs/train_pdebench_2task_baseline_ddp.yaml:1-140` – DDP sizing/optimization knobs (batch/accum/compile/BF16/CPU offload/checkpointing).
- `tests/integration/test_lightning_parity.py:1-360` – Coverage of single/2/4-GPU, FSDP, compile toggle, checkpoint compatibility, pipeline runner.
- `docs/lightning_training.md:1-189,400-472` – Usage guide, performance guidance, and limitations.
- `docs/phase6_completion_summary.md:1-170,200-250` – Status of Lightning migration and remaining enhancements.
- `docs/distributed_training_outstanding_issues.md:360-417` – Recorded FSDP/Lightning crash context on certain stacks.

## Architecture Documentation
- Lightning path wraps the existing latent-state operator: data flows through `build_latent_pair_loader` (with caching, task-aware samplers) into `UPSDataModule`, feeding `OperatorLightningModule` that directly reuses the native loss bundle and optimizer stack. Trainer strategy/precision are derived from the same YAML configs used by native training, enabling the fast-to-SOTA orchestrator to swap training backends via `--use-lightning` while keeping validation/evaluation/leaderboard logic unchanged. Run_lightning_pipeline is a thin sequential driver awaiting additional stages.

## Historical Context (from thoughts/)
- Lightning migration plan and success criteria, including parity targets and staged implementation order (thoughts/shared/plans/2025-11-12-distributed-training-multi-task.md:2100-2260).
- Distributed crash investigations noting Lightning FSDP failures on specific PyTorch/CUDA builds (docs/distributed_training_outstanding_issues.md:360-417) and mitigation steps (docs/distributed_training_SIGSEGV_solution.md:260-520).
- Performance optimization research (e.g., BF16, DDP tuning, compile guidance) referenced in multi-GPU configs (thoughts/shared/research/2025-11-13-massive-training-speed-optimization.md:240-340).

## Related Research
- `thoughts/shared/research/2025-11-12-distributed-training-analysis.md` – Early distributed training assessment.
- `thoughts/shared/research/2025-11-13-ddp-resolution.md` – DDP fix documentation.
- `docs/phase6_completion_summary.md` – Lightning migration summary and open enhancements.

## Open Questions
- Additional stages: Lightning implementations for diffusion residual, consistency distillation, and steady prior are not present; run_lightning_pipeline/run_fast_to_sota would need to be extended to keep parity with multi-stage native training.
- Validation/checkpointing: Trainer disables val loaders while monitoring `val/nrmse`; how should Lightning handle validation/early-stop in workflows that rely on WandB artifacts vs on-the-fly val splits?
- Distributed stability: Documentation records past FSDP/DDP crashes with specific PyTorch/CUDA/Lightning versions; confirm current stack parity and whether compile should remain enabled for multi-GPU in production configs.
- Logging parity: Lightning uses WandbLogger directly, whereas native training relies on WandBContext; decide whether orchestration artifacts/messages (fast_to_sota gate tables) should be mirrored for Lightning runs.
