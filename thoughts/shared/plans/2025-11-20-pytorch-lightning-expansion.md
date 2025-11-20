# PyTorch Lightning Expansion Plan

## Overview
Expand the Lightning path so it can own training, evaluation orchestration, logging, and built-in optimization/tuning. Deliver parity with native multi-stage pipelines (operator → diffusion/consistency), enable Lightning-native validation/checkpoint gating, surface performance callbacks, and expose tuner hooks for batch-size/LR search—all while retaining the existing data pipeline, caching, and multi-task support.

## Current State Analysis
- Operator-only Lightning is live: `scripts/train_lightning.py` maps config → Trainer (DDP/FSDP, AMP/compile, grad clip/accum) and uses `UPSDataModule` → `build_latent_pair_loader` for task-aware, cache-backed loading (scripts/train_lightning.py:35-138; src/ups/data/lightning_datamodule.py:14-45; src/ups/data/latent_pairs.py:700-1139).
- Lightning modules mirror native losses/optimizers (Muon hybrid, schedulers, spectral/rollout/inverse toggles) and log per-task NRMSE (src/ups/training/lightning_modules.py:80-333).
- Orchestration integration: `run_fast_to_sota.py` can swap in Lightning via `--use-lightning`; `run_lightning_pipeline.py` runs operator only (scripts/run_fast_to_sota.py:751-814; scripts/run_lightning_pipeline.py:32-58).
- Validation/logging gaps: Trainer disables val loaders (`limit_val_batches=0`) but still monitors `val/nrmse`, so ModelCheckpoint/EarlyStopping lack signal; eval/gating still rely on native evaluate.py.
- Distributed stability notes: past FSDP/compile crashes on some PyTorch/CUDA builds (docs/distributed_training_outstanding_issues.md:360-417); Lightning tests cover single/2/4 GPU, FSDP, compile toggle (tests/integration/test_lightning_parity.py:1-360).
- Optimization/tuning: Massive speed plan exists for native path; Lightning does not expose built-in tuner (batch size/LR find) or profiling callbacks.

## Desired End State
- Lightning runs full training pipeline stages (operator, diffusion residual, consistency/steady prior) with preserved data loaders, samplers, caches, and task-aware logging.
- Trainer uses real validation splits for checkpointing/early-stop, or explicitly wires external eval metrics into callbacks.
- Fast-to-SOTA + vast_launch can choose Lightning end-to-end with matching WandB artifacts/gate tables.
- Performance knobs surfaced: compile/grad-clip/accum/activation-checkpointing/CPU offload/worker/prefetch remain config-driven; callbacks for throughput/profiler are available.
- Hyperparameter tuning hooks (Lightning tuner for batch-size/LR) optional via CLI flags.

### Key Discoveries
- Validation disabled but monitored (`limit_val_batches=0` + ModelCheckpoint on `val/nrmse`) (scripts/train_lightning.py:95-138).
- Lightning data module already preserves custom distributed samplers (`replace_sampler_ddp=False`) (src/ups/data/lightning_datamodule.py:17-45).
- Multi-task loaders carry `task_names` for per-task metrics; eval still external (src/ups/data/latent_pairs.py:700-1139).

## What We're NOT Doing
- No architectural changes to operators/encoders/decoders.
- No replacement of native path; Lightning remains alternate path until parity proven.
- No new external tuning services (Ray/Optuna); only Lightning-native tuner hooks.
- No multi-node support changes (single-node DDP/FSDP only).

## Implementation Approach
Incremental phases: restore validation + eval/logging parity; add Lightning modules for remaining stages; wire orchestration; surface optimization/tuning/profiling hooks; add tests/docs. Keep native untouched.

## Phase 1: Enable Validation + Checkpoint/Early-Stop Signals
### Overview
Re-enable validation loaders and align metrics so Lightning callbacks have real signals; allow optional external-eval gating when val split absent.
### Changes Required
1) Trainer validation config  
**File**: `scripts/train_lightning.py`  
**Changes**: remove `limit_val_batches=0`; gate via config flag `training.skip_lightning_val` default false. If true, disable ModelCheckpoint/EarlyStopping and rely on external eval.  
2) Data splits  
**File**: `src/ups/data/lightning_datamodule.py`  
**Changes**: ensure `cfg_copy["data"]["split"]="val"` builds a valid loader; if split missing, fallback to train with warning.  
3) Metric naming  
**File**: `src/ups/training/lightning_modules.py`  
**Changes**: add validation-step logging for per-task when `task_names` present; ensure keys align with checkpoint monitor (e.g., `val/nrmse`).  
### Success Criteria
#### Automated
- `python scripts/validate_config.py <config>` passes with `training.skip_lightning_val` toggled.
- `pytest tests/integration/test_lightning_parity.py::test_lightning_vs_native_single_gpu` passes with val enabled.
#### Manual
- With val enabled, trainer saves top-k checkpoints based on `val/nrmse`; early stop triggers when patience hit.
- With `skip_lightning_val=true`, no Lightning checkpoints/early-stop, external eval handles gating; training completes.

## Phase 2: Implement Lightning Stages for Diffusion/Consistency
### Overview
Add Lightning modules for diffusion residual and consistency/steady-prior to mirror native stages and wire into pipeline runner.
### Changes Required
1) New Lightning modules  
**File**: `src/ups/training/lightning_modules.py`  
**Changes**: add `DiffusionLightningModule` and `ConsistencyLightningModule` reusing native configs/losses; support compile/AMP/grad clip; log stage-specific metrics.  
2) Trainer entrypoint routing  
**File**: `scripts/train_lightning.py`  
**Changes**: allow `--stage` ∈ {operator,diff_residual,consistency_distill,steady_prior}; map to corresponding module; load stage cfg (epochs/patience/optimizer/scheduler).  
3) Pipeline runner  
**File**: `scripts/run_lightning_pipeline.py`  
**Changes**: run stages sequentially when epochs>0; preserve global_step continuity via `ckpt_path` between stages; optional `--devices` override; exit nonzero on stage failure.  
4) Orchestrator integration  
**File**: `scripts/run_fast_to_sota.py`  
**Changes**: when `--use-lightning`, route diffusion/consistency stages through train_lightning or pipeline; ensure WANDB_CONTEXT_FILE passed.  
### Success Criteria
#### Automated
- `pytest tests/integration/test_lightning_parity.py::test_multi_stage_pipeline` enhanced to cover diffusion stub passes (or new targeted test).
- Stage CLI runs: `python scripts/train_lightning.py --config <cfg> --stage diff_residual --devices 1` exits 0.
#### Manual
- Multi-stage Lightning run completes operator→diffusion (dummy epochs=1), checkpoints per stage saved, no duplicate WandB runs.

## Phase 3: Logging & Gating Parity with Native Pipeline
### Overview
Align Lightning runs with native fast-to-SOTA artifact/gate reporting and WandB context handling.
### Changes Required
1) WandB context/artifacts  
**File**: `scripts/run_fast_to_sota.py`  
**Changes**: when Lightning used, propagate FAST_TO_SOTA_WANDB_INFO/WANDB_CONTEXT_FILE; capture summary/gate tables from evaluate stages and attach to Lightning run (single run).  
2) Evaluate integration audit  
**File**: `scripts/evaluate.py`  
**Changes**: ensure Lightning-produced checkpoints compatible (already via state_dict strip); document expected ckpt naming.  
3) Metrics alignment  
**File**: `src/ups/training/lightning_modules.py`  
**Changes**: standardize metric names (`train/loss`, `val/nrmse`, `train/<task>/nrmse`) for gating ingestion; optional hook to log wall-clock/throughput.  
### Success Criteria
#### Automated
- Lightning checkpoint loads in native eval: `python scripts/evaluate.py --config <cfg> --operator checkpoints/operator-epoch=00.ckpt` passes smoke test.
- `pytest tests/integration/test_lightning_parity.py::test_checkpoint_compatibility` passes with new ckpt names.  
#### Manual
- Fast-to-SOTA with `--use-lightning` produces summary.json including gate results; WandB artifacts include summary/leaderboard/eval metrics.

## Phase 4: Lightning-Native Evaluation Pipeline
### Overview
Add a Lightning-native eval path so small/full eval, gating, and leaderboard updates can run without the native evaluator, using the same Trainer/Module stack and WandB context.
### Changes Required
1) Eval module  
**File**: `src/ups/training/lightning_modules.py`  
**Changes**: add an eval mode (new class or reuse operator module) to load checkpoints, run forward-only over the eval split, compute metrics/extra, and emit WandB logs/rows. Ensure state_dict stripping handles Lightning ckpts for operator/diffusion/consistency.
2) Eval entrypoint  
**File**: `scripts/evaluate_lightning.py` (NEW)  
**Changes**: CLI `--config --checkpoint --device --output-prefix --leaderboard-* --wandb-*`; uses Lightning Trainer `test`/`predict` to generate metrics JSON and optional leaderboard CSV/HTML mirroring `scripts/evaluate.py`.  
3) Orchestrator integration  
**File**: `scripts/run_fast_to_sota.py`  
**Changes**: add `--lightning-eval` (or auto when `--use-lightning`) to call `evaluate_lightning.py` for small/full stages; propagate WANDB_CONTEXT_FILE; collect metrics for gate checks and write summary.json as today.  
4) Config mapping  
Ensure eval honors `training.num_gpus`, AMP/compile (or forces eval-safe defaults), TTC/eval-specific settings, and task lists; document expectations.  
### Success Criteria
#### Automated
- `python scripts/evaluate_lightning.py --config <cfg> --checkpoint <ckpt> --output-prefix /tmp/out` writes results JSON; no crash on single-GPU smoke.
- New integration test (e.g., `tests/integration/test_lightning_parity.py::test_lightning_eval_smoke`) compares key metrics to native evaluate within tolerance.  
#### Manual
- fast-to-SOTA with `--use-lightning --lightning-eval` completes train + small/full eval; gate decisions and leaderboard outputs match native evaluator for a reference checkpoint/config (within expected numerical noise).

## Phase 5: Performance/Profiling & Tuning Hooks
### Overview
Expose Lightning-native tuning/profiling and align config knobs with optimization plans.
### Changes Required
1) Tuner hooks  
**File**: `scripts/train_lightning.py`  
**Changes**: add flags `--tune-batch-size`, `--tune-lr`; when set, run `trainer.tuner.scale_batch_size` / `tuner.lr_find` before `fit`, apply results back to cfg (with opt-out).  
2) Profiling callbacks  
**File**: `scripts/train_lightning.py`  
**Changes**: optional callbacks (Timer, LearningRateMonitor, PyTorchProfiler) gated by config `training.lightning_profile` (mode: none/timer/profiler).  
3) Config surfacing for optimization knobs  
Ensure Lightning path honors `training.cpu_offload_optimizer`, `training.compile_mode`, `training.num_workers/prefetch_factor/pin_memory`, `use_activation_checkpoint` already set; document supported knobs in docs.  
### Success Criteria
#### Automated
- CLI dry run: `python scripts/train_lightning.py --config <cfg> --stage operator --tune-batch-size --devices 1` executes tuner without crash (limited epochs).  
- Unit test for tuner flag stubs (guarded): add small test in `tests/unit/test_lightning_tuner.py` to assert flags call mocked tuner.  
#### Manual
- Timer/profiler outputs generated when `training.lightning_profile=profiler`; logs show batch-size/lr suggestions; no slowdown when disabled.

## Phase 6: Docs, Tests, and Stability Guardrails
### Overview
Document new Lightning capabilities, add targeted tests, and encode stability guardrails for compile/DDP/FSDP.
### Changes Required
1) Documentation  
**File**: `docs/lightning_training.md`  
**Changes**: update usage with new stages, validation toggle, tuner/profile flags, fast-to-SOTA flow, known stability matrix (compile vs DDP/FSDP).  
**File**: `docs/phase6_completion_summary.md`  
**Changes**: note expanded stages and validation behavior.  
2) Tests  
**Files**: `tests/integration/test_lightning_parity.py` (add val-enabled case, tuner flag noop), add small unit tests for module selection and metric naming.  
3) Stability guards  
**File**: `scripts/train_lightning.py`  
**Changes**: environment checks: if devices>1 and compile enabled with known-bad versions (per docs/distributed_training_outstanding_issues.md), emit warning or disable compile unless `training.force_compile_distributed`.  
### Success Criteria
#### Automated
- `pytest tests/integration/test_lightning_parity.py -m "not slow"` passes.
- `python scripts/train_lightning.py --config <cfg> --stage operator --devices 2` emits compile warning when guarded.  
#### Manual
- Docs reflect new CLI/stage workflow; reviewers can follow to run Lightning end-to-end with or without validation and tuning.

## Testing Strategy
- Unit: tuner flag path; module selection by stage; metric naming for val/train; stability guard toggles; eval script option parsing.  
- Integration: single-GPU operator with val enabled; multi-stage pipeline smoke (operator+diffusion epochs=1); Lightning eval smoke vs native; checkpoint compatibility; fast-to-SOTA `--use-lightning` smoke (can be marked slow).  
- Manual: full-stage run with validation; DDP/FSDP run with compile on/off; tuner flag outputs; profiler callback output; fast-to-SOTA promotion with Lightning; Lightning eval path parity with native gating.

## Performance Considerations
- Keep compile optional for multi-GPU; warn/fallback on known unstable stacks.  
- Tuner runs add overhead—gated via flags only.  
- Profilers can slow runs; default off.  
- Validate that per-task logging remains lightweight; avoid excessive sync_dist on high-frequency metrics.

## Migration Notes
- Backward compatible: native path unchanged; operator-only Lightning default remains.  
- When enabling validation, ensure configs specify a val split or set `training.skip_lightning_val=true`.  
- Checkpoint formats: Lightning ckpts remain loadable by native eval after state_dict key stripping (test included).  
- fast-to-SOTA users: use `--use-lightning` to run Lightning; gating still via evaluate.py but with Lightning-produced checkpoints.

## References
- Readiness research: `thoughts/shared/research/2025-11-20-pytorch-lightning-readiness.md`  
- Performance plan (native): `thoughts/shared/plans/2025-11-13-massive-training-speed-optimization.md`  
- Multi-dataset scaling (data/loading context): `thoughts/shared/plans/2025-11-07-pdebench-multi-dataset-scaling.md`  
- Docs/tests: `docs/lightning_training.md`, `docs/phase6_completion_summary.md`, `tests/integration/test_lightning_parity.py`  
- Stability context: `docs/distributed_training_outstanding_issues.md`
