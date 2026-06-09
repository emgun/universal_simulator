# Foundation Branch Archaeology: `codex/foundation-performance-roadmap`

Date: 2026-06-09

Status: research notes only. This document records what the divergent foundation-performance branch contains, what should be salvaged for the north-star roadmap phases, and what is retired. It adds no claim evidence and authorizes no held-out test access.

## Branch Facts

- Branch: `codex/foundation-performance-roadmap`, now pushed to origin for preservation (it previously existed only on one machine).
- Divergence point: `447513a` (2025-10-14, "Add TTC evaluation with config includes and checkpoint fixes").
- Last commit: `91f4294` (2026-03-30, "Refactor training pipeline and cheap experiment workflow").
- Scope: 576 commits not on `main`; roughly 38k insertions across 215 files in `src/`, `scripts/`, and `configs/`.
- Relationship to `main`: `main` continued from the divergence point into the sota-loop/claim-evidence line (PRs #44–#75). The branch is a parallel GPU-scale training-infrastructure effort that predates the current claim protocol and shares none of its evidence discipline.

## Why This Matters Now

Phase 1 of `docs/superpowers/plans/2026-06-09-universal-simulator-north-star-roadmap.md` requires GPU training at medium-v1 scale with capacity and data sweeps. This branch already built most of that infrastructure and ran the experiments that should set Phase 1 priors. Salvaging it is cheaper than rebuilding it.

## Salvage List (map to roadmap phases)

Training infrastructure (Phase 1, P1.1–P1.4):

- `src/ups/training/lightning_modules.py` (~880 lines): Lightning multi-GPU training modules, including FSDP fixes (prefix stripping, rank checks, checkpoint save/load paths).
- FSDP + optimizer stack: `src/ups/training/muon.py`, `muon_factory.py`, `hybrid_optimizer.py`, `optim_factory.py`, `param_groups.py`, `distributed_utils.py`. Includes a local Muon implementation that works under FSDP (upstream `torch.optim.Muon` is 2D-only) and a HybridOptimizer state-dict format compatible with Lightning/FSDP checkpoints.
- `LazyPDEBenchDataset` (in `src/ups/data/pdebench.py` on the branch): lazy HDF5 loading for datasets too large for memory. Needed for Phase 1 data-budget sweeps beyond the 32/512-sample caps.
- `src/ups/utils/checkpoint_manager.py`, `checkpoint_utils.py`, `stage_tracker.py`, `wandb_context.py`, `leaderboard.py`: run/checkpoint lifecycle management for long GPU runs.
- Vast launcher improvements: feature-branch checkout for remote runs, `EVAL_ONLY` mode preferring run IDs over artifacts.
- Known-good environment pin: torch 2.5.1 to avoid an NCCL deadlock observed with newer versions.

Architecture and objectives (Phase 1 reference model, explore-track bets):

- `src/ups/models/pure_transformer.py`: alternative plain-transformer backbone, useful as a Phase 1 capacity-sweep candidate.
- `src/ups/training/physics_losses.py`, expanded `losses.py` (+600 lines), `query_sampling.py`: loss-surface options including spectral loss (with the bfloat16 FFT float32 cast fix).
- TTC (test-time conditioning) stack: `src/ups/inference/rollout_ttc.py`, `src/ups/eval/reward_models.py`, TTC eval configs. Relevant to explore-track corrector bets; note the branch's own analyses found TTC reward-model pitfalls (see learnings).

Experiment evidence to treat as Phase 1 priors (not claim evidence):

- Token-count ablation (64/128/256 latent tokens, Burgers-scale): 128 tokens optimal (`0.0577` NRMSE vs `0.0748` at 64 and `0.0596` at 256); 64 under-capacity despite best training-loss optics; 256 overfits the operator. Source: `reports/research/2025-10-30-token-ablation-conclusions.md` (main checkout reports/).
- Data-saturation analysis: at that scale the bottleneck was model capacity and optimization, not data volume. Source: `reports/research/2025-10-30-data-saturation-analysis.md`.
- Diffusion-residual overfitting analyses and ARM/TTC failure analyses under `reports/` in the main checkout: a catalogue of dead ends that Phase 1 and the explore track should not repeat.

## Retire List

- `configs/deprecated/**` on the branch (Burgers 32-dim experiment family, archived experiment configs): superseded; do not port.
- Branch-era eval configs that bypass the current claim protocol (`eval_2task_*`, `ablation_upt_*` as-is): the settings are useful as references, but any future run must be re-expressed under the frozen light-v1/medium-v1/universal-v1 contracts.
- The branch's training entrypoint wiring where it conflicts with `scripts/train.py` on `main`: salvage modules, not the pipeline topology, and port them through small reviewed PRs.

## Decisions

- The branch is preserved on origin and formally retired as a line of development. No further commits should land on it.
- Salvage happens by porting specific modules into `main` through normal PRs with tests, starting when Phase 1 (P1.1 GPU pipeline smoke run) begins. Nothing is ported wholesale.
- The experiment learnings above are planning priors only. None of the branch's numbers are claim-comparable: different protocol, splits, metrics, and era of the codebase.
- The main checkout still has this branch checked out locally; switching it to `main` is blocked only by another worktree holding `main` and is cosmetic now that the branch is pushed.
