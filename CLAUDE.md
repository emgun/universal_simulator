# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Universal Physics Stack (UPS)** is a unified latent simulator with discretization-agnostic I/O, transformer core, few-step diffusion residual, steady-state latent prior, physics guards, and test-time conditioning (TTC).

The project uses a **latent space approach**: PDEs are encoded into a low-dimensional latent representation, evolved with a PDE-Transformer operator, optionally refined with a diffusion residual model, then decoded back to physical space at arbitrary query points.

## Key Commands

### Training

**Production Training (VastAI - Recommended):**
```bash
# One-time setup (configure credentials)
python scripts/vast_launch.py setup-env

# Launch training with auto-shutdown
python scripts/vast_launch.py launch \
  --config configs/train_burgers_golden.yaml \
  --auto-shutdown
```

**Local Training (Native PyTorch):**
```bash
# Full pipeline (all stages)
python scripts/train.py --config configs/train_burgers_golden.yaml --stage all

# Single stage
python scripts/train.py --config configs/train_burgers_golden.yaml --stage operator
# Stages: operator, diff_residual, consistency_distill, steady_prior
```

**Local Training (PyTorch Lightning - Alternative):**
```bash
# Single-GPU training
python scripts/train_lightning.py \
  --config configs/train_burgers_golden.yaml \
  --stage operator

# Multi-GPU training (DDP)
torchrun --nproc_per_node=2 scripts/train_lightning.py \
  --config configs/train_pdebench_2task_baseline_ddp.yaml \
  --stage operator

# Multi-GPU training (FSDP)
# First enable FSDP in config: training.use_fsdp2: true
torchrun --nproc_per_node=4 scripts/train_lightning.py \
  --config configs/train_pdebench_2task_baseline_ddp_4gpu.yaml \
  --stage operator

# See docs/lightning_training.md for full documentation
```

**Fast-to-SOTA Pipeline:**
```bash
# Complete pipeline: training → small eval → full eval
python scripts/run_fast_to_sota.py \
  --train-config configs/train_burgers_golden.yaml \
  --small-eval-config configs/small_eval_rerun_txxoc8a8.yaml \
  --full-eval-config configs/full_eval_rerun_txxoc8a8.yaml
```

### Validation & Testing

```bash
# Validate configuration before launch
python scripts/validate_config.py configs/train_burgers_32dim.yaml

# Validate data integrity
python scripts/validate_data.py configs/train_burgers_32dim.yaml

# Dry-run (test + cost estimate)
python scripts/dry_run.py configs/train_burgers_32dim.yaml --estimate-only

# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run specific test
pytest tests/unit/test_leaderboard.py -v

# Run tests in parallel
pytest -n auto
```

### Linting & Formatting

```bash
# Run pre-commit hooks
pre-commit run --all-files

# Format with black
black src/ tests/ scripts/

# Sort imports
isort src/ tests/ scripts/

# Lint with ruff
ruff check src/ tests/ scripts/
```

### Evaluation

```bash
# Evaluate a checkpoint
python scripts/evaluate.py \
  --checkpoint checkpoints/op_latest.ckpt \
  --config configs/eval_burgers_32dim_practical.yaml

# Analyze a training run
python scripts/analyze_run.py <run_id> --output reports/analysis.md

# Compare multiple runs
python scripts/compare_runs.py run1_id run2_id run3_id
```

### Data Management

```bash
# Precompute latent cache (with timeouts)
python scripts/precompute_latent_cache.py \
  --config configs/cache_precompute_defaults.yaml \
  --tasks burgers1d --splits train \
  --dataloader-timeout 120 \
  --hdf5-timeout 60 \
  --watchdog-timeout 120

# Copy data to local storage on VastAI/Vultr (10x faster I/O)
bash scripts/copy_data_to_local.sh /root/data/pdebench /workspace/data_local/pdebench

# Then precompute with local data
python scripts/precompute_latent_cache.py \
  --config configs/cache_precompute_defaults.yaml \
  --root /workspace/data_local/pdebench \
  --tasks burgers1d --splits train
```

### Experiment Management

```bash
# Archive completed/failed experiments
python scripts/archive_experiments.py --status all --dry-run
python scripts/archive_experiments.py --status all

# List experiment status
python scripts/archive_experiments.py --list-only

# Promote successful experiment to production
python scripts/promote_config.py \
  experiments/2025-01-22-my-experiment/config.yaml \
  --production-dir configs/ \
  --rename train_burgers_improved.yaml \
  --update-leaderboard

# Catalog all configs
python scripts/config_catalog.py

# Deprecate outdated config
python scripts/deprecate_config.py configs/old_config.yaml \
  --reason "Superseded by golden config" \
  --replacement train_burgers_golden.yaml \
  --move-to-deprecated
```

### Monitoring

```bash
# Show VastAI instances
vastai show instances

# Watch logs (live)
vastai logs <instance_id> -f

# SSH into instance
vastai ssh <instance_id>

# Monitor instance startup and logs
scripts/monitor_instance.sh <instance_id>
```

## Architecture

### Package Structure

All code is namespaced under `src/ups/` to avoid stdlib collisions:

- **`src/ups/core/`** - Core abstractions
  - `latent_state.py` - Latent state representation
  - `blocks_pdet.py` - PDE-Transformer blocks (shifted window attention)
  - `conditioning.py` - Adaptive layer norm conditioning
  - `shifted_window.py` - Window-based attention for efficiency

- **`src/ups/io/`** - Discretization-agnostic I/O
  - `enc_grid.py` - Grid encoder (Burgers, NS, etc.)
  - `enc_mesh_particle.py` - Mesh/particle encoder
  - `decoder_anypoint.py` - Query-based decoder for arbitrary points

- **`src/ups/models/`** - Core models
  - `latent_operator.py` - Latent evolution operator (PDE-Transformer backbone)
  - `diffusion_residual.py` - Diffusion model for uncertainty/refinement
  - `steady_prior.py` - Steady-state prior model
  - `physics_guards.py` - Physics-based constraints
  - `multiphysics_factor_graph.py` - Multiphysics coupling
  - `particles_contacts.py` - Particle dynamics

- **`src/ups/training/`** - Training infrastructure
  - `loop_train.py` - Training loops
  - `losses.py` - Loss functions (NRMSE, spectral energy, etc.)
  - `consistency_distill.py` - Consistency distillation for few-step diffusion

- **`src/ups/data/`** - Data handling
  - `datasets.py` - Dataset abstractions
  - `pdebench.py` - PDEBench data loader
  - `latent_pairs.py` - Latent state pair generation
  - `parallel_cache.py` - Parallel latent cache computation
  - `schemas.py` - Data schemas

- **`src/ups/inference/`** - Inference modes
  - `rollout_transient.py` - Transient rollout (autoregressive)
  - `rollout_ttc.py` - Test-time conditioning rollout
  - `da_latent.py` - Data assimilation
  - `control_safe.py` - Safe control

- **`src/ups/eval/`** - Evaluation tools
  - `metrics.py` - Error metrics (MSE, NRMSE, etc.)
  - `calibration.py` - Uncertainty calibration
  - `gates.py` - Physics gate checks
  - `physics_checks.py` - Conservation law verification
  - `reward_models.py` - Physics-based reward models for TTC
  - `pdebench_runner.py` - PDEBench evaluation runner
  - `reports.py` - Report generation

- **`src/ups/utils/`** - Utilities
  - `config_loader.py` - YAML config loading
  - `leaderboard.py` - Leaderboard tracking
  - `wandb_context.py` - Clean WandB integration (one run per pipeline)

### Script Organization

**Core Pipeline Scripts:**
- **`scripts/run_fast_to_sota.py`** - Main orchestrator (validation → train → eval → gating → leaderboard)
- **`scripts/train.py`** - Training engine (standalone or called by orchestrator)
- **`scripts/evaluate.py`** - Evaluation engine (standalone or called by orchestrator)
- **`scripts/vast_launch.py`** - VastAI provisioning (generates onstart scripts that call orchestrator)

**Validation & Quality:**
- **`scripts/validate_config.py`** - Config validation
- **`scripts/validate_data.py`** - Data integrity checks
- **`scripts/dry_run.py`** - Dry-run testing with cost estimates

**Experiment Management:**
- **`scripts/archive_experiments.py`** - Archive completed/failed experiments
- **`scripts/promote_config.py`** - Promote successful configs to production
- **`scripts/config_catalog.py`** - Generate config catalog CSV

**Analysis & Utilities:**
- **`scripts/analyze_run.py`** - Post-run analysis
- **`scripts/compare_runs.py`** - Multi-run comparison
- **`scripts/precompute_latent_cache.py`** - Pre-encode training data
- **`scripts/monitor_instance.sh`** - VastAI instance monitoring

**Architecture Notes:**
- `run_fast_to_sota.py` delegates to `train.py` and `evaluate.py` via subprocess calls
- `train.py` and `evaluate.py` can be used standalone for debugging
- `vast_launch.py` generates `.vast/onstart.sh` which calls `run_fast_to_sota.py`

### Key Architectural Concepts

1. **Latent Space Evolution**: Physical fields → Encoder → Latent tokens → Operator → Latent tokens → Decoder → Physical fields

2. **Training Stages** (sequential pipeline):
   - **Operator**: Train deterministic latent evolution model (PDE-Transformer)
   - **Diffusion Residual**: Train diffusion model for uncertainty/correction
   - **Consistency Distillation**: Distill diffusion model to few-step sampler
   - **Steady Prior**: Train steady-state prior (optional)

3. **Test-Time Conditioning (TTC)**: At inference, use analytical physics rewards (mass/energy conservation) to guide sampling and select best rollout candidates

4. **Latent Caching**: Pre-encode training data to latent space once, then train directly on latent pairs (faster training, less GPU memory)

5. **Physics Guards**: Enforce conservation laws and boundary conditions during training/inference

## Configuration System

Configs are YAML files in `configs/` using a hierarchical structure:

**Production Config**: `configs/train_burgers_32dim.yaml`
- 32-dim latent space
- 25 epochs operator training
- Enhanced TTC (8 candidates, beam width 3)
- Expected NRMSE: ~0.09 with TTC (vs ~0.78 baseline)
- Training time: ~25 min on A100

**Key Config Sections**:
- `data` - Dataset configuration (root, split, download settings)
- `latent` - Latent space dimensions (dim, tokens)
- `operator` - PDE-Transformer architecture (hidden_dim, depths, num_heads)
- `diffusion` - Diffusion model architecture
- `training` - Training hyperparameters (batch_size, lr, optimizer, etc.)
- `stages` - Per-stage training config (operator, diff_residual, etc.)
- `ttc` - Test-time conditioning parameters
- `logging` - WandB logging configuration

**Important**: `latent.dim` must match `operator.pdet.input_dim`, `diffusion.latent_dim`, and `ttc.decoder.latent_dim`

**Config Management**: See `configs/README.md` for:
- Golden config identification (which configs to use)
- Deprecated config list (which to avoid)
- Config naming conventions
- Promotion workflow

## Development Workflow

### Adding New Features

1. **Add unit tests first** in `tests/unit/`
2. **Implement feature** in appropriate `src/ups/` module
3. **Add integration test** in `tests/integration/` if needed
4. **Update config** if feature requires new parameters
5. **Run validation**: `pytest tests/ && pre-commit run --all-files`

### Configuration Changes

- Always validate configs: `python scripts/validate_config.py <config_path>`
- Keep dimension consistency: `latent.dim == operator.pdet.input_dim == diffusion.latent_dim`
- Test locally with `--stage operator --epochs 1` before full runs

### Experiment Lifecycle Management

**Organized iteration** to prevent experimental configs from piling up:

```bash
# 1. Start new experiment
mkdir -p experiments/$(date +%Y-%m-%d)-my-experiment
cp configs/train_burgers_golden.yaml experiments/$(date +%Y-%m-%d)-my-experiment/config.yaml

# 2. Run experiment
python scripts/vast_launch.py launch \
  --config experiments/2025-01-22-my-experiment/config.yaml \
  --auto-shutdown

# 3. Document results in experiments/2025-01-22-my-experiment/notes.md

# 4a. If successful: Promote to production
python scripts/promote_config.py \
  experiments/2025-01-22-my-experiment/config.yaml \
  --production-dir configs/ \
  --rename train_burgers_64dim.yaml \
  --update-leaderboard

# 4b. If completed/failed: Archive
python scripts/archive_experiments.py --status all
```

**Automated archiving** keeps `experiments/` clean:

```bash
# List experiment status
python scripts/archive_experiments.py --list-only

# Archive completed experiments (dry-run first)
python scripts/archive_experiments.py --status all --dry-run

# Actually archive
python scripts/archive_experiments.py --status all
```

**Experiment status classifications**:
- **active**: Recent runs (< 7 days) or in-progress
- **success**: Passed validation gates, ready to promote
- **failed**: Errors or failing gates
- **stale**: No activity for > 30 days

See `experiments/README.md` for detailed workflow.

### Production Training Checklist

1. **Validate config**: `python scripts/validate_config.py <config>`
2. **Validate data**: `python scripts/validate_data.py <config>`
3. **Dry-run estimate**: `python scripts/dry_run.py <config> --estimate-only`
4. **Launch**: `python scripts/vast_launch.py launch --config <config> --auto-shutdown`
5. **Monitor**: Check WandB dashboard + `vastai logs <instance_id>`
6. **Analyze**: `python scripts/analyze_run.py <run_id>` after completion

### VastAI Best Practices

- **Always use `--auto-shutdown`** to avoid idle costs
- **Use Git clone method** for development/debugging (2-3 min startup)
- **Use Docker method** for production (1-2 min startup, requires GitHub Actions build)
- **Global env-vars** are set once via `python scripts/vast_launch.py setup-env`
- **Training data** is downloaded by the onstart script from B2 cloud storage
- **Test/val data** is downloaded from WandB artifacts during pipeline

## Important Technical Details

### Multi-Stage Training

Training happens in sequential stages, each with its own checkpoint:

1. **Operator** (`checkpoints/op_latest.ckpt`): Deterministic latent evolution
2. **Diffusion** (`checkpoints/diff_latest.ckpt`): Loads operator, trains diffusion residual
3. **Distill** (`checkpoints/distill_latest.ckpt`): Loads diffusion, distills to few-step
4. **Steady** (`checkpoints/steady_latest.ckpt`): Optional steady-state prior

Each stage loads the previous stage's checkpoint as a starting point.

### Latent Cache

- Precomputed latent pairs stored in `data/latent_cache/`
- Speeds up training by ~3-5x (no re-encoding each epoch)
- Cache is dataset + encoder specific
- Set `RESET_CACHE=1` (default) to regenerate cache on VastAI runs

### Test-Time Conditioning (TTC)

TTC improves predictions by using physics-based rewards:
- **Analytical rewards**: Mass/energy conservation checks
- **Beam search**: Evaluate multiple candidate trajectories
- **Best selection**: Pick rollout with best physics scores

Key parameters (`ttc` section in config):
- `candidates`: Number of rollout candidates to evaluate (default: 8)
- `beam_width`: Beam width for search (default: 3)
- `max_evaluations`: Maximum beam evaluations (default: 150)

### WandB Integration

**Clean Architecture (New):**
- **One Pipeline = One WandB Run** (no run proliferation!)
- Uses `WandBContext` for centralized logging (`src/ups/utils/wandb_context.py`)
- Proper data types:
  - Time series (training): `ctx.log_training_metric(stage, metric, value, step)`
  - Final scalars (eval): `ctx.log_eval_summary(metrics, prefix="eval")`
  - Metadata: `ctx.update_config(updates)`
  - Tables: `ctx.log_table(name, columns, data)`
- Context passed to subprocesses via `WANDB_CONTEXT_FILE` environment variable
- See `WANDB_IMPLEMENTATION_SUMMARY.md` for details

**Setup:**
- Required env vars: `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`
- Disable with: `logging.wandb.enabled=false` in config or `mode="disabled"`
- Artifacts: Datasets and checkpoints uploaded automatically

### Python Environment

- **Python 3.10+** required
- Install: `pip install -e .[dev]` for development
- Key dependencies: PyTorch 2.3+, einops, xarray, zarr, wandb
- Mixed precision (bf16) is default: `training.amp=true`
- torch.compile enabled by default: `training.compile=true`

### Code Style

- Line length: 100 characters
- Formatter: black
- Import sorter: isort (black-compatible profile)
- Linter: ruff
- Type hints: Use `from __future__ import annotations` for forward references

## Common Issues

### Dimension Mismatch Errors

**Problem**: `latent.dim != operator.pdet.input_dim`
**Solution**: Ensure all dimension fields match in config:
```yaml
latent:
  dim: 32
operator:
  pdet:
    input_dim: 32  # Must match latent.dim
diffusion:
  latent_dim: 32  # Must match latent.dim
ttc:
  decoder:
    latent_dim: 32  # Must match latent.dim
```

### Out of Memory (OOM)

**Solutions**:
- Reduce `training.batch_size`
- Reduce `latent.tokens` or `latent.dim`
- Reduce `operator.pdet.hidden_dim`
- Disable compile: `training.compile=false`

### Poor Convergence

**Check**:
- Operator final loss should be < 0.001 (typically ~0.0002)
- Learning rate schedule (cosine annealing is default)
- Gradient clipping: `training.grad_clip=1.0`
- See `parallel_runs_playbook.md` for hyperparameter sweep guidance

### VastAI Instance Issues

**Instance stuck in "loading"**:
```bash
vastai destroy instance <ID>
# Relaunch
```

**Training errors**:
```bash
# SSH and check logs
vastai ssh <instance_id>
tail -100 /workspace/universal_simulator/logs/*.log
```

## Documentation

- **README.md**: Project overview and quick start
- **QUICKSTART.md**: 5-minute quick start guide
- **PRODUCTION_WORKFLOW.md**: VastAI training workflow
- **parallel_runs_playbook.md**: Hyperparameter sweep guidance
- **docs/production_playbook.md**: Best practices and decision trees
- **docs/runbook.md**: Operational procedures
- **docs/vastai_env_setup.md**: VastAI credential setup

## Reference Performance (32-dim Burgers)

Expected metrics for `configs/train_burgers_32dim.yaml`:

| Metric | Value |
|--------|-------|
| Baseline NRMSE | ~0.78 |
| TTC NRMSE | ~0.09 |
| Improvement | 88% |
| Training time | ~25 min (A100) |
| Cost | ~$1.25 @ $1.89/hr |
| Operator final loss | ~0.00023 |

Reference run: `emgun-morpheus-space/universal-simulator/rv86k4w1`
