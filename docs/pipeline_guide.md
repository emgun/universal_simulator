# Universal Simulator Training & Evaluation Pipeline

**Last Updated:** October 14, 2025

This guide documents the standard, streamlined pipeline for training and evaluating the Universal Simulator on Burgers 1D and other PDE tasks.

## Quick Start

### Local Smoke Test
```bash
# Run quick end-to-end test on CPU with synthetic data
bash scripts/smoke_test.sh
```

### Remote Training (Full Pipeline)
```bash
# Set environment variables
export WANDB_API_KEY=your_key_here
export B2_KEY_ID=your_b2_key_id
export B2_APP_KEY=your_b2_app_key
export B2_BUCKET=your_bucket
export WANDB_PROJECT=universal-simulator
export WANDB_DATASETS=burgers1d_full_v1

# Launch training on Vast.ai or other remote GPU
TRAIN_CONFIG=configs/train_burgers_quality_v3.yaml \
bash scripts/run_remote_scale.sh
```

### Evaluation Only (Using Existing Checkpoints)
```bash
# Evaluate with checkpoints from W&B
EVAL_ONLY=1 \
OPERATOR_ARTIFACT=run-mt7rckc8-history:v0 \
DIFFUSION_ARTIFACT=run-pp0c2k31-history:v0 \
EVAL_CONFIG=configs/eval_burgers_512dim_ttc_val.yaml \
EVAL_TEST_CONFIG=configs/eval_burgers_512dim_ttc_test.yaml \
bash scripts/run_remote_scale.sh
```

## Pipeline Architecture

### Core Components

1. **Training Script:** `scripts/train.py`
   - Multi-stage training: operator → diffusion → consistency distillation
   - Supports resume from checkpoints
   - W&B logging and artifact management

2. **Evaluation Script:** `scripts/evaluate.py`
   - Latent operator evaluation with optional TTC
   - Comprehensive metrics and visualizations
   - Saves results as JSON/CSV/HTML/PNG

3. **Remote Orchestrator:** `scripts/run_remote_scale.sh`
   - Handles data hydration from B2 or W&B
   - Runs full training pipeline or eval-only mode
   - Auto-manages checkpoints and artifacts

4. **Configuration System:**
   - Base configs with overlays (via `include:`)
   - YAML-based, validated with Pydantic
   - See `configs/train_burgers_quality_v3.yaml` as canonical example

## Standard Workflows

### Workflow 1: Full Training Run

```bash
# 1. Prepare environment
export WANDB_API_KEY=...
export B2_KEY_ID=...
export B2_APP_KEY=...
export B2_BUCKET=pdebench

# 2. Set dataset and config
export WANDB_DATASETS=burgers1d_full_v1
export TRAIN_CONFIG=configs/train_burgers_quality_v3.yaml

# 3. Run training (auto-evaluates at end)
bash scripts/run_remote_scale.sh

# Results:
# - Checkpoints in checkpoints/scale/*.pt
# - Reports in reports/*.json
# - W&B run with all metrics logged
```

**What happens:**
1. Hydrates dataset from B2 to `data/pdebench/`
2. Precomputes latent cache (optional, controlled by `PRECOMPUTE_LATENT`)
3. Trains operator (stage 1)
4. Trains diffusion residual (stage 2)
5. Trains consistency distillation (stage 3, optional)
6. Evaluates on validation split
7. Evaluates on test split
8. Uploads checkpoints to W&B as artifacts

### Workflow 2: Evaluation with Existing Checkpoints

```bash
# 1. Set evaluation mode
export EVAL_ONLY=1

# 2. Specify checkpoint artifacts from W&B
export OPERATOR_ARTIFACT=run-mt7rckc8-history:v0
export DIFFUSION_ARTIFACT=run-pp0c2k31-history:v0

# 3. Specify evaluation configs
export EVAL_CONFIG=configs/eval_burgers_512dim_ttc_val.yaml
export EVAL_TEST_CONFIG=configs/eval_burgers_512dim_ttc_test.yaml

# 4. Run evaluation
bash scripts/run_remote_scale.sh

# Results:
# - Downloaded checkpoints in checkpoints/scale/
# - Reports in reports/*.json
# - W&B run with evaluation metrics
```

**What happens:**
1. Downloads checkpoints from W&B artifacts
2. Skips training entirely
3. Evaluates on validation split
4. Evaluates on test split
5. Uploads evaluation results to W&B

### Workflow 3: TTC Evaluation

TTC (Test-Time Computation) uses trajectory sampling and reward models to select better predictions at inference time.

```bash
# Use TTC-enabled configs
export EVAL_CONFIG=configs/eval_burgers_512dim_ttc_val.yaml
export EVAL_TEST_CONFIG=configs/eval_burgers_512dim_ttc_test.yaml

# Run evaluation
EVAL_ONLY=1 bash scripts/run_remote_scale.sh

# View TTC-specific outputs:
# - reports/*_ttc_step_logs.json (trajectory selection details)
# - reports/*_ttc_rewards.png (reward progression plots)
```

**TTC Configuration Options:**
- `ttc.candidates`: Number of trajectories to sample (default: 6)
- `ttc.beam_width`: Beam search width (1 = greedy, 2+ = beam search)
- `ttc.reward.grid`: Resolution for reward model decoder (e.g., [64, 64])
- `ttc.sampler.noise_std`: Noise level for trajectory diversity

See [docs/ttc_analysis_lowmem.md](ttc_analysis_lowmem.md) for detailed TTC analysis.

## Directory Structure

```
universal_simulator/
├── configs/              # Configuration files
│   ├── defaults.yaml    # Base configuration
│   ├── train_burgers_quality_v3.yaml  # Latest training config
│   ├── eval_burgers_512dim_ttc_val.yaml  # TTC evaluation
│   └── ...
├── scripts/              # Pipeline scripts
│   ├── train.py         # Main training script
│   ├── evaluate.py      # Evaluation script
│   ├── run_remote_scale.sh  # Remote orchestrator
│   ├── smoke_test.sh    # Quick validation test
│   └── ...
├── src/ups/              # Core library code
│   ├── data/            # Data loaders
│   ├── models/          # Model architectures
│   ├── eval/            # Evaluation runners
│   └── ...
├── archive/              # Deprecated configs/scripts
│   ├── configs/         # Old configuration files
│   └── scripts/         # Old/deprecated scripts
├── docs/                 # Documentation
│   ├── pipeline_guide.md  # This file
│   └── ttc_analysis_lowmem.md  # TTC analysis
├── checkpoints/          # Model checkpoints (gitignored)
├── reports/              # Evaluation results (gitignored)
└── data/                 # Datasets (gitignored)
```

## Configuration System

### Base Configuration Pattern

Configs use `include:` to build on base configs:

```yaml
# configs/my_experiment.yaml
include: train_burgers_quality_v3.yaml

# Override specific settings
training:
  batch_size: 8

logging:
  wandb:
    run_name: my-experiment
```

### Key Configuration Sections

1. **data:** Dataset settings (task, split, root, patch_size)
2. **latent:** Latent space dimensions (dim, tokens)
3. **training:** Training hyperparameters (batch_size, epochs, learning rates)
4. **operator:** Operator model architecture (PDET settings)
5. **stages:** Training stage configurations (epochs per stage)
6. **ttc:** Test-time computation settings (optional)
7. **logging:** W&B and output settings

See `configs/defaults.yaml` for full schema.

## Environment Variables

### Required for Training
- `WANDB_API_KEY`: Weights & Biases API key
- `WANDB_PROJECT`: W&B project name (default: universal-simulator)

### Optional for B2 Data Hydration
- `B2_KEY_ID`: Backblaze B2 application key ID
- `B2_APP_KEY`: Backblaze B2 application key
- `B2_BUCKET`: B2 bucket name (default: pdebench)
- `WANDB_DATASETS`: Comma-separated list of dataset names to download

### Pipeline Control
- `EVAL_ONLY=1`: Skip training, only run evaluation
- `PRECOMPUTE_LATENT=1`: Precompute latent cache before training (default: 1)
- `RESET_CACHE=1`: Clear latent cache and checkpoints before run (default: 1)
- `CLEANUP_AFTER_RUN=1`: Remove temp files after completion (default: 1)
- `FIX_LIBCUDA=1`: Fix libcuda.so symlink on remote (default: 1)

### Configuration Overrides
- `TRAIN_CONFIG`: Path to training config (default: configs/train_burgers_quality_v2.yaml)
- `EVAL_CONFIG`: Path to validation eval config
- `EVAL_TEST_CONFIG`: Path to test eval config
- `TRAIN_STAGE`: Which stage to run (default: all, options: operator, diff_residual, consistency_distill)

### Checkpoint Management
- `OPERATOR_ARTIFACT`: W&B artifact path for operator checkpoint
- `DIFFUSION_ARTIFACT`: W&B artifact path for diffusion checkpoint
- `CONSISTENCY_ARTIFACT`: W&B artifact path for consistency checkpoint (optional)

## Common Tasks

### Resume Training from Checkpoint

```bash
# Download checkpoint from W&B
PYTHONPATH=src python scripts/download_checkpoints_from_wandb.py \
  --dest checkpoints/scale \
  --operator-artifact run-abc123-history:v0

# Resume training from specific stage
TRAIN_STAGE=diff_residual bash scripts/run_remote_scale.sh
```

### Evaluate on Custom Split

Edit your eval config:
```yaml
data:
  split: custom_test  # or val, test, train
  root: data/pdebench
```

Then run:
```bash
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/my_eval.yaml \
  --operator checkpoints/scale/operator.pt \
  --diffusion checkpoints/scale/diffusion_residual.pt \
  --output-prefix reports/my_eval
```

### Upload Results as W&B Artifact

```bash
PYTHONPATH=src python scripts/upload_artifact.py \
  my-results-v1 \
  evaluation \
  reports/my_eval.json \
  reports/my_eval.csv \
  reports/my_eval*.png \
  --project universal-simulator \
  --metadata '{"model": "512dim", "split": "val"}'
```

### Compare TTC vs Non-TTC

```bash
# 1. Run without TTC
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/eval_burgers_512dim.yaml \
  --operator checkpoints/scale/operator.pt \
  --diffusion checkpoints/scale/diffusion_residual.pt \
  --output-prefix reports/baseline_notttc

# 2. Run with TTC
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/eval_burgers_512dim_ttc_val.yaml \
  --operator checkpoints/scale/operator.pt \
  --diffusion checkpoints/scale/diffusion_residual.pt \
  --output-prefix reports/baseline_ttc

# 3. Compare metrics
echo "No TTC:"
cat reports/baseline_notttc.json
echo "With TTC:"
cat reports/baseline_ttc.json
```

## Troubleshooting

### Issue: CUDA Out of Memory during TTC Evaluation

**Symptoms:** `torch.OutOfMemoryError` with large memory allocation (50GB+)

**Solution:** Use low-memory TTC config
```yaml
# In your eval config
ttc:
  reward:
    grid: [32, 32]  # Reduce from [64, 64]
  decoder:
    hidden_dim: 128  # Reduce from 256
  candidates: 4  # Reduce from 6
  beam_width: 1  # Reduce from 2
training:
  batch_size: 16  # Reduce from 32
```

See [docs/ttc_analysis_lowmem.md](ttc_analysis_lowmem.md) for detailed analysis.

### Issue: Dataset Download Fails

**Symptoms:** `FileNotFoundError` for dataset files

**Solutions:**
1. **For B2:** Ensure `B2_KEY_ID`, `B2_APP_KEY`, `B2_BUCKET` are set correctly
2. **For W&B:** Ensure `WANDB_API_KEY` is valid and dataset artifact exists
3. **Manual download:** Use `scripts/fetch_datasets_b2.sh` with dataset names

### Issue: Checkpoint Load Failure

**Symptoms:** `KeyError` or `RuntimeError` when loading checkpoints

**Common causes:**
- Checkpoint saved with `torch.compile` (has `_orig_mod.` prefix)
- Architecture mismatch between config and checkpoint

**Solution:** The evaluate.py script automatically handles `torch.compile` prefixes. If issues persist:
```python
# Manually strip prefix
checkpoint = torch.load('checkpoint.pt')
if any(k.startswith('_orig_mod.') for k in checkpoint.keys()):
    checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()}
```

### Issue: W&B Authentication Error

**Symptoms:** `CommError: 401 Unauthorized`

**Solution:**
```bash
# Re-login to W&B
wandb login YOUR_API_KEY

# Or disable W&B for local testing
export WANDB_MODE=disabled
```

## Performance Tips

1. **Precompute Latent Cache:** Set `PRECOMPUTE_LATENT=1` to cache encoder outputs (saves ~30% training time)
2. **Use Batch Accumulation:** For large models, set `training.accum_steps` to accumulate gradients
3. **Enable Compilation:** Set `training.compile: true` for ~20% speedup (PyTorch 2.0+)
4. **Adjust Workers:** Set `training.num_workers` based on CPU cores (0 for GPU-only, 4-8 for multi-core)
5. **Profile First:** Run with a few epochs to identify bottlenecks before full training

## Recent Changes (October 2025)

- ✅ TTC evaluation infrastructure implemented
- ✅ Low-memory TTC configs for large reward models
- ✅ B2 data hydration for faster remote setup
- ✅ Checkpoint download from W&B artifacts
- ✅ EVAL_ONLY mode for checkpoint-based evaluation
- ✅ Comprehensive evaluation reports (JSON/CSV/HTML/PNG)
- ✅ Codebase cleanup: 8 configs + 9 scripts archived
- ✅ Fixed encoder view→reshape bug for PyTorch compatibility

## References

- [TTC Analysis Report](ttc_analysis_lowmem.md) - Detailed TTC evaluation analysis
- [Archive Directory](../archive/README.md) - Deprecated configs/scripts
- Main training config: [configs/train_burgers_quality_v3.yaml](../configs/train_burgers_quality_v3.yaml)
- Remote orchestrator: [scripts/run_remote_scale.sh](../scripts/run_remote_scale.sh)

## Support

For issues or questions:
1. Check [docs/](.) for relevant guides
2. Review [archive/](../archive/) for historical context
3. Check W&B runs for similar experiments
4. Review recent commits: `git log --oneline --since="2 weeks ago"`

---

## SOTA Metrics (October 14, 2025 Update)

### New Evaluation Metrics

The evaluation pipeline now includes **SOTA-comparable metrics** for direct comparison with published PDE ML benchmarks:

- ✅ **nRMSE** (Normalized RMSE) - Scale-invariant metric
- ✅ **Relative L2** - Standard PDE benchmarking metric
- ✅ **Per-sample Relative L2** - For distribution analysis

### Baseline vs TTC Comparison

**Baseline Evaluation (No TTC):**
```bash
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/eval_burgers_512dim_val_baseline.yaml \
  --operator checkpoints/scale/operator.pt \
  --diffusion checkpoints/scale/diffusion_residual.pt \
  --output-prefix reports/baseline_val
```

**TTC Evaluation:**
```bash
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/eval_burgers_512dim_ttc_val.yaml \
  --operator checkpoints/scale/operator.pt \
  --diffusion checkpoints/scale/diffusion_residual.pt \
  --output-prefix reports/ttc_val
```

**Compare Results:**
```python
import json

with open('reports/baseline_val.json') as f:
    baseline = json.load(f)
with open('reports/ttc_val.json') as f:
    ttc = json.load(f)

baseline_rel_l2 = baseline['metrics']['rel_l2']
ttc_rel_l2 = ttc['metrics']['rel_l2']
improvement = (baseline_rel_l2 - ttc_rel_l2) / baseline_rel_l2 * 100

print(f"Baseline Relative L2: {baseline_rel_l2:.6f}")
print(f"TTC Relative L2:      {ttc_rel_l2:.6f}")
print(f"Improvement:          {improvement:+.2f}%")
```

### SOTA Benchmarks (PDEBench Burgers 1D)

| Model | Relative L2 ↓ | Year | Params |
|-------|--------------|------|--------|
| FNO-2D | 0.0180 | 2021 | 2.4M |
| U-Net | 0.0250 | 2020 | ~2M |
| ResNet | 0.0310 | 2019 | ~1M |
| **Ours (Target)** | **< 0.020** | 2025 | 2.1M |

See [SOTA Comparison Guide](sota_comparison_guide.md) for detailed methodology.

