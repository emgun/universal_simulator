# Unified Training + Evaluation Pipeline

## Overview

The training pipeline has been enhanced to automatically run evaluation (both baseline and TTC) after training completes, with all results logged to a single WandB run for comprehensive tracking.

## Key Features

### 1. Automatic Evaluation
- Evaluation runs automatically after training completes
- No need to run separate evaluation scripts
- Can be disabled via config if needed

### 2. Single WandB Run
- Training and evaluation logged to same run
- Better organization and easier comparison
- Unified metrics and charts

### 3. Comprehensive Reporting
- Baseline evaluation metrics
- TTC evaluation metrics (if enabled)
- Automatic improvement calculations
- Summary tables and comparisons
- WandB artifacts for easy sharing

### 4. Zero Configuration Required
- Works with existing training configs
- Just add `evaluation.enabled: true` and `ttc` config
- Everything else is automatic

## Usage

### Basic Training with Evaluation

Simply run training as before:

```bash
python scripts/train.py --config configs/train_burgers_32dim_pru2jxc4.yaml --stage all
```

The pipeline will automatically:
1. Train operator → diffusion → distillation
2. Run baseline evaluation on test set
3. Run TTC evaluation (if `ttc.enabled: true`)
4. Log all metrics to WandB
5. Generate summary report

### Configuration

Add these sections to your training config:

```yaml
# Enable automatic evaluation after training
evaluation:
  enabled: true  # Set to false to skip evaluation

# TTC configuration (optional, enables TTC evaluation)
ttc:
  enabled: true
  steps: 1
  candidates: 6
  beam_width: 2
  # ... other TTC settings
  decoder:
    latent_dim: 64  # Match your model's latent dim
    hidden_dim: 128  # Scale appropriately
    # ... other decoder settings
```

### Disabling Evaluation

If you want to train without evaluation:

```yaml
evaluation:
  enabled: false
```

Or omit the `evaluation` section entirely (defaults to enabled).

## WandB Integration

### Metrics Logged

**Training Metrics** (per stage):
- `operator/loss`, `operator/lr`, `operator/grad_norm`, etc.
- `diffusion_residual/loss`, `diffusion_residual/lr`, etc.
- `consistency_distill/loss`, `consistency_distill/lr`, etc.

**Evaluation Metrics**:
- `eval/baseline_mse`, `eval/baseline_mae`, `eval/baseline_rmse`, `eval/baseline_nrmse`
- `eval/ttc_mse`, `eval/ttc_mae`, `eval/ttc_rmse`, `eval/ttc_nrmse`
- `eval/ttc_improvement_pct` (if TTC enabled)

**Summary Metrics**:
- `summary/total_training_complete`
- `summary/operator_checkpoint_size_mb`
- `summary/diffusion_checkpoint_size_mb`

### Artifacts

Evaluation reports are saved as WandB artifacts:
- `eval-baseline-{run_id}`: Baseline evaluation results
- `eval-ttc-{run_id}`: TTC evaluation results (if enabled)

### Summary Table

A markdown summary table is automatically generated showing:
| Metric | Baseline | TTC | Improvement |
|--------|----------|-----|-------------|
| MSE    | ...      | ... | ...%        |
| MAE    | ...      | ... | ...%        |
| RMSE   | ...      | ... | ...%        |
| NRMSE  | ...      | ... | ...%        |

Available in WandB under `evaluation_summary`.

### Comparison Table

A WandB Table is created for easy filtering and comparison across runs.

## Pipeline Stages

The unified pipeline now has 5 stages:

### Stage 1: Operator Training
- Trains the latent operator
- Logs: `operator/*` metrics
- Saves: `checkpoints/operator.pt`

### Stage 2: Diffusion Residual Training
- Trains the diffusion model
- Logs: `diffusion_residual/*` metrics
- Saves: `checkpoints/diffusion_residual.pt`, `diffusion_residual_ema.pt`

### Stage 3: Consistency Distillation
- Distills to faster inference
- Logs: `consistency_distill/*` metrics
- Updates: `checkpoints/diffusion_residual.pt`

### Stage 4: Steady Prior (Optional)
- Trains steady-state prior
- Logs: `steady_prior/*` metrics
- Saves: `checkpoints/steady_prior.pt`

### Stage 5: Evaluation (NEW!)
- Runs baseline evaluation on test set
- Optionally runs TTC evaluation
- Logs: `eval/*` metrics
- Saves: evaluation reports as artifacts

## Examples

### Example 1: 32-dim Model with TTC

```yaml
# configs/train_burgers_32dim_pru2jxc4.yaml
include: train_burgers_quality_v2.yaml

latent:
  dim: 32
  tokens: 16

# ... training settings ...

evaluation:
  enabled: true

ttc:
  enabled: true
  # ... TTC settings ...
  decoder:
    latent_dim: 32
    hidden_dim: 64

logging:
  wandb:
    enabled: true
    run_name: burgers32-pru2jxc4-unified
    tags: [32dim, unified_pipeline]
```

Run:
```bash
python scripts/train.py --config configs/train_burgers_32dim_pru2jxc4.yaml --stage all
```

Output:
```
==================================================
STAGE 1/4: Training Operator
==================================================
...

==================================================
STAGE 2/4: Training Diffusion Residual
==================================================
...

==================================================
STAGE 3/4: Consistency Distillation
==================================================
...

==================================================
STAGE 5/5: Evaluation on Test Set
==================================================

📊 Running baseline evaluation...
Baseline Results:
  MSE:   0.145562
  MAE:   0.237588
  RMSE:  0.381526
  NRMSE: 0.784469

📊 Running TTC evaluation...
TTC Results:
  MSE:   0.002242
  MAE:   0.037100
  RMSE:  0.047347
  NRMSE: 0.092096

  TTC Improvement: 88.3%

==================================================
📝 WandB Summary Generated
==================================================
View full results at: https://wandb.ai/...

==================================================
✅ All training stages complete!
==================================================
```

### Example 2: Quick Training without Evaluation

```yaml
# configs/train_burgers_quick.yaml
include: train_burgers_quality_v2.yaml

stages:
  operator:
    epochs: 5  # Quick test
  diff_residual:
    epochs: 2
  consistency_distill:
    epochs: 2

evaluation:
  enabled: false  # Skip evaluation

logging:
  wandb:
    enabled: true
    run_name: burgers-quick-test
```

### Example 3: Baseline Only (No TTC)

```yaml
# configs/train_burgers_baseline_only.yaml
include: train_burgers_quality_v2.yaml

evaluation:
  enabled: true

ttc:
  enabled: false  # Skip TTC evaluation

logging:
  wandb:
    enabled: true
    run_name: burgers-baseline-only
```

## Benefits

### Before (Separate Scripts)
```bash
# 1. Train
python scripts/train.py --config config.yaml --stage all

# 2. Wait for training to complete

# 3. Run baseline eval
python scripts/evaluate.py --config eval_config.yaml --operator checkpoints/operator.pt --diffusion checkpoints/diffusion_residual.pt --output-prefix reports/eval_baseline

# 4. Run TTC eval
python scripts/evaluate.py --config eval_ttc_config.yaml --operator checkpoints/operator.pt --diffusion checkpoints/diffusion_residual.pt --output-prefix reports/eval_ttc

# 5. Manually compare results

# Problems:
# - Multiple WandB runs
# - Manual comparison needed
# - Easy to forget steps
# - Harder to track lineage
```

### After (Unified Pipeline)
```bash
# 1. Train (evaluation runs automatically)
python scripts/train.py --config config.yaml --stage all

# Done! Everything in one WandB run with automatic comparison.

# Benefits:
# ✅ Single WandB run
# ✅ Automatic evaluation
# ✅ Built-in comparison
# ✅ Complete lineage tracking
# ✅ Less manual work
```

## Advanced Usage

### Programmatic Access

You can also call the training function directly:

```python
from scripts.train import train_all_stages, load_config

cfg = load_config("configs/train_burgers_32dim_pru2jxc4.yaml")
train_all_stages(cfg)
```

### Custom Evaluation Logic

If you need custom evaluation, you can disable auto-eval and run separately:

```yaml
evaluation:
  enabled: false
```

Then use `scripts/evaluate.py` as before for custom evaluation logic.

## Migration Guide

### Updating Existing Configs

1. **Add evaluation section** (optional, defaults to enabled):
```yaml
evaluation:
  enabled: true
```

2. **Add TTC config if you want TTC evaluation**:
```yaml
ttc:
  enabled: true
  # ... TTC settings from eval config ...
  decoder:
    latent_dim: <match your model>
    hidden_dim: <scale appropriately>
```

3. **Update WandB tags** (optional):
```yaml
logging:
  wandb:
    tags: [<existing tags>, unified_pipeline]
```

### Backward Compatibility

- Old configs still work (evaluation defaults to enabled)
- Can disable evaluation to get old behavior
- Separate evaluation scripts still work if needed

## Troubleshooting

### Evaluation Fails

If evaluation fails, training still completes successfully. Error is logged to WandB:
- Check `eval/error` metric in WandB
- Review console output for error details
- Verify TTC config matches model dimensions

### Missing TTC Results

If TTC evaluation is skipped:
- Check `ttc.enabled: true` in config
- Verify TTC decoder dimensions match model
- Check for errors in console output

### WandB Not Logging

If WandB integration isn't working:
- Verify `logging.wandb.enabled: true`
- Check WandB API key is set
- Review console for WandB initialization errors

## Performance

### Evaluation Overhead

- Baseline evaluation: ~30-60 seconds
- TTC evaluation: ~3-5 minutes (due to beam search)
- Total overhead: ~5-6 minutes for full pipeline

### Memory Usage

Evaluation uses similar memory to training. If OOM:
- Reduce batch size in training config
- Disable TTC evaluation
- Run evaluation separately on smaller GPU

## See Also

- `scripts/train.py` - Main training script
- `scripts/evaluate.py` - Standalone evaluation (still available)
- `configs/train_burgers_32dim_pru2jxc4.yaml` - Example unified config
- `configs/train_burgers_64dim_pru2jxc4.yaml` - Another example

