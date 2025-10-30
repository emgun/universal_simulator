# Training Metrics Documentation

This document describes all metrics logged during the Universal Physics Stack (UPS) training pipeline. Metrics are organized by training stage and logging frequency.

## Overview

The training pipeline uses a clean WandB architecture where:
- **Time series metrics** (per-epoch/per-batch) are logged via `wandb_ctx.log_training_metric(stage, metric, value, step)`
- **Summary metrics** (final scalars) are logged via `wandb_ctx.log_eval_summary(metrics, prefix)`
- **Structured data** (tables, comparisons) are logged via `wandb_ctx.log_table(name, columns, data)`

All training stages share a single WandB run for clean organization.

---

## Stage 1: Operator Training

**Location**: `scripts/train.py::train_operator()`

### Per-Epoch Metrics (logged every epoch)

Logged via `TrainingLogger.log()` at line 643-651:

| Metric | Key | Description | When Logged |
|--------|-----|-------------|-------------|
| Loss | `training/operator/loss` | Mean epoch loss (averaged across all batches) | Every epoch |
| Learning Rate | `training/operator/lr` | Current learning rate from optimizer | Every epoch |
| Epochs Since Improve | `training/operator/epochs_since_improve` | Patience counter (epochs without loss improvement) | Every epoch |
| Gradient Norm | `training/operator/grad_norm` | Mean gradient norm (averaged across gradient steps) | Every epoch |
| Epoch Time | `training/operator/epoch_time_sec` | Wall-clock time for epoch (seconds) | Every epoch |
| Best Loss | `training/operator/best_loss` | Best loss seen so far (checkpoint threshold) | Every epoch |

### Per-Batch Loss Components (logged every 10 batches)

Logged at line 612-614 (`wandb_ctx.log_training_metric("operator", name, value.item(), step=logger.get_global_step())`):

| Component | Key | Description | When Logged | Controlled By |
|-----------|-----|-------------|-------------|---------------|
| Forward Loss | `training/operator/L_forward` | One-step prediction MSE loss | Every 10 batches | Always computed |
| Inverse Encoding Loss | `training/operator/L_inv_enc` | Ensures latent is decodable (decoder reconstruction error) | Every 10 batches | `lambda_inv_enc > 0` |
| Inverse Decoding Loss | `training/operator/L_inv_dec` | Ensures decoded fields are re-encodable (encoder roundtrip error) | Every 10 batches | `lambda_inv_dec > 0` |
| Spectral Loss | `training/operator/L_spec` | Relative spectral energy difference (FFT domain) | Every 10 batches | `lambda_spectral > 0` |
| Rollout Loss | `training/operator/L_rollout` | Multi-step rollout MSE (optional) | Every 10 batches | `lambda_rollout > 0` |

**Loss Component Details** (from `src/ups/training/losses.py`):

- **L_forward**: `F.mse_loss(pred_next, target_next)` - Core one-step prediction loss
- **L_inv_enc**: `F.mse_loss(decoder(latent), input_fields)` - Physical space reconstruction accuracy
- **L_inv_dec**: `F.mse_loss(encoder(decoder(latent)), latent)` - Latent space roundtrip consistency
- **L_spec**: `abs(energy(pred_fft) - energy(target_fft)) / energy(target_fft)` - Spectral energy conservation
- **L_rollout**: `F.mse_loss(pred_seq, target_seq)` - Multi-step trajectory accuracy

**Curriculum Learning**: Inverse losses use curriculum scheduling (lines 204-217 in `losses.py`):
- Epochs 0-15: `weight = 0` (pure forward training)
- Epochs 15-30: Linear ramp from 0 to `lambda_inv_*`
- Epochs 30+: Full weight (capped at `max_weight=0.05`)

### Training Configuration Logged

Logged during `train_all_stages()` at line 1559-1564:

- `gpu_name`: GPU device name (e.g., "NVIDIA A100-SXM4-80GB")
- `gpu_count`: Number of GPUs available
- `cuda_version`: CUDA version string

---

## Stage 2: Diffusion Residual Training

**Location**: `scripts/train.py::train_diffusion()`

### Per-Epoch Metrics (logged every epoch)

Logged via `TrainingLogger.log()` at line 825-833:

| Metric | Key | Description | When Logged |
|--------|-----|-------------|-------------|
| Loss | `training/diffusion_residual/loss` | Mean epoch loss (MSE + auxiliary losses) | Every epoch |
| Learning Rate | `training/diffusion_residual/lr` | Current learning rate from optimizer | Every epoch |
| Epochs Since Improve | `training/diffusion_residual/epochs_since_improve` | Patience counter | Every epoch |
| Gradient Norm | `training/diffusion_residual/grad_norm` | Mean gradient norm | Every epoch |
| Epoch Time | `training/diffusion_residual/epoch_time_sec` | Wall-clock time for epoch (seconds) | Every epoch |
| Best Loss | `training/diffusion_residual/best_loss` | Best loss seen so far | Every epoch |

### Loss Computation

Logged at line 782-788:

- **Base Loss**: `F.mse_loss(drift, residual_target)` where `residual_target = target - operator(state)`
- **Spectral Loss**: `lambda_spectral * spectral_energy_loss(drift, residual_target)` (optional)
- **Relative Loss**: `lambda_relative * nrmse(drift, residual_target)` (optional)

**Total Loss**: `base + spectral + relative`

---

## Stage 3: Consistency Distillation

**Location**: `scripts/train.py::train_consistency()`

### Per-Epoch Metrics (logged every epoch)

Logged via `TrainingLogger.log()` at line 1153-1161:

| Metric | Key | Description | When Logged |
|--------|-----|-------------|-------------|
| Loss | `training/consistency_distill/loss` | Mean epoch distillation loss | Every epoch |
| Learning Rate | `training/consistency_distill/lr` | Current learning rate from optimizer | Every epoch |
| Epochs Since Improve | `training/consistency_distill/epochs_since_improve` | Patience counter | Every epoch |
| Gradient Norm | `training/consistency_distill/grad_norm` | Mean gradient norm | Every epoch |
| Epoch Time | `training/consistency_distill/epoch_time_sec` | Wall-clock time for epoch (seconds) | Every epoch |
| Best Loss | `training/consistency_distill/best_loss` | Best loss seen so far | Every epoch |

### Loss Computation

Computed at line 1112 via `_distill_forward_and_loss_compiled()`:

- **Teacher**: Pre-computed operator forward pass (cached once per batch)
- **Student**: Diffusion model applied with vectorized tau values
- **Loss**: `F.mse_loss(student_z, z_tiled)` - Self-consistency loss

**Optimization**: Teacher predictions computed once per batch (line 1095), then reused across micro-batches.

---

## Stage 4: Steady Prior Training

**Location**: `scripts/train.py::train_steady_prior()`

### Per-Epoch Metrics (logged every epoch)

Logged via `TrainingLogger.log()` at line 1268-1276:

| Metric | Key | Description | When Logged |
|--------|-----|-------------|-------------|
| Loss | `training/steady_prior/loss` | Mean epoch loss (MSE) | Every epoch |
| Learning Rate | `training/steady_prior/lr` | Current learning rate from optimizer | Every epoch |
| Epochs Since Improve | `training/steady_prior/epochs_since_improve` | Patience counter | Every epoch |
| Gradient Norm | `training/steady_prior/grad_norm` | Mean gradient norm | Every epoch |
| Epoch Time | `training/steady_prior/epoch_time_sec` | Wall-clock time for epoch (seconds) | Every epoch |
| Best Loss | `training/steady_prior/best_loss` | Best loss seen so far | Every epoch |

### Loss Computation

Computed at line 1252:

- **Loss**: `F.mse_loss(refined.z, z1)` - Simple MSE between refined state and target

---

## Evaluation Metrics

**Location**: 
- `scripts/train.py::_run_evaluation()` (lines 1311-1385)
- `scripts/evaluate.py` (lines 519-524)
- `src/ups/eval/pdebench_runner.py::evaluate_latent_operator()` (lines 217-226)

### Baseline Evaluation

Logged via `wandb_ctx.log_eval_summary()` at lines 1402-1415:

| Metric | Key | Description | Type |
|--------|-----|-------------|------|
| MSE | `eval/baseline_mse` | Mean squared error | Summary scalar |
| MAE | `eval/baseline_mae` | Mean absolute error | Summary scalar |
| RMSE | `eval/baseline_rmse` | Root mean squared error | Summary scalar |
| NRMSE | `eval/baseline_nrmse` | Normalized RMSE (SOTA comparison metric) | Summary scalar |
| Relative L2 | `eval/baseline_rel_l2` | Relative L2 norm | Summary scalar |

### Physics Metrics (optional)

Logged when available:

| Metric | Key | Description | Type |
|--------|-----|-------------|------|
| Conservation Gap | `eval/baseline_conservation_gap` | Mass/energy conservation violation | Summary scalar |
| BC Violation | `eval/baseline_bc_violation` | Boundary condition violation | Summary scalar |
| Negativity Penalty | `eval/baseline_negativity_penalty` | Negative value penalty (for positive-definite fields) | Summary scalar |

### TTC (Test-Time Conditioning) Evaluation

Logged when `ttc.enabled=true`:

| Metric | Key | Description | Type |
|--------|-----|-------------|------|
| MSE | `eval/ttc_mse` | Mean squared error with TTC | Summary scalar |
| MAE | `eval/ttc_mae` | Mean absolute error with TTC | Summary scalar |
| RMSE | `eval/ttc_rmse` | Root mean squared error with TTC | Summary scalar |
| NRMSE | `eval/ttc_nrmse` | Normalized RMSE with TTC | Summary scalar |
| Relative L2 | `eval/ttc_rel_l2` | Relative L2 norm with TTC | Summary scalar |
| Improvement | `eval/ttc_improvement_pct` | `(baseline_nrmse - ttc_nrmse) / baseline_nrmse * 100` | Summary scalar |

**TTC Physics Metrics** (optional):
- `eval/ttc_conservation_gap`
- `eval/ttc_bc_violation`
- `eval/ttc_negativity_penalty`

### Per-Sample Metrics (detailed evaluation)

Stored in evaluation details dict (not logged to WandB by default):

- `per_sample_mse`: List of MSE values for each test sample
- `per_sample_mae`: List of MAE values for each test sample
- `per_sample_rel_l2`: List of relative L2 values for each sample

### TTC Step Logs (detailed evaluation)

Stored when TTC enabled:

```python
ttc_step_logs = [
    {
        "step": int,           # TTC step number
        "chosen": int,         # Index of chosen candidate
        "totals": List[float], # Total reward for each candidate
        "components": Dict[str, List[float]]  # Individual reward components
    },
    ...
]
```

---

## Evaluation Summary Tables

**Location**: `scripts/train.py::_log_evaluation_summary()` (lines 1483-1513)

### Accuracy Comparison Table

Logged via `wandb_ctx.log_table()` at lines 1483-1487:

**Table**: `"Training Evaluation Summary"`

| Column | Description |
|--------|-------------|
| Metric | Metric name (MSE, MAE, RMSE, NRMSE, REL_L2) |
| Baseline | Baseline metric value |
| TTC | TTC metric value (if enabled) |
| Improvement | Percentage improvement (`(base - ttc) / base * 100%`) |

### Physics Diagnostics Table

Logged when physics metrics available (lines 1490-1513):

**Table**: `"Training Physics Diagnostics"`

| Column | Description |
|--------|-------------|
| Physics Check | Check name (Conservation Gap, BC Violation, etc.) |
| Baseline | Baseline value |
| TTC | TTC value (if enabled) |

---

## Final Summary Metrics

**Location**: `scripts/train.py::train_all_stages()` (lines 1713-1719)

Logged at end of full pipeline:

| Metric | Key | Description |
|--------|-----|-------------|
| Training Complete | `summary/total_training_complete` | Always 1 (completion flag) |
| Operator Checkpoint Size | `summary/operator_checkpoint_size_mb` | Checkpoint file size (MB) |
| Diffusion Checkpoint Size | `summary/diffusion_checkpoint_size_mb` | Checkpoint file size (MB) |
| Steady Prior Checkpoint Size | `summary/steady_prior_checkpoint_size_mb` | Checkpoint file size (MB) |

---

## WandB Configuration

**Location**: `src/ups/utils/wandb_context.py::create_wandb_context()` (lines 308-379)

### Architecture Config

Logged during WandB run initialization:

- `latent_dim`: Latent space dimensionality
- `latent_tokens`: Number of latent tokens
- `operator_hidden_dim`: Operator transformer hidden dimension
- `operator_num_heads`: Number of attention heads
- `operator_depths`: Transformer depth per stage
- `operator_group_size`: Window size for shifted attention
- `diffusion_latent_dim`: Diffusion model input dimension
- `diffusion_hidden_dim`: Diffusion model hidden dimension

### Training Config

- `batch_size`: Training batch size
- `time_stride`: Temporal stride between samples
- `grad_clip`: Gradient clipping threshold
- `amp`: Automatic mixed precision enabled
- `compile`: torch.compile enabled
- `ema_decay`: Exponential moving average decay rate
- `accum_steps`: Gradient accumulation steps

### Per-Stage Config

For each stage (operator, diffusion, consistency):
- `{stage}_epochs`: Number of training epochs
- `{stage}_lr`: Learning rate
- `{stage}_weight_decay`: Weight decay
- `{stage}_optimizer`: Optimizer name (adam, adamw, sgd)

### TTC Config

When TTC enabled:
- `ttc_enabled`: True
- `ttc_candidates`: Number of candidate trajectories
- `ttc_beam_width`: Beam search width
- `ttc_steps`: Number of TTC refinement steps
- `ttc_noise_std`: Noise standard deviation for sampling

---

## File Outputs

### Training Logs

**JSONL Format** (one entry per epoch): `reports/training_log.jsonl`

Written by `TrainingLogger.log()` at lines 169-192:

```json
{
  "stage": "operator",
  "loss": 0.000234,
  "epoch": 24,
  "lr": 0.0001,
  "global_step": 24,
  "epochs_since_improve": 0,
  "grad_norm": 0.456,
  "epoch_time_sec": 15.3,
  "best_loss": 0.000234
}
```

### Evaluation Reports

**JSON Format**: `reports/eval_baseline.json`, `reports/eval_ttc.json`

Structure:
```json
{
  "metrics": {
    "mse": 0.001234,
    "mae": 0.0234,
    "rmse": 0.0351,
    "nrmse": 0.089,
    "rel_l2": 0.089
  },
  "extra": {
    "task": "burgers_1d",
    "split": "test",
    "num_samples": 3072,
    "ttc": false
  }
}
```

---

## Metric Step Relationships

**Location**: `src/ups/utils/wandb_context.py::create_wandb_context()` (lines 397-408)

WandB metric relationships are defined during initialization:

```python
# Each stage uses its own step metric
wandb.define_metric("training/operator/step")
wandb.define_metric("training/diffusion_residual/step")
wandb.define_metric("training/consistency_distill/step")
wandb.define_metric("training/steady_prior/step")

# All metrics in each namespace use their stage's step
wandb.define_metric("training/operator/*", step_metric="training/operator/step")
wandb.define_metric("training/diffusion_residual/*", step_metric="training/diffusion_residual/step")
# ... etc
```

This ensures:
- Each stage's metrics are plotted against its own epoch count
- Independent x-axis scales per stage
- Clean separation in WandB UI

---

## Example Metric Flow

### Operator Training (Epoch 10)

1. **Batch 0-99**: Loss components computed
2. **Batch 0, 10, 20, ..., 90**: Individual loss components logged (`L_forward`, `L_inv_enc`, etc.)
3. **End of Epoch**: Aggregate metrics logged:
   - `training/operator/loss = 0.00234` (mean of all batches)
   - `training/operator/lr = 0.0001`
   - `training/operator/grad_norm = 0.456`
   - `training/operator/epoch_time_sec = 15.3`
   - `training/operator/step = 10` (epoch number)

### Evaluation (Final)

1. **Baseline rollout**: Compute predictions without TTC
2. **Aggregate baseline metrics**: MSE, MAE, RMSE, NRMSE, rel_l2
3. **TTC rollout** (if enabled): Compute predictions with TTC
4. **Aggregate TTC metrics**: MSE, MAE, RMSE, NRMSE, rel_l2
5. **Log to WandB summary**:
   - `eval/baseline_nrmse = 0.78`
   - `eval/ttc_nrmse = 0.09`
   - `eval/ttc_improvement_pct = 88.5`
6. **Create comparison tables**: Logged to WandB Tables view

---

## Notes

### Logging Frequency

- **Per-epoch metrics**: Logged once per epoch for all stages
- **Per-batch loss components**: Logged every 10 batches (operator stage only)
- **Evaluation metrics**: Logged once at end of training (if evaluation enabled)
- **Summary metrics**: Logged once at end of full pipeline

### Data Types

- **Time series** (charts): `wandb_ctx.log_training_metric()`
  - Includes explicit step parameter for x-axis
  - Examples: loss curves, learning rate schedules
  
- **Summary scalars** (single values): `wandb_ctx.log_eval_summary()`
  - No step parameter
  - Examples: final NRMSE, improvement percentage
  
- **Tables** (comparisons): `wandb_ctx.log_table()`
  - Structured multi-column data
  - Examples: baseline vs TTC comparison

### Curriculum Learning

Inverse losses (`L_inv_enc`, `L_inv_dec`) use curriculum scheduling to prevent gradient explosion:
- Start with pure forward training (weight=0)
- Gradually ramp up over 15 epochs
- Cap at maximum weight (0.05) to maintain stability

See `src/ups/training/losses.py::compute_inverse_loss_curriculum_weight()` for implementation.

### Physics-Informed Metrics

Conservation gap, boundary condition violations, and negativity penalties are computed when:
- Dataset provides metadata for physics checks
- TTC reward models are configured
- Evaluation includes physics diagnostics

These are optional and depend on the PDE type (e.g., Burgers, Navier-Stokes).

---

## References

- **Training Script**: `scripts/train.py`
- **Loss Functions**: `src/ups/training/losses.py`
- **Evaluation Runner**: `src/ups/eval/pdebench_runner.py`
- **WandB Context**: `src/ups/utils/wandb_context.py`
- **Metrics**: `src/ups/eval/metrics.py`
- **Reports**: `src/ups/eval/reports.py`
