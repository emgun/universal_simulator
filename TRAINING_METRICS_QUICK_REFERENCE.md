# Training Metrics Quick Reference

## Logged Metrics by Stage

### Operator Stage (`training/operator/*`)

**Per-Epoch** (every epoch):
- `loss` - Mean epoch loss
- `lr` - Learning rate
- `epochs_since_improve` - Patience counter
- `grad_norm` - Mean gradient norm
- `epoch_time_sec` - Epoch duration (seconds)
- `best_loss` - Best loss seen so far

**Per-Batch** (every 10 batches):
- `L_forward` - One-step prediction MSE
- `L_inv_enc` - Inverse encoding loss (if enabled)
- `L_inv_dec` - Inverse decoding loss (if enabled)
- `L_spec` - Spectral loss (if enabled)
- `L_rollout` - Multi-step rollout loss (if enabled)

### Diffusion Residual Stage (`training/diffusion_residual/*`)

**Per-Epoch** (every epoch):
- `loss` - Mean epoch loss (MSE + auxiliaries)
- `lr` - Learning rate
- `epochs_since_improve` - Patience counter
- `grad_norm` - Mean gradient norm
- `epoch_time_sec` - Epoch duration (seconds)
- `best_loss` - Best loss seen so far

### Consistency Distillation Stage (`training/consistency_distill/*`)

**Per-Epoch** (every epoch):
- `loss` - Mean distillation loss
- `lr` - Learning rate
- `epochs_since_improve` - Patience counter
- `grad_norm` - Mean gradient norm
- `epoch_time_sec` - Epoch duration (seconds)
- `best_loss` - Best loss seen so far

### Steady Prior Stage (`training/steady_prior/*`)

**Per-Epoch** (every epoch):
- `loss` - Mean MSE loss
- `lr` - Learning rate
- `epochs_since_improve` - Patience counter
- `grad_norm` - Mean gradient norm
- `epoch_time_sec` - Epoch duration (seconds)
- `best_loss` - Best loss seen so far

---

## Evaluation Metrics (Summary)

### Baseline (`eval/*`)

**Accuracy**:
- `baseline_mse` - Mean squared error
- `baseline_mae` - Mean absolute error
- `baseline_rmse` - Root mean squared error
- `baseline_nrmse` - Normalized RMSE (SOTA metric)
- `baseline_rel_l2` - Relative L2 norm

**Physics** (optional):
- `baseline_conservation_gap` - Mass/energy conservation violation
- `baseline_bc_violation` - Boundary condition violation
- `baseline_negativity_penalty` - Negative value penalty

### TTC (`eval/*`, if enabled)

**Accuracy**:
- `ttc_mse`, `ttc_mae`, `ttc_rmse`, `ttc_nrmse`, `ttc_rel_l2`
- `ttc_improvement_pct` - Improvement over baseline

**Physics** (optional):
- `ttc_conservation_gap`, `ttc_bc_violation`, `ttc_negativity_penalty`

---

## Configuration Logged (`wandb.config`)

### Architecture
- `latent_dim`, `latent_tokens`
- `operator_hidden_dim`, `operator_num_heads`, `operator_depths`
- `diffusion_hidden_dim`

### Training
- `batch_size`, `time_stride`, `grad_clip`
- `amp`, `compile`, `ema_decay`, `accum_steps`

### Per-Stage
- `{stage}_epochs`, `{stage}_lr`, `{stage}_weight_decay`, `{stage}_optimizer`

### TTC (if enabled)
- `ttc_enabled`, `ttc_candidates`, `ttc_beam_width`, `ttc_steps`

---

## File Outputs

- **Training log**: `reports/training_log.jsonl` (per-epoch JSONL)
- **Baseline eval**: `reports/eval_baseline.json` (final metrics)
- **TTC eval**: `reports/eval_ttc.json` (final metrics, if enabled)

---

## Key Loss Formulas

### Operator Stage

- **L_forward**: `MSE(pred_next, target_next)` - Always used (weight=1.0)
- **L_inv_enc**: `MSE(decoder(latent), input_fields)` - Curriculum scheduled
- **L_inv_dec**: `MSE(encoder(decoder(latent)), latent)` - Curriculum scheduled
- **L_spec**: `|energy(pred_fft) - energy(target_fft)| / energy(target_fft)`
- **L_rollout**: `MSE(pred_seq, target_seq)` - Multi-step accuracy

**Total**: `L_forward + λ_inv_enc*L_inv_enc + λ_inv_dec*L_inv_dec + λ_spec*L_spec + λ_rollout*L_rollout`

### Diffusion Stage

- **Base**: `MSE(drift, residual_target)` where `residual = target - operator(state)`
- **Spectral**: `λ_spec * spectral_energy_loss(drift, residual)`
- **Relative**: `λ_rel * NRMSE(drift, residual)`

**Total**: `base + spectral + relative`

### Consistency Stage

- **Loss**: `MSE(student_z, teacher_z)` - Self-consistency over tau

### Steady Prior Stage

- **Loss**: `MSE(refined.z, target)` - Simple refinement loss

---

## Typical Values (32-dim Burgers)

**Operator Training (25 epochs)**:
- Final loss: ~0.00023
- Epoch time: ~15 sec/epoch (A100)
- Best grad_norm: ~0.4-0.6

**Diffusion Training (10 epochs)**:
- Final loss: ~0.0001-0.0005
- Epoch time: ~20 sec/epoch (A100)

**Consistency Distillation (5 epochs)**:
- Final loss: ~0.0001
- Epoch time: ~12 sec/epoch (A100)

**Evaluation**:
- Baseline NRMSE: ~0.78
- TTC NRMSE: ~0.09
- Improvement: ~88%

---

## See Also

- Full documentation: `TRAINING_METRICS_DOCUMENTATION.md`
- Loss implementations: `src/ups/training/losses.py`
- Evaluation runner: `src/ups/eval/pdebench_runner.py`
- WandB context: `src/ups/utils/wandb_context.py`
