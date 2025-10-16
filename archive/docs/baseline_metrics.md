# Baseline Metrics - Phase 0 Validation

This document tracks baseline metrics for the Universal Physics Stack as we scale up.

## Phase 0.3 - Pipeline Validation (Oct 1, 2025)

### Setup
- **Dataset**: burgers1d_subset_v1 (16 train samples, 201 timesteps, 1024 spatial points)
- **Model**: Latent dim=128, tokens=64, PDE-T with depths=[1,1,1]
- **Training**: 2 epochs per stage (operator, diff_residual, consistency_distill, steady_prior)
- **Hardware**: CPU (local development)
- **Config**: `configs/train_pdebench_test.yaml`

### Training Results

**Operator Stage** (2 epochs):
- Initial loss: 0.9986
- Final loss: 0.9944
- Improvement: 0.42%
- Checkpoint size: 7.1 MB

**Diffusion Residual** (2 epochs):
- Checkpoint size: 519 KB
- Status: ✅ Completed

**Consistency Distillation** (2 epochs):
- Status: ✅ Completed

**Steady Prior** (2 epochs):
- Checkpoint size: 519 KB
- Status: ✅ Completed

### Evaluation Metrics (Test Set)

```json
{
  "mse": 0.9639,
  "mae": 0.8284,
  "rmse": 0.9818
}
```

**Test Configuration**:
- Rollout steps: 10
- Samples evaluated: ~3.3M points
- Tau: 0.5

### Pipeline Validation Status

✅ **Artifact Extraction**: Successfully extracted burgers1d from tarball
✅ **Data Loading**: PDEBench HDF5 format verified (16×201×1024×1)
✅ **Training Pipeline**: All 4 stages execute without errors
✅ **Checkpoint Creation**: Operator, diffusion, and steady prior checkpoints saved
✅ **Evaluation Pipeline**: Generates JSON, CSV, HTML, PNG outputs
✅ **Visualizations**: Latent heatmaps, spectra, error histograms created

### Generated Outputs

- `checkpoints/test/operator.pt` (7.1 MB)
- `checkpoints/test/diffusion_residual.pt` (519 KB)
- `checkpoints/test/steady_prior.pt` (519 KB)
- `reports/test_eval.json` (metrics)
- `reports/test_eval.html` (full report)
- `reports/test_eval_*.png` (5 visualization plots)
- `reports/test_eval_preview.npz` (latent states)

### Next Steps (Phase 1)

1. **Scale Model Capacity**: Increase latent_dim to 256-512, tokens to 128-256
2. **Prepare More Datasets**: Add Advection1D, Burgers2D, Navier-Stokes2D
3. **GPU Training**: Enable AMP and run on GPU for faster iteration
4. **Longer Training**: Run full 30/15/10/20 epoch schedule
5. **Baseline Comparison**: Compare against FNO, UNet, DiT baselines

### Notes

- This is a minimal validation run to verify the pipeline works end-to-end
- Only 2 epochs per stage limits model performance (expected low accuracy)
- Loss did decrease, indicating learning is occurring
- CPU-only training is slow but validates the workflow
- All infrastructure components (data, training, eval, visualization) are functional

**Conclusion**: Phase 0.3 pipeline validation successful ✅
Ready to proceed with Phase 1 scaling.

