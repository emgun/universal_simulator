# SOTA Comparison Guide - PDE Benchmarking

**Last Updated:** October 14, 2025

This guide documents how to properly compare the Universal Simulator against State-of-the-Art (SOTA) PDE solvers using standardized metrics from the research literature.

## Standard Metrics for PDE Benchmarking

### Primary Metrics

1. **Relative L2 Error**
   - **Formula:** `||u_pred - u_true||_L2 / ||u_true||_L2`
   - **Why:** Scale-invariant, accounts for solution magnitude
   - **Used by:** PDEBench, FNO, Neuraloperator papers
   - **Implementation:** `sqrt(MSE(pred, true)) / sqrt(mean(true^2))`

2. **nRMSE (Normalized Root Mean Square Error)**
   - **Formula:** `RMSE / RMS(u_true)`
   - **Why:** Normalized by target magnitude, comparable across datasets
   - **Used by:** Many ML + PDE papers
   - **Implementation:** `sqrt(MSE) / sqrt(mean(true^2))`
   - **Note:** For element-wise comparison, nRMSE ≡ relative L2

3. **MSE (Mean Squared Error)**
   - **Formula:** `mean((u_pred - u_true)^2)`
   - **Why:** Standard ML metric, differentiable
   - **Limitation:** Not scale-invariant

4. **RMSE (Root Mean Square Error)**
   - **Formula:** `sqrt(MSE)`
   - **Why:** Same units as solution
   - **Limitation:** Not normalized

### Secondary Metrics

5. **MAE (Mean Absolute Error)**
   - **Formula:** `mean(|u_pred - u_true|)`
   - **Why:** Robust to outliers
   - **Used by:** Some benchmarks

## Metrics Now Implemented

As of October 14, 2025, our evaluation pipeline ([src/ups/eval/pdebench_runner.py](../src/ups/eval/pdebench_runner.py)) computes:

- ✅ **MSE** - Mean Squared Error
- ✅ **MAE** - Mean Absolute Error
- ✅ **RMSE** - Root Mean Square Error
- ✅ **nRMSE** - Normalized RMSE (SOTA)
- ✅ **Relative L2** - Normalized L2 error (SOTA)

**Per-sample metrics also available:**
- `per_sample_mse`
- `per_sample_mae`
- `per_sample_rel_l2` (for distribution analysis)

## Running SOTA-Comparable Evaluations

### Step 1: Baseline Evaluation (No TTC)

Run evaluation without TTC to establish baseline performance:

```bash
# Validation split
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/eval_burgers_512dim_val_baseline.yaml \
  --operator checkpoints/scale/operator.pt \
  --diffusion checkpoints/scale/diffusion_residual.pt \
  --output-prefix reports/baseline_val_notttc \
  --print-json

# Test split
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/eval_burgers_512dim_test_baseline.yaml \
  --operator checkpoints/scale/operator.pt \
  --diffusion checkpoints/scale/diffusion_residual.pt \
  --output-prefix reports/baseline_test_notttc \
  --print-json
```

### Step 2: TTC Evaluation

Run evaluation with TTC for comparison:

```bash
# Validation split with TTC
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/eval_burgers_512dim_ttc_val.yaml \
  --operator checkpoints/scale/operator.pt \
  --diffusion checkpoints/scale/diffusion_residual.pt \
  --output-prefix reports/ttc_val \
  --print-json

# Test split with TTC
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/eval_burgers_512dim_ttc_test.yaml \
  --operator checkpoints/scale/operator.pt \
  --diffusion checkpoints/scale/diffusion_residual.pt \
  --output-prefix reports/ttc_test \
  --print-json
```

### Step 3: Compare Results

```bash
# View baseline results
echo "=== BASELINE (No TTC) ==="
cat reports/baseline_val_notttc.json
cat reports/baseline_test_notttc.json

# View TTC results
echo "=== WITH TTC ==="
cat reports/ttc_val.json
cat reports/ttc_test.json
```

Expected output format:
```json
{
  "metrics": {
    "mse": 0.001216,
    "mae": 0.024571,
    "rmse": 0.034876,
    "nrmse": 0.012345,      // NEW: For SOTA comparison
    "rel_l2": 0.012345      // NEW: For SOTA comparison
  },
  "extra": {
    "samples": 2621440000,
    "tau": 0.5,
    "ttc": false
  }
}
```

## SOTA Benchmarks from Literature

### PDEBench - Burgers Equation (1D)

**Paper:** "PDEBench: An Extensive Benchmark for Scientific Machine Learning" (2022)

| Model | Relative L2 ↓ | Year |
|-------|--------------|------|
| FNO-2D | 0.0180 | 2021 |
| U-Net | 0.0250 | 2020 |
| ResNet | 0.0310 | 2019 |
| Baseline CNN | 0.0450 | 2022 |

**Dataset:** PDEBench Burgers 1D, 1024 spatial points, 100 time steps

### Neural Operator Benchmarks

**Paper:** "Fourier Neural Operator for Parametric PDEs" (2021)

| Model | Test Error ↓ | Parameters |
|-------|-------------|------------|
| FNO-2D (Best) | 0.0135 | 2.4M |
| FNO-1D | 0.0200 | 0.8M |
| DeepONet | 0.0280 | 1.2M |
| PINN | 0.0450 | 0.5M |

**Metric:** Relative L2 error on test set

### Our Target Performance

Based on current architecture (512-dim latent operator):
- **Parameters:** ~2M (operator) + ~100K (diffusion) = ~2.1M total
- **Target Relative L2:** < 0.020 (competitive with FNO)
- **Target nRMSE:** < 0.020 (competitive with FNO)

## Comparison Methodology

### Fair Comparison Requirements

1. **Same Dataset Split**
   - Use standard PDEBench train/val/test splits
   - Document any preprocessing (normalization, etc.)

2. **Same Evaluation Protocol**
   - Report metrics on TEST split (not validation)
   - Use relative L2 or nRMSE for scale-invariance
   - Report mean ± std over multiple runs if stochastic

3. **Comparable Model Size**
   - Report total parameters
   - Report inference time (ms/sample)
   - Report GPU memory requirements

4. **Reproducibility**
   - Specify random seed
   - Specify batch size
   - Specify precision (fp32, bf16, etc.)
   - Include config files in repo

### Reporting Template

```markdown
## Results on PDEBench Burgers 1D

**Model:** Universal Simulator (512-dim latent operator + diffusion)
**Checkpoints:**
- Operator: run-mt7rckc8-history:v0
- Diffusion: run-pp0c2k31-history:v0

### Validation Set
- **Relative L2:** 0.XXXX ± 0.YYYY
- **nRMSE:** 0.XXXX ± 0.YYYY
- **MSE:** 0.XXXX
- **Samples:** 2,621,440,000 elements (13 trajectories)

### Test Set
- **Relative L2:** 0.XXXX ± 0.YYYY
- **nRMSE:** 0.XXXX ± 0.YYYY
- **MSE:** 0.XXXX
- **Samples:** 2,621,440,000 elements (13 trajectories)

### Model Details
- **Parameters:** 2.1M (operator) + 100K (diffusion)
- **Latent Dimension:** 512
- **Latent Tokens:** 128
- **Architecture:** PDET (Patch Diffusion Transformer)
- **Inference Time:** XX ms/step (H200 GPU)
- **Memory:** XX GB (batch size 32)

### TTC Improvement
- **Baseline (no TTC):** Relative L2 = 0.XXXX
- **With TTC:** Relative L2 = 0.YYYY
- **Improvement:** -X.XX% (lower is better)

### Configuration
- Precision: bf16
- Batch Size: 32
- Random Seed: 17
- Config: configs/eval_burgers_512dim_ttc_test.yaml
```

## TTC Impact on SOTA Metrics

### Measuring TTC Benefit

The key question: **Does TTC improve SOTA metrics?**

**Hypothesis:** TTC trajectory selection should reduce relative L2 by choosing physics-consistent predictions.

**Test:**
1. Run baseline evaluation (TTC disabled)
2. Run TTC evaluation (same checkpoints)
3. Compare relative L2 and nRMSE
4. Compute improvement percentage

**Expected Results:**
- If TTC works: relative L2 decreases by 5-15%
- If TTC overhead: relative L2 stays same or increases slightly
- If TTC harmful: relative L2 increases (back to drawing board)

### Analysis Script

```python
import json

# Load results
with open('reports/baseline_test_notttc.json') as f:
    baseline = json.load(f)
with open('reports/ttc_test.json') as f:
    ttc = json.load(f)

# Compare
baseline_rel_l2 = baseline['metrics']['rel_l2']
ttc_rel_l2 = ttc['metrics']['rel_l2']
improvement = (baseline_rel_l2 - ttc_rel_l2) / baseline_rel_l2 * 100

print(f"Baseline Relative L2: {baseline_rel_l2:.6f}")
print(f"TTC Relative L2:      {ttc_rel_l2:.6f}")
print(f"Improvement:          {improvement:+.2f}%")

if improvement > 5:
    print("✅ TTC provides significant improvement!")
elif improvement > 0:
    print("⚠️  TTC provides marginal improvement")
else:
    print("❌ TTC does not improve metrics")
```

## Next Steps for SOTA Comparison

### Immediate Actions

1. **Run Baseline Evaluation**
   ```bash
   # Use configs created above
   PYTHONPATH=src python scripts/evaluate.py \
     --config configs/eval_burgers_512dim_val_baseline.yaml \
     --operator checkpoints/scale/operator.pt \
     --diffusion checkpoints/scale/diffusion_residual.pt \
     --output-prefix reports/baseline_val_notttc \
     --print-json
   ```

2. **Re-run TTC with Updated Metrics**
   ```bash
   PYTHONPATH=src python scripts/evaluate.py \
     --config configs/eval_burgers_512dim_ttc_val.yaml \
     --operator checkpoints/scale/operator.pt \
     --diffusion checkpoints/scale/diffusion_residual.pt \
     --output-prefix reports/ttc_val_with_sota_metrics \
     --print-json
   ```

3. **Compare and Document**
   - Compute TTC improvement
   - Compare against PDEBench baselines
   - Document results in paper/report

### Future Work

1. **Extended Benchmarks**
   - Evaluate on other PDEBench tasks (advection, reaction-diffusion)
   - Compare on different spatial resolutions
   - Test on longer rollout horizons

2. **Ablation Studies**
   - Effect of latent dimension (256 vs 512 vs 1024)
   - Effect of diffusion correction
   - Effect of TTC parameters (candidates, beam width, grid size)

3. **Publication-Ready Results**
   - Multiple random seeds for error bars
   - Full dataset evaluation (not limited samples)
   - Inference timing benchmarks
   - Memory profiling

## References

### Key Papers

1. **PDEBench** (Takamoto et al., 2022)
   - https://arxiv.org/abs/2210.07182
   - Standard benchmark suite for PDE ML

2. **Fourier Neural Operator** (Li et al., 2021)
   - https://arxiv.org/abs/2010.08895
   - SOTA neural operator architecture

3. **Neural Operator Survey** (Kovachki et al., 2021)
   - https://arxiv.org/abs/2108.08481
   - Overview of neural operator methods

### Our Implementation

- **Evaluation Code:** [src/ups/eval/pdebench_runner.py](../src/ups/eval/pdebench_runner.py)
- **Baseline Configs:**
  - [configs/eval_burgers_512dim_val_baseline.yaml](../configs/eval_burgers_512dim_val_baseline.yaml)
  - [configs/eval_burgers_512dim_test_baseline.yaml](../configs/eval_burgers_512dim_test_baseline.yaml)
- **TTC Configs:**
  - [configs/eval_burgers_512dim_ttc_val.yaml](../configs/eval_burgers_512dim_ttc_val.yaml)
  - [configs/eval_burgers_512dim_ttc_test.yaml](../configs/eval_burgers_512dim_ttc_test.yaml)

---

**Updated:** October 14, 2025
**Status:** ✅ Metrics implemented, ready for SOTA comparison runs
**Next:** Run baseline and TTC evaluations, compare results
