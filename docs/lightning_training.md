# PyTorch Lightning Training Guide

This document describes how to use PyTorch Lightning for training the Universal Physics Stack (UPS) models.

## Overview

The Lightning implementation provides an alternative training path alongside the native PyTorch DDP implementation. Lightning offers:

- **Easier strategy switching**: Toggle between DDP, FSDP, and other strategies via config
- **Cleaner code**: Reduced boilerplate for distributed training
- **Better ecosystem integration**: Easy integration with callbacks, loggers, and profilers
- **Backward compatibility**: Checkpoints can be loaded into native models

## Quick Start

### Single-GPU Training

```bash
python scripts/train_lightning.py \
  --config configs/train_burgers_golden.yaml \
  --stage operator
```

### Multi-GPU Training (DDP)

```bash
torchrun --nproc_per_node=2 scripts/train_lightning.py \
  --config configs/train_pdebench_2task_baseline_ddp.yaml \
  --stage operator
```

### Multi-GPU Training (FSDP)

```bash
# Enable FSDP in your config:
# training:
#   use_fsdp2: true
#   num_gpus: 4

torchrun --nproc_per_node=4 scripts/train_lightning.py \
  --config configs/train_pdebench_2task_baseline_ddp_4gpu.yaml \
  --stage operator
```

### Multi-Stage Pipeline

```bash
python scripts/run_lightning_pipeline.py \
  --train-config configs/train_burgers_golden.yaml
```

## Configuration

### Basic Configuration

Lightning training uses the same config files as native training. Key parameters:

```yaml
training:
  num_gpus: 2              # Number of GPUs (1 = single-GPU, 2+ = DDP/FSDP)
  use_fsdp2: false         # Enable FSDP (Fully Sharded Data Parallel)
  batch_size: 8            # Per-GPU batch size
  accum_steps: 6           # Gradient accumulation steps
  compile: true            # Enable torch.compile
  compile_mode: "default"  # Compile mode: default, reduce-overhead, max-autotune
  amp: true                # Enable automatic mixed precision
  amp_dtype: "bfloat16"    # AMP dtype: bfloat16 or float16
  grad_clip: 1.0           # Gradient clipping value
  dt: 0.1                  # Time step for operator

stages:
  operator:
    epochs: 25
    patience: 5            # Early stopping patience

optimizer:
  name: "adam"             # Optimizer: adam, adamw, sgd, muon_hybrid
  lr: 1e-3
  weight_decay: 0.0

logging:
  wandb:
    enabled: true
    project: "universal-simulator"
    entity: "your-entity"
```

### Strategy Selection

Lightning automatically selects the strategy based on config:

| Config | Strategy |
|--------|----------|
| `num_gpus: 1` | Single-GPU (auto) |
| `num_gpus: 2+` | DDP |
| `num_gpus: 2+, use_fsdp2: true` | FSDP |

### Compile Settings

Lightning supports torch.compile with fallback:

```yaml
training:
  compile: true              # Enable compilation
  compile_mode: "default"    # Options: default, reduce-overhead, max-autotune
```

If compilation fails, Lightning will gracefully fall back to eager mode.

## Command-Line Interface

### train_lightning.py

```bash
python scripts/train_lightning.py \
  --config <path-to-config> \
  --stage <operator|diff_residual|steady_prior> \
  [--devices <num-gpus>]
```

**Arguments:**
- `--config`: Path to training config YAML (required)
- `--stage`: Training stage (currently only `operator` is supported)
- `--devices`: Override device count (defaults to `training.num_gpus` from config)

### run_lightning_pipeline.py

```bash
python scripts/run_lightning_pipeline.py \
  --train-config <path-to-config> \
  [--devices <num-gpus>]
```

**Arguments:**
- `--train-config`: Path to training config YAML (required)
- `--devices`: Override device count for all stages

## Distributed Training

### DDP (Distributed Data Parallel)

**Best for**: 2-8 GPUs, models that fit in single-GPU memory

```bash
torchrun --nproc_per_node=2 scripts/train_lightning.py \
  --config configs/train_pdebench_2task_baseline_ddp.yaml \
  --stage operator
```

**Features:**
- Each GPU maintains a full model copy
- Gradients are synchronized after backward pass
- Scales well up to 8 GPUs
- Simple and reliable

### FSDP (Fully Sharded Data Parallel)

**Best for**: 4+ GPUs, memory-constrained models

```yaml
training:
  use_fsdp2: true
  num_gpus: 4
```

```bash
torchrun --nproc_per_node=4 scripts/train_lightning.py \
  --config configs/train_pdebench_2task_baseline_ddp_4gpu.yaml \
  --stage operator
```

**Features:**
- Model parameters are sharded across GPUs
- Reduces memory usage (~30-40% savings)
- Better scaling for large models
- Requires more communication bandwidth

**Trade-offs:**
- FSDP: Lower memory, slightly slower
- DDP: Higher memory, faster

## Checkpointing

### Checkpoint Format

Lightning saves checkpoints in PyTorch Lightning format:

```
checkpoints/
  operator-epoch=01-val_nrmse=0.1234.ckpt
  operator-epoch=02-val_nrmse=0.0987.ckpt
  operator-epoch=03-val_nrmse=0.0876.ckpt
```

### Loading Lightning Checkpoints into Native Models

Lightning checkpoints can be loaded into native models for inference:

```python
import torch
from ups.models.latent_operator import LatentOperator, LatentOperatorConfig

# Load Lightning checkpoint
ckpt = torch.load("checkpoints/operator-epoch=10.ckpt")

# Extract state dict
lightning_state = ckpt["state_dict"]

# Remove 'operator.' prefix
operator_state = {}
for key, value in lightning_state.items():
    if key.startswith("operator."):
        operator_state[key.replace("operator.", "", 1)] = value

# Load into native model
operator = LatentOperator(config)
operator.load_state_dict(operator_state, strict=False)
```

### Loading Native Checkpoints into Lightning

Native checkpoints can also be loaded into Lightning models:

```python
from ups.training.lightning_modules import OperatorLightningModule

# Load native checkpoint
native_state = torch.load("checkpoints/operator.pt")

# Create Lightning model
model = OperatorLightningModule(cfg)

# Load operator state
model.operator.load_state_dict(native_state["operator"], strict=False)
```

## Logging

### WandB Integration

Lightning automatically integrates with WandB:

```yaml
logging:
  wandb:
    enabled: true
    project: "universal-simulator"
    entity: "your-entity"
    run_name: "my-experiment"
    tags: ["lightning", "ddp"]
```

**Features:**
- Single WandB run (not per-rank)
- Automatic metric logging
- Model checkpointing to WandB (optional)
- Distributed training metadata

### Metrics

Lightning logs the following metrics:

**Training:**
- `train/loss`: Total training loss
- `train/forward_loss`: Forward prediction loss
- `train/spectral_loss`: Spectral energy loss (if enabled)
- `train/rollout_loss`: Rollout loss (if enabled)
- `train/<task>/nrmse`: Per-task NRMSE (multi-task only)

**Validation:**
- `val/nrmse`: Validation NRMSE

## Performance

### Expected Performance

**Single-GPU vs Multi-GPU (2 GPUs):**
- Throughput: ~1.7-1.9× faster
- Memory per GPU: ~50% of single-GPU
- Batch size: 2× larger

**DDP vs FSDP (4 GPUs):**
- DDP: Faster, higher memory per GPU
- FSDP: Slower (~10-15%), lower memory per GPU (~30% savings)

### Optimization Tips

1. **Increase batch size**: With more GPUs, increase per-GPU batch size to maximize utilization
2. **Enable compile**: Use `compile: true` for 10-30% speedup (when stable)
3. **Reduce gradient accumulation**: With more GPUs, reduce `accum_steps`
4. **Enable AMP**: Use `amp: true` with `bfloat16` for memory savings
5. **Tune num_workers**: Increase `num_workers` for parallel data loading

## Troubleshooting

### Common Issues

**1. "RuntimeError: NCCL error"**

Solution: Check GPU connectivity, reduce batch size, or disable compile:
```yaml
training:
  compile: false
```

**2. "Out of memory" errors**

Solution: Reduce batch size or enable FSDP:
```yaml
training:
  batch_size: 4  # Reduce
  use_fsdp2: true  # Enable FSDP
```

**3. "Checkpoint compatibility issues"**

Solution: Use the state dict extraction code above to convert between formats.

**4. "Multiple WandB runs created"**

This should not happen with Lightning. Check that `replace_sampler_ddp=False` in the trainer config.

## Testing

### Running Tests

```bash
# All Lightning tests
pytest tests/integration/test_lightning_parity.py -v

# Specific test
pytest tests/integration/test_lightning_parity.py::test_lightning_vs_native_single_gpu -v

# GPU tests only
pytest tests/integration/test_lightning_parity.py -m gpu -v

# Slow tests (2-GPU, 4-GPU)
pytest tests/integration/test_lightning_parity.py -m slow -v
```

### Test Coverage

The test suite covers:
- ✅ Single-GPU parity with native training
- ✅ 2-GPU DDP training
- ✅ 4-GPU DDP training
- ✅ FSDP support
- ✅ Checkpoint compatibility (Lightning ↔ native)
- ✅ Compile toggle
- ✅ Multi-stage pipeline

## Comparison: Lightning vs Native

| Feature | Lightning | Native |
|---------|-----------|--------|
| **Code complexity** | Lower | Higher |
| **Flexibility** | Medium | High |
| **Strategy switching** | Easy (config-driven) | Manual |
| **Ecosystem** | Rich (callbacks, loggers) | Limited |
| **Performance** | Similar (~5% slower) | Baseline |
| **Debugging** | Harder (abstraction layers) | Easier |
| **Checkpoint format** | Lightning-specific | PyTorch standard |
| **Multi-stage support** | Partial (operator only) | Full |

### When to Use Lightning

**Use Lightning when:**
- You want easy strategy switching (DDP ↔ FSDP)
- You need Lightning ecosystem features (callbacks, profilers)
- You prefer cleaner, less boilerplate code
- You're experimenting with different distributed strategies

**Use Native when:**
- You need full control over training loop
- You're running production workloads (proven stability)
- You need all stages (diff_residual, steady_prior)
- You prefer PyTorch standard checkpoints

## Current Limitations

Lightning support is currently **partial**:

| Feature | Status |
|---------|--------|
| Operator stage | ✅ Implemented |
| Diffusion residual stage | ❌ Not implemented |
| Consistency distillation | ❌ Not implemented |
| Steady prior stage | ❌ Not implemented |
| Multi-stage pipeline | ⚠️ Operator only |
| Checkpoint resume | ✅ Supported |
| VastAI integration | ⚠️ Manual setup |

**Recommendation**: Use Lightning for operator-only experiments. For full pipelines, use native training.

## Future Enhancements

Planned improvements:
- [ ] Diffusion residual Lightning module
- [ ] Consistency distillation Lightning module
- [ ] Steady prior Lightning module
- [ ] VastAI launcher integration
- [ ] DeepSpeed strategy support
- [ ] Advanced profiling and optimization

## References

- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [DDP Guide](https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html)
- [FSDP Guide](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html)
- [Native Training Guide](../PRODUCTION_WORKFLOW.md)
