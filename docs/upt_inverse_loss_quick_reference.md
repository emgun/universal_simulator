# UPT Inverse Loss: Quick Reference Guide

## Enable Inverse Losses

### Configuration (YAML)

```yaml
training:
  # Enable inverse losses
  use_inverse_losses: true
  lambda_inv_enc: 0.01              # Encoding reconstruction weight
  lambda_inv_dec: 0.01              # Decoding reconstruction weight
  
  # Frequency control
  inverse_loss_frequency: 1         # Apply every batch (1) or every N batches
  
  # Curriculum learning
  inverse_loss_warmup_epochs: 5     # Warmup period before full strength
  inverse_loss_max_weight: 0.05     # Cap on maximum weight
  
  # Query sampling for memory efficiency
  query_sampling:
    enabled: true
    num_queries: 2048               # Sample N points (vs. full 4096 for 64×64)
    strategy: uniform               # "uniform" or "stratified"
```

## Memory Impact

### Query Sampling Savings

| Setting | Points | Memory (B=10) | Reduction |
|---------|--------|---------------|-----------|
| No sampling | 4096 | 10.5 GB | Baseline |
| With sampling | 2048 | 5.2 GB | -50% |
| With AMP | 2048 | 2.6 GB | -75% |

### Typical Configuration

For **64×64 grids** (4096 points):
- Set `num_queries: 2048` for 50% decoder memory savings
- Apply only to `inverse_encoding_loss` (inverse_decoding requires full grid)
- Provides 20-30% training speedup

For **multi-task training** (slower machines):
- Reduce to `num_queries: 1024` for additional 2x speedup

## Implementation Details

### Two Inverse Loss Components

#### 1. Inverse Encoding Loss
```
input_fields → [encoded] → decoder → reconstructed → MSE loss
```
- Ensures decoder can reconstruct input fields from latent
- Supports query sampling (50% reduction possible)
- Location: `src/ups/training/losses.py:25-82`

#### 2. Inverse Decoding Loss
```
latent → decoder → fields → [re-encode] → latent' → MSE loss
```
- Ensures encoder can re-encode decoded fields
- No query sampling (encoder requires full grid)
- Location: `src/ups/training/losses.py:85-135`

### Curriculum Learning Schedule

```
Epoch 0-5:      weight = 0        (pure forward training)
Epoch 5-10:     weight = 0 → 0.01 (linear ramp)
Epoch 10+:      weight = 0.01     (full strength)
```

Maximum weight is capped at `inverse_loss_max_weight: 0.05` to prevent instability.

## Key Files

| Component | File | Lines |
|-----------|------|-------|
| Inverse encoding | `src/ups/training/losses.py` | 25-82 |
| Inverse decoding | `src/ups/training/losses.py` | 85-135 |
| Loss bundle | `src/ups/training/losses.py` | 267-369 |
| Curriculum schedule | `src/ups/training/losses.py` | 233-264 |
| Query sampling | `src/ups/training/query_sampling.py` | 106-153 |
| Training loop | `scripts/train.py` | 498-774 |
| Decoder model | `src/ups/io/decoder_anypoint.py` | 58-160 |
| Encoder model | `src/ups/io/enc_grid.py` | 22-137 |

## Common Patterns

### Check if Inverse Losses Enabled

```python
use_inverse_losses = (
    bool(cfg.get("training", {}).get("use_inverse_losses", False)) or
    cfg.get("training", {}).get("lambda_inv_enc", 0.0) > 0 or
    cfg.get("training", {}).get("lambda_inv_dec", 0.0) > 0
)
```

### Extract Lambda Parameters

```python
loss_weights = {
    "lambda_inv_enc": float(train_cfg.get("lambda_inv_enc", 0.0)),
    "lambda_inv_dec": float(train_cfg.get("lambda_inv_dec", 0.0)),
    "inverse_loss_warmup_epochs": int(train_cfg.get("inverse_loss_warmup_epochs", 15)),
    "inverse_loss_max_weight": float(train_cfg.get("inverse_loss_max_weight", 0.05)),
}
```

### Apply Query Sampling

```python
query_sample_cfg = train_cfg.get("query_sampling", {})
use_query_sampling = query_sample_cfg.get("enabled", False)
num_queries = query_sample_cfg.get("num_queries", None) if use_query_sampling else None
query_strategy = query_sample_cfg.get("strategy", "uniform")
```

### Compute Loss Bundle

```python
loss_bundle = compute_operator_loss_bundle(
    input_fields=input_fields_physical if use_inv_now else None,
    encoded_latent=state.z if use_inv_now else None,
    decoder=decoder if use_inv_now else None,
    input_positions=coords if use_inv_now else None,
    encoder=encoder if use_inv_now else None,
    query_positions=coords if use_inv_now else None,
    coords=coords if use_inv_now else None,
    meta=meta if use_inv_now else None,
    pred_next=next_state.z,
    target_next=target,
    weights=loss_weights,
    current_epoch=epoch,
    num_queries=num_queries,
    query_strategy=query_strategy,
    grid_shape=grid_shape,
)
```

## Performance Tips

### Memory Optimization
1. **Enable query sampling**: Reduce `num_queries` from 4096 to 2048 (50% savings)
2. **Use AMP**: Enable `training.amp: true` for bfloat16 (another 50% savings)
3. **Reduce frequency**: Set `inverse_loss_frequency: 2` to apply every 2 batches (further 50% savings)

### Speed Optimization
1. **Query sampling strategy**: Use `uniform` for simplicity, `stratified` for better coverage
2. **Batch size**: Inverse losses are O(B × Q), so reduce batch size if OOM
3. **Accumulation steps**: Use gradient accumulation instead of larger batches

### Quality Optimization
1. **Warmup period**: 5-15 epochs for stable convergence (avoid gradient explosion)
2. **Lambda weights**: 0.01 is typical starting point, tune up to 0.05 maximum
3. **Application frequency**: Use `inverse_loss_frequency: 1` for consistent training signal

## Debugging

### Check if Inverse Losses Are Applied

```python
# In training loop
if wandb_ctx and i % 10 == 0:
    for name, value in loss_bundle.components.items():
        print(f"{name}: {value.item():.6f}")
        # Should see "L_inv_enc" and "L_inv_dec" if enabled
```

### Verify Query Sampling

```python
# Check input dimensions
print(f"Full grid size: {coords.shape[1]}")      # Should be H×W = 4096
print(f"Sampled size: {coords_sampled.shape[1]}") # Should be 2048
```

### Monitor Curriculum Weight

```python
from ups.training.losses import compute_inverse_loss_curriculum_weight

for epoch in range(40):
    weight = compute_inverse_loss_curriculum_weight(
        epoch, 0.01, warmup_epochs=5, max_weight=0.05
    )
    print(f"Epoch {epoch}: weight={weight:.6f}")
```

## References

- Full analysis: `/Users/emerygunselman/Code/universal_simulator/docs/upt_inverse_loss_analysis.md`
- Example config: `/Users/emerygunselman/Code/universal_simulator/configs/train_burgers_upt_full.yaml`
- Multi-task config: `/Users/emerygunselman/Code/universal_simulator/configs/train_pdebench_2task_192d.yaml`
