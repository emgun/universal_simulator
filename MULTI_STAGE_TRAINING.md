# Multi-Stage Training Pipeline Implementation

## Overview

The Universal Physics Stack (UPS) implements a sequential four-stage training pipeline for latent space physics operators. Each stage builds on the previous stage's checkpoint, enabling progressive refinement of the model.

**Training Stages:**
1. **Operator**: Deterministic latent evolution model (PDE-Transformer)
2. **Diffusion Residual**: Diffusion model for uncertainty/refinement
3. **Consistency Distillation**: Few-step distillation of diffusion model
4. **Steady Prior**: Steady-state latent prior (optional)

The pipeline is orchestrated by `scripts/train.py` with optional integration via `scripts/run_fast_to_sota.py`.

---

## Stage 1: Operator Training

### Implementation Location
**File**: `/Users/emerygunselman/Code/universal_simulator/scripts/train.py` (lines 400-693)  
**Function**: `train_operator(cfg, wandb_ctx=None, global_step=0)`

### Architecture
The operator is a latent evolution model (`LatentOperator`) with a PDE-Transformer backbone (`PDETransformerConfig`).

**Model Creation**:
```python
def make_operator(cfg: dict) -> LatentOperator:
    latent_cfg = cfg.get("latent", {})
    dim = latent_cfg.get("dim", 32)
    pdet_cfg = cfg.get("operator", {}).get("pdet", {})
    # Builds PDETransformerConfig with:
    # - input_dim: matches latent.dim
    # - hidden_dim: typically dim * 2
    # - depths: list of layer depths per block (e.g., [1, 1, 1])
    # - group_size: window attention group size
    # - num_heads: attention heads
    config = LatentOperatorConfig(
        latent_dim=dim,
        pdet=PDETransformerConfig(**pdet_cfg),
        time_embed_dim=dim,
    )
    return LatentOperator(config)
```

### Input/Output Format
**Input**:
- Latent state `z0`: shape `(B, tokens, latent_dim)` at time t=0
- Conditioning `cond`: optional dict of conditioning tensors

**Output**:
- Predicted latent state `z1`: shape `(B, tokens, latent_dim)` at time t=dt
- Temporal step: fixed dt (typically 0.1)

### Loss Functions

#### Primary Loss: Forward Prediction Loss
```
L_forward = MSE(predicted_z1, target_z1)
```
Ensures one-step ahead prediction accuracy. Always computed.

#### Optional Auxiliary Losses

1. **Inverse Encoding Loss** (`L_inv_enc`)
   - Ensures latent is decodable back to original fields
   - Flow: `input_fields → decoder → reconstructed_fields`
   - Loss: `MSE(reconstructed, input_fields)` in physical space
   - Weight: `lambda_inv_enc` (default: 0.0)

2. **Inverse Decoding Loss** (`L_inv_dec`)
   - Ensures decoded fields can be re-encoded
   - Flow: `latent → decoder → fields → encoder → reconstructed_latent`
   - Loss: `MSE(reconstructed_latent, latent)` in latent space
   - Weight: `lambda_inv_dec` (default: 0.0)

3. **Rollout Loss** (`L_rollout`)
   - Multi-step trajectory prediction loss
   - Loss: `MSE(predicted_rollout, target_rollout)`
   - Weight: `lambda_rollout` (default: 0.0)

4. **Spectral Loss** (`L_spec`)
   - Frequency-domain loss for energy preservation
   - Implementation:
     ```python
     pred_fft = torch.fft.rfft(pred, dim=1)
     target_fft = torch.fft.rfft(target, dim=1)
     energy_pred = |pred_fft|^2
     energy_target = |target_fft|^2
     L_spec = |energy_pred - energy_target| / (energy_target + eps)
     ```
   - Weight: `lambda_spectral` (default: 0.0)

**Loss Bundle Computation** (`src/ups/training/losses.py`):
```python
loss_bundle = compute_operator_loss_bundle(
    input_fields=input_fields_physical,
    encoded_latent=state.z,
    decoder=decoder,
    input_positions=coords,
    encoder=encoder,
    query_positions=coords,
    coords=coords,
    meta=meta,
    pred_next=next_state.z,
    target_next=target,
    pred_rollout=rollout_pred,
    target_rollout=rollout_tgt,
    spectral_pred=next_state.z,
    spectral_target=target,
    weights={
        "lambda_forward": 1.0,
        "lambda_inv_enc": float(...),
        "lambda_inv_dec": float(...),
        "lambda_spectral": float(...),
        "lambda_rollout": float(...),
    },
    current_epoch=epoch,
)
loss = loss_bundle.total
```

### Curriculum Learning for Inverse Losses

Inverse loss weights follow a curriculum schedule:

```python
def compute_inverse_loss_curriculum_weight(
    epoch: int,
    base_weight: float,
    warmup_epochs: int = 15,
    max_weight: float = 0.05,
) -> float:
    if epoch < warmup_epochs:
        return 0.0  # Pure forward training
    elif epoch < warmup_epochs * 2:
        progress = (epoch - warmup_epochs) / warmup_epochs
        return min(base_weight * progress, max_weight)  # Linear ramp
    else:
        return min(base_weight, max_weight)  # Full weight (capped)
```

**Purpose**: Prevents gradient explosion early in training when encoder/decoder are still poorly conditioned.

### Training Loop

**Configuration** (config.stages.operator):
- `epochs`: Total training epochs
- `optimizer`: {name, lr, weight_decay}
- `scheduler`: {name, step_size, gamma, ...}
- `patience`: Early stopping patience (epochs without improvement)

**Key Components**:

1. **Gradient Accumulation**:
   ```python
   accum_steps = cfg.get("training", {}).get("accum_steps", 1)
   optimizer.zero_grad(set_to_none=True)
   
   for i, batch in enumerate(loader):
       ...
       (loss / accum_steps).backward()
       
       do_step = ((i + 1) % accum_steps == 0) or ((i + 1) == num_batches)
       if do_step:
           grad_norm = torch.nn.utils.clip_grad_norm_(
               operator.parameters(),
               float('inf') if clip_val is None else clip_val
           )
           optimizer.step()
           optimizer.zero_grad(set_to_none=True)
   ```

2. **Automatic Mixed Precision (AMP)**:
   ```python
   use_amp = cfg.get("training", {}).get("amp", False)
   scaler = GradScaler(enabled=use_amp)
   
   with autocast(enabled=use_amp):
       loss = compute_loss(...)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

3. **Exponential Moving Average (EMA)**:
   ```python
   ema_decay = cfg.get("training", {}).get("ema_decay")
   if ema_decay:
       ema_model = copy.deepcopy(operator)
       for p_ema, p in zip(ema_model.parameters(), operator.parameters()):
           p_ema.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)
   ```

4. **Gradient Norm Tracking**:
   ```python
   grad_norm = torch.nn.utils.clip_grad_norm_(
       operator.parameters(),
       float('inf') if clip_val is None else clip_val
   )
   total_grad_norm += float(grad_norm)
   mean_grad_norm = total_grad_norm / grad_steps
   ```
   - Logged per epoch to detect training instabilities
   - If `grad_clip` is set, clipping is applied

5. **Early Stopping**:
   ```python
   if mean_loss + 1e-6 < best_loss:
       best_loss = mean_loss
       best_state = copy.deepcopy(operator.state_dict())
       epochs_since_improve = 0
   else:
       epochs_since_improve += 1
       if patience is not None and epochs_since_improve > patience:
           break
   ```

### Checkpointing & Artifacts

**Saved Checkpoints**:
- `checkpoints/operator.pt`: Best operator state
- `checkpoints/operator_ema.pt`: Best EMA state (if enabled)

**Uploaded to W&B**: via `wandb_ctx.save_file()`

**Logged Metrics** (per epoch):
- `loss`: Mean epoch loss
- `lr`: Learning rate
- `epochs_since_improve`: Patience counter
- `grad_norm`: Mean gradient norm
- `epoch_time_sec`: Epoch duration
- `best_loss`: Best loss so far
- Individual loss components: `L_forward`, `L_inv_enc`, `L_inv_dec`, `L_spec`, `L_rollout`

---

## Stage 2: Diffusion Residual Training

### Implementation Location
**File**: `/Users/emerygunselman/Code/universal_simulator/scripts/train.py` (lines 695-880)  
**Function**: `train_diffusion(cfg, wandb_ctx=None, global_step=0)`

### Architecture
The diffusion model learns a residual correction to the deterministic operator:

**Model Definition** (`src/ups/models/diffusion_residual.py`):
```python
@dataclass
class DiffusionResidualConfig:
    latent_dim: int
    hidden_dim: int = 128
    cond_dim: int = 0
    residual_guidance_weight: float = 1.0

class DiffusionResidual(nn.Module):
    def __init__(self, cfg):
        self.network = nn.Sequential(
            nn.Linear(latent_dim + 1 + cond_dim, hidden_dim),  # z + tau + cond
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    
    def forward(self, state: LatentState, tau: torch.Tensor) -> torch.Tensor:
        # Returns drift: z + drift ≈ z_target
        z = state.z  # (B, tokens, latent_dim)
        tau = tau.view(B, 1, 1)  # (B, 1, 1) expanded to (B, tokens, 1)
        return self.network(torch.cat([z, tau], dim=-1))
```

**Key Mechanism**: 
- `tau` is a per-sample scalar in (0,1) controlling diffusion strength
- Sampled at training time: `tau ~ Uniform(0,1)` or `Beta(alpha, beta)`
- This broadens supervision across the diffusion space

### Loss Functions

**Primary Loss: MSE of Residual**:
```
residual_target = z_target - operator(z_current)
drift = diffusion(z_current, tau)
L_base = MSE(drift, residual_target)
```

The diffusion model learns to predict the gap between:
- What the operator predicts: `operator(z_t)`
- What we observe: `z_{t+1}`

**Optional Auxiliary Losses**:

1. **Spectral Loss** (`lambda_spectral`):
   ```
   L_spec = spectral_energy_loss(drift, residual_target)
   ```

2. **NRMSE Loss** (`lambda_relative`):
   ```
   L_rel = NRMSE(drift, residual_target) = sqrt(MSE / mean(target^2))
   ```

**Total Loss**:
```python
loss = L_base + lambda_spectral * L_spec + lambda_relative * L_rel
```

### Teacher Model (Frozen Operator)

The operator from Stage 1 is loaded as a frozen teacher:

```python
operator = make_operator(cfg)
op_path = checkpoint_dir / "operator.pt"
operator_state = torch.load(op_path, map_location="cpu")
operator_state = _strip_compiled_prefix(operator_state)  # Remove torch.compile wrapper
operator.load_state_dict(operator_state)
operator.eval()  # Freeze
operator = _maybe_compile(operator, cfg, "operator_teacher")
```

### Training Loop

Similar structure to operator training but with key differences:

1. **Batch Processing**:
   ```python
   for i, batch in enumerate(loader):
       z0, z1, cond = unpack_batch(batch)
       state = LatentState(z=z0.to(device), cond=cond)
       
       # Get operator prediction (frozen)
       with torch.no_grad():
           predicted = operator(state, dt_tensor)
       
       # Compute residual target
       residual_target = z1 - predicted.z
       
       # Sample tau for each sample in batch
       tau_tensor = _sample_tau(z0.size(0), device, cfg)
       
       # Diffusion forward
       drift = diff(LatentState(z=predicted.z, cond=cond), tau_tensor)
       
       # Compute loss
       loss = F.mse_loss(drift, residual_target)
   ```

2. **Tau Sampling**:
   ```python
   def _sample_tau(batch_size: int, device, cfg) -> torch.Tensor:
       dist_cfg = cfg.get("training", {}).get("tau_distribution")
       if dist_cfg and dist_cfg.get("type") == "beta":
           alpha = dist_cfg.get("alpha", 1.0)
           beta = dist_cfg.get("beta", 1.0)
           beta_dist = torch.distributions.Beta(alpha, beta)
           return beta_dist.sample((batch_size,)).to(device)
       return torch.rand(batch_size, device=device)  # Default: uniform
   ```

3. **Checkpoint Intervals** (Optional):
   ```python
   checkpoint_interval = cfg.get("training", {}).get("checkpoint_interval", 0)
   if checkpoint_interval and (epoch + 1) % checkpoint_interval == 0:
       epoch_ckpt = checkpoint_dir / f"diffusion_residual_epoch_{epoch + 1}.pt"
       torch.save(diff.state_dict(), epoch_ckpt)
   ```

### Checkpointing & Artifacts

**Saved Checkpoints**:
- `checkpoints/diffusion_residual.pt`: Best diffusion model
- `checkpoints/diffusion_residual_ema.pt`: Best EMA state (if enabled)
- Optional: `checkpoints/diffusion_residual_epoch_*.pt` (periodic checkpoints)

**Uploaded to W&B**: via `wandb_ctx.save_file()`

**Logged Metrics** (per epoch):
- `loss`: Mean epoch loss
- `lr`: Learning rate
- `grad_norm`: Mean gradient norm
- `epoch_time_sec`: Epoch duration

---

## Stage 3: Consistency Distillation

### Implementation Location
**File**: `/Users/emerygunselman/Code/universal_simulator/scripts/train.py` (lines 942-1204)  
**Function**: `train_consistency(cfg, wandb_ctx=None, global_step=0)`

### Motivation

The diffusion model requires many sampling steps at inference for good accuracy. Consistency distillation trains a "student" to match the "teacher" (diffusion) predictions in fewer steps.

**Key Insight**: The student diffusion model IS the diffusion model from Stage 2, being distilled via teacher forcing.

### Teacher Predictions (Operator + Diffusion)

**Teacher Flow**:
```
z_t → operator → z_pred → + diffusion_residual(z_pred, tau) → z_t+1
```

The teacher predicts z_{t+1} by:
1. Running operator to get base prediction
2. Adding diffusion residual at various tau values

### Loss Functions

The distillation loss is computed over multiple tau samples:

```python
def _distill_forward_and_loss_compiled(
    teacher_z_chunk: torch.Tensor,      # (B, tokens, latent_dim)
    teacher_cond_chunk: dict,
    num_taus: int,
    diff_model: nn.Module,
    t_value: torch.Tensor,
    tau_seed: torch.Tensor,             # (num_taus,)
    device: torch.device,
) -> torch.Tensor:
    
    # Expand batch across tau values
    z_tiled = teacher_z_chunk.unsqueeze(1) \
        .expand(B, num_taus, T, D) \
        .reshape(B * num_taus, T, D)
    
    # Replicate conditioning
    cond_tiled = {k: v.repeat_interleave(num_taus, dim=0) 
                  for k, v in teacher_cond_chunk.items()}
    
    # Expand tau across batch
    tau_flat = tau_seed.repeat(B)
    
    # Student forward: predict z_{t+1} = z + diff(z, tau)
    tiled_state = LatentState(z=z_tiled, t=t_value, cond=cond_tiled)
    drift = diff_model(tiled_state, tau_flat)
    student_z = z_tiled + drift
    
    # Loss: match base prediction (teacher is implicit)
    loss = MSE(student_z, z_tiled)
    return loss
```

**Loss Strategy**:
- For each batch, sample multiple tau values
- Expand batch across taus (vectorized)
- Compute student predictions
- MSE between student predictions and input (which has absorbed the teacher effect)

### Configuration Parameters

**Key Config Section** (config.stages.consistency_distill):
- `epochs`: Number of distillation epochs
- `batch_size`: Micro-batch size (default: 8, smaller than operator)
- `distill_num_taus`: Number of tau values to sample per batch (default: 3)
- `distill_micro_batch`: Micro-batch size for gradient accumulation
- `tau_schedule`: Optional per-epoch tau adjustment (list of num_taus values)
- `target_loss`: Optional early stopping if loss reaches threshold
- `compile`: Whether to torch.compile the distillation function

### Optimizations

1. **Teacher Caching** (OPTIMIZATION #1):
   ```python
   # Compute teacher predictions ONCE per batch (not per micro-batch)
   with torch.no_grad(), autocast(enabled=use_amp):
       teacher_full = operator(full_batch_state, dt_tensor)
   
   # Reuse teacher predictions across micro-batches
   for start in range(0, batch_size, micro):
       teacher_z_chunk = teacher_full.z[start:end]
       teacher_cond_chunk = {k: v[start:end] for k, v in teacher_full.cond.items()}
       # ... compute loss with this chunk
   ```
   **Expected speedup**: ~2x (reduces teacher calls by 50% when micro < batch_size)

2. **Async GPU Transfers**:
   ```python
   z0_device = z0.to(device, non_blocking=True)
   cond_device = {k: v.to(device, non_blocking=True) for k, v in cond.items()}
   ```

3. **AMP for Teacher**:
   ```python
   with torch.no_grad(), autocast(enabled=use_amp):
       teacher_full = operator(full_batch_state, dt_tensor)
   ```
   **Rationale**: Reduces teacher forward time ~20%, overall ~8% speedup

4. **Compiled Distillation Function**:
   ```python
   if should_compile:
       distill_fn = torch.compile(
           _distill_forward_and_loss,
           mode="default",  # Safe mode without CUDA graphs
           fullgraph=False,
       )
   ```
   **Expected speedup**: ~1.3-1.5x via kernel fusion

5. **Persistent Workers**:
   ```python
   training_cfg["persistent_workers"] = True if num_workers > 0 else False
   training_cfg["prefetch_factor"] = 4
   ```
   Avoids DataLoader worker respawning overhead

6. **Adaptive Tau Schedule**:
   ```python
   if tau_schedule:
       num_taus_epoch = int(tau_schedule[min(epoch, len(tau_schedule)-1)])
   ```
   Allows gradual increase of distillation difficulty

### Training Loop Structure

```python
for epoch in range(epochs):
    for batch in loader:
        # 1. Compute teacher ONCE per batch
        with torch.no_grad(), autocast(enabled=use_amp):
            teacher_full = operator(full_batch_state, dt_tensor)
        
        # 2. Process micro-batches
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        
        for micro_start in range(0, batch_size, micro_batch):
            micro_end = min(micro_start + micro_batch, batch_size)
            chunk_weight = (micro_end - micro_start) / batch_size
            
            # Slice pre-computed teacher
            teacher_z_chunk = teacher_full.z[micro_start:micro_end]
            teacher_cond_chunk = {k: v[micro_start:micro_end] 
                                 for k, v in teacher_full.cond.items()}
            
            # Sample tau values for this chunk
            tau_seed = _sample_tau(num_taus_epoch, device, cfg)
            
            # Compute loss (optionally compiled)
            with autocast(enabled=use_amp):
                loss_chunk = distill_fn(
                    teacher_z_chunk,
                    teacher_cond_chunk,
                    num_taus_epoch,
                    diff,
                    teacher_full.t,
                    tau_seed,
                    device,
                )
            
            # Scale loss by chunk weight (for proper averaging)
            scaler.scale(loss_chunk * chunk_weight).backward()
            batch_loss_value += loss_chunk.item() * chunk_weight
        
        # 3. Optimizer step
        scaler.unscale_(optimizer) if use_amp else None
        grad_norm = torch.nn.utils.clip_grad_norm_(diff.parameters(), clip_val)
        scaler.step(optimizer) if use_amp else optimizer.step()
        
        # 4. EMA update
        if ema_model:
            _update_ema(ema_model, diff, ema_decay)
```

### Early Stopping with Target Loss

```python
target_loss = cfg.get("stages", {}).get("consistency_distill", {}).get("target_loss", 0.0)
if target_loss and best_loss <= target_loss:
    print(f"Reached target loss {best_loss:.6f} <= {target_loss:.6f}, stopping early")
    break
```

### Checkpointing & Artifacts

**Saved Checkpoints**:
- `checkpoints/diffusion_residual.pt`: Updated (distilled) diffusion model
- `checkpoints/diffusion_residual_ema.pt`: EMA state (if enabled)

**Uploaded to W&B**: via `wandb_ctx.save_file()`

**Memory Management**:
```python
del operator  # Clean up teacher from memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

---

## Stage 4: Steady Prior Training (Optional)

### Implementation Location
**File**: `/Users/emerygunselman/Code/universal_simulator/scripts/train.py` (lines 1206-1309)  
**Function**: `train_steady_prior(cfg, wandb_ctx=None, global_step=0)`

### Architecture
Learns a prior distribution over steady-state latent representations:

**Model Definition** (`src/ups/models/steady_prior.py`):
```python
@dataclass
class SteadyPriorConfig:
    latent_dim: int
    hidden_dim: int = 128
    num_steps: int = 6      # Number of refinement steps
    cond_dim: int = 0

class SteadyPrior(nn.Module):
    def __init__(self, cfg):
        self.drift = nn.Sequential(
            nn.Linear(latent_dim + cond_dim + 1, hidden_dim),  # z + step_idx + cond
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    
    def forward(self, state: LatentState, cond=None) -> LatentState:
        current = state.z.clone()
        
        # Iterative refinement
        for step in range(self.num_steps):
            tau = step / max(self.num_steps - 1, 1)  # [0, 1] schedule
            tau_tensor = torch.full((B, T, 1), tau, device=current.device)
            
            drift = self.drift(torch.cat([current, tau_tensor], dim=-1))
            current = current + drift
        
        return LatentState(z=current, t=state.t, cond=state.cond)
```

**Key Design**:
- Runs for fixed `num_steps` (typically 4-6)
- Each step refines the latent via learnable drift
- Time schedule: tau goes from 0 to 1 across steps

### Loss Function

**Simple MSE**:
```python
def forward(self, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    state = LatentState(z=z0.to(device), cond=cond)
    refined = prior(state)
    loss = F.mse_loss(refined.z, z1.to(device))
    return loss
```

**Purpose**: Prior learns to refine initial latents toward better representations.

### Training Loop

Simplest of all stages - basic epoch-based training:

```python
for epoch in range(epochs):
    epoch_loss = 0.0
    total_grad_norm = 0.0
    
    for i, batch in enumerate(loader):
        z0, z1, cond = unpack_batch(batch)
        state = LatentState(z=z0.to(device), cond=cond)
        
        refined = prior(state)
        loss = F.mse_loss(refined.z, z1.to(device))
        
        (loss / accum_steps).backward()
        
        do_step = ((i + 1) % accum_steps == 0) or ((i + 1) == num_batches)
        if do_step:
            grad_norm = torch.nn.utils.clip_grad_norm_(prior.parameters(), float('inf'))
            total_grad_norm += grad_norm.item()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        epoch_loss += loss.item()
    
    mean_loss = epoch_loss / batches
    mean_grad_norm = total_grad_norm / grad_steps
    
    # Early stopping
    if mean_loss + 1e-6 < best_loss:
        best_loss = mean_loss
        epochs_since_improve = 0
    else:
        epochs_since_improve += 1
        if patience and epochs_since_improve > patience:
            break
```

### Configuration

**Enable/Disable**:
```yaml
stages:
  steady_prior:
    epochs: 0  # Set to 0 to skip entirely
```

### Checkpointing & Artifacts

**Saved Checkpoints**:
- `checkpoints/steady_prior.pt`: Best steady prior model

**Uploaded to W&B**: via `wandb_ctx.save_file()`

---

## Checkpoint Passing & State Management

### Checkpoint Loading Flow

```
Stage 1 (Operator)
    ↓ saves to checkpoints/operator.pt
    ↓
Stage 2 (Diffusion)
    ↓ loads checkpoints/operator.pt as frozen teacher
    ↓ trains diffusion, saves to checkpoints/diffusion_residual.pt
    ↓
Stage 3 (Consistency Distillation)
    ↓ loads checkpoints/operator.pt (teacher) and checkpoints/diffusion_residual.pt (student)
    ↓ trains diffusion, overwrites checkpoints/diffusion_residual.pt
    ↓
Stage 4 (Steady Prior)
    ↓ no checkpoint dependencies, trains independently
    ↓ saves to checkpoints/steady_prior.pt
```

### Compiled Prefix Stripping

When loading checkpoints from torch.compile models:

```python
def _strip_compiled_prefix(state_dict: dict) -> dict:
    """Strip _orig_mod. prefix from state dict keys (from torch.compile)."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "") if key.startswith("_orig_mod.") else key
        new_state_dict[new_key] = value
    return new_state_dict

# Usage
operator_state = torch.load(op_path, map_location="cpu")
operator_state = _strip_compiled_prefix(operator_state)
operator.load_state_dict(operator_state)
```

### Device Management

Aggressive device placement for multi-GPU safety:

```python
def _ensure_model_on_device(model: nn.Module, device: torch.device) -> None:
    """Aggressively ensure all model parameters and buffers are on the correct device."""
    model.to(device)
    for param in model.parameters():
        param.data = param.data.to(device)
        if param.grad is not None:
            param.grad.data = param.grad.data.to(device)
    for buffer in model.buffers():
        buffer.data = buffer.data.to(device)
```

---

## Global Step & Logging

### WandB Context Integration

Each stage receives:
- `wandb_ctx`: WandBContext instance for clean logging
- `global_step`: Cumulative step counter across stages

```python
def train_all_stages(cfg: dict, wandb_ctx=None) -> None:
    global_step = 0
    
    # Stage 1
    train_operator(cfg, wandb_ctx=wandb_ctx, global_step=global_step)
    global_step += op_epochs
    
    # Stage 2
    train_diffusion(cfg, wandb_ctx=wandb_ctx, global_step=global_step)
    global_step += diff_epochs
    
    # Stage 3
    train_consistency(cfg, wandb_ctx=wandb_ctx, global_step=global_step)
    global_step += distill_epochs
    
    # Stage 4
    train_steady_prior(cfg, wandb_ctx=wandb_ctx, global_step=global_step)
```

### TrainingLogger Class

```python
class TrainingLogger:
    def __init__(self, cfg, stage, global_step=0, wandb_ctx=None):
        self.stage = stage
        self.global_step = global_step
        self.wandb_ctx = wandb_ctx
        self.log_path = Path(cfg.get("training", {}).get("log_path", "reports/training_log.jsonl"))
    
    def log(self, *, epoch, loss, optimizer, patience_counter=None, grad_norm=None, ...):
        self.global_step += 1
        
        # Log to JSONL file
        entry = {
            "stage": self.stage,
            "epoch": epoch,
            "loss": loss,
            "global_step": self.global_step,
            "grad_norm": grad_norm,
            ...
        }
        self.log_path.write_text(json.dumps(entry) + "\n")
        
        # Log to WandB using clean context
        if self.wandb_ctx:
            self.wandb_ctx.log_training_metric(
                self.stage, "loss", loss, step=self.global_step
            )
            self.wandb_ctx.log_training_metric(
                self.stage, "grad_norm", grad_norm, step=self.global_step
            )
```

### Gradient Norm Tracking Details

**Per-Step Tracking**:
```python
total_grad_norm = 0.0
grad_steps = 0

if do_step:
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        float('inf') if clip_val is None else clip_val
    )
    total_grad_norm += float(grad_norm)
    grad_steps += 1

mean_grad_norm = total_grad_norm / max(grad_steps, 1)
```

**What Triggers**:
- Called after every gradient accumulation step
- Returns value before clipping (for diagnostics)
- Clipping applied if `grad_clip` is set

**Purpose**:
- Detect exploding/vanishing gradients
- Monitor training stability
- Verify optimizer effectiveness

---

## Integration with Fast-to-SOTA Orchestrator

### Entry Points

**Standalone**:
```bash
python scripts/train.py --config config.yaml --stage all
```

**Via Orchestrator**:
```bash
python scripts/run_fast_to_sota.py \
    --train-config config.yaml \
    --train-stage all
```

### WandB Context Passing

Training script saves context for evaluation subprocesses:

```python
# In train_all_stages()
context_file_path = os.environ.get("WANDB_CONTEXT_FILE")
if context_file_path and wandb_ctx and wandb_ctx.enabled:
    save_wandb_context(wandb_ctx, Path(context_file_path))
    print(f"✓ Saved WandB context to {context_file_path}")

# Evaluation subprocess loads context
# from ups.utils.wandb_context import load_wandb_context
# wandb_ctx = load_wandb_context(Path(context_file_path))
```

### Summary & Evaluation

After all training stages (if enabled):

```python
if run_eval:
    # Run baseline evaluation
    baseline_results = _run_evaluation(cfg, checkpoint_dir, eval_mode="baseline", wandb_ctx=wandb_ctx)
    
    # Run TTC evaluation (if enabled)
    if cfg.get("ttc", {}).get("enabled", False):
        ttc_results = _run_evaluation(cfg, checkpoint_dir, eval_mode="ttc", wandb_ctx=wandb_ctx)
        
        # Log comparison
        _log_evaluation_summary(wandb_ctx, baseline_results, ttc_results)
```

---

## Configuration Structure

### Complete Multi-Stage Config Template

```yaml
# Data
data:
  task: "burgers1d"
  root: "data/pdebench"
  split: "train"
  download: true

# Latent space
latent:
  dim: 32          # Must match operator.pdet.input_dim
  tokens: 48       # Spatial token count

# Operator architecture
operator:
  pdet:
    input_dim: 32          # Must match latent.dim
    hidden_dim: 64         # 2x latent_dim typical
    depths: [1, 1, 1]
    group_size: 16
    num_heads: 8

# Diffusion model
diffusion:
  hidden_dim: 64

# Training hyperparameters
training:
  dt: 0.1                  # Time step
  batch_size: 32
  num_workers: 4
  pin_memory: true
  amp: true                # Automatic mixed precision
  compile: true            # torch.compile
  compile_mode: "default"
  accum_steps: 1           # Gradient accumulation
  grad_clip: 1.0
  ema_decay: 0.999         # Exponential moving average
  
  # Loss weights
  lambda_spectral: 0.0
  lambda_relative: 0.0
  lambda_rollout: 0.0
  lambda_inv_enc: 0.0
  lambda_inv_dec: 0.0
  
  # Inverse loss curriculum
  inverse_loss_warmup_epochs: 15
  inverse_loss_max_weight: 0.05
  inverse_loss_frequency: 1  # Frequency of computing inverse losses
  
  # Diffusion tau sampling
  tau_distribution:
    type: "uniform"  # or "beta"
    alpha: 1.0       # Only for beta
    beta: 1.0

# Per-stage configuration
stages:
  operator:
    epochs: 25
    patience: 5
    optimizer:
      name: "adamw"
      lr: 0.001
      weight_decay: 1e-5
    scheduler:
      name: "cosineannealinglr"
      t_max: 25
      eta_min: 1e-6
  
  diff_residual:
    epochs: 15
    optimizer:
      name: "adamw"
      lr: 0.0005
      weight_decay: 1e-5
    scheduler:
      name: "cosineannealinglr"
      t_max: 15
      eta_min: 1e-7
    checkpoint_interval: 0  # Disabled by default
  
  consistency_distill:
    epochs: 5
    batch_size: 8
    distill_num_taus: 3
    distill_micro_batch: 8
    tau_schedule: [3, 3, 3, 3, 3]  # num_taus per epoch
    compile: true
    optimizer:
      name: "adamw"
      lr: 0.0005
  
  steady_prior:
    epochs: 0  # Set to 0 to skip

# Evaluation
evaluation:
  enabled: true

# Test-time conditioning
ttc:
  enabled: false
  steps: 1
  candidates: 4
  beam_width: 1
```

---

## Gradient Norm Analysis & Debugging

### What Gradient Norms Tell You

| Grad Norm Range | Interpretation |
|-----------------|-----------------|
| `< 0.01` | Vanishing gradients, poor optimization |
| `0.01 - 1.0` | Healthy (typical range) |
| `> 1.0` | Need clipping enabled or learning rate reduction |
| `> 10.0` | Exploding gradients, loss instability likely |

### Checking Logs

**JSONL Log File**:
```bash
# View grad_norm over time
jq '.grad_norm' reports/training_log.jsonl | head -20

# Detect NaN steps
jq 'select(.grad_norm > 1000)' reports/training_log.jsonl
```

**WandB Charts**:
- Plot `{stage}/grad_norm` over time
- Compare across stages
- Identify convergence issues

### Common Issues & Solutions

**Issue**: Grad norm suddenly spikes
- **Cause**: Learning rate too high, loss becomes unstable
- **Fix**: Enable `grad_clip`, reduce learning rate, or use scheduler

**Issue**: Grad norm always < 0.001
- **Cause**: Model not training effectively
- **Fix**: Increase learning rate, check data loading, verify loss is computed

**Issue**: Inverse losses cause grad norm explosion
- **Cause**: Encoder/decoder not properly scaled
- **Fix**: Use curriculum learning (enabled by default), increase warmup_epochs

---

## References & File Locations

### Source Code Structure

```
src/ups/
├── core/
│   ├── latent_state.py       # LatentState container
│   └── blocks_pdet.py         # PDETransformer blocks
├── models/
│   ├── latent_operator.py     # Stage 1 model
│   ├── diffusion_residual.py  # Stage 2 model
│   └── steady_prior.py        # Stage 4 model
├── training/
│   ├── loop_train.py          # CurriculumConfig, LatentTrainer (legacy)
│   └── losses.py              # LossBundle, loss functions
└── data/
    └── latent_pairs.py        # build_latent_pair_loader()

scripts/
├── train.py                   # Main training entrypoint
└── run_fast_to_sota.py        # Orchestration pipeline
```

### Key Metrics & Checkpoints

| Artifact | Location | Stage | Purpose |
|----------|----------|-------|---------|
| operator.pt | checkpoints/ | 1 | Deterministic evolution model |
| operator_ema.pt | checkpoints/ | 1 | EMA version of operator |
| diffusion_residual.pt | checkpoints/ | 2,3 | Residual correction model |
| diffusion_residual_ema.pt | checkpoints/ | 2,3 | EMA version |
| steady_prior.pt | checkpoints/ | 4 | Steady-state prior |
| training_log.jsonl | reports/ | All | Per-epoch metrics (JSONL format) |

