# Universal Physics Stack Training Infrastructure Analysis

## Overview

The UPS training system is a multi-stage, single-device training pipeline with orchestration through `run_fast_to_sota.py`. The infrastructure supports operator training, diffusion residual training, consistency distillation, and steady-state priors as sequential stages, all using a centralized WandB context for clean logging.

---

## 1. Training Orchestration Architecture

### Entry Point: `scripts/run_fast_to_sota.py`

**File**: `/Users/emerygunselman/Code/universal_simulator/scripts/run_fast_to_sota.py`

**Key Orchestration Functions** (Lines 350-1240):
- **`main()`** (Line 351): Orchestrates the complete fast-to-SOTA pipeline
  - Parses CLI arguments for config paths, eval configs, gates, leaderboard settings
  - Creates run directory structure: `artifacts/runs/{run_id}/` with subdirs: `configs/`, `analysis/`, `artifacts/`
  - Manages metadata: stores config hashes, architecture fingerprints, and training status in `checkpoints/metadata.json` (Lines 92-120)
  - Implements sequential subprocess calls: validation → training → small eval → full eval → promotion
  - **Key subprocess environment variables**:
    - `WANDB_MODE`: Controls offline/online/disabled logging (Line 704)
    - `FAST_TO_SOTA_WANDB_INFO`: Path where training subprocess saves WandB run info (Line 705)
    - `WANDB_CONTEXT_FILE`: Shared context file for WandB logging across subprocesses (Line 706)

**Training Invocation** (Lines 693-712):
```python
train_cmd = [PYTHON, "scripts/train.py", "--config", str(resolved_train_config), "--stage", args.train_stage]
train_env = {
    "WANDB_MODE": args.wandb_mode,
    "FAST_TO_SOTA_WANDB_INFO": str(wandb_info_path),
    "WANDB_CONTEXT_FILE": str(wandb_context_file),
}
_run_command(train_cmd, env=train_env, desc="train")
```

**Metadata Management** (Lines 575-586):
- Tracks `trained`, `trained_at`, `last_small_eval`, `last_full_eval` timestamps
- Stores architecture fingerprint via `_extract_arch_fingerprint()` (Lines 74-89)
- Allows skipping training if `metadata.trained=True` (unless `--force-train` is set)

**Gating System** (Lines 249-301):
- `_check_gates()`: Evaluates improvement metrics (delta-based) and ratio limits against baseline
- Default ratio gates: `conservation_gap`, `bc_violation`, `ece`, `wall_clock` (all capped at 1.0x baseline, Line 540-543)
- Compares candidate metrics to baseline via leaderboard CSV (Lines 878-883, 1015-1020)

---

## 2. Training Script Architecture

### Entry Point: `scripts/train.py`

**File**: `/Users/emerygunselman/Code/universal_simulator/scripts/train.py` (26,539 lines)

**Main Stages**:
1. **Operator Training** → `train_operator()` (Line 480)
2. **Diffusion Residual** → `train_diffusion()` (Line 896)
3. **Consistency Distillation** → `train_consistency_distill()` 
4. **Steady-State Prior** → `train_steady_prior()`

### Device Management

**Single-Device Architecture** (Lines 491-493, 901-902):
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
operator.to(device)
```

- **No distributed training**: Uses single CUDA device if available, falls back to CPU
- **Explicit device transfers**: Every tensor moved to device during forward pass (Lines 640-648)
- **OOM handling**: Catches and skips batches on CUDA OOM (Lines 779-783):
  ```python
  except RuntimeError as e:
      if "out of memory" in str(e).lower():
          if torch.cuda.is_available():
              torch.cuda.empty_cache()
          print("Warning: OOM encountered in operator step, skipping batch")
          continue
  ```

### Environment Initialization (Lines 32-47)

**PyTorch Configuration**:
```python
# torch.compile safe defaults
import torch._dynamo as _dynamo
_dynamo.config.suppress_errors = True
_dynamo.config.error_on_recompile = False

# Prevent Inductor subprocess crashes
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

# Safe multiprocessing start method for CUDA + DataLoader workers
mp.set_start_method("spawn", force=True)  # Line 45
```

---

## 3. Training Loop Implementation

### Core Training Loop: `train_operator()` (Lines 480-894)

**Loop Structure** (Lines 599-868):
```python
for epoch in range(epochs):
    epoch_start = time.time()
    epoch_loss = 0.0
    total_grad_norm = 0.0
    batches = 0
    grad_steps = 0
    
    optimizer.zero_grad(set_to_none=True)  # Line 610
    for i, batch in enumerate(loader):
        unpacked = unpack_batch(batch)
        # Forward pass with autocast
        with autocast(enabled=use_amp):
            next_state = operator(state, dt_tensor)
            loss = compute_operator_loss_bundle(...)
        
        # Backward with gradient accumulation
        loss_value = loss.detach().item()
        if use_amp:
            scaler.scale(loss / accum_steps).backward()
        else:
            (loss / accum_steps).backward()
        
        # Accumulated gradient step
        do_step = ((i + 1) % accum_steps == 0) or ((i + 1) == num_batches)
        if do_step:
            if use_amp:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(...)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(...)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            grad_steps += 1
            if ema_model is not None:
                _update_ema(ema_model, operator, ema_decay)
```

**Gradient Accumulation** (Lines 568, 810-824):
- Configured via `training.accum_steps` (default: 1)
- Step only after accumulating `accum_steps` batches (Line 810)
- Enables effective batch size increase without OOM

**Mixed Precision (AMP)** (Lines 559-560, 806-818):
- Enabled via `training.amp: true` and CUDA availability (Line 379)
- Uses `GradScaler` for loss scaling (Line 560)
- Spectral loss explicitly disables autocast for FFT stability (Lines 74-80)

**EMA (Exponential Moving Average)** (Lines 561-563, 825-826):
- Initialized if `training.ema_decay > 0` (Line 561)
- Applied after each step (Line 826):
  ```python
  @torch.no_grad()
  def _update_ema(ema_model, model, decay):
      for p_ema, p in zip(ema_model.parameters(), model.parameters()):
          p_ema.mul_(decay).add_(p.data, alpha=1.0 - decay)
  ```

**Early Stopping** (Lines 853-862):
- Tracks `best_loss` and `epochs_since_improve`
- Patience via `training.patience` or stage-specific `stages.{stage}.patience`
- Stops if `epochs_since_improve > patience`

**Checkpoint Saving** (Lines 868-893):
- Saves best state to `checkpoints/operator.pt` (Line 871)
- Saves EMA checkpoint to `checkpoints/operator_ema.pt` if EMA enabled (Line 875)
- Uploads to WandB via `wandb_ctx.save_file()` if WandB enabled (Line 882)
- Sends alert via `wandb_ctx.alert()` (Line 889)

---

## 4. Data Loading Pipeline

### DataLoader Factory: `build_latent_pair_loader()`

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/data/latent_pairs.py` (Lines 711-892)

**DataLoader Configuration** (Lines 732-741):
```python
loader_kwargs = {
    "batch_size": batch,
    "shuffle": True,
    "collate_fn": latent_pair_collate,
    "num_workers": num_workers,              # Line 721: max(1, cpu_count // 4)
    "pin_memory": pin_memory,                # Line 722: True if CUDA available
    "persistent_workers": num_workers > 0,   # Line 738
}
if prefetch_factor is not None and num_workers > 0:
    loader_kwargs["prefetch_factor"] = prefetch_factor  # Line 741
```

**Worker Configuration** (Lines 720-723):
- `num_workers`: Auto-calculated as `max(1, os.cpu_count() // 4)` (Line 720)
  - Can be overridden via `training.num_workers`
- `pin_memory`: True if CUDA available (Line 722)
  - Can be overridden via `training.pin_memory`
- `prefetch_factor`: Optional, from `training.prefetch_factor`
- `persistent_workers`: True when `num_workers > 0` to avoid worker restart overhead

**Multi-Task Support** (Lines 743-843):
- Accepts single task or list of tasks via `data.task`
- For each task:
  - Grid tasks: `GridLatentPairDataset` with on-demand or cached encoding
  - Mesh/Particle tasks: `GraphLatentPairDataset` with on-demand or cached encoding
- Combines via `ConcatDataset` for mixed-task training (Line 841)
- **Preloaded cache mode**: Uses `PreloadedCacheDataset` when cache is complete and RAM is sufficient (Lines 769-792)

**Encoder Initialization** (Lines 756-831):
- **Grid encoders**: `GridEncoder` placed on device during loader construction (Line 765)
- **Graph encoders**: `MeshParticleEncoder` placed on device during loader construction (Line 831)
- Encoders set to `.eval()` mode (frozen during operator training)

**Cache System** (Lines 724-728):
- Root: `training.latent_cache_dir` (optional)
- Per-task cache: `{cache_root}/{task_name}_{split}/sample_{idx:05d}.pt`
- Dtype: `training.latent_cache_dtype` (default: "float16")
- Latent pairs cached in compressed form for faster subsequent epochs

**Custom Collate** (Lines 895-988):
- `latent_pair_collate()`: Handles variable-length trajectories and optional inverse loss fields
- Flattens trajectory pairs: `(num_pairs, tokens, dim)` → `(batch*num_pairs, tokens, dim)` (Line 908)
- Handles mixed-task batches with dimension mismatches (Lines 939-943)
- Returns dict: `{"z0", "z1", "cond", "future", "input_fields", "coords", "meta", "task_names"}`

**Batch Unpacking** (Lines 991-1003):
- `unpack_batch()`: Detects dict vs. tuple format
- Backward-compatible with legacy tuple format: `(z0, z1, cond)` or `(z0, z1, cond, future)`

---

## 5. Optimizer and Scheduler Configuration

### Optimizer Factory: `_create_optimizer()` (Lines 258-338)

**Supported Optimizers**:
1. **Adam** (Line 267): `torch.optim.Adam(lr, weight_decay)`
2. **AdamW** (Line 269): `torch.optim.AdamW(lr, weight_decay)`
3. **SGD** (Line 271): `torch.optim.SGD(lr, momentum, weight_decay)`
4. **Muon Hybrid** (Lines 276-336): Custom hybrid Muon+AdamW optimizer
   - Splits parameters: 2D+ parameters → Muon, 1D parameters → AdamW
   - Uses `ups.training.param_groups.build_param_groups()` to split (Line 291)
   - Muon config: `muon_momentum`, `muon_ns_steps`, `muon_backend` (Lines 295-297)
   - Falls back to AdamW if no Muon backends available (Lines 284-286)
   - Returns `HybridOptimizer` wrapper if both optimizers exist (Line 336)

**Configuration** (Lines 260-265):
- Stage-specific overrides via `stages.{stage}.optimizer`
- Global fallback via `optimizer` config section
- Learning rate: `optimizer.lr` (default: 1e-3)
- Weight decay: `optimizer.weight_decay` (default: 0.0)

### Scheduler Factory: `_create_scheduler()` (Lines 341-375)

**Supported Schedulers**:
1. **StepLR** (Line 347): Decays by `gamma` every `step_size` epochs
2. **CosineAnnealingLR** (Line 351): Cosine annealing with optional `eta_min`
3. **ReduceLROnPlateau** (Line 355): Reduces on validation plateau with `factor`, `patience`, `threshold`

**Configuration**:
- Stage-specific via `stages.{stage}.scheduler`
- Global fallback via `optimizer.scheduler`

**Step Timing** (Lines 863-867):
```python
if scheduler is not None:
    if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
        scheduler.step(mean_loss)  # Metric-based
    else:
        scheduler.step()  # Epoch-based
```

---

## 6. Model Initialization and Compilation

### Model Factory: `make_operator()` (Lines 226-255)

**Operator Creation**:
```python
latent_dim = cfg.get("latent", {}).get("dim", 32)
operator_cfg = cfg.get("operator", {})
pdet_cfg = operator_cfg.get("pdet", {})
architecture_type = operator_cfg.get("architecture_type", "pdet_unet")

# Create appropriate config
if architecture_type == "pdet_stack":
    from ups.models.pure_transformer import PureTransformerConfig
    pdet_config = PureTransformerConfig(**pdet_cfg)
else:  # pdet_unet (default)
    pdet_config = PDETransformerConfig(**pdet_cfg)

config = LatentOperatorConfig(
    latent_dim=latent_dim,
    pdet=pdet_config,
    architecture_type=architecture_type,
    time_embed_dim=latent_dim,
)
return LatentOperator(config)
```

**Optional Compilation** (Lines 382-413):
```python
def _maybe_compile(model, cfg, name):
    compile_enabled = bool(cfg.get("training", {}).get("compile", False))
    if not compile_enabled:
        return model
    
    # Skip teacher models to avoid CUDA graph issues (Line 395)
    if "teacher" in name:
        return model
    
    # Use "default" mode by default; allow override via training.compile_mode
    training_cfg = cfg.get("training", {})
    user_mode = str(training_cfg.get("compile_mode", "")).lower()
    compile_mode = user_mode if user_mode in {"default", "reduce-overhead", "max-autotune"} else "default"
    
    try:
        compiled = torch.compile(model, mode=compile_mode, fullgraph=False)
        return compiled
    except Exception:
        return model  # Silent fallback on error
```

**torch.compile Configuration**:
- Enabled via `training.compile: true` (default: false)
- Mode via `training.compile_mode` (default: "default")
- Suppresses compilation errors: `_dynamo.config.suppress_errors = True` (Line 35)
- Single compile thread: `TORCHINDUCTOR_COMPILE_THREADS=1` (Line 41)

---

## 7. Logging and Monitoring

### WandB Integration: `src/ups/utils/wandb_context.py`

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/utils/wandb_context.py` (Lines 1-277)

**WandBContext Dataclass** (Lines 27-43):
```python
@dataclass
class WandBContext:
    run: Any  # wandb.Run object
    run_id: str
    enabled: bool = True
```

**Key Methods**:
1. **`log_training_metric(stage, metric, value, step)`** (Lines 45-80):
   - Logs time series: `training/{stage}/{metric} = value` at `step`
   - Also logs step metric for chart definition (Line 78)
   - Used in operator training: `logger.log()` (Lines 834-842)

2. **`log_eval_summary(metrics, prefix="eval")`** (Lines 82-109):
   - Logs scalars to run summary (not time series)
   - Used for final eval results

3. **`log_table(name, columns, data)`** (Lines 111-141):
   - Creates structured tables for comparisons

4. **`save_file(file_path)`** (Lines 161-176):
   - Uploads checkpoints and reports to WandB artifacts

5. **`update_config(updates)`** (Lines 178-200):
   - Updates run metadata/config

6. **`alert(title, text, level)`** (Lines 239-250):
   - Sends dashboard notifications

### Training Logger: `TrainingLogger` (Lines 135-214)

**File**: `/Users/emerygunselman/Code/universal_simulator/scripts/train.py` (Lines 135-214)

**Logging Architecture**:
```python
class TrainingLogger:
    def __init__(self, cfg, stage, global_step=0, wandb_ctx=None):
        self.stage = stage
        self.global_step = global_step
        self.wandb_ctx = wandb_ctx
        self.log_path = Path(cfg.get("training", {}).get("log_path", "reports/training_log.jsonl"))
    
    def log(self, *, epoch, loss, optimizer, patience_counter=None, grad_norm=None, 
            epoch_time=None, best_loss=None):
        # Log to JSONL file
        if self.log_path:
            entry = {
                "stage": self.stage,
                "loss": loss,
                "epoch": epoch,
                "lr": lr,
                "global_step": self.global_step,
                ...
            }
            self.log_path.open("a").write(json.dumps(entry) + "\n")
        
        # Log to WandB
        if self.wandb_ctx:
            self.wandb_ctx.log_training_metric(self.stage, "loss", loss, step=self.global_step)
            self.wandb_ctx.log_training_metric(self.stage, "lr", lr, step=self.global_step)
            ...
```

**Dual-Output Logging**:
- **JSONL file**: `reports/training_log.jsonl` (per-epoch metrics)
- **WandB**: Time series in `training/{stage}/{metric}` namespace

---

## 8. Seeding and Determinism

### Seed Configuration: `set_seed()` (Lines 97-125)

```python
def set_seed(cfg):
    seed = cfg.get("seed", 17)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    deterministic = cfg.get("deterministic", False)
    benchmark = cfg.get("benchmark", True)
    
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = benchmark
```

**Configuration**:
- `seed`: Default 17 (Line 103)
- `deterministic`: Enable deterministic algorithms (default: False)
- `benchmark`: Enable cuDNN benchmarking (default: True, ignored if deterministic=True)

---

## 9. Loss Computation and Auxiliary Functions

### Loss Functions (Lines 61-80)

**NRMSE Loss** (Lines 61-64):
```python
def _nrmse(pred, target, eps=1e-8):
    mse = torch.mean((pred - target) ** 2)
    denom = torch.mean(target ** 2) + eps
    return torch.sqrt(mse / denom)
```

**Spectral Energy Loss** (Lines 66-80):
```python
def _spectral_energy_loss(pred, target, dim=1, eps=1e-8):
    with torch.cuda.amp.autocast(enabled=False):  # Force float32 for FFT
        pred_fft = torch.fft.rfft(pred.float(), dim=dim)
        tgt_fft = torch.fft.rfft(target.float(), dim=dim)
        pred_energy = torch.mean(pred_fft.abs() ** 2)
        tgt_energy = torch.mean(tgt_fft.abs() ** 2)
        loss = torch.abs(pred_energy - tgt_energy) / (tgt_energy + eps)
    return loss.to(pred.dtype)
```

### Loss Bundle Computation (Lines 656-774)

**Loss Weights Configuration**:
```python
loss_weights = {
    "lambda_forward": 1.0,
    "lambda_inv_enc": float(train_cfg.get("lambda_inv_enc", 0.0)),
    "lambda_inv_dec": float(train_cfg.get("lambda_inv_dec", 0.0)),
    "lambda_spectral": lam_spec,
    "lambda_rollout": lam_rollout,
    "inverse_loss_warmup_epochs": int(train_cfg.get("inverse_loss_warmup_epochs", 15)),
    "inverse_loss_max_weight": float(train_cfg.get("inverse_loss_max_weight", 0.05)),
}
```

**Physics Priors** (Lines 580-594):
```python
physics_weights = {
    "lambda_divergence": physics_cfg.get("lambda_divergence", 0.0),
    "lambda_conservation": physics_cfg.get("lambda_conservation", 0.0),
    "lambda_boundary": physics_cfg.get("lambda_boundary", 0.0),
    "lambda_positivity": physics_cfg.get("lambda_positivity", 0.0),
    "bc_value": physics_cfg.get("bc_value", 0.0),
    "bc_type": physics_cfg.get("bc_type", "all"),
    "lambda_latent_norm": physics_cfg.get("lambda_latent_norm", 0.0),
    "lambda_latent_diversity": physics_cfg.get("lambda_latent_diversity", 0.0),
}
```

**Loss Bundle Selection** (Lines 705-774):
- If `training.physics_priors.enabled`: Uses `compute_operator_loss_bundle_with_physics()`
- Otherwise: Uses standard `compute_operator_loss_bundle()`

---

## 10. Batch Processing and Conditioning

### Batch Unpacking (Lines 611-648)

```python
unpacked = unpack_batch(batch)

if isinstance(unpacked, dict):
    z0 = unpacked["z0"]
    z1 = unpacked["z1"]
    cond = unpacked.get("cond", {})
    future = unpacked.get("future")
    input_fields_physical = unpacked.get("input_fields")
    coords = unpacked.get("coords")
    meta = unpacked.get("meta")
    task_names = unpacked.get("task_names")
elif len(unpacked) == 4:
    z0, z1, cond, future = unpacked
    # ... other fields = None
else:
    z0, z1, cond = unpacked
    # ... other fields = None
```

### Conditioning Preparation (Lines 640-648)

```python
cond_device = {k: v.to(device) for k, v in cond.items()}
state = LatentState(z=z0.to(device), t=torch.tensor(0.0, device=device), cond=cond_device)
target = z1.to(device)

# Move inverse loss fields to device if present
if input_fields_physical is not None:
    input_fields_physical = {k: v.to(device) for k, v in input_fields_physical.items()}
if coords is not None:
    coords = coords.to(device)
```

### Per-Task Metrics (Lines 788-851)

**Optional Tracking** (Lines 789-800):
```python
if task_names is not None and train_cfg.get("log_per_task_metrics", False):
    task_counts = defaultdict(int)
    for task_name in task_names:
        if task_name:
            task_counts[task_name] += 1
    
    batch_size = len(task_names)
    for task_name, count in task_counts.items():
        task_metrics[task_name]["loss"].append(loss_value)
        task_metrics[task_name]["count"] += count
```

**Epoch-End Logging** (Lines 844-851):
```python
if wandb_ctx and train_cfg.get("log_per_task_metrics", False):
    for task_name, metrics in task_metrics.items():
        if metrics["count"] > 0:
            avg_loss = sum(metrics["loss"]) / len(metrics["loss"])
            wandb_ctx.log_training_metric("operator", f"{task_name}/loss", avg_loss, step=epoch)
            wandb_ctx.log_training_metric("operator", f"{task_name}/sample_count", metrics["count"], step=epoch)
```

---

## 11. Configuration Management

### Config Loading: `load_config()` (Lines 92-94)

```python
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
```

**Advanced Config Loading** (via `src/ups/utils/config_loader.py`, Line 44):
- Supports recursive includes via `!include` directives
- Called by orchestrator: `load_config_with_includes(train_config)` (Line 517)

### Stage-Specific Configuration

**Stages Dictionary** (Lines 485-486, 342):
```python
stages:
  operator:
    epochs: 25
    optimizer:
      name: adam
      lr: 1e-3
    scheduler:
      name: cosineannealinglr
      t_max: 25
    patience: 5
  diffusion_residual:
    epochs: 10
  consistency_distill:
    epochs: 5
  steady_prior:
    epochs: 0  # Disabled by default
```

**Stage Config Resolution** (Lines 260-261, 342-343, 415-427):
```python
def _get_patience(cfg, stage):
    stage_cfg = cfg.get("stages", {}).get(stage, {})
    if "patience" in stage_cfg:
        return stage_cfg["patience"]
    training_cfg = cfg.get("training", {})
    return training_cfg.get("patience")
```

---

## 12. Current Limitations and Design Notes

### No Distributed Training Support

1. **Single-device only**: All models and data on one device (Line 491-493)
2. **No multi-GPU support**: No `DataParallel`, `DistributedDataParallel`, or sharding
3. **No model parallelism**: No gradient accumulation across devices
4. **Batch size limitations**: Limited by single GPU memory

### Data Loading Characteristics

1. **Synchronous encoding**: Latent pairs encoded during `__getitem__()` or cached offline
2. **No prefetching optimization for GPUs**: Workers compute on CPU, transfer to GPU per batch
3. **Potential bottleneck**: Encoder initialization happens at loader construction time, not per-worker

### Checkpoint and State Management

1. **No resumable training**: Checkpoints save final model, not training state (optimizer, scheduler, epoch number)
2. **No distributed checkpoint format**: Just `torch.save(state_dict)`
3. **No gradient state persistence**: EMA and optimizer states reset between runs

### Loss Computation

1. **No gradient checkpointing**: Full computation graph retained during backprop
2. **Spectral loss forces float32**: Disables autocast for FFT stability (Line 74)
3. **No reduction option**: All losses computed on full batch before backprop

---

## 13. Key Configuration Parameters for Distributed Training Preparation

If adapting for distributed training, these parameters would need updates:

**DataLoader Parameters** (Lines 732-741):
- `num_workers`: Currently auto-calculated for single machine
- `batch_size`: Single device batch size
- `pin_memory`: Device-specific

**Device Selection** (Lines 491-493):
- Hardcoded to single device
- Would need rank/world_size for distributed setup

**Model Initialization** (Lines 382-413):
- torch.compile disabled on teacher models
- Would need backend selection for distributed compile

**Checkpointing** (Lines 868-878):
- Saves only model state, not optimizer/scheduler
- No distributed checkpoint coordination

---

## 14. Summary Table: Training Loop Phases

| Phase | Location | Key Code | Responsibility |
|-------|----------|----------|-----------------|
| Config Load | `train.py` L92 | `load_config()` | Parse YAML config |
| Seed Set | `train.py` L97 | `set_seed()` | Reproducibility |
| DataLoader Build | `latent_pairs.py` L711 | `build_latent_pair_loader()` | Create iterator with encoders |
| Model Create | `train.py` L226 | `make_operator()` | Instantiate LatentOperator |
| Optimizer Create | `train.py` L258 | `_create_optimizer()` | Setup Adam/AdamW/SGD/Muon |
| Scheduler Create | `train.py` L341 | `_create_scheduler()` | Setup LR schedule |
| Model to Device | `train.py` L493 | `operator.to(device)` | Single device placement |
| Optional Compile | `train.py` L382 | `_maybe_compile()` | Enable torch.compile |
| EMA Init | `train.py` L562 | `_init_ema()` | Create EMA copy |
| Epoch Loop | `train.py` L599 | `for epoch in range(epochs)` | Main training loop |
| Batch Iterate | `train.py` L611 | `for batch in loader` | Pull batches from DataLoader |
| Unpack Batch | `train.py` L612 | `unpack_batch()` | Convert to z0, z1, cond, future |
| Device Transfer | `train.py` L640 | `.to(device)` | Move tensors to device |
| Forward Pass | `train.py` L652 | `operator(state, dt_tensor)` | Compute predictions |
| Loss Compute | `train.py` L696 | `compute_operator_loss_bundle()` | Calculate loss |
| Backward | `train.py` L806 | `.backward()` | Backpropagation |
| Grad Clip | `train.py` L815 | `clip_grad_norm_()` | Gradient clipping |
| Optimizer Step | `train.py` L817 | `optimizer.step()` | Update parameters |
| EMA Update | `train.py` L826 | `_update_ema()` | Update EMA copy |
| Epoch Log | `train.py` L834 | `logger.log()` | Log metrics to file and WandB |
| Scheduler Step | `train.py` L863 | `scheduler.step()` | Update learning rate |
| Checkpoint Save | `train.py` L871 | `torch.save(operator_path)` | Save best model |

