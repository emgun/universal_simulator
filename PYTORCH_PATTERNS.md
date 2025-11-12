# PyTorch Patterns in Universal Simulator Codebase

## Overview
This document catalogs existing PyTorch patterns found in the codebase as of the current state. These are documented examples of how the project currently implements distributed training, device management, optimization, checkpointing, and mixed precision training.

---

## 1. Distributed & Multi-Device Patterns

### Status: NOT USED
**torch.nn.DataParallel** and **torch.distributed** are not used in this codebase.

The project uses:
- Single-GPU training (determined at runtime via `torch.cuda.is_available()`)
- Dataloader worker processes with pinned memory for async transfers

**Evidence:**
- `/Users/emerygunselman/Code/universal_simulator/scripts/train.py:25` - Sets multiprocessing method to "spawn"
- No DataParallel or DistributedDataParallel wrappers in model creation

---

## 2. Device Placement Patterns

### Pattern: Runtime Device Detection

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Line 491 (train_operator):**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
operator.to(device)
```

**Line 901 (train_diffusion):**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

**Line 1161 (train_consistency_distill):**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Line 1423 (train_steady_prior):**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prior.to(device)
```

### Pattern: Aggressive Device Placement with Gradient Handling

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Lines 1083-1093 (_ensure_model_on_device function):**
```python
def _ensure_model_on_device(model: nn.Module, device: torch.device) -> None:
    """Aggressively ensure all model parameters and buffers are on the correct device."""
    model.to(device)
    # Force all parameters to device
    for param in model.parameters():
        param.data = param.data.to(device)
        if param.grad is not None:
            param.grad.data = param.grad.data.to(device)
    # Force all buffers to device
    for buffer in model.buffers():
        buffer.data = buffer.data.to(device)
```

### Pattern: Batch-Level Device Transfer

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Line 640-642 (train_operator):**
```python
cond_device = {k: v.to(device) for k, v in cond.items()}
state = LatentState(z=z0.to(device), t=torch.tensor(0.0, device=device), cond=cond_device)
target = z1.to(device)
```

### Pattern: Non-Blocking Device Transfers

**File:** `/Users/emerygunselman/Code/universal_simulator/src/ups/data/latent_pairs.py`

**Lines 195-206 (encoder device handling):**
```python
flattened = flattened.to(encoder_device, non_blocking=True)
coord_batch = coord_batch.to(encoder_device, non_blocking=True)
params_device = {k: v.to(encoder_device, non_blocking=True) for k, v in params.items()}
bc_device = {k: v.to(encoder_device, non_blocking=True) for k, v in bc.items()}
```

**Lines 1290-1292 (train_consistency_distill):**
```python
z0_device = z0.to(device, non_blocking=True)
cond_device = {k: v.to(device, non_blocking=True) for k, v in cond.items()}
full_batch_state = LatentState(z=z0_device, t=torch.tensor(0.0, device=device), cond=cond_device)
```

### Pattern: CUDA Cache Clearing for OOM Recovery

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Lines 780-783 (train_operator OOM handling):**
```python
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Warning: OOM encountered in operator step, skipping batch")
        continue
```

**Lines 972-976 (train_diffusion OOM handling):**
```python
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Warning: OOM encountered in operator forward (teacher), skipping batch")
        continue
```

### Pattern: Tensor Device Creation

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Line 555 (train_operator):**
```python
dt_tensor = torch.tensor(dt, device=device)
```

**Line 462 (_sample_tau):**
```python
return torch.rand(batch_size, device=device)
```

---

## 3. Optimizer & Scheduler Setup

### Pattern: Standard PyTorch Optimizers

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Lines 258-338 (_create_optimizer function):**

**Adam:**
```python
if name == "adam":
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
```

**AdamW:**
```python
if name == "adamw":
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
```

**SGD:**
```python
if name == "sgd":
    momentum = opt_cfg.get("momentum", 0.9)
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
```

### Pattern: Hybrid Muon+AdamW Optimizer

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Lines 276-336 (muon_hybrid/muon optimizer creation):**
```python
if name == "muon_hybrid" or name == "muon":
    from ups.training.param_groups import build_param_groups, print_param_split_summary
    from ups.training.hybrid_optimizer import HybridOptimizer
    from ups.training.muon_factory import create_muon_optimizer, get_available_backends

    backends = get_available_backends()
    if not backends:
        print("WARNING: No Muon implementation available, falling back to AdamW")
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    muon_params, adamw_params = build_param_groups(model)
    print_param_split_summary(model)

    muon_momentum = opt_cfg.get("muon_momentum", 0.95)
    muon_ns_steps = opt_cfg.get("muon_ns_steps", 5)
    muon_backend = opt_cfg.get("muon_backend", "auto")

    adamw_betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    adamw_eps = opt_cfg.get("eps", 1e-8)

    optimizers = []

    if len(muon_params) > 0:
        muon_opt, backend_name = create_muon_optimizer(
            muon_params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=muon_momentum,
            nesterov=True,
            ns_steps=muon_ns_steps,
            backend=muon_backend,
        )
        optimizers.append(muon_opt)
        print(f"  Muon ({backend_name}): {len(muon_params)} parameter groups")

    if len(adamw_params) > 0:
        adamw_opt = torch.optim.AdamW(
            adamw_params,
            lr=lr,
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=weight_decay,
        )
        optimizers.append(adamw_opt)
        print(f"  AdamW: {len(adamw_params)} parameter groups")

    if len(optimizers) == 1:
        return optimizers[0]

    return HybridOptimizer(optimizers)
```

### Pattern: Parameter Group Splitting

**File:** `/Users/emerygunselman/Code/universal_simulator/src/ups/training/param_groups.py`

**Lines 23-59 (build_param_groups function):**
```python
def build_param_groups(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split model parameters into Muon-compatible and AdamW groups."""
    muon_params = []
    adamw_params = []

    for _name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if is_muon_compatible(p):  # Check if ndim >= 2
            muon_params.append(p)
        else:
            adamw_params.append(p)

    return muon_params, adamw_params
```

**Muon Compatibility Check (lines 7-20):**
```python
def is_muon_compatible(p: nn.Parameter) -> bool:
    """Check if parameter is Muon-compatible (2D or higher)."""
    return p.requires_grad and p.ndim >= 2
```

### Pattern: Learning Rate Schedulers

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Lines 341-375 (_create_scheduler function):**

**StepLR:**
```python
if name == "steplr":
    step_size = sched_cfg.get("step_size", 1)
    gamma = sched_cfg.get("gamma", 0.5)
    return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
```

**CosineAnnealingLR:**
```python
if name == "cosineannealinglr":
    t_max = sched_cfg.get("t_max", 10)
    eta_min = sched_cfg.get("eta_min", 0.0)
    return lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
```

**ReduceLROnPlateau:**
```python
if name in {"reducelronplateau", "reducelr", "plateau"}:
    mode = sched_cfg.get("mode", "min")
    factor = sched_cfg.get("factor", 0.5)
    patience = sched_cfg.get("patience", 3)
    threshold = sched_cfg.get("threshold", 1e-3)
    threshold_mode = sched_cfg.get("threshold_mode", "rel")
    cooldown = sched_cfg.get("cooldown", 0)
    min_lr = sched_cfg.get("min_lr", 0.0)
    eps = sched_cfg.get("eps", 1e-8)
    return lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        threshold=threshold,
        threshold_mode=threshold_mode,
        cooldown=cooldown,
        min_lr=min_lr,
        eps=eps,
    )
```

### Pattern: Scheduler Step Patterns

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Lines 863-867 (train_operator):**
```python
if scheduler is not None:
    if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
        scheduler.step(mean_loss)
    else:
        scheduler.step()
```

### Pattern: Optimizer Param Groups Access

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Line 167 (TrainingLogger.log):**
```python
lr = optimizer.param_groups[0].get("lr") if optimizer.param_groups else None
```

---

## 4. Mixed Precision Training (AMP)

### Pattern: AMP Availability Check

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Lines 378-379 (_amp_enabled function):**
```python
def _amp_enabled(cfg: Dict) -> bool:
    return bool(cfg.get("training", {}).get("amp", False)) and torch.cuda.is_available()
```

### Pattern: GradScaler Setup

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Lines 559-560 (train_operator):**
```python
use_amp = _amp_enabled(cfg)
scaler = GradScaler(enabled=use_amp)
```

**Lines 934-935 (train_diffusion):**
```python
use_amp = _amp_enabled(cfg)
scaler = GradScaler(enabled=use_amp)
```

### Pattern: Autocast Context Manager

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Line 22 (imports):**
```python
from torch.cuda.amp import autocast, GradScaler
```

**Line 651 (train_operator forward pass):**
```python
with autocast(enabled=use_amp):
    # Forward prediction (always computed)
    next_state = operator(state, dt_tensor)
    # ... loss computation ...
```

**Line 981 (train_diffusion):**
```python
with autocast(enabled=use_amp):
    drift = diff(predicted, tau_tensor)
    base = F.mse_loss(drift, residual_target)
    # ...
```

### Pattern: Gradient Scaling and Unscaling

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Lines 806-823 (train_operator backward and step):**
```python
if use_amp:
    scaler.scale(loss / accum_steps).backward()
else:
    (loss / accum_steps).backward()

do_step = ((i + 1) % accum_steps == 0) or ((i + 1) == num_batches)
if do_step:
    if use_amp:
        if clip_val is not None:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(operator.parameters(), 
                                                    float('inf') if clip_val is None else clip_val)
        total_grad_norm += float(grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(operator.parameters(), 
                                                    float('inf') if clip_val is None else clip_val)
        total_grad_norm += grad_norm.item()
        optimizer.step()
```

**Lines 999-1010 (train_diffusion):**
```python
if use_amp:
    scaler.scale(loss / accum_steps).backward()
else:
    (loss / accum_steps).backward()

do_step = ((i + 1) % accum_steps == 0) or ((i + 1) == num_batches)
if do_step:
    if use_amp:
        if clip_val is not None:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(diff.parameters(), float('inf') if clip_val is None else clip_val)
        total_grad_norm += float(grad_norm)
        scaler.step(optimizer)
        scaler.update()
```

**Lines 1330-1340 (train_consistency_distill):**
```python
scaler.scale(loss_chunk * chunk_weight).backward()

if (step + 1) % accum_steps == 0 or (step + 1) == num_total_steps:
    if clip_val is not None:
        scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(distill_model.parameters(), float('inf') if clip_val is None else clip_val)
    total_grad_norm += float(grad_norm)
    scaler.step(optimizer)
    scaler.update()
```

### Pattern: Spectral Loss with Autocast Disabled

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Lines 66-80 (_spectral_energy_loss function):**
```python
def _spectral_energy_loss(pred: torch.Tensor, target: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """Relative spectral energy difference along the given axis.

    cuFFT requires power-of-two signal sizes when using half precision. 
    Temporarily disable autocast and promote to float32 before taking the FFT 
    so non-power-of-two token counts (e.g., 48) do not trigger runtime errors. 
    Cast the result back to the original dtype for downstream losses.
    """
    with torch.cuda.amp.autocast(enabled=False):
        pred_fft = torch.fft.rfft(pred.float(), dim=dim)
        tgt_fft = torch.fft.rfft(target.float(), dim=dim)
        pred_energy = torch.mean(pred_fft.abs() ** 2)
        tgt_energy = torch.mean(tgt_fft.abs() ** 2)
        loss = torch.abs(pred_energy - tgt_energy) / (tgt_energy + eps)
    return loss.to(pred.dtype)
```

### Pattern: BF16 Support via Training Config

**File:** `/Users/emerygunselman/Code/universal_simulator/CLAUDE.md`

Referenced in documentation (line 82):
```
Mixed precision (bf16) is default: training.amp=true
```

The config enables AMP via `training.amp` boolean flag.

---

## 5. torch.compile Usage

### Pattern: torch.compile with Mode Selection

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Lines 382-413 (_maybe_compile function):**
```python
def _maybe_compile(model: nn.Module, cfg: Dict, name: str) -> nn.Module:
    """Optionally compile a model with torch.compile when enabled and available.

    Controlled by training.compile bool. Falls back silently if unavailable.
    """
    try:
        compile_enabled = bool(cfg.get("training", {}).get("compile", False))
    except Exception:
        compile_enabled = False
    if not compile_enabled:
        return model
    
    # Skip compilation for teacher models (eval-only) to avoid CUDA graph issues
    if "teacher" in name:
        return model
        
    try:
        import torch

        # Safer default: "default" mode; allow override via training.compile_mode
        training_cfg = cfg.get("training", {})
        user_mode = str(training_cfg.get("compile_mode", "")).lower()
        if user_mode in {"default", "reduce-overhead", "max-autotune"}:
            compile_mode = user_mode
        else:
            compile_mode = "default"

        compiled = torch.compile(model, mode=compile_mode, fullgraph=False)
        return compiled
    except Exception:
        # If torch.compile is unavailable or fails, just return the original model
        return model
```

### Pattern: Compiled Prefix Stripping from State Dict

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Lines 83-89 (_strip_compiled_prefix function):**
```python
def _strip_compiled_prefix(state_dict: dict) -> dict:
    """Strip _orig_mod. prefix from state dict keys (from torch.compile)."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "") if key.startswith("_orig_mod.") else key
        new_state_dict[new_key] = value
    return new_state_dict
```

### Pattern: Compile Configuration

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Lines 32-41 (torch._dynamo configuration):**
```python
try:
    import torch._dynamo as _dynamo
    _dynamo.config.suppress_errors = True  # Avoid hard-crash on backend failures
    _dynamo.config.error_on_recompile = False
except Exception:
    pass

# Avoid inductor subprocess crashes by compiling in the main process
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
```

### Usage Examples:

**Line 494 (train_operator):**
```python
operator = _maybe_compile(operator, cfg, "operator")
```

**Line 912 (train_diffusion):**
```python
operator = _maybe_compile(operator, cfg, "operator_teacher")
```

**Line 921 (train_diffusion):**
```python
diff = _maybe_compile(diff, cfg, "diffusion_residual")
```

---

## 6. Checkpoint Saving/Loading Patterns

### Pattern: State Dict Saving

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Lines 868-878 (train_operator checkpoint save):**
```python
operator.load_state_dict(best_state)
logger.close()
checkpoint_dir = ensure_checkpoint_dir(cfg)
operator_path = checkpoint_dir / "operator.pt"
torch.save(operator.state_dict(), operator_path)
print(f"Saved operator checkpoint to {operator_path}")
if ema_model is not None:
    operator_ema_path = checkpoint_dir / "operator_ema.pt"
    to_save = best_ema_state if best_ema_state is not None else ema_model.state_dict()
    torch.save(to_save, operator_ema_path)
    print(f"Saved operator EMA checkpoint to {operator_ema_path}")
```

### Pattern: State Dict Loading with Device Mapping

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Lines 906-911 (train_diffusion checkpoint load):**
```python
op_path = checkpoint_dir / "operator.pt"
if op_path.exists():
    operator_state = torch.load(op_path, map_location="cpu")
    operator_state = _strip_compiled_prefix(operator_state)
    operator.load_state_dict(operator_state)
_ensure_model_on_device(operator, device)
```

**Lines 1168-1174 (train_consistency_distill):**
```python
op_path = checkpoint_dir / "operator.pt"
if op_path.exists():
    operator_state = torch.load(op_path, map_location="cpu")
    operator_state = _strip_compiled_prefix(operator_state)
    operator.load_state_dict(operator_state)
```

### Pattern: EMA State Dict Tracking

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Lines 557-563 (train_operator EMA initialization):**
```python
best_state = copy.deepcopy(operator.state_dict())
# AMP + EMA setup
use_amp = _amp_enabled(cfg)
scaler = GradScaler(enabled=use_amp)
ema_decay = _get_ema_decay(cfg, "operator")
ema_model = _init_ema(operator) if ema_decay else None
best_ema_state = copy.deepcopy(ema_model.state_dict()) if ema_model is not None else None
```

**Lines 855-857 (track best EMA state):**
```python
best_state = copy.deepcopy(operator.state_dict())
if ema_model is not None:
    best_ema_state = copy.deepcopy(ema_model.state_dict())
```

### Pattern: Checkpoint Metadata Stripping

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/evaluate.py`

**Lines 39-61 (_load_state_dict_compat function):**
```python
def _load_state_dict_compat(model: torch.nn.Module, ckpt_path: str, 
                            *, prefix_to_strip: str = "_orig_mod.") -> None:
    """Load a checkpoint while stripping an optional prefix from keys (e.g., from torch.compile()).

    This makes loading robust across compiled/non-compiled training runs.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise RuntimeError(f"Unsupported checkpoint format at {ckpt_path}: {type(ckpt)}")

    if prefix_to_strip:
        fixed = {}
        for k, v in state_dict.items():
            if k.startswith(prefix_to_strip):
                fixed[k[len(prefix_to_strip) :]] = v
            else:
                fixed[k] = v
        state_dict = fixed

    model.load_state_dict(state_dict)
```

### Pattern: Checkpoint Manager for WandB

**File:** `/Users/emerygunselman/Code/universal_simulator/src/ups/utils/checkpoint_manager.py`

**Lines 36-123 (download_checkpoints_from_run):**
Downloads checkpoint files from WandB runs with optional force re-download:
```python
def download_checkpoints_from_run(self, run_id: str, checkpoint_files: Optional[List[str]] = None, 
                                  force: bool = False) -> List[Path]:
    # Default checkpoint files to download
    if checkpoint_files is None:
        checkpoint_files = [
            "checkpoints/operator.pt",
            "checkpoints/operator_ema.pt",
            "checkpoints/diffusion_residual.pt",
            "checkpoints/diffusion_residual_ema.pt",
            "checkpoints/scale/input_stats.pt",
            "checkpoints/scale/latent_stats.pt",
        ]
    # ... authentication and download logic ...
```

---

## 7. Gradient Management Patterns

### Pattern: Gradient Clipping

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Lines 815-816 (with AMP):**
```python
if clip_val is not None:
    scaler.unscale_(optimizer)
grad_norm = torch.nn.utils.clip_grad_norm_(operator.parameters(), 
                                            float('inf') if clip_val is None else clip_val)
```

**Lines 820-821 (without AMP):**
```python
grad_norm = torch.nn.utils.clip_grad_norm_(operator.parameters(), 
                                            float('inf') if clip_val is None else clip_val)
total_grad_norm += grad_norm.item()
```

### Pattern: Zero Grad with set_to_none

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Line 610 (train_operator):**
```python
optimizer.zero_grad(set_to_none=True)
```

**Line 823 (train_operator per-step):**
```python
optimizer.zero_grad(set_to_none=True)
```

**Line 952 (train_diffusion):**
```python
optimizer.zero_grad(set_to_none=True)
```

**File:** `/Users/emerygunselman/Code/universal_simulator/src/ups/training/hybrid_optimizer.py`

**Lines 72-80 (HybridOptimizer.zero_grad):**
```python
def zero_grad(self, set_to_none: bool = True) -> None:
    """Zero gradients for all child optimizers."""
    for opt in self.optimizers:
        opt.zero_grad(set_to_none=set_to_none)
```

### Pattern: EMA Model Update

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Lines 438-441 (_update_ema function):**
```python
@torch.no_grad()
def _update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.mul_(decay).add_(p.data, alpha=1.0 - decay)
```

**Line 826 (train_operator):**
```python
if ema_model is not None and ema_decay is not None:
    _update_ema(ema_model, operator, ema_decay)
```

### Pattern: EMA Model Initialization

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Lines 430-435 (_init_ema function):**
```python
def _init_ema(model: nn.Module) -> nn.Module:
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    ema.eval()
    return ema
```

---

## 8. Model Training/Eval Mode Patterns

### Pattern: Model Mode Management

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Line 913 (train_diffusion - teacher in eval):**
```python
operator.eval()
```

**Line 536 (train_operator - encoder frozen):**
```python
encoder.eval()  # Encoder is frozen during operator stage
```

**Line 551 (train_operator - decoder frozen):**
```python
decoder.eval()  # Decoder not trained during operator stage
```

---

## 9. Multiprocessing & Determinism Patterns

### Pattern: Multiprocessing Start Method

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Lines 44-47:**
```python
# Ensure CUDA + DataLoader workers use a safe start method
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass
```

### Pattern: Determinism Configuration

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Lines 97-126 (set_seed function):**
```python
def set_seed(cfg: Dict) -> None:
    """Set random seed and configure determinism settings."""
    seed = cfg.get("seed", 17)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    deterministic = cfg.get("deterministic", False)
    benchmark = cfg.get("benchmark", True)

    if deterministic:
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"✓ Deterministic mode enabled (seed={seed}, CUBLAS workspace configured)")
    else:
        torch.backends.cudnn.benchmark = benchmark
        print(f"✓ Seed set to {seed} (deterministic={deterministic}, benchmark={benchmark})")
```

---

## 10. Hybrid Optimizer Pattern

### Pattern: Multi-Optimizer Wrapper

**File:** `/Users/emerygunselman/Code/universal_simulator/src/ups/training/hybrid_optimizer.py`

**Lines 1-136 (HybridOptimizer class):**

A custom wrapper class that manages multiple optimizers (Muon + AdamW) as a single unit:

**Constructor (lines 35-52):**
```python
def __init__(self, optimizers: list[Optimizer]):
    if not optimizers:
        raise ValueError("Must provide at least one optimizer")

    self.optimizers = optimizers
    self._param_groups = []

    # Initialize with dummy parameter to satisfy PyTorch base class
    dummy_param = [{'params': []}]
    super().__init__(dummy_param, {})
```

**Step Implementation (lines 54-70):**
```python
def step(self, closure: callable | None = None) -> float | None:
    loss = None
    for opt in self.optimizers:
        if closure is not None:
            loss = opt.step(closure)
        else:
            opt.step()
    return loss
```

**State Dict Implementation (lines 82-104):**
```python
def state_dict(self) -> dict[str, Any]:
    return {
        f"optimizer_{i}": opt.state_dict()
        for i, opt in enumerate(self.optimizers)
    }

def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    for i, opt in enumerate(self.optimizers):
        key = f"optimizer_{i}"
        if key in state_dict:
            opt.load_state_dict(state_dict[key])
```

**Param Groups Property (lines 106-121):**
```python
@property
def param_groups(self) -> list[dict[str, Any]]:
    """Get all param_groups from all child optimizers.
    
    This property is required for:
    - AMP GradScaler.unscale_() 
    - LR schedulers
    """
    groups = []
    for opt in self.optimizers:
        groups.extend(opt.param_groups)
    return groups
```

---

## 11. Gradient Accumulation Pattern

### Pattern: Gradient Accumulation with Step Control

**File:** `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

**Lines 568 & 810 (train_operator):**
```python
accum_steps = max(1, int(cfg.get("training", {}).get("accum_steps", 1)))
# ...
do_step = ((i + 1) % accum_steps == 0) or ((i + 1) == num_batches)
if do_step:
    # ... gradient scaling, unscaling, clipping ...
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

---

## Summary

The codebase implements a sophisticated PyTorch training system with:

1. **Single-GPU Training**: No DataParallel or DistributedDataParallel (uses CUDA when available)
2. **Flexible Device Management**: Runtime detection with aggressive placement verification
3. **Multiple Optimizer Support**: Standard (Adam, AdamW, SGD) and hybrid Muon+AdamW
4. **Mixed Precision Training**: torch.cuda.amp with GradScaler and automatic fallback
5. **torch.compile Integration**: Optional model compilation with error recovery
6. **Comprehensive Checkpointing**: State dict management with prefix stripping for compiled models
7. **Advanced Optimization**: EMA models, gradient accumulation, multiple schedulers
8. **Safety Mechanisms**: OOM recovery, determinism configuration, multiprocessing safety

