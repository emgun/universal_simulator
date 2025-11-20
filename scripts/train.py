#!/usr/bin/env python
from __future__ import annotations

"""Training entrypoint for latent operator stages."""

print("[IMPORT-DEBUG-START] ⭐ train.py script START (before stdlib imports)", flush=True)
print("[IMPORT-DEBUG] About to import argparse...", flush=True)
import argparse
print("[IMPORT-DEBUG] ✓ argparse imported", flush=True)
import copy
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
print("[IMPORT-DEBUG] ✓ Standard library + torch imports complete")

try:
    import wandb
except ImportError:
    wandb = None
print("[IMPORT-DEBUG] ✓ wandb import complete")

# Safer compile defaults: compile in main process and fall back to eager on failures
try:
    import torch._dynamo as _dynamo

    _dynamo.config.suppress_errors = True  # Avoid hard-crash on backend failures
    _dynamo.config.error_on_recompile = False
    # Tame CUDA graphs reuse issues by disabling cudagraph capture in Inductor.
    try:
        import torch._inductor.config as _inductor_config

        _inductor_config.triton.cudagraphs = False
        _inductor_config.freezing = True
    except Exception:
        pass
except Exception:
    pass
print("[IMPORT-DEBUG] ✓ torch._dynamo config complete")

# Avoid inductor subprocess crashes by compiling in the main process
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

# Ensure CUDA + DataLoader workers use a safe start method
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass
print("[IMPORT-DEBUG] ✓ mp.set_start_method complete")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
print("[IMPORT-DEBUG] ✓ sys.path.insert complete")

print("[IMPORT-DEBUG] About to import ups.core.blocks_pdet...")
from ups.core.blocks_pdet import PDETransformerConfig
print("[IMPORT-DEBUG] ✓ ups.core.blocks_pdet complete")

print("[IMPORT-DEBUG] About to import ups.core.latent_state...")
from ups.core.latent_state import LatentState
print("[IMPORT-DEBUG] ✓ ups.core.latent_state complete")

print("[IMPORT-DEBUG] About to import ups.data.latent_pairs...")
from ups.data.latent_pairs import build_latent_pair_loader, unpack_batch
print("[IMPORT-DEBUG] ✓ ups.data.latent_pairs complete")

print("[IMPORT-DEBUG] About to import ups.models.diffusion_residual...")
from ups.models.diffusion_residual import DiffusionResidual, DiffusionResidualConfig
print("[IMPORT-DEBUG] ✓ ups.models.diffusion_residual complete")

print("[IMPORT-DEBUG] About to import ups.models.latent_operator...")
from ups.models.latent_operator import LatentOperator, LatentOperatorConfig
print("[IMPORT-DEBUG] ✓ ups.models.latent_operator complete")

print("[IMPORT-DEBUG] About to import ups.models.steady_prior...")
from ups.models.steady_prior import SteadyPrior, SteadyPriorConfig
print("[IMPORT-DEBUG] ✓ ups.models.steady_prior complete")

print("[IMPORT-DEBUG] About to import ups.training.distributed_utils...")
from ups.training.distributed_utils import (
    maybe_empty_cache,
    maybe_trigger_simulated_oom,
    sync_error_flag,
)
print("[IMPORT-DEBUG] ✓ ups.training.distributed_utils complete")

print("[IMPORT-DEBUG] ✅ ALL IMPORTS COMPLETE")


# ---- Distributed Training Setup ----
def setup_distributed():
    """Initialize distributed training if RANK environment variable is set.

    Returns:
        tuple: (device, is_distributed, rank, world_size, local_rank)
    """
    import torch.distributed as dist
    import traceback

    # Log environment variables at entry
    print("[DDP-DEBUG] setup_distributed() called")
    print(f"[DDP-DEBUG] RANK={os.environ.get('RANK', 'NOT_SET')}")
    print(f"[DDP-DEBUG] LOCAL_RANK={os.environ.get('LOCAL_RANK', 'NOT_SET')}")
    print(f"[DDP-DEBUG] WORLD_SIZE={os.environ.get('WORLD_SIZE', 'NOT_SET')}")
    print(f"[DDP-DEBUG] MASTER_ADDR={os.environ.get('MASTER_ADDR', 'NOT_SET')}")
    print(f"[DDP-DEBUG] MASTER_PORT={os.environ.get('MASTER_PORT', 'NOT_SET')}")

    # CRITICAL FIX: Check if process group already initialized (multi-stage training)
    # If already initialized, return existing distributed context
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        print(f"[DDP-DEBUG] Process group already initialized - reusing existing context")
        print(f"[DDP-DEBUG] Rank: {rank}, World size: {world_size}, Device: {device}")
        return device, True, rank, world_size, local_rank

    if "RANK" in os.environ:
        print("[DDP-DEBUG] RANK env var detected, initializing DDP...")

        # Parse environment variables before DDP init
        try:
            local_rank = int(os.environ["LOCAL_RANK"])
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            print(
                f"[DDP-DEBUG] Parsed env vars: rank={rank}, local_rank={local_rank}, world_size={world_size}"
            )
        except (KeyError, ValueError) as e:
            print(f"[DDP-ERROR] Failed to parse environment variables: {e}")
            traceback.print_exc()
            raise

        # Check GPU availability
        try:
            print(f"[DDP-DEBUG] Checking GPU availability...")
            print(f"[DDP-DEBUG] torch.cuda.is_available() = {torch.cuda.is_available()}")
            print(f"[DDP-DEBUG] torch.cuda.device_count() = {torch.cuda.device_count()}")
            if torch.cuda.is_available():
                print(f"[DDP-DEBUG] CUDA version: {torch.version.cuda}")
                print(
                    f"[DDP-DEBUG] NCCL available: {torch.cuda.nccl.is_available() if hasattr(torch.cuda, 'nccl') else 'N/A'}"
                )
                if hasattr(torch.cuda, "nccl") and torch.cuda.nccl.is_available():
                    print(f"[DDP-DEBUG] NCCL version: {torch.cuda.nccl.version()}")
        except Exception as e:
            print(f"[DDP-WARNING] Error checking GPU info: {e}")

        # Initialize process group with error handling and fallback
        # Allow backend override via DDP_BACKEND env var (e.g., export DDP_BACKEND=gloo)
        preferred_backend = os.environ.get("DDP_BACKEND", "nccl").lower()
        backend_to_use = None

        # Try preferred backend first
        for backend in [preferred_backend, "gloo"]:
            if backend_to_use is not None:
                break  # Already succeeded

            try:
                print(f"[DDP-DEBUG] Attempting dist.init_process_group(backend='{backend}')...")
                dist.init_process_group(backend=backend)
                print(
                    f"[DDP-DEBUG] dist.init_process_group(backend='{backend}') completed successfully"
                )
                backend_to_use = backend
                print(f"[DDP-DEBUG] dist.is_initialized() = {dist.is_initialized()}")
                print(f"[DDP-DEBUG] dist.get_backend() = {dist.get_backend()}")
                print(f"[DDP-DEBUG] dist.get_rank() = {dist.get_rank()}")
                print(f"[DDP-DEBUG] dist.get_world_size() = {dist.get_world_size()}")

                # CRITICAL FIX: Add barrier after init_process_group to sync all ranks
                # Without this, ranks can be out of sync during model creation
                dist.barrier()
                print(f"[DDP-DEBUG] Barrier passed after init_process_group on rank {dist.get_rank()}")

                if backend == "gloo" and preferred_backend == "nccl":
                    print(f"[DDP-WARNING] Fell back to Gloo backend (NCCL failed)")
                    print(f"[DDP-WARNING] Gloo is CPU-based and slower - for debugging only!")
                break

            except Exception as e:
                print(f"[DDP-ERROR] dist.init_process_group(backend='{backend}') failed: {e}")
                print(f"[DDP-ERROR] Exception type: {type(e).__name__}")
                traceback.print_exc()

                if backend == preferred_backend and backend != "gloo":
                    print(f"[DDP-WARNING] {backend.upper()} failed, trying Gloo fallback...")
                    print("[DDP-DEBUG] Printing NCCL/CUDA environment variables:")
                    for key in sorted(os.environ.keys()):
                        if "NCCL" in key or "CUDA" in key:
                            print(f"[DDP-DEBUG]   {key}={os.environ[key]}")
                else:
                    # All backends failed
                    print(f"[DDP-ERROR] All backends failed (tried: {preferred_backend}, gloo)")
                    raise

        # Set CUDA device
        try:
            print(f"[DDP-DEBUG] Setting CUDA device to {local_rank}...")
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            print(f"[DDP-DEBUG] Device set to {device}")
            print(f"[DDP-DEBUG] torch.cuda.current_device() = {torch.cuda.current_device()}")
        except Exception as e:
            print(f"[DDP-ERROR] Failed to set CUDA device: {e}")
            traceback.print_exc()
            raise

        if rank == 0:
            print(f"[DDP-INFO] Distributed training initialized: {world_size} GPUs")

        print(f"[DDP-DEBUG] setup_distributed() returning successfully for rank {rank}")
        return device, True, rank, world_size, local_rank
    else:
        # Single-GPU mode (backward compatible)
        print("[DDP-DEBUG] RANK env var not found, using single-GPU mode")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DDP-INFO] Using device: {device} (single-GPU mode)")
        return device, False, 0, 1, None


def aggregate_metrics(
    metrics: dict[str, float],
    world_size: int,
    rank: int,
) -> dict[str, float]:
    """Aggregate metrics across all ranks (mean).

    Args:
        metrics: Dictionary of metric name to value
        world_size: Number of distributed processes
        rank: Current process rank

    Returns:
        Dictionary of aggregated metrics (mean across all ranks)
    """
    if world_size == 1:
        return metrics  # Single-GPU, no aggregation needed

    import torch.distributed as dist

    aggregated = {}
    for key, value in metrics.items():
        # Convert to tensor
        tensor = torch.tensor(value, device=f"cuda:{rank}")

        # All-reduce (sum)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        # Average
        aggregated[key] = (tensor / world_size).item()

    return aggregated


# ---- Auxiliary training losses ----
def _nrmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    denom = torch.mean(target**2) + eps
    return torch.sqrt(mse / denom)


def _spectral_energy_loss(
    pred: torch.Tensor, target: torch.Tensor, dim: int = 1, eps: float = 1e-8
) -> torch.Tensor:
    """Relative spectral energy difference along the given axis (default: token axis).

    cuFFT requires power-of-two signal sizes when using half precision. Temporarily
    disable autocast and promote to float32 before taking the FFT so non-power-of-two
    token counts (e.g., 48) do not trigger runtime errors. Cast the result back to the
    original dtype for downstream losses.
    """
    with torch.cuda.amp.autocast(enabled=False):
        pred_fft = torch.fft.rfft(pred.float(), dim=dim)
        tgt_fft = torch.fft.rfft(target.float(), dim=dim)
        pred_energy = torch.mean(pred_fft.abs() ** 2)
        tgt_energy = torch.mean(tgt_fft.abs() ** 2)
        loss = torch.abs(pred_energy - tgt_energy) / (tgt_energy + eps)
    return loss.to(pred.dtype)


def _strip_compiled_prefix(state_dict: dict) -> dict:
    """Strip _orig_mod. prefix from state dict keys (from torch.compile)."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "") if key.startswith("_orig_mod.") else key
        new_state_dict[new_key] = value
    return new_state_dict


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def set_seed(cfg: dict) -> None:
    """Set random seed and configure determinism settings.

    Args:
        cfg: Config dict with optional 'seed', 'deterministic', and 'benchmark' keys
    """
    seed = cfg.get("seed", 17)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Configure PyTorch determinism
    deterministic = cfg.get("deterministic", False)
    benchmark = cfg.get("benchmark", True)

    if deterministic:
        # Set CUBLAS workspace config for deterministic CuBLAS operations
        import os

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"✓ Deterministic mode enabled (seed={seed}, CUBLAS workspace configured)")
    else:
        torch.backends.cudnn.benchmark = benchmark
        print(f"✓ Seed set to {seed} (deterministic={deterministic}, benchmark={benchmark})")


def ensure_checkpoint_dir(cfg: dict) -> Path:
    ckpt_cfg = cfg.get("checkpoint", {})
    directory = Path(ckpt_cfg.get("dir", "checkpoints"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


class TrainingLogger:
    def __init__(
        self, cfg: dict[str, dict], stage: str, global_step: int = 0, wandb_ctx=None
    ) -> None:
        """Training logger that writes to file and optionally to WandB.

        Args:
            cfg: Training configuration
            stage: Training stage name (operator, diffusion_residual, etc.)
            global_step: Initial global step counter
            wandb_ctx: Optional WandBContext for logging (recommended)
        """
        training_cfg = cfg.get("training", {})
        log_path = training_cfg.get("log_path", "reports/training_log.jsonl")
        self.stage = stage
        self.global_step = global_step
        self.wandb_ctx = wandb_ctx
        self.log_path = Path(log_path) if log_path else None

        # Create log file if specified
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        epoch: int,
        loss: float,
        optimizer: torch.optim.Optimizer,
        patience_counter: int | None = None,
        grad_norm: float | None = None,
        epoch_time: float | None = None,
        best_loss: float | None = None,
    ) -> None:
        lr = optimizer.param_groups[0].get("lr") if optimizer.param_groups else None
        self.global_step += 1

        # Log to file (JSONL format)
        if self.log_path:
            entry = {
                "stage": self.stage,
                "loss": loss,
                "epoch": epoch,
                "lr": lr,
                "global_step": self.global_step,
            }
            if patience_counter is not None:
                entry["epochs_since_improve"] = patience_counter
            if grad_norm is not None:
                entry["grad_norm"] = grad_norm
            if epoch_time is not None:
                entry["epoch_time_sec"] = epoch_time
            if best_loss is not None:
                entry["best_loss"] = best_loss

            try:
                with self.log_path.open("a", encoding="utf-8") as fh:
                    import json

                    fh.write(json.dumps(entry) + "\n")
            except Exception:
                pass

        # Log to WandB using clean context (proper time series!)
        if self.wandb_ctx:
            self.wandb_ctx.log_training_metric(self.stage, "loss", loss, step=self.global_step)
            if lr is not None:
                self.wandb_ctx.log_training_metric(self.stage, "lr", lr, step=self.global_step)
            if patience_counter is not None:
                self.wandb_ctx.log_training_metric(
                    self.stage, "epochs_since_improve", patience_counter, step=self.global_step
                )
            if grad_norm is not None:
                self.wandb_ctx.log_training_metric(
                    self.stage, "grad_norm", grad_norm, step=self.global_step
                )
            if epoch_time is not None:
                self.wandb_ctx.log_training_metric(
                    self.stage, "epoch_time_sec", epoch_time, step=self.global_step
                )
            if best_loss is not None:
                self.wandb_ctx.log_training_metric(
                    self.stage, "best_loss", best_loss, step=self.global_step
                )

    def close(self) -> None:
        # No longer owns wandb run - orchestrator manages it
        pass

    def get_global_step(self) -> int:
        return self.global_step


def dataset_loader(cfg: dict) -> DataLoader:
    data_cfg = cfg.get("data", {})
    if not (data_cfg.get("task") or data_cfg.get("kind")):
        raise ValueError(
            "Training now requires a real dataset configuration. Set data.task for PDEBench or data.kind for Zarr datasets."
        )
    return build_latent_pair_loader(cfg)


def make_operator(cfg: dict) -> LatentOperator:
    latent_cfg = cfg.get("latent", {})
    dim = latent_cfg.get("dim", 32)
    operator_cfg = cfg.get("operator", {})
    pdet_cfg = operator_cfg.get("pdet", {})
    architecture_type = operator_cfg.get("architecture_type", "pdet_unet")

    if not pdet_cfg:
        pdet_cfg = {
            "input_dim": dim,
            "hidden_dim": dim * 2,
            "depths": [1, 1, 1],
            "group_size": max(dim // 2, 4),
            "num_heads": 4,
        }

    # Create appropriate config based on architecture type
    if architecture_type == "pdet_stack":
        from ups.models.pure_transformer import PureTransformerConfig

        pdet_config = PureTransformerConfig(**pdet_cfg)
    else:  # pdet_unet (default)
        pdet_config = PDETransformerConfig(**pdet_cfg)

    config = LatentOperatorConfig(
        latent_dim=dim,
        pdet=pdet_config,
        architecture_type=architecture_type,
        time_embed_dim=dim,
    )
    return LatentOperator(config)


def _create_optimizer(cfg: dict, model: nn.Module, stage: str) -> torch.optim.Optimizer:
    """Create optimizer with optional hybrid Muon+AdamW support and CPU offload."""
    from ups.training.hybrid_optimizer import wrap_optimizer_with_cpu_offload

    stage_cfg = cfg.get("stages", {}).get(stage, {}) if isinstance(cfg.get("stages"), dict) else {}
    opt_cfg = stage_cfg.get("optimizer") or cfg.get("optimizer", {})
    name = opt_cfg.get("name", "adam").lower()
    lr = opt_cfg.get("lr", 1e-3)
    weight_decay = opt_cfg.get("weight_decay", 0.0)

    # Check if CPU offload is enabled
    cpu_offload_enabled = cfg.get("training", {}).get("cpu_offload_optimizer", False)

    # Standard optimizers (original behavior)
    if name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        return wrap_optimizer_with_cpu_offload(optimizer, cpu_offload_enabled)
    if name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            fused=True,  # Enable fused kernels (PyTorch 2.0+)
        )
        return wrap_optimizer_with_cpu_offload(optimizer, cpu_offload_enabled)
    if name == "sgd":
        momentum = opt_cfg.get("momentum", 0.9)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        return wrap_optimizer_with_cpu_offload(optimizer, cpu_offload_enabled)

    # NEW: Hybrid Muon+AdamW optimizer
    if name == "muon_hybrid" or name == "muon":
        from ups.training.hybrid_optimizer import HybridOptimizer
        from ups.training.muon_factory import create_muon_optimizer, get_available_backends
        from ups.training.param_groups import build_param_groups, print_param_split_summary

        # Log available backends
        backends = get_available_backends()
        if not backends:
            print("WARNING: No Muon implementation available, falling back to AdamW")
            print(
                "  Install: pip install torch>=2.9 or pip install git+https://github.com/nil0x9/flash-muon.git"
            )
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                fused=True,  # Enable fused kernels (PyTorch 2.0+)
            )
            return wrap_optimizer_with_cpu_offload(optimizer, cpu_offload_enabled)

        print(f"Available Muon backends: {', '.join(backends)}")

        # IMPORTANT: CPU offload is incompatible with Muon optimizer (torchao issue)
        # Muon optimizer is not iterable, causing TypeError in CPUOffloadOptimizer.__init__
        if cpu_offload_enabled:
            print(
                "⚠️  WARNING: cpu_offload_optimizer is incompatible with muon_hybrid optimizer"
            )
            print("   Disabling CPU offload for this run (torchao compatibility issue)")
            print("   See: https://github.com/pytorch/ao/issues/2919")
            cpu_offload_enabled = False

        # Split parameters into Muon (2D+) and AdamW (1D) groups
        muon_params, adamw_params = build_param_groups(model)
        print_param_split_summary(model)

        # Muon-specific hyperparameters (with defaults from research)
        muon_momentum = opt_cfg.get("muon_momentum", 0.95)  # Nesterov momentum
        muon_ns_steps = opt_cfg.get("muon_ns_steps", 5)  # Newton-Schulz iterations
        muon_backend = opt_cfg.get("muon_backend", "auto")  # Backend selection

        # AdamW hyperparameters
        adamw_betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        adamw_eps = opt_cfg.get("eps", 1e-8)

        optimizers = []

        # Create Muon optimizer if there are 2D+ parameters
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

        # Create AdamW optimizer for 1D parameters
        if len(adamw_params) > 0:
            adamw_opt = torch.optim.AdamW(
                adamw_params,
                lr=lr,
                betas=adamw_betas,
                eps=adamw_eps,
                weight_decay=weight_decay,
                fused=True,  # Enable fused kernels (PyTorch 2.0+)
            )
            optimizers.append(adamw_opt)
            print(f"  AdamW: {len(adamw_params)} parameter groups")

        # If only one optimizer, return it directly
        if len(optimizers) == 1:
            return optimizers[0]

        # Return hybrid wrapper
        return HybridOptimizer(optimizers)

    raise ValueError(f"Unsupported optimizer '{name}'")


def _create_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, stage: str):
    stage_cfg = cfg.get("stages", {}).get(stage, {})
    sched_cfg = stage_cfg.get("scheduler") or cfg.get("optimizer", {}).get("scheduler")
    if not sched_cfg:
        return None
    name = sched_cfg.get("name", "steplr").lower()
    if name == "steplr":
        step_size = sched_cfg.get("step_size", 1)
        gamma = sched_cfg.get("gamma", 0.5)
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "cosineannealinglr":
        t_max = sched_cfg.get("t_max", 10)
        eta_min = sched_cfg.get("eta_min", 0.0)
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
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
    raise ValueError(f"Unsupported scheduler '{name}'")


def _amp_enabled(cfg: dict) -> bool:
    return bool(cfg.get("training", {}).get("amp", False)) and torch.cuda.is_available()


def _get_amp_dtype(cfg: dict) -> tuple[torch.dtype, bool]:
    """Get AMP dtype and whether to use GradScaler.

    Returns:
        tuple: (autocast_dtype, use_scaler)
            - autocast_dtype: torch.bfloat16, torch.float16, or torch.float32
            - use_scaler: True only for float16 (BF16 doesn't need gradient scaling)
    """
    if not _amp_enabled(cfg):
        return torch.float32, False

    amp_dtype_str = cfg.get("training", {}).get("amp_dtype", "bfloat16")

    if amp_dtype_str == "bfloat16":
        return torch.bfloat16, False  # BF16 doesn't need gradient scaling
    elif amp_dtype_str == "float16":
        return torch.float16, True    # FP16 requires GradScaler
    else:
        # Default to bfloat16 for unknown values
        return torch.bfloat16, False


def _maybe_compile(model: nn.Module, cfg: dict, name: str) -> nn.Module:
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


def setup_fsdp2(model: nn.Module, cfg: dict, local_rank: int) -> nn.Module:
    """
    Wrap model with FSDP2 (Fully Sharded Data Parallel v2).

    Requires PyTorch 2.4+. Shards parameters across GPUs to save memory.

    Args:
        model: Model to wrap
        cfg: Config dict with FSDP settings
        local_rank: Local GPU rank

    Returns:
        FSDP-wrapped model
    """
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    except ImportError:
        print("[FSDP2] Warning: FSDP not available, falling back to DDP")
        from torch.nn.parallel import DistributedDataParallel as DDP
        return DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            static_graph=True,
        )

    # Configure sharding strategy
    strategy = ShardingStrategy.FULL_SHARD  # Shard params, gradients, and optimizer state

    # Configure mixed precision (match training config)
    training_cfg = cfg.get("training", {})
    amp_enabled = training_cfg.get("amp", False)
    amp_dtype_str = training_cfg.get("amp_dtype", "bfloat16")

    if amp_enabled:
        if amp_dtype_str == "bfloat16":
            param_dtype = torch.bfloat16
        elif amp_dtype_str == "float16":
            param_dtype = torch.float16
        else:
            param_dtype = torch.float32

        mixed_precision_policy = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=param_dtype,
            buffer_dtype=torch.float32,  # Keep buffers in FP32 for stability
        )
    else:
        mixed_precision_policy = None

    # Optional CPU offload (not used by default for speed)
    cpu_offload = CPUOffload(offload_params=False)

    # Auto-wrap policy: wrap layers with >100M parameters
    # Use functools.partial to create the policy correctly for PyTorch 2.x
    from functools import partial
    auto_wrap_policy = partial(
        size_based_auto_wrap_policy,
        min_num_params=100_000_000
    )

    print(f"[FSDP2] Wrapping model with FSDP2, strategy={strategy}, mixed_precision={amp_dtype_str if amp_enabled else 'disabled'}")

    # Apply FSDP2 wrapper
    model = FSDP(
        model,
        sharding_strategy=strategy,
        cpu_offload=cpu_offload,
        mixed_precision=mixed_precision_policy,
        auto_wrap_policy=auto_wrap_policy,
        device_id=local_rank,
    )

    return model


def save_checkpoint_fsdp(model: nn.Module, path: Path, is_distributed: bool, rank: int = 0) -> None:
    """Save checkpoint compatible with FSDP.

    Args:
        model: Model to save (may be FSDP-wrapped or DDP-wrapped)
        path: Path to save checkpoint
        is_distributed: Whether using distributed training
        rank: Process rank (only rank 0 saves for FSDP)
    """
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig

        if isinstance(model, FSDP):
            # Use FSDP state dict API
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                state_dict = model.state_dict()

                # Only rank 0 saves
                if rank == 0:
                    torch.save(state_dict, path)
                    print(f"[FSDP2] Saved FSDP checkpoint to {path}")
        else:
            # Standard DDP or single-GPU checkpoint
            if not is_distributed or rank == 0:
                # Unwrap DDP if needed
                model_state = model.module.state_dict() if is_distributed else model.state_dict()
                torch.save(model_state, path)
    except ImportError:
        # FSDP not available, use standard save
        if not is_distributed or rank == 0:
            model_state = model.module.state_dict() if is_distributed else model.state_dict()
            torch.save(model_state, path)


def load_checkpoint_fsdp(model: nn.Module, path: Path, is_distributed: bool) -> None:
    """Load checkpoint compatible with FSDP.

    Args:
        model: Model to load into (may be FSDP-wrapped or DDP-wrapped)
        path: Path to load checkpoint from
        is_distributed: Whether using distributed training
    """
    if not path.exists():
        print(f"[FSDP2] Warning: Checkpoint not found at {path}")
        return

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType

        checkpoint = torch.load(path, map_location="cpu")

        if isinstance(model, FSDP):
            # Use FSDP load API
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                model.load_state_dict(checkpoint)
                print(f"[FSDP2] Loaded FSDP checkpoint from {path}")
        else:
            # Standard DDP or single-GPU load
            if is_distributed:
                model.module.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
    except ImportError:
        # FSDP not available, use standard load
        checkpoint = torch.load(path, map_location="cpu")
        if is_distributed:
            model.module.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)


def _grad_clip_value(cfg: dict, stage: str) -> float | None:
    # Stage-specific override takes precedence; fallback to training.grad_clip
    stage_cfg = cfg.get("stages", {}).get(stage, {}) if isinstance(cfg.get("stages"), dict) else {}
    if "grad_clip" in stage_cfg:
        return stage_cfg.get("grad_clip")
    return cfg.get("training", {}).get("grad_clip")


def _get_ema_decay(cfg: dict, stage: str) -> float | None:
    stage_cfg = cfg.get("stages", {}).get(stage, {}) if isinstance(cfg.get("stages"), dict) else {}
    if "ema_decay" in stage_cfg:
        return stage_cfg.get("ema_decay")
    return cfg.get("training", {}).get("ema_decay")


def _init_ema(model: nn.Module) -> nn.Module | None:
    # FSDP-wrapped models are not deepcopy-safe; skip EMA in that case
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore[attr-defined]
    except Exception:
        FSDP = None  # type: ignore[assignment]

    if FSDP is not None and isinstance(model, FSDP):
        print("[EMA] Skipping EMA for FSDP-wrapped model (deepcopy not supported)")
        return None  # Caller guards on None

    # For standard modules, ensure we detach parameters to avoid view/inplace issues
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
        if p.grad is not None:
            p.grad = None
    ema.eval()
    return ema


@torch.no_grad()
def _update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for p_ema, p in zip(ema_model.parameters(), model.parameters(), strict=False):
        p_ema.mul_(decay).add_(p.data, alpha=1.0 - decay)


def _get_patience(cfg: dict, stage: str) -> int | None:
    stage_cfg = cfg.get("stages", {}).get(stage, {})
    if "patience" in stage_cfg:
        return stage_cfg["patience"]
    training_cfg = cfg.get("training", {})
    return training_cfg.get("patience")


def _sample_tau(batch_size: int, device: torch.device, cfg: dict) -> torch.Tensor:
    dist_cfg = cfg.get("training", {}).get("tau_distribution")
    if dist_cfg:
        dist_type = str(dist_cfg.get("type", "")).lower()
        if dist_type == "beta":
            alpha = float(dist_cfg.get("alpha", 1.0))
            beta = float(dist_cfg.get("beta", 1.0))
            beta_dist = torch.distributions.Beta(alpha, beta)
            samples = beta_dist.sample((batch_size,))
            return samples.to(device=device)
    return torch.rand(batch_size, device=device)


def _should_stop(patience: int | None, epochs_since_improve: int) -> bool:
    if patience is None:
        return False
    return epochs_since_improve > patience


def _stage_epochs(cfg: dict, stage: str) -> int:
    """Helper to read configured epochs for a stage; defaults to 0 when unset."""
    try:
        value = cfg.get("stages", {}).get(stage, {}).get("epochs", 0)
        return int(value) if value is not None else 0
    except Exception:
        return 0


def train_operator(cfg: dict, wandb_ctx=None, global_step: int = 0) -> None:
    # CRITICAL FIX: Setup distributed training FIRST
    # This ensures DistributedSampler is created properly in dataset_loader()
    device, is_distributed, rank, world_size, local_rank = setup_distributed()

    # Now create data loader (will use DistributedSampler if distributed)
    loader = dataset_loader(cfg)
    operator = make_operator(cfg)
    train_cfg = cfg.get("training", {})
    dt = train_cfg.get("dt", 0.1)
    stage_cfg = cfg.get("stages", {}).get("operator", {})
    epochs = stage_cfg.get("epochs", 1)
    optimizer = _create_optimizer(cfg, operator, "operator")
    scheduler = _create_scheduler(optimizer, cfg, "operator")
    patience = _get_patience(cfg, "operator")
    logger = TrainingLogger(cfg, stage="operator", global_step=global_step, wandb_ctx=wandb_ctx)

    operator.to(device)

    # Wrap with FSDP2 or DDP if distributed
    if is_distributed:
        use_fsdp = cfg.get("training", {}).get("use_fsdp2", False)
        num_gpus = cfg.get("training", {}).get("num_gpus", 1)

        # Use FSDP2 for 2+ GPUs if enabled
        if use_fsdp and num_gpus >= 2:
            if rank == 0:
                print(f"Using FSDP2 for {num_gpus}-GPU distributed training")
            operator = setup_fsdp2(operator, cfg, local_rank)
        else:
            from torch.nn.parallel import DistributedDataParallel as DDP

            if rank == 0:
                print("Using DDP for distributed training")
            operator = DDP(
                operator,
                device_ids=[local_rank],
                output_device=local_rank,
                static_graph=True,  # Required for torch.compile compatibility
                find_unused_parameters=False,  # All params used in operator
            )
            if rank == 0:
                print(f"Operator wrapped with DDP on device {local_rank}")

        # CRITICAL FIX: Add barrier after model wrapping to ensure all ranks synchronized
        # Without this, torch.compile can start on different ranks at different times
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()
            if rank == 0:
                print("[DDP-DEBUG] Barrier passed after model wrapping")

    operator = _maybe_compile(operator, cfg, "operator")

    # Instantiate encoder and decoder for inverse losses
    use_inverse_losses = (
        bool(cfg.get("training", {}).get("use_inverse_losses", False))
        or cfg.get("training", {}).get("lambda_inv_enc", 0.0) > 0
        or cfg.get("training", {}).get("lambda_inv_dec", 0.0) > 0
    )

    encoder = None
    decoder = None
    if use_inverse_losses:
        from ups.io.decoder_anypoint import AnyPointDecoder, AnyPointDecoderConfig
        from ups.io.enc_grid import GridEncoder, GridEncoderConfig

        # Create encoder (same config as used in data preprocessing)
        data_cfg = cfg.get("data", {})
        latent_cfg = cfg.get("latent", {})

        # Infer field channels from dataset
        # For Burgers: {"u": 1}
        # TODO: Make this more robust by reading from dataset metadata
        field_channels = {"u": 1}  # Burgers 1D default
        if "field_channels" in data_cfg:
            field_channels = data_cfg["field_channels"]

        encoder_cfg = GridEncoderConfig(
            latent_len=latent_cfg.get("tokens", 32),
            latent_dim=latent_cfg.get("dim", 16),
            field_channels=field_channels,
            patch_size=data_cfg.get("patch_size", 4),
        )
        # Prefer using the same encoder instance used by the dataset if available
        dataset_obj = getattr(loader, "dataset", None)
        shared_encoder = None
        if dataset_obj is not None:
            # Handle ConcatDataset by peeking first child
            if hasattr(dataset_obj, "encoder"):
                shared_encoder = dataset_obj.encoder
            elif (
                hasattr(dataset_obj, "datasets")
                and len(dataset_obj.datasets) > 0
                and hasattr(dataset_obj.datasets[0], "encoder")
            ):
                shared_encoder = dataset_obj.datasets[0].encoder
        encoder = (shared_encoder or GridEncoder(encoder_cfg)).to(device)
        encoder.eval()  # Encoder is frozen during operator stage

        # Create decoder (matches TTC decoder config or use sensible defaults)
        ttc_decoder_cfg = cfg.get("ttc", {}).get("decoder", {})
        decoder_cfg = AnyPointDecoderConfig(
            latent_dim=latent_cfg.get("dim", 16),
            query_dim=2,  # 2D spatial coords for Burgers
            hidden_dim=ttc_decoder_cfg.get("hidden_dim", latent_cfg.get("dim", 16) * 4),
            num_layers=ttc_decoder_cfg.get("num_layers", 2),
            num_heads=ttc_decoder_cfg.get("num_heads", 4),
            frequencies=tuple(ttc_decoder_cfg.get("frequencies", [1.0, 2.0, 4.0])),
            mlp_hidden_dim=ttc_decoder_cfg.get("mlp_hidden_dim", 128),
            output_channels=field_channels,
        )
        decoder = AnyPointDecoder(decoder_cfg).to(device)
        decoder.eval()  # Decoder not trained during operator stage

        print("✓ Initialized encoder and decoder for inverse losses")

    dt_tensor = torch.tensor(dt, device=device)
    best_loss = float("inf")
    # Save unwrapped state to avoid DDP "module." prefix issues
    best_state = copy.deepcopy(
        operator.module.state_dict() if is_distributed else operator.state_dict()
    )
    # AMP + EMA setup
    use_amp = _amp_enabled(cfg)
    autocast_dtype, use_scaler = _get_amp_dtype(cfg)
    scaler = GradScaler(enabled=use_scaler)
    ema_decay = _get_ema_decay(cfg, "operator")
    ema_model = _init_ema(operator) if ema_decay else None
    best_ema_state = copy.deepcopy(ema_model.state_dict()) if ema_model is not None else None
    clip_val = _grad_clip_value(cfg, "operator")
    epochs_since_improve = 0

    import time

    accum_steps = max(1, int(cfg.get("training", {}).get("accum_steps", 1)))
    lam_spec = float(cfg.get("training", {}).get("lambda_spectral", 0.0) or 0.0)
    lam_rel = float(cfg.get("training", {}).get("lambda_relative", 0.0) or 0.0)
    lam_rollout = float(cfg.get("training", {}).get("lambda_rollout", 0.0) or 0.0)

    # Extract query sampling config
    query_sample_cfg = train_cfg.get("query_sampling", {})
    use_query_sampling = query_sample_cfg.get("enabled", False)
    num_queries = query_sample_cfg.get("num_queries", None) if use_query_sampling else None
    query_strategy = query_sample_cfg.get("strategy", "uniform")

    # Extract physics prior config
    physics_cfg = train_cfg.get("physics_priors", {})
    use_physics_priors = physics_cfg.get("enabled", False)

    # Physics loss weights (Phase 4.2 + 4.3)
    physics_weights = {
        "lambda_divergence": physics_cfg.get("lambda_divergence", 0.0),
        "lambda_conservation": physics_cfg.get("lambda_conservation", 0.0),
        "lambda_boundary": physics_cfg.get("lambda_boundary", 0.0),
        "lambda_positivity": physics_cfg.get("lambda_positivity", 0.0),
        "bc_value": physics_cfg.get("bc_value", 0.0),
        "bc_type": physics_cfg.get("bc_type", "all"),
        # Latent regularization (Phase 4.3)
        "lambda_latent_norm": physics_cfg.get("lambda_latent_norm", 0.0),
        "lambda_latent_diversity": physics_cfg.get("lambda_latent_diversity", 0.0),
    }

    # Reference fields storage (for conservation checks)
    reference_fields_storage = {}

    print(f"[TRAIN-DEBUG] About to start training loop: epochs={epochs}, rank={rank}")
    print(f"[TRAIN-DEBUG] Loader type: {type(loader)}")
    print(f"[TRAIN-DEBUG] Loader length: {len(loader)}")

    for epoch in range(epochs):
        print(f"[TRAIN-DEBUG] Starting epoch {epoch+1}/{epochs} on rank {rank}")

        # For distributed training, set epoch for sampler (enables shuffling)
        if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
            print(f"[TRAIN-DEBUG] Set epoch {epoch} on sampler")

        epoch_start = time.time()
        epoch_loss = 0.0
        total_grad_norm = 0.0
        batches = 0
        grad_steps = 0
        num_batches = len(loader)

        # Per-task metric tracking
        task_metrics = defaultdict(lambda: {"loss": [], "count": 0})

        optimizer.zero_grad(set_to_none=True)

        print(f"[TRAIN-DEBUG] About to iterate DataLoader on rank {rank}, num_batches={num_batches}")
        batch_count = 0

        for i, batch in enumerate(loader):
            if i == 0:
                print(f"[TRAIN-DEBUG] Got first batch on rank {rank}, batch keys: {batch.keys() if isinstance(batch, dict) else 'not a dict'}")
            batch_count += 1

            unpacked = unpack_batch(batch)

            # Handle both dict and tuple formats
            if isinstance(unpacked, dict):
                z0 = unpacked["z0"]
                z1 = unpacked["z1"]
                cond = unpacked.get("cond", {})
                future = unpacked.get("future")
                # Extract inverse loss fields from dict
                input_fields_physical = unpacked.get("input_fields")
                coords = unpacked.get("coords")
                meta = unpacked.get("meta")
                # Extract task names for per-task metrics
                task_names = unpacked.get("task_names")
            elif len(unpacked) == 4:
                z0, z1, cond, future = unpacked
                input_fields_physical = None
                coords = None
                meta = None
                task_names = None
            else:
                z0, z1, cond = unpacked
                future = None
                input_fields_physical = None
                coords = None
                meta = None
                task_names = None

            cond_device = {k: v.to(device) for k, v in cond.items()}
            state = LatentState(
                z=z0.to(device), t=torch.tensor(0.0, device=device), cond=cond_device
            )
            target = z1.to(device)

            # Move inverse loss fields to device if present
            if input_fields_physical is not None:
                input_fields_physical = {k: v.to(device) for k, v in input_fields_physical.items()}
            if coords is not None:
                coords = coords.to(device)

            maybe_trigger_simulated_oom("operator", i, rank)
            loss_bundle = None
            loss = None
            batch_failed = False
            try:
                with autocast(enabled=use_amp, dtype=autocast_dtype):
                    # Forward prediction (always computed)
                    next_state = operator(state, dt_tensor)

                    # Build loss weights dict
                    loss_weights = {
                        "lambda_forward": 1.0,  # Always weight forward loss at 1.0
                        "lambda_inv_enc": float(train_cfg.get("lambda_inv_enc", 0.0)),
                        "lambda_inv_dec": float(train_cfg.get("lambda_inv_dec", 0.0)),
                        "lambda_spectral": lam_spec,
                        "lambda_rollout": lam_rollout,
                        # Curriculum schedule parameters (optional)
                        "inverse_loss_warmup_epochs": int(
                            train_cfg.get("inverse_loss_warmup_epochs", 15)
                        ),
                        "inverse_loss_max_weight": float(
                            train_cfg.get("inverse_loss_max_weight", 0.05)
                        ),
                    }

                    # Prepare rollout targets if needed
                    rollout_pred = None
                    rollout_tgt = None
                    if lam_rollout > 0.0 and future is not None and future.numel() > 0:
                        rollout_targets = future.to(device)
                        rollout_state = next_state
                        rollout_preds = []
                        steps = rollout_targets.shape[1]
                        for step in range(steps):
                            rollout_state = operator(rollout_state, dt_tensor)
                            rollout_preds.append(rollout_state.z)
                        rollout_pred = torch.stack(rollout_preds, dim=1)  # (B, steps, tokens, dim)
                        rollout_tgt = rollout_targets

                    # Compute loss bundle (handles optional inverse losses)
                    # Optionally subsample inverse loss computation to reduce overhead
                    inv_freq = int(train_cfg.get("inverse_loss_frequency", 1) or 1)
                    use_inv_now = use_inverse_losses and (inv_freq > 0) and (i % inv_freq == 0)

                    # Get grid shape from meta if available
                    grid_shape = meta.get("grid_shape", None) if meta else None

                    # Decode fields for physics checks if physics priors enabled
                    decoded_fields = None
                    reference_fields = None
                    if use_physics_priors and any(
                        physics_weights[k] > 0
                        for k in [
                            "lambda_divergence",
                            "lambda_conservation",
                            "lambda_boundary",
                            "lambda_positivity",
                        ]
                    ):
                        # Decode current latent state to physical space
                        if decoder is not None and coords is not None:
                            with torch.no_grad():
                                decoded_fields = decoder(coords, state.z)

                            # Store reference fields at first batch (for conservation checks)
                            if not reference_fields_storage and input_fields_physical is not None:
                                reference_fields_storage.update(
                                    {
                                        k: v.detach().clone()
                                        for k, v in input_fields_physical.items()
                                    }
                                )
                            reference_fields = reference_fields_storage

                    # Use physics-aware loss bundle if physics priors enabled, otherwise standard
                    if use_physics_priors:
                        from ups.training.losses import compute_operator_loss_bundle_with_physics

                        loss_bundle = compute_operator_loss_bundle_with_physics(
                            # Inverse encoding inputs (optional)
                            input_fields=input_fields_physical if use_inv_now else None,
                            encoded_latent=state.z if use_inv_now else None,
                            decoder=decoder if use_inv_now else None,
                            input_positions=coords if use_inv_now else None,
                            # Inverse decoding inputs (optional)
                            encoder=encoder if use_inv_now else None,
                            query_positions=coords if use_inv_now else None,
                            coords=coords if use_inv_now else None,
                            meta=meta if use_inv_now else None,
                            # Forward prediction (always)
                            pred_next=next_state.z,
                            target_next=target,
                            # Rollout (optional)
                            pred_rollout=rollout_pred,
                            target_rollout=rollout_tgt,
                            # Spectral (optional)
                            spectral_pred=next_state.z if lam_spec > 0 else None,
                            spectral_target=target if lam_spec > 0 else None,
                            # Weights
                            weights=loss_weights,
                            # Curriculum learning
                            current_epoch=epoch,
                            # Query sampling parameters
                            num_queries=num_queries,
                            query_strategy=query_strategy,
                            grid_shape=grid_shape,
                            # Physics prior arguments
                            decoded_fields=decoded_fields,
                            decoded_coords=coords,
                            reference_fields=reference_fields,
                            physics_weights=physics_weights,
                        )
                    else:
                        from ups.training.losses import compute_operator_loss_bundle

                        loss_bundle = compute_operator_loss_bundle(
                            # Inverse encoding inputs (optional)
                            input_fields=input_fields_physical if use_inv_now else None,
                            encoded_latent=state.z if use_inv_now else None,
                            decoder=decoder if use_inv_now else None,
                            input_positions=coords if use_inv_now else None,
                            # Inverse decoding inputs (optional)
                            encoder=encoder if use_inv_now else None,
                            query_positions=coords if use_inv_now else None,
                            coords=coords if use_inv_now else None,
                            meta=meta if use_inv_now else None,
                            # Forward prediction (always)
                            pred_next=next_state.z,
                            target_next=target,
                            # Rollout (optional)
                            pred_rollout=rollout_pred,
                            target_rollout=rollout_tgt,
                            # Spectral (optional)
                            spectral_pred=next_state.z if lam_spec > 0 else None,
                            spectral_target=target if lam_spec > 0 else None,
                            # Weights
                            weights=loss_weights,
                            # Curriculum learning
                            current_epoch=epoch,
                            # Query sampling parameters
                            num_queries=num_queries,
                            query_strategy=query_strategy,
                            grid_shape=grid_shape,
                        )

                    loss = loss_bundle.total

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    batch_failed = True
                    maybe_empty_cache(True)
                    if rank == 0:
                        print(
                            "Warning: OOM encountered in operator step, skipping batch (all ranks)"
                        )
                else:
                    raise

            skip_batch = sync_error_flag(batch_failed, device, is_distributed)
            if skip_batch:
                if rank == 0 and not batch_failed:
                    print(
                        "Warning: Remote OOM detected in operator step, skipping batch (all ranks)"
                    )
                optimizer.zero_grad(set_to_none=True)
                continue

            loss_value = loss.detach().item()

            # Accumulate per-task metrics (if enabled and task_names available)
            if task_names is not None and train_cfg.get("log_per_task_metrics", False):
                # Count samples per task in this batch
                task_counts = defaultdict(int)
                for task_name in task_names:
                    if task_name:
                        task_counts[task_name] += 1

                # Attribute loss to each task (simple approach: batch loss weighted by sample count)
                batch_size = len(task_names)
                for task_name, count in task_counts.items():
                    task_metrics[task_name]["loss"].append(loss_value)
                    task_metrics[task_name]["count"] += count

            # Log individual loss components to WandB
            if wandb_ctx and i % 10 == 0:  # Log every 10 batches
                for name, value in loss_bundle.components.items():
                    wandb_ctx.log_training_metric(
                        "operator", name, value.item(), step=logger.get_global_step()
                    )
            if use_scaler:
                scaler.scale(loss / accum_steps).backward()
            else:
                (loss / accum_steps).backward()
            do_step = ((i + 1) % accum_steps == 0) or ((i + 1) == num_batches)
            if do_step:
                if use_scaler:
                    if clip_val is not None:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        operator.parameters(), float("inf") if clip_val is None else clip_val
                    )
                    total_grad_norm += float(grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        operator.parameters(), float("inf") if clip_val is None else clip_val
                    )
                    total_grad_norm += grad_norm.item()
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                grad_steps += 1
                if ema_model is not None and ema_decay is not None:
                    _update_ema(ema_model, operator, ema_decay)
            epoch_loss += loss_value
            batches += 1

        print(f"[TRAIN-DEBUG] Finished epoch {epoch+1} on rank {rank}: processed {batches} batches, batch_count={batch_count}")

        epoch_time = time.time() - epoch_start
        mean_loss = epoch_loss / max(batches, 1)
        mean_grad_norm = total_grad_norm / max(grad_steps, 1)

        # Aggregate metrics across ranks in distributed mode
        metrics = aggregate_metrics(
            {
                "loss": mean_loss,
                "grad_norm": mean_grad_norm,
                "epoch_time": epoch_time,
            },
            world_size,
            rank,
        )
        mean_loss = metrics["loss"]
        mean_grad_norm = metrics["grad_norm"]
        epoch_time = metrics["epoch_time"]

        logger.log(
            epoch=epoch,
            loss=mean_loss,
            optimizer=optimizer,
            patience_counter=epochs_since_improve,
            grad_norm=mean_grad_norm,
            epoch_time=epoch_time,
            best_loss=best_loss,
        )

        # Log per-task metrics at epoch end (if enabled)
        if wandb_ctx and train_cfg.get("log_per_task_metrics", False):
            for task_name, metrics in task_metrics.items():
                if metrics["count"] > 0:
                    # Average loss for this task across the epoch
                    avg_loss = (
                        sum(metrics["loss"]) / len(metrics["loss"]) if metrics["loss"] else 0.0
                    )
                    wandb_ctx.log_training_metric(
                        "operator", f"{task_name}/loss", avg_loss, step=epoch
                    )
                    wandb_ctx.log_training_metric(
                        "operator", f"{task_name}/sample_count", metrics["count"], step=epoch
                    )

        if mean_loss + 1e-6 < best_loss:
            best_loss = mean_loss
            # Save unwrapped state to avoid DDP "module." prefix issues
            best_state = copy.deepcopy(
                operator.module.state_dict() if is_distributed else operator.state_dict()
            )
            if ema_model is not None:
                best_ema_state = copy.deepcopy(ema_model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if _should_stop(patience, epochs_since_improve):
                break
        if scheduler is not None:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(mean_loss)
            else:
                scheduler.step()
    # Load best state (DDP-aware)
    if is_distributed:
        operator.module.load_state_dict(best_state)
    else:
        operator.load_state_dict(best_state)

    logger.close()
    checkpoint_dir = ensure_checkpoint_dir(cfg)

    # Save checkpoints using FSDP-aware save function (rank 0 only in distributed mode)
    operator_path = checkpoint_dir / "operator.pt"
    save_checkpoint_fsdp(operator, operator_path, is_distributed, rank)
    if rank == 0:
        print(f"Saved operator checkpoint to {operator_path}")

        if ema_model is not None:
            operator_ema_path = checkpoint_dir / "operator_ema.pt"
            to_save = best_ema_state if best_ema_state is not None else ema_model.state_dict()
            torch.save(to_save, operator_ema_path)
            print(f"Saved operator EMA checkpoint to {operator_ema_path}")

    # Upload checkpoint to W&B (rank 0 only)
    if wandb_ctx and (not is_distributed or rank == 0):
        wandb_ctx.save_file(operator_path)
        print("Uploaded operator checkpoint to W&B")
        if ema_model is not None:
            wandb_ctx.save_file(operator_ema_path)

    # Send W&B alert (rank 0 only)
    if wandb_ctx and (not is_distributed or rank == 0):
        wandb_ctx.alert(
            title="✅ Operator Training Complete",
            text=f"Final loss: {best_loss:.6f} | Ready for diffusion stage",
            level="INFO",
        )


def train_diffusion(cfg: dict, wandb_ctx=None, global_step: int = 0) -> None:
    loader = dataset_loader(cfg)
    checkpoint_dir = ensure_checkpoint_dir(cfg)

    # Setup distributed training
    device, is_distributed, rank, world_size, local_rank = setup_distributed()

    # Create operator and load checkpoint directly to target device
    operator = make_operator(cfg)
    op_path = checkpoint_dir / "operator.pt"
    if op_path.exists():
        operator_state = torch.load(op_path, map_location="cpu")
        operator_state = _strip_compiled_prefix(operator_state)
        operator.load_state_dict(operator_state)
    _ensure_model_on_device(operator, device)

    # Wrap operator with FSDP2 or DDP if distributed (teacher model)
    if is_distributed:
        use_fsdp = cfg.get("training", {}).get("use_fsdp2", False)
        num_gpus = cfg.get("training", {}).get("num_gpus", 1)

        # Use FSDP2 for 4+ GPUs if enabled
        if use_fsdp and num_gpus >= 2:
            if rank == 0:
                print(f"Using FSDP2 for operator (teacher), {num_gpus}-GPU distributed training")
            operator = setup_fsdp2(operator, cfg, local_rank)
        else:
            from torch.nn.parallel import DistributedDataParallel as DDP

            if rank == 0:
                print("Using DDP for operator (teacher)")
            operator = DDP(
                operator,
                device_ids=[local_rank],
                output_device=local_rank,
                static_graph=True,
                find_unused_parameters=False,
            )
            if rank == 0:
                print(f"Operator (teacher) wrapped with DDP on device {local_rank}")

    operator = _maybe_compile(operator, cfg, "operator_teacher")
    operator.eval()

    latent_dim = cfg.get("latent", {}).get("dim", 32)
    stage_cfg = cfg.get("stages", {}).get("diff_residual", {})
    # Read hidden_dim from config, fallback to latent_dim * 2 for backward compatibility
    hidden_dim = cfg.get("diffusion", {}).get("hidden_dim", latent_dim * 2)
    diff = DiffusionResidual(DiffusionResidualConfig(latent_dim=latent_dim, hidden_dim=hidden_dim))
    _ensure_model_on_device(diff, device)

    # Wrap diffusion with FSDP2 or DDP if distributed
    if is_distributed:
        use_fsdp = cfg.get("training", {}).get("use_fsdp2", False)
        num_gpus = cfg.get("training", {}).get("num_gpus", 1)

        # Use FSDP2 for 4+ GPUs if enabled
        if use_fsdp and num_gpus >= 2:
            if rank == 0:
                print(f"Using FSDP2 for diffusion, {num_gpus}-GPU distributed training")
            diff = setup_fsdp2(diff, cfg, local_rank)
        else:
            from torch.nn.parallel import DistributedDataParallel as DDP

            if rank == 0:
                print("Using DDP for diffusion")
            diff = DDP(
                diff,
                device_ids=[local_rank],
                output_device=local_rank,
                static_graph=True,
            find_unused_parameters=False,
        )
        if rank == 0:
            print(f"Diffusion model wrapped with DDP on device {local_rank}")

    diff = _maybe_compile(diff, cfg, "diffusion_residual")

    optimizer = _create_optimizer(cfg, diff, "diff_residual")
    scheduler = _create_scheduler(optimizer, cfg, "diff_residual")
    patience = _get_patience(cfg, "diff_residual")
    dt = cfg.get("training", {}).get("dt", 0.1)
    epochs = stage_cfg.get("epochs", 1)
    checkpoint_interval = int(cfg.get("training", {}).get("checkpoint_interval", 0) or 0)
    logger = TrainingLogger(
        cfg, stage="diffusion_residual", global_step=global_step, wandb_ctx=wandb_ctx
    )
    dt_tensor = torch.tensor(dt, device=device)
    best_loss = float("inf")
    # Save unwrapped state to avoid DDP "module." prefix issues
    best_state = copy.deepcopy(diff.module.state_dict() if is_distributed else diff.state_dict())
    # AMP + EMA setup
    use_amp = _amp_enabled(cfg)
    autocast_dtype, use_scaler = _get_amp_dtype(cfg)
    scaler = GradScaler(enabled=use_scaler)
    ema_decay = _get_ema_decay(cfg, "diff_residual")
    ema_model = _init_ema(diff) if ema_decay else None
    best_ema_state = copy.deepcopy(ema_model.state_dict()) if ema_model is not None else None
    clip_val = _grad_clip_value(cfg, "diff_residual")
    epochs_since_improve = 0

    import time

    accum_steps = max(1, int(cfg.get("training", {}).get("accum_steps", 1)))
    lam_spec = float(cfg.get("training", {}).get("lambda_spectral", 0.0) or 0.0)
    lam_rel = float(cfg.get("training", {}).get("lambda_relative", 0.0) or 0.0)
    for epoch in range(epochs):
        # For distributed training, set epoch for sampler (enables shuffling)
        if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        epoch_start = time.time()
        epoch_loss = 0.0
        total_grad_norm = 0.0
        batches = 0
        grad_steps = 0
        optimizer.zero_grad(set_to_none=True)
        num_batches = len(loader)
        for i, batch in enumerate(loader):
            unpacked = unpack_batch(batch)
            if isinstance(unpacked, dict):
                z0 = unpacked["z0"]
                z1 = unpacked["z1"]
                cond = unpacked.get("cond", {})
            elif len(unpacked) == 4:
                z0, z1, cond, _ = unpacked
            else:
                z0, z1, cond = unpacked
            cond_device = {k: v.to(device) for k, v in cond.items()}
            state = LatentState(
                z=z0.to(device), t=torch.tensor(0.0, device=device), cond=cond_device
            )
            target = z1.to(device)
            maybe_trigger_simulated_oom("diffusion", i, rank)
            batch_failed = False
            try:
                with torch.no_grad():
                    predicted = operator(state, dt_tensor)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    batch_failed = True
                    maybe_empty_cache(True)
                    if rank == 0:
                        print(
                            "Warning: OOM encountered in operator forward (teacher), skipping batch (all ranks)"
                        )
                else:
                    raise
            residual_target = None
            loss = None
            if not batch_failed:
                residual_target = target - predicted.z
                tau_tensor = _sample_tau(z0.size(0), device, cfg)
                try:
                    with autocast(enabled=use_amp, dtype=autocast_dtype):
                        drift = diff(predicted, tau_tensor)
                        base = F.mse_loss(drift, residual_target)
                        extra = 0.0
                        if lam_spec > 0.0:
                            extra = extra + lam_spec * _spectral_energy_loss(
                                drift, residual_target, dim=1
                            )
                        if lam_rel > 0.0:
                            extra = extra + lam_rel * _nrmse(drift, residual_target)
                        loss = base + extra
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        batch_failed = True
                        maybe_empty_cache(True)
                        if rank == 0:
                            print(
                                "Warning: OOM encountered in diffusion step, skipping batch (all ranks)"
                            )
                    else:
                        raise
            skip_batch = sync_error_flag(batch_failed, device, is_distributed)
            if skip_batch:
                if rank == 0 and not batch_failed:
                    print(
                        "Warning: Remote OOM encountered in diffusion step, skipping batch (all ranks)"
                    )
                optimizer.zero_grad(set_to_none=True)
                continue
            loss_value = loss.detach().item()
            if use_scaler:
                scaler.scale(loss / accum_steps).backward()
            else:
                (loss / accum_steps).backward()
            do_step = ((i + 1) % accum_steps == 0) or ((i + 1) == num_batches)
            if do_step:
                if use_scaler:
                    if clip_val is not None:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        diff.parameters(), float("inf") if clip_val is None else clip_val
                    )
                    total_grad_norm += float(grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        diff.parameters(), float("inf") if clip_val is None else clip_val
                    )
                    total_grad_norm += grad_norm.item()
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                grad_steps += 1
                if ema_model is not None and ema_decay is not None:
                    _update_ema(ema_model, diff, ema_decay)
            epoch_loss += loss.item()
            batches += 1

        epoch_time = time.time() - epoch_start
        mean_loss = epoch_loss / max(batches, 1)
        mean_grad_norm = total_grad_norm / max(grad_steps, 1)

        # Aggregate metrics across ranks in distributed mode
        metrics = aggregate_metrics(
            {
                "loss": mean_loss,
                "grad_norm": mean_grad_norm,
                "epoch_time": epoch_time,
            },
            world_size,
            rank,
        )
        mean_loss = metrics["loss"]
        mean_grad_norm = metrics["grad_norm"]
        epoch_time = metrics["epoch_time"]

        logger.log(
            epoch=epoch,
            loss=mean_loss,
            optimizer=optimizer,
            patience_counter=epochs_since_improve,
            grad_norm=mean_grad_norm,
            epoch_time=epoch_time,
            best_loss=best_loss,
        )
        if mean_loss + 1e-6 < best_loss:
            best_loss = mean_loss
            # Save unwrapped state to avoid DDP "module." prefix issues
            best_state = copy.deepcopy(
                diff.module.state_dict() if is_distributed else diff.state_dict()
            )
            if ema_model is not None:
                best_ema_state = copy.deepcopy(ema_model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if _should_stop(patience, epochs_since_improve):
                break
        if scheduler is not None:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(mean_loss)
            else:
                scheduler.step()

        # Intermediate checkpoints (rank 0 only)
        if checkpoint_interval and (epoch + 1) % checkpoint_interval == 0:
            if not is_distributed or rank == 0:
                epoch_ckpt = checkpoint_dir / f"diffusion_residual_epoch_{epoch + 1}.pt"
                model_state = diff.module.state_dict() if is_distributed else diff.state_dict()
                torch.save(model_state, epoch_ckpt)
                if ema_model is not None:
                    ema_epoch_ckpt = checkpoint_dir / f"diffusion_residual_ema_epoch_{epoch + 1}.pt"
                    torch.save(ema_model.state_dict(), ema_epoch_ckpt)

    # Load best state (DDP-aware)
    if is_distributed:
        diff.module.load_state_dict(best_state)
    else:
        diff.load_state_dict(best_state)

    logger.close()

    # Save final checkpoints (rank 0 only)
    if not is_distributed or rank == 0:
        diffusion_path = checkpoint_dir / "diffusion_residual.pt"
        model_state = diff.module.state_dict() if is_distributed else diff.state_dict()
        torch.save(model_state, diffusion_path)
        print(f"Saved diffusion residual checkpoint to {diffusion_path}")

        if ema_model is not None:
            diffusion_ema_path = checkpoint_dir / "diffusion_residual_ema.pt"
            torch.save(
                best_ema_state if best_ema_state is not None else ema_model.state_dict(),
                diffusion_ema_path,
            )
            print(f"Saved diffusion EMA checkpoint to {diffusion_ema_path}")

    # Upload checkpoint to W&B (rank 0 only)
    if wandb_ctx and (not is_distributed or rank == 0):
        wandb_ctx.save_file(diffusion_path)
        print("Uploaded diffusion checkpoint to W&B")
        if ema_model is not None:
            wandb_ctx.save_file(diffusion_ema_path)

    # Send W&B alert (rank 0 only)
    if wandb_ctx and (not is_distributed or rank == 0):
        wandb_ctx.alert(
            title="✅ Diffusion Residual Training Complete",
            text=f"Final loss: {best_loss:.6f} | Ready for consistency distillation",
            level="INFO",
        )


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


def _distill_forward_and_loss_compiled(
    teacher_z_chunk: torch.Tensor,
    teacher_cond_chunk: dict,
    num_taus: int,
    diff_model: nn.Module,
    t_value: torch.Tensor,
    tau_seed: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """OPTIMIZATION #1: Compiled distillation forward pass.

    Fuses tau expansion + diffusion forward + loss into single compiled graph.
    Expected speedup: ~1.3-1.5x via kernel fusion and reduced Python overhead.
    """
    Bc, T, D = teacher_z_chunk.shape

    # Tau expansion
    z_tiled = (
        teacher_z_chunk.unsqueeze(1)
        .expand(Bc, num_taus, T, D)
        .reshape(Bc * num_taus, T, D)
        .contiguous()
    )

    cond_tiled = {k: v.repeat_interleave(num_taus, dim=0) for k, v in teacher_cond_chunk.items()}

    tau_flat = tau_seed.repeat(Bc).to(z_tiled.dtype)

    # Diffusion forward
    tiled_state = LatentState(z=z_tiled, t=t_value, cond=cond_tiled)
    drift = diff_model(tiled_state, tau_flat)

    # Loss computation
    z_tiled_cast = z_tiled.to(drift.dtype)
    student_z = z_tiled_cast + drift
    loss = torch.nn.functional.mse_loss(student_z, z_tiled_cast)

    return loss


# Distillation function - will be optionally compiled based on config
_distill_forward_and_loss = _distill_forward_and_loss_compiled


def train_consistency(cfg: dict, wandb_ctx=None, global_step: int = 0) -> None:
    # Use smaller batch size for consistency stage to avoid OOM
    # This stage needs both operator and diffusion models loaded
    cfg_copy = copy.deepcopy(cfg)
    original_batch_size = cfg_copy.get("training", {}).get("batch_size", 32)
    consistency_batch_size = (
        cfg_copy.get("stages", {}).get("consistency_distill", {}).get("batch_size", 8)
    )
    training_cfg_copy = cfg_copy.setdefault("training", {})
    training_cfg_copy["batch_size"] = consistency_batch_size
    # Enable pin_memory for faster transfers
    training_cfg_copy["pin_memory"] = True
    # OPTIMIZATION #3: Persistent workers to avoid respawning overhead
    training_cfg_copy["persistent_workers"] = (
        True if training_cfg_copy.get("num_workers", 0) > 0 else False
    )
    training_cfg_copy["prefetch_factor"] = 4  # Increase prefetch

    loader = dataset_loader(cfg_copy)
    checkpoint_dir = ensure_checkpoint_dir(cfg)

    # Setup distributed training
    device, is_distributed, rank, world_size, local_rank = setup_distributed()

    # Create operator and load checkpoint directly to target device
    operator = make_operator(cfg)
    op_path = checkpoint_dir / "operator.pt"
    if op_path.exists():
        operator_state = torch.load(op_path, map_location="cpu")
        operator_state = _strip_compiled_prefix(operator_state)
        operator.load_state_dict(operator_state)
    _ensure_model_on_device(operator, device)

    # Wrap operator with FSDP2 or DDP if distributed (teacher model)
    if is_distributed:
        use_fsdp = cfg.get("training", {}).get("use_fsdp2", False)
        num_gpus = cfg.get("training", {}).get("num_gpus", 1)

        # Use FSDP2 for 4+ GPUs if enabled
        if use_fsdp and num_gpus >= 2:
            if rank == 0:
                print(f"Using FSDP2 for operator (teacher) in consistency stage, {num_gpus}-GPU distributed training")
            operator = setup_fsdp2(operator, cfg, local_rank)
        else:
            from torch.nn.parallel import DistributedDataParallel as DDP

            if rank == 0:
                print("Using DDP for operator (teacher) in consistency stage")
            operator = DDP(
                operator,
                device_ids=[local_rank],
                output_device=local_rank,
                static_graph=True,
                find_unused_parameters=False,
            )
            if rank == 0:
                print(f"Operator (teacher) wrapped with DDP on device {local_rank}")

    operator = _maybe_compile(operator, cfg, "operator_teacher")
    operator.eval()

    # Create diffusion model and load checkpoint directly to target device
    latent_dim = cfg.get("latent", {}).get("dim", 32)
    stage_cfg = cfg.get("stages", {}).get("consistency_distill", {})
    tau_schedule = stage_cfg.get("tau_schedule")
    target_loss = float(
        stage_cfg.get("target_loss")
        or cfg.get("training", {}).get("distill_target_loss", 0.0)
        or 0.0
    )
    # Read hidden_dim from config, fallback to latent_dim * 2 for backward compatibility
    hidden_dim = cfg.get("diffusion", {}).get("hidden_dim", latent_dim * 2)
    diff = DiffusionResidual(DiffusionResidualConfig(latent_dim=latent_dim, hidden_dim=hidden_dim))
    diff_path = checkpoint_dir / "diffusion_residual.pt"
    if diff_path.exists():
        diff_state = torch.load(diff_path, map_location="cpu")
        diff_state = _strip_compiled_prefix(diff_state)
        diff.load_state_dict(diff_state)
    _ensure_model_on_device(diff, device)

    # Wrap diffusion with FSDP2 or DDP if distributed
    if is_distributed:
        use_fsdp = cfg.get("training", {}).get("use_fsdp2", False)
        num_gpus = cfg.get("training", {}).get("num_gpus", 1)

        # Use FSDP2 for 4+ GPUs if enabled
        if use_fsdp and num_gpus >= 2:
            if rank == 0:
                print(f"Using FSDP2 for diffusion in consistency stage, {num_gpus}-GPU distributed training")
            diff = setup_fsdp2(diff, cfg, local_rank)
        else:
            from torch.nn.parallel import DistributedDataParallel as DDP

            if rank == 0:
                print("Using DDP for diffusion in consistency stage")
            diff = DDP(
                diff,
                device_ids=[local_rank],
                output_device=local_rank,
                static_graph=True,
                find_unused_parameters=False,
            )
            if rank == 0:
                print(f"Diffusion model wrapped with DDP on device {local_rank}")

    diff = _maybe_compile(diff, cfg, "diffusion_residual")

    print("Consistency distillation optimizations enabled:")
    print("  - Async GPU transfers: enabled")
    print(f"  - Adaptive tau schedule: {tau_schedule if tau_schedule else 'using base num_taus'}")
    print(f"  - Micro-batch size: {cfg.get('training', {}).get('distill_micro_batch', 'auto')}")

    epochs = stage_cfg.get("epochs", 1)
    optimizer = _create_optimizer(cfg, diff, "consistency_distill")
    scheduler = _create_scheduler(optimizer, cfg, "consistency_distill")
    patience = _get_patience(cfg, "consistency_distill")
    logger = TrainingLogger(
        cfg, stage="consistency_distill", global_step=global_step, wandb_ctx=wandb_ctx
    )
    dt = cfg.get("training", {}).get("dt", 0.1)

    dt_tensor = torch.tensor(dt, device=device)

    # Teacher/student are inlined below to enable reuse and vectorized taus

    best_loss = float("inf")
    # Save unwrapped state to avoid DDP "module." prefix issues
    best_state = copy.deepcopy(diff.module.state_dict() if is_distributed else diff.state_dict())
    use_amp = _amp_enabled(cfg)
    autocast_dtype, use_scaler = _get_amp_dtype(cfg)
    scaler = GradScaler(enabled=use_scaler)
    ema_decay = _get_ema_decay(cfg, "consistency_distill")
    ema_model = _init_ema(diff) if ema_decay else None
    clip_val = _grad_clip_value(cfg, "consistency_distill")
    epochs_since_improve = 0

    # Get micro-batch size for gradient accumulation
    distill_micro = cfg.get("training", {}).get("distill_micro_batch")
    base_num_taus = int(cfg.get("training", {}).get("distill_num_taus", 3) or 3)

    # Conditionally compile the distillation function if enabled
    # Allow per-stage override via stages.consistency_distill.compile
    distill_fn = _distill_forward_and_loss  # Default to uncompiled
    distill_compile_enabled = False

    # Check stage-specific compile setting first, fall back to global training.compile
    stage_compile = cfg.get("stages", {}).get("consistency_distill", {}).get("compile")
    global_compile = cfg.get("training", {}).get("compile", False)
    should_compile = stage_compile if stage_compile is not None else global_compile

    if should_compile:
        try:
            # Use "default" mode instead of "reduce-overhead" to avoid CUDA graph bugs
            # CUDA graphs can cause "accessing tensor output that has been overwritten" errors
            # when tensors are reused across iterations (issue #27338341)
            distill_fn = torch.compile(
                _distill_forward_and_loss,
                mode="default",  # Safe mode without CUDA graphs
                fullgraph=False,
            )
            distill_compile_enabled = True
            print("✓ torch.compile enabled for consistency distillation function (safe mode)")
        except Exception as e:
            print(f"⚠ torch.compile failed for distill function: {e}")
            distill_fn = _distill_forward_and_loss

    # Log optimizations applied
    print("Consistency distillation optimizations:")
    print("  - Teacher caching: ENABLED (computed once per batch)")
    print(f"  - AMP for teacher: {'ENABLED' if use_amp else 'DISABLED'}")
    print("  - Async GPU transfers: ENABLED")
    print(f"  - torch.compile: {'ENABLED' if distill_compile_enabled else 'DISABLED'}")
    print(f"  - Persistent workers: {training_cfg_copy.get('persistent_workers', False)}")
    print(f"  - Prefetch factor: {training_cfg_copy.get('prefetch_factor', 2)}")
    print(f"  - Micro-batch size: {distill_micro}")
    print(f"  - Base num taus: {base_num_taus}")
    if tau_schedule:
        print(f"  - Tau schedule: {tau_schedule}")

    import time

    for epoch in range(epochs):
        # For distributed training, set epoch for sampler (enables shuffling)
        if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        epoch_start = time.time()
        epoch_loss = 0.0
        total_grad_norm = 0.0
        batches = 0

        num_taus_epoch = base_num_taus
        if tau_schedule:
            idx = min(epoch, len(tau_schedule) - 1)
            scheduled = tau_schedule[idx]
            if scheduled:
                num_taus_epoch = int(scheduled)

        for batch_idx, batch in enumerate(loader):
            unpacked = unpack_batch(batch)
            if isinstance(unpacked, dict):
                z0 = unpacked["z0"]
                cond = unpacked.get("cond", {})
            elif len(unpacked) == 4:
                z0, _, cond, _ = unpacked
            else:
                z0, _, cond = unpacked
            batch_size = z0.shape[0]
            micro = distill_micro or batch_size
            optimizer.zero_grad(set_to_none=True)
            batch_loss_value = 0.0
            maybe_trigger_simulated_oom("consistency", batch_idx, rank)
            batch_failed = False

            # OPTIMIZATION #1: Compute teacher predictions ONCE per batch (outside micro-batch loop)
            # Moves teacher forward from inside loop (called N times) to outside (called 1 time)
            # Expected speedup: ~2x (reduces teacher calls by 50% when micro < batch_size)
            z0_device = z0.to(device, non_blocking=True)
            cond_device = {k: v.to(device, non_blocking=True) for k, v in cond.items()}
            full_batch_state = LatentState(
                z=z0_device, t=torch.tensor(0.0, device=device), cond=cond_device
            )

            # OPTIMIZATION #6: Use AMP for teacher forward (even though no gradients)
            # Reduces teacher forward time by ~20%, overall ~8% speedup
            try:
                with torch.no_grad(), autocast(enabled=use_amp, dtype=autocast_dtype):
                    teacher_full = operator(full_batch_state, dt_tensor)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    batch_failed = True
                    maybe_empty_cache(True)
                    if rank == 0:
                        print(
                            "Warning: OOM in consistency distill teacher forward, skipping batch (all ranks)"
                        )
                else:
                    raise

            for start in range(0, batch_size, micro):
                if batch_failed:
                    break
                end = min(start + micro, batch_size)
                chunk_weight = (end - start) / batch_size

                # Slice pre-computed teacher predictions
                teacher_z_chunk = teacher_full.z[start:end]
                teacher_cond_chunk = {k: v[start:end] for k, v in teacher_full.cond.items()}

                try:
                    # Sample tau values
                    tau_seed = _sample_tau(num_taus_epoch, device, cfg)

                    # Use (optionally compiled) forward function
                    with autocast(enabled=use_amp, dtype=autocast_dtype):
                        loss_chunk = distill_fn(
                            teacher_z_chunk,
                            teacher_cond_chunk,
                            num_taus_epoch,
                            diff,
                            teacher_full.t,
                            tau_seed,
                            device,
                        )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        batch_failed = True
                        maybe_empty_cache(True)
                        if rank == 0:
                            print(
                                "Warning: OOM in consistency distill chunk, skipping chunk (all ranks)"
                            )
                        break
                    else:
                        raise
                if batch_failed:
                    break
                if use_scaler:
                    scaler.scale(loss_chunk * chunk_weight).backward()
                else:
                    (loss_chunk * chunk_weight).backward()
                batch_loss_value += loss_chunk.item() * chunk_weight
            skip_batch = sync_error_flag(batch_failed, device, is_distributed)
            if skip_batch:
                if rank == 0 and not batch_failed:
                    print(
                        "Warning: Remote OOM in consistency distillation, skipping batch (all ranks)"
                    )
                optimizer.zero_grad(set_to_none=True)
                continue
            if use_scaler:
                if clip_val is not None:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    diff.parameters(), float("inf") if clip_val is None else clip_val
                )
                total_grad_norm += float(grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    diff.parameters(), float("inf") if clip_val is None else clip_val
                )
                total_grad_norm += grad_norm.item()
                optimizer.step()
            if ema_model is not None and ema_decay is not None:
                _update_ema(ema_model, diff, ema_decay)
            epoch_loss += batch_loss_value
            batches += 1

        epoch_time = time.time() - epoch_start
        mean_loss = epoch_loss / max(batches, 1)
        mean_grad_norm = total_grad_norm / max(batches, 1)

        # Aggregate metrics across ranks in distributed mode
        metrics = aggregate_metrics(
            {
                "loss": mean_loss,
                "grad_norm": mean_grad_norm,
                "epoch_time": epoch_time,
            },
            world_size,
            rank,
        )
        mean_loss = metrics["loss"]
        mean_grad_norm = metrics["grad_norm"]
        epoch_time = metrics["epoch_time"]

        logger.log(
            epoch=epoch,
            loss=mean_loss,
            optimizer=optimizer,
            patience_counter=epochs_since_improve,
            grad_norm=mean_grad_norm,
            epoch_time=epoch_time,
            best_loss=best_loss,
        )
        if mean_loss + 1e-6 < best_loss:
            best_loss = mean_loss
            # Save unwrapped state to avoid DDP "module." prefix issues
            best_state = copy.deepcopy(
                diff.module.state_dict() if is_distributed else diff.state_dict()
            )
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if _should_stop(patience, epochs_since_improve):
                break
        if target_loss and best_loss <= target_loss:
            print(
                f"Consistency distill reached target loss {best_loss:.6f} <= {target_loss:.6f}; stopping early"
            )
            break
        if scheduler is not None:
            scheduler.step()
    # Load best state (DDP-aware)
    if is_distributed:
        diff.module.load_state_dict(best_state)
    else:
        diff.load_state_dict(best_state)

    logger.close()

    # Save checkpoints (rank 0 only)
    if not is_distributed or rank == 0:
        diffusion_path = checkpoint_dir / "diffusion_residual.pt"
        model_state = diff.module.state_dict() if is_distributed else diff.state_dict()
        torch.save(model_state, diffusion_path)
        print(f"Updated diffusion residual via consistency distillation to {diffusion_path}")

        if ema_model is not None:
            diffusion_ema_path = checkpoint_dir / "diffusion_residual_ema.pt"
            torch.save(ema_model.state_dict(), diffusion_ema_path)
            print(f"Saved diffusion EMA checkpoint to {diffusion_ema_path}")

    # Upload updated checkpoint to W&B (rank 0 only)
    if wandb_ctx and (not is_distributed or rank == 0):
        wandb_ctx.save_file(diffusion_path)
        print("Uploaded updated diffusion checkpoint to W&B")
        if ema_model is not None:
            wandb_ctx.save_file(diffusion_ema_path)

    # Clean up operator from memory
    del operator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Send W&B alert
    if wandb_ctx:
        wandb_ctx.alert(
            title="✅ Consistency Distillation Complete",
            text=f"Final loss: {best_loss:.6f} | Ready for steady prior training",
            level="INFO",
        )


def train_steady_prior(cfg: dict, wandb_ctx=None, global_step: int = 0) -> None:
    latent_dim = cfg.get("latent", {}).get("dim", 32)
    stage_cfg = cfg.get("stages", {}).get("steady_prior", {})
    epochs = stage_cfg.get("epochs", 0)

    # Early exit when disabled
    if epochs <= 0:
        print("Skipping steady_prior stage (epochs<=0)")
        return

    loader = dataset_loader(cfg)
    prior = SteadyPrior(
        SteadyPriorConfig(latent_dim=latent_dim, hidden_dim=latent_dim * 2, num_steps=4)
    )
    optimizer = _create_optimizer(cfg, prior, "steady_prior")
    scheduler = _create_scheduler(optimizer, cfg, "steady_prior")
    patience = _get_patience(cfg, "steady_prior")
    logger = TrainingLogger(cfg, stage="steady_prior", global_step=global_step, wandb_ctx=wandb_ctx)

    # Setup distributed training
    device, is_distributed, rank, world_size, local_rank = setup_distributed()

    prior.to(device)

    # Wrap prior with FSDP2 or DDP if distributed
    if is_distributed:
        use_fsdp = cfg.get("training", {}).get("use_fsdp2", False)
        num_gpus = cfg.get("training", {}).get("num_gpus", 1)

        # Use FSDP2 for 4+ GPUs if enabled
        if use_fsdp and num_gpus >= 2:
            if rank == 0:
                print(f"Using FSDP2 for steady prior, {num_gpus}-GPU distributed training")
            prior = setup_fsdp2(prior, cfg, local_rank)
        else:
            from torch.nn.parallel import DistributedDataParallel as DDP

            if rank == 0:
                print("Using DDP for steady prior")
            prior = DDP(
                prior,
                device_ids=[local_rank],
                output_device=local_rank,
                static_graph=True,
                find_unused_parameters=False,
            )
            if rank == 0:
                print(f"Steady prior wrapped with DDP on device {local_rank}")

    best_loss = float("inf")
    # Save unwrapped state to avoid DDP "module." prefix issues
    best_state = copy.deepcopy(prior.module.state_dict() if is_distributed else prior.state_dict())
    epochs_since_improve = 0

    import time

    accum_steps = max(1, int(cfg.get("training", {}).get("accum_steps", 1)))
    for epoch in range(epochs):
        # For distributed training, set epoch for sampler (enables shuffling)
        if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        epoch_start = time.time()
        epoch_loss = 0.0
        total_grad_norm = 0.0
        batches = 0
        grad_steps = 0
        optimizer.zero_grad(set_to_none=True)
        num_batches = len(loader)
        for i, batch in enumerate(loader):
            unpacked = unpack_batch(batch)
            if isinstance(unpacked, dict):
                z0 = unpacked["z0"]
                z1 = unpacked["z1"]
                cond = unpacked.get("cond", {})
            elif len(unpacked) == 4:
                z0, z1, cond, _ = unpacked
            else:
                z0, z1, cond = unpacked
            cond_device = {k: v.to(device) for k, v in cond.items()}
            state = LatentState(
                z=z0.to(device), t=torch.tensor(0.0, device=device), cond=cond_device
            )
            refined = prior(state)
            loss = F.mse_loss(refined.z, z1.to(device))
            (loss / accum_steps).backward()
            do_step = ((i + 1) % accum_steps == 0) or ((i + 1) == num_batches)
            if do_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(prior.parameters(), float("inf"))
                total_grad_norm += grad_norm.item()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                grad_steps += 1
            epoch_loss += loss.item()
            batches += 1

        epoch_time = time.time() - epoch_start
        mean_loss = epoch_loss / max(batches, 1)
        mean_grad_norm = total_grad_norm / max(grad_steps, 1)

        # Aggregate metrics across ranks in distributed mode
        metrics = aggregate_metrics(
            {
                "loss": mean_loss,
                "grad_norm": mean_grad_norm,
                "epoch_time": epoch_time,
            },
            world_size,
            rank,
        )
        mean_loss = metrics["loss"]
        mean_grad_norm = metrics["grad_norm"]
        epoch_time = metrics["epoch_time"]

        logger.log(
            epoch=epoch,
            loss=mean_loss,
            optimizer=optimizer,
            patience_counter=epochs_since_improve,
            grad_norm=mean_grad_norm,
            epoch_time=epoch_time,
            best_loss=best_loss,
        )
        if mean_loss + 1e-6 < best_loss:
            best_loss = mean_loss
            # Save unwrapped state to avoid DDP "module." prefix issues
            best_state = copy.deepcopy(
                prior.module.state_dict() if is_distributed else prior.state_dict()
            )
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if _should_stop(patience, epochs_since_improve):
                break
        if scheduler is not None:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(mean_loss)
            else:
                scheduler.step()
    # Load best state (DDP-aware)
    if is_distributed:
        prior.module.load_state_dict(best_state)
    else:
        prior.load_state_dict(best_state)

    logger.close()
    checkpoint_dir = ensure_checkpoint_dir(cfg)

    # Save checkpoint (rank 0 only)
    if not is_distributed or rank == 0:
        prior_path = checkpoint_dir / "steady_prior.pt"
        model_state = prior.module.state_dict() if is_distributed else prior.state_dict()
        torch.save(model_state, prior_path)
        print(f"Saved steady prior checkpoint to {prior_path}")

    # Upload checkpoint to W&B (rank 0 only)
    if wandb_ctx and (not is_distributed or rank == 0):
        wandb_ctx.save_file(prior_path)
        print("Uploaded steady prior checkpoint to W&B")

    # Send W&B alert (rank 0 only)
    if wandb_ctx and (not is_distributed or rank == 0):
        wandb_ctx.alert(
            title="🎉 All Training Stages Complete!",
            text=f"Steady prior final loss: {best_loss:.6f} | Full pipeline ready for evaluation",
            level="INFO",
        )


def _run_evaluation(
    cfg: dict, checkpoint_dir: Path, eval_mode: str = "baseline", wandb_ctx=None
) -> dict:
    """Run evaluation and return metrics. Mode can be 'baseline' or 'ttc'."""
    from ups.eval.pdebench_runner import evaluate_latent_operator
    from ups.inference.rollout_ttc import TTCConfig, build_reward_model_from_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    operator = make_operator(cfg)
    op_path = checkpoint_dir / "operator.pt"
    if op_path.exists():
        operator_state = torch.load(op_path, map_location="cpu")
        operator_state = _strip_compiled_prefix(operator_state)
        operator.load_state_dict(operator_state)
    operator = operator.to(device)
    operator.eval()

    # Get latent_dim for potential TTC reward model (needed regardless of diffusion)
    latent_dim = cfg.get("latent", {}).get("dim", 32)

    # Load diffusion if available
    diffusion = None
    diff_path = checkpoint_dir / "diffusion_residual.pt"
    if diff_path.exists():
        hidden_dim = cfg.get("diffusion", {}).get("hidden_dim", latent_dim * 2)
        diffusion = DiffusionResidual(
            DiffusionResidualConfig(latent_dim=latent_dim, hidden_dim=hidden_dim)
        )
        diff_state = torch.load(diff_path, map_location="cpu")
        diff_state = _strip_compiled_prefix(diff_state)
        diffusion.load_state_dict(diff_state)
        diffusion = diffusion.to(device)
        diffusion.eval()

    # Setup TTC if requested
    tau = cfg.get("training", {}).get("tau", 0.5)
    ttc_cfg = None
    reward_model = None

    if eval_mode == "ttc" and cfg.get("ttc", {}).get("enabled"):
        ttc_dict = cfg.get("ttc", {})
        # Use direct constructor like other places in codebase (evaluate.py, pdebench_runner.py)
        ttc_cfg = TTCConfig(
            steps=ttc_dict.get("steps", 1),
            dt=ttc_dict.get("dt", cfg.get("training", {}).get("dt", 0.1)),
            candidates=ttc_dict.get("candidates", 4),
            beam_width=ttc_dict.get("beam_width", 1),
            horizon=ttc_dict.get("horizon", 1),
            tau_range=(
                tuple(ttc_dict.get("tau_range", [0.3, 0.7]))
                if "tau_range" in ttc_dict
                else (0.3, 0.7)
            ),
            residual_threshold=ttc_dict.get("residual_threshold"),
            max_evaluations=ttc_dict.get("max_evaluations"),
            gamma=ttc_dict.get("gamma", 1.0),
            device=device,
        )
        reward_model = build_reward_model_from_config(ttc_dict, latent_dim, device)

    # Change data split to test for evaluation
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg["data"]["split"] = "test"

    # Run evaluation
    print(f"\nRunning evaluation (mode: {eval_mode})...")
    report, details = evaluate_latent_operator(
        cfg=eval_cfg,
        operator=operator,
        diffusion=diffusion,
        tau=tau,
        device=device,
        ttc_config=ttc_cfg,
        reward_model=reward_model,
        return_details=True,
    )

    # Add TTC flag to report
    report.extra["ttc"] = eval_mode == "ttc"

    return {"report": report, "details": details}


def _log_evaluation_summary(wandb_ctx, baseline_metrics: dict, ttc_metrics: dict = None) -> None:
    """Log evaluation results and summary to WandB using WandBContext.

    Args:
        wandb_ctx: WandBContext instance (or None to skip logging)
        baseline_metrics: Baseline evaluation results
        ttc_metrics: Optional TTC evaluation results
    """
    if not wandb_ctx or not wandb_ctx.enabled:
        return

    baseline_report = baseline_metrics["report"]

    # Log baseline metrics to SUMMARY (scalars, not time series!)
    baseline_vals = {
        "baseline_mse": baseline_report.metrics.get("mse"),
        "baseline_mae": baseline_report.metrics.get("mae"),
        "baseline_rmse": baseline_report.metrics.get("rmse"),
        "baseline_nrmse": baseline_report.metrics.get("nrmse"),
        "baseline_rel_l2": baseline_report.metrics.get("rel_l2"),
    }

    # Add physics metrics if present
    for physics_key in ["conservation_gap", "bc_violation", "negativity_penalty"]:
        if physics_key in baseline_report.metrics:
            baseline_vals[f"baseline_{physics_key}"] = baseline_report.metrics[physics_key]

    wandb_ctx.log_eval_summary(baseline_vals, prefix="eval")

    # Log TTC metrics if available
    ttc_improvement_pct = None
    if ttc_metrics:
        ttc_report = ttc_metrics["report"]
        ttc_vals = {
            "ttc_mse": ttc_report.metrics.get("mse"),
            "ttc_mae": ttc_report.metrics.get("mae"),
            "ttc_rmse": ttc_report.metrics.get("rmse"),
            "ttc_nrmse": ttc_report.metrics.get("nrmse"),
            "ttc_rel_l2": ttc_report.metrics.get("rel_l2"),
        }

        # Add TTC physics metrics
        for physics_key in ["conservation_gap", "bc_violation", "negativity_penalty"]:
            if physics_key in ttc_report.metrics:
                ttc_vals[f"ttc_{physics_key}"] = ttc_report.metrics[physics_key]

        wandb_ctx.log_eval_summary(ttc_vals, prefix="eval")

        # Compute TTC improvement
        baseline_nrmse = baseline_report.metrics.get("nrmse", 1.0)
        ttc_nrmse = ttc_report.metrics.get("nrmse", 1.0)
        ttc_improvement_pct = ((baseline_nrmse - ttc_nrmse) / baseline_nrmse) * 100

        wandb_ctx.log_eval_summary({"ttc_improvement_pct": ttc_improvement_pct}, prefix="eval")

    # Create summary table
    summary_md = "## Evaluation Summary\n\n"
    summary_md += "| Metric | Baseline | TTC | Improvement |\n"
    summary_md += "|--------|----------|-----|-------------|\n"

    for metric_name in ["mse", "mae", "rmse", "nrmse", "rel_l2"]:
        baseline_val = baseline_metrics["report"].metrics.get(metric_name, 0)
        if ttc_metrics:
            ttc_val = ttc_metrics["report"].metrics.get(metric_name, 0)
            improv = ((baseline_val - ttc_val) / baseline_val) * 100 if baseline_val > 0 else 0
            summary_md += (
                f"| {metric_name.upper()} | {baseline_val:.6f} | {ttc_val:.6f} | {improv:.1f}% |\n"
            )
        else:
            summary_md += f"| {metric_name.upper()} | {baseline_val:.6f} | - | - |\n"

    # Log comprehensive comparison tables
    if wandb_ctx:
        # Accuracy metrics table
        accuracy_rows = []
        for metric in ["mse", "mae", "rmse", "nrmse", "rel_l2"]:
            base_val = baseline_metrics["report"].metrics.get(metric)
            if ttc_metrics:
                ttc_val = ttc_metrics["report"].metrics.get(metric)
                if base_val is not None and ttc_val is not None and base_val != 0:
                    improvement_pct = ((base_val - ttc_val) / base_val) * 100.0
                else:
                    improvement_pct = None
                accuracy_rows.append(
                    [
                        metric.upper(),
                        f"{base_val:.6f}" if base_val is not None else "N/A",
                        f"{ttc_val:.6f}" if ttc_val is not None else "N/A",
                        f"{improvement_pct:.1f}%" if improvement_pct is not None else "N/A",
                    ]
                )
            else:
                accuracy_rows.append(
                    [
                        metric.upper(),
                        f"{base_val:.6f}" if base_val is not None else "N/A",
                        "N/A",
                        "N/A",
                    ]
                )

        wandb_ctx.log_table(
            "Training Evaluation Summary",
            columns=["Metric", "Baseline", "TTC", "Improvement"],
            data=accuracy_rows,
        )

        # Physics metrics table (if present)
        physics_rows = []
        for physics_key in ["conservation_gap", "bc_violation", "negativity_penalty"]:
            base_val = baseline_report.metrics.get(physics_key)
            if base_val is not None:
                if ttc_metrics:
                    ttc_val = ttc_metrics["report"].metrics.get(physics_key)
                    physics_rows.append(
                        [
                            physics_key.replace("_", " ").title(),
                            f"{base_val:.6f}",
                            f"{ttc_val:.6f}" if ttc_val is not None else "N/A",
                        ]
                    )
                else:
                    physics_rows.append(
                        [
                            physics_key.replace("_", " ").title(),
                            f"{base_val:.6f}",
                            "N/A",
                        ]
                    )

        if physics_rows:
            wandb_ctx.log_table(
                "Training Physics Diagnostics",
                columns=["Physics Check", "Baseline", "TTC"],
                data=physics_rows,
            )


def train_all_stages(cfg: dict, wandb_ctx=None) -> None:
    """Run all training stages in sequence with clean WandB context.

    Args:
        cfg: Training configuration
        wandb_ctx: Optional WandBContext (if not provided, will try to load from env)
    """
    # Load or create WandB context
    if wandb_ctx is None:
        # Try to load from environment (subprocess mode)
        import datetime
        import json
        import os
        from pathlib import Path

        from ups.utils.wandb_context import create_wandb_context, save_wandb_context

        # Standalone mode: create new WandB context
        # IMPORTANT: Only rank 0 should initialize WandB in DDP mode
        # Check RANK env var set by torchrun (DDP not initialized yet at this point)
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

        logging_cfg = cfg.get("logging", {})
        wandb_cfg = logging_cfg.get("wandb", {})
        if wandb_cfg.get("enabled", True) and rank == 0:
            run_id = f"train-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb_ctx = create_wandb_context(cfg, run_id=run_id, mode="online")

            # Save context to file for evaluation subprocess
            context_file_path = os.environ.get("WANDB_CONTEXT_FILE")
            if context_file_path and wandb_ctx and wandb_ctx.enabled:
                save_wandb_context(wandb_ctx, Path(context_file_path))
                print(f"✓ Saved WandB context to {context_file_path}")

            # Save WandB info for orchestrator
            wandb_info_path = os.environ.get("FAST_TO_SOTA_WANDB_INFO")
            if wandb_info_path and wandb_ctx and wandb_ctx.run:
                wandb_info = {
                    "id": wandb_ctx.run.id,
                    "name": wandb_ctx.run.name,
                    "project": wandb_ctx.run.project,
                    "entity": wandb_ctx.run.entity,
                    "url": wandb_ctx.run.url,
                }
                Path(wandb_info_path).write_text(json.dumps(wandb_info, indent=2))
                print(f"✓ Saved WandB info to {wandb_info_path}")

    # Initialize stage tracker
    from ups.utils.stage_tracker import StageTracker

    checkpoint_dir = ensure_checkpoint_dir(cfg)
    tracker = StageTracker(checkpoint_dir)

    # Auto-resume logic (if enabled via CLI flag)
    import sys

    if "--auto-resume" in sys.argv:
        print("\n" + "=" * 50)
        print("AUTO-RESUME ENABLED")
        print("=" * 50)

        completed_stages = tracker.get_completed_stages()
        if completed_stages:
            print(f"✓ Found completed stages: {', '.join(completed_stages)}")
            print("  Setting epochs=0 for completed stages to skip them")

            # Override config to skip completed stages
            for stage_name in completed_stages:
                if "stages" not in cfg:
                    cfg["stages"] = {}
                if stage_name not in cfg["stages"]:
                    cfg["stages"][stage_name] = {}

                original_epochs = cfg["stages"][stage_name].get("epochs", 0)
                cfg["stages"][stage_name]["epochs"] = 0
                print(f"  - {stage_name}: {original_epochs} epochs → 0 epochs (skipping)")
        else:
            print("  No completed stages found, starting from beginning")
        print("=" * 50 + "\n")

    # Log system info to config
    if wandb_ctx and wandb_ctx.enabled and torch.cuda.is_available():
        wandb_ctx.update_config(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
            }
        )

    global_step = 0

    # Stage 1: Operator
    op_epochs = _stage_epochs(cfg, "operator")
    if op_epochs > 0:
        print("\n" + "=" * 50)
        print("STAGE 1/4: Training Operator")
        print("=" * 50)

        # Mark stage as started
        tracker.mark_stage_started("operator", total_epochs=op_epochs)

        try:
            train_operator(cfg, wandb_ctx=wandb_ctx, global_step=global_step)

            # Mark stage as completed
            tracker.mark_stage_completed("operator", checkpoint="operator_ema.pt")
        except Exception as e:
            # Mark stage as failed
            tracker.mark_stage_failed("operator", error_message=str(e))
            raise

        global_step += op_epochs
    else:
        print("\n" + "=" * 50)
        print("STAGE 1/4: Skipping Operator (epochs<=0)")
        print("=" * 50)

    # Clear GPU cache between stages
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ Cleared GPU cache")

    # Stage 2: Diffusion Residual
    diff_epochs = _stage_epochs(cfg, "diff_residual")
    if diff_epochs > 0:
        print("\n" + "=" * 50)
        print("STAGE 2/4: Training Diffusion Residual")
        print("=" * 50)

        # Mark stage as started
        tracker.mark_stage_started("diff_residual", total_epochs=diff_epochs)

        try:
            train_diffusion(cfg, wandb_ctx=wandb_ctx, global_step=global_step)

            # Mark stage as completed
            tracker.mark_stage_completed("diff_residual", checkpoint="diffusion_residual_ema.pt")
        except Exception as e:
            # Mark stage as failed
            tracker.mark_stage_failed("diff_residual", error_message=str(e))
            raise

        global_step += diff_epochs
    else:
        print("\n" + "=" * 50)
        print("STAGE 2/4: Skipping Diffusion Residual (epochs<=0)")
        print("=" * 50)

    # Clear GPU cache between stages
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ Cleared GPU cache")

    # Stage 3: Consistency Distillation
    distill_epochs = _stage_epochs(cfg, "consistency_distill")
    if distill_epochs > 0:
        print("\n" + "=" * 50)
        print("STAGE 3/4: Consistency Distillation")
        print("=" * 50)

        # Mark stage as started
        tracker.mark_stage_started("consistency_distill", total_epochs=distill_epochs)

        try:
            train_consistency(cfg, wandb_ctx=wandb_ctx, global_step=global_step)

            # Mark stage as completed (Consistency overwrites diffusion checkpoint)
            tracker.mark_stage_completed(
                "consistency_distill", checkpoint="diffusion_residual_ema.pt"
            )
        except Exception as e:
            # Mark stage as failed
            tracker.mark_stage_failed("consistency_distill", error_message=str(e))
            raise

        global_step += distill_epochs
    else:
        print("\n" + "=" * 50)
        print("STAGE 3/4: Skipping Consistency Distillation (epochs<=0)")
        print("=" * 50)

    # Clear GPU cache between stages
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ Cleared GPU cache")

    # Stage 4: Steady Prior
    steady_epochs = _stage_epochs(cfg, "steady_prior")
    if steady_epochs > 0:
        print("\n" + "=" * 50)
        print("STAGE 4/4: Training Steady Prior")
        print("=" * 50)

        # Mark stage as started
        tracker.mark_stage_started("steady_prior", total_epochs=steady_epochs)

        try:
            train_steady_prior(cfg, wandb_ctx=wandb_ctx, global_step=global_step)

            # Mark stage as completed
            tracker.mark_stage_completed("steady_prior", checkpoint="steady_prior.pt")
        except Exception as e:
            # Mark stage as failed
            tracker.mark_stage_failed("steady_prior", error_message=str(e))
            raise
    else:
        print("\n" + "=" * 50)
        print("STAGE 4/4: Skipping Steady Prior (epochs<=0)")
        print("=" * 50)

    # Stage 5: Evaluation (optional, controlled by config)
    checkpoint_dir = ensure_checkpoint_dir(cfg)
    run_eval = cfg.get("evaluation", {}).get("enabled", True)  # Default to True for convenience

    if run_eval:
        print("\n" + "=" * 50)
        print("STAGE 5/5: Evaluation on Test Set")
        print("=" * 50)

        try:
            # Run baseline evaluation
            print("\n📊 Running baseline evaluation...")
            baseline_results = _run_evaluation(
                cfg, checkpoint_dir, eval_mode="baseline", wandb_ctx=wandb_ctx
            )
            baseline_report = baseline_results["report"]

            print("Baseline Results:")
            print(f"  MSE:   {baseline_report.metrics.get('mse', 0):.6f}")
            print(f"  MAE:   {baseline_report.metrics.get('mae', 0):.6f}")
            print(f"  RMSE:  {baseline_report.metrics.get('rmse', 0):.6f}")
            print(f"  NRMSE: {baseline_report.metrics.get('nrmse', 0):.6f}")

            # Run TTC evaluation if configured
            ttc_results = None
            if cfg.get("ttc", {}).get("enabled", False):
                print("\n📊 Running TTC evaluation...")
                ttc_results = _run_evaluation(
                    cfg, checkpoint_dir, eval_mode="ttc", wandb_ctx=wandb_ctx
                )
                ttc_report = ttc_results["report"]

                print("TTC Results:")
                print(f"  MSE:   {ttc_report.metrics.get('mse', 0):.6f}")
                print(f"  MAE:   {ttc_report.metrics.get('mae', 0):.6f}")
                print(f"  RMSE:  {ttc_report.metrics.get('rmse', 0):.6f}")
                print(f"  NRMSE: {ttc_report.metrics.get('nrmse', 0):.6f}")

                # Compute improvement
                baseline_nrmse = baseline_report.metrics.get("nrmse", 1.0)
                ttc_nrmse = ttc_report.metrics.get("nrmse", 1.0)
                improvement = ((baseline_nrmse - ttc_nrmse) / baseline_nrmse) * 100
                print(f"\n  TTC Improvement: {improvement:.1f}%")

            # Log to WandB using clean context
            if wandb_ctx:
                _log_evaluation_summary(wandb_ctx, baseline_results, ttc_results)

                # Save reports as artifacts
                import os

                report_dir = Path("reports")
                report_dir.mkdir(parents=True, exist_ok=True)

                # Save baseline report
                baseline_json = report_dir / "eval_baseline.json"
                baseline_report.to_json(baseline_json)
                if wandb_ctx and wandb_ctx.run is not None:
                    artifact = wandb.Artifact(
                        name=f"eval-baseline-{wandb_ctx.run.id}", type="evaluation"
                    )
                    artifact.add_file(str(baseline_json))
                    wandb_ctx.log_artifact(artifact)

                # Save TTC report if available
                if ttc_results:
                    ttc_json = report_dir / "eval_ttc.json"
                    ttc_report.to_json(ttc_json)
                    if wandb_ctx and wandb_ctx.run is not None:
                        artifact = wandb.Artifact(
                            name=f"eval-ttc-{wandb_ctx.run.id}", type="evaluation"
                        )
                        artifact.add_file(str(ttc_json))
                        wandb_ctx.log_artifact(artifact)

        except Exception as e:
            print(f"\n⚠️  Evaluation failed: {e}")
            if wandb_ctx and wandb_ctx.enabled:
                wandb_ctx.log_eval_summary({"error": str(e)}, prefix="eval")
    else:
        print("\n" + "=" * 50)
        print("STAGE 5/5: Skipping Evaluation (disabled in config)")
        print("=" * 50)

    # Log final summary
    if wandb_ctx and wandb_ctx.enabled:
        # Load final checkpoints to get model sizes
        import os

        summary = {
            "total_training_complete": 1,
            "operator_checkpoint_size_mb": (
                os.path.getsize(checkpoint_dir / "operator.pt") / 1e6
                if (checkpoint_dir / "operator.pt").exists()
                else 0
            ),
            "diffusion_checkpoint_size_mb": (
                os.path.getsize(checkpoint_dir / "diffusion_residual.pt") / 1e6
                if (checkpoint_dir / "diffusion_residual.pt").exists()
                else 0
            ),
            "steady_prior_checkpoint_size_mb": (
                os.path.getsize(checkpoint_dir / "steady_prior.pt") / 1e6
                if (checkpoint_dir / "steady_prior.pt").exists()
                else 0
            ),
        }
        wandb_ctx.log_eval_summary(summary, prefix="summary")

        # Generate final report summary
        print("\n" + "=" * 50)
        print("📝 WandB Summary Generated")
        print("=" * 50)
        print(f"View full results at: {wandb_ctx.run.url}")

        # Training run owns its own lifecycle - call finish()
        wandb_ctx.finish()

    print("\n" + "=" * 50)
    print("✅ All training stages complete!")
    print("=" * 50)

    # Force cleanup of CUDA resources and DataLoader workers
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main() -> None:
    print("[IMPORT-DEBUG] ✅ Entered main() function")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_multi_pde.yaml")
    parser.add_argument(
        "--stage",
        choices=["operator", "diff_residual", "consistency_distill", "steady_prior", "all"],
        required=True,
        help="Training stage to run, or 'all' to run full pipeline",
    )
    parser.add_argument(
        "--resume-from-wandb",
        type=str,
        default=None,
        help="WandB run ID to resume from (e.g., 'train-20251027_193043')",
    )
    parser.add_argument(
        "--resume-mode",
        type=str,
        default="allow",
        choices=["allow", "must", "never"],
        help="WandB resume mode (allow, must, never). Default: allow",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Automatically detect completed stages and skip them (resume from last incomplete stage)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg)
    stage = args.stage

    # Handle checkpoint resumption from WandB if requested
    if args.resume_from_wandb:
        from ups.utils.checkpoint_manager import CheckpointManager

        checkpoint_dir = Path(cfg.get("checkpoint", {}).get("dir", "checkpoints"))
        print(f"\n=== Resuming from WandB run: {args.resume_from_wandb} ===")

        try:
            manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

            # Download checkpoints from WandB
            downloaded_files = manager.download_checkpoints_from_run(
                run_id=args.resume_from_wandb,
                checkpoint_files=None,  # Download all common checkpoints
                force=False,  # Skip already-downloaded files
            )

            # Setup WandB environment for resumption
            manager.setup_wandb_resume(run_id=args.resume_from_wandb, resume_mode=args.resume_mode)

            # Verify critical checkpoints exist
            if stage == "all" or stage == "operator":
                if not manager.verify_checkpoints(["operator.pt", "operator_ema.pt"]):
                    print("⚠️  Warning: Operator checkpoints missing")

            print(f"✓ Ready to resume training from {args.resume_from_wandb}\n")

        except Exception as e:
            print(f"❌ Failed to resume from WandB: {e}")
            print("Continuing with fresh training...")
            import traceback

            traceback.print_exc()

    if stage == "all":
        train_all_stages(cfg)
    elif stage == "operator":
        train_operator(cfg)
    elif stage == "diff_residual":
        train_diffusion(cfg)
    elif stage == "consistency_distill":
        train_consistency(cfg)
    elif stage == "steady_prior":
        train_steady_prior(cfg)

    # Final cleanup to ensure clean exit
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Give time for any background threads to finish
    import time

    time.sleep(2)


print("[IMPORT-DEBUG] ✅ About to check if __name__ == '__main__'")
if __name__ == "__main__":
    print("[IMPORT-DEBUG] ✅ __name__ == '__main__' is True, calling main()")
    main()
else:
    print(f"[IMPORT-DEBUG] ⚠️  __name__ == {__name__!r}, NOT calling main()")
