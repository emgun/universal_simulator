"""Factory for creating Muon optimizers with automatic fallback chain."""
from __future__ import annotations

import torch.nn as nn
from torch.optim import Optimizer


def create_muon_optimizer(
    params: list[nn.Parameter],
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    backend: str = "auto",
) -> tuple[Optimizer, str]:
    """
    Create Muon optimizer with automatic fallback chain.

    Tries implementations in order:
    1. flash-muon (Triton-optimized, 35% faster)
    2. torch.optim.Muon (official PyTorch 2.9+)
    3. Raises error if none available

    Args:
        params: List of parameters to optimize
        lr: Learning rate (use same as AdamW baseline)
        weight_decay: L2 weight decay (decoupled, like AdamW)
        momentum: Nesterov momentum coefficient (default: 0.95)
        nesterov: Whether to use Nesterov momentum
        ns_steps: Newton-Schulz iterations for orthogonalization (default: 5)
        backend: "auto" (try fallback), "flash" (flash-muon only), "torch" (torch.optim.Muon only)

    Returns:
        (optimizer, backend_name) - Optimizer instance and string indicating which backend was used

    Raises:
        RuntimeError: If no Muon implementation is available

    Example:
        >>> optimizer, backend = create_muon_optimizer(muon_params, lr=1e-3, weight_decay=0.03)
        >>> print(f"Using {backend}")
        Using flash-muon
    """
    # Backend-specific mode
    if backend == "flash":
        return _create_flash_muon(params, lr, weight_decay, momentum, nesterov, ns_steps)
    elif backend == "torch":
        return _create_torch_muon(params, lr, weight_decay, momentum, nesterov, ns_steps)
    elif backend != "auto":
        raise ValueError(f"Unknown backend: {backend}. Use 'auto', 'flash', or 'torch'")

    # Auto mode: Try fallback chain
    # Try 1: flash-muon (Triton-optimized, 35% faster)
    try:
        return _create_flash_muon(params, lr, weight_decay, momentum, nesterov, ns_steps)
    except ImportError:
        pass

    # Try 2: torch.optim.Muon (official PyTorch 2.9+)
    try:
        return _create_torch_muon(params, lr, weight_decay, momentum, nesterov, ns_steps)
    except (ImportError, AttributeError):
        pass

    # No implementation available
    raise RuntimeError(
        "No Muon optimizer implementation found. Install one of:\n"
        "  1. flash-muon: pip install git+https://github.com/nil0x9/flash-muon.git\n"
        "  2. torch>=2.9: pip install --upgrade torch>=2.9\n"
    )


def _create_flash_muon(
    params: list[nn.Parameter],
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_steps: int,
) -> tuple[Optimizer, str]:
    """Create flash-muon optimizer (Triton-optimized)."""
    from flash_muon import Muon

    optimizer = Muon(
        params,
        lr=lr,
        momentum=momentum,
        nesterov=nesterov,
        backend_steps=ns_steps,
        weight_decay=weight_decay,
    )
    return optimizer, "flash-muon"


def _create_torch_muon(
    params: list[nn.Parameter],
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_steps: int,
) -> tuple[Optimizer, str]:
    """Create official torch.optim.Muon optimizer."""
    from torch.optim import Muon

    optimizer = Muon(
        params,
        lr=lr,
        momentum=momentum,
        nesterov=nesterov,
        ns_steps=ns_steps,
        weight_decay=weight_decay,
    )
    return optimizer, "torch.optim.Muon"


def get_available_backends() -> list[str]:
    """
    Check which Muon backends are available.

    Returns:
        List of available backend names: ["flash-muon", "torch.optim.Muon"]

    Example:
        >>> backends = get_available_backends()
        >>> print(f"Available: {', '.join(backends)}")
        Available: flash-muon, torch.optim.Muon
    """
    available = []

    # Check flash-muon
    try:
        import flash_muon  # noqa: F401
        available.append("flash-muon")
    except ImportError:
        pass

    # Check torch.optim.Muon
    try:
        from torch.optim import Muon  # noqa: F401
        available.append("torch.optim.Muon")
    except (ImportError, AttributeError):
        pass

    return available
