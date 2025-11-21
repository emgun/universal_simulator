"""Parameter grouping utilities for hybrid Muon+AdamW optimization."""
from __future__ import annotations

import torch.nn as nn


def is_muon_compatible(p: nn.Parameter) -> bool:
    """
    Check if parameter is Muon-compatible (2D or higher).

    Args:
        p: Parameter to check

    Returns:
        True if parameter should use Muon (ndim >= 2), False otherwise
    """
    return p.requires_grad and p.ndim >= 2


def build_param_groups(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """
    Split model parameters into Muon-compatible and AdamW groups.

    Parameter Classification:
        Muon (2D+):
            - Linear layer weights (2D)
            - Conv2d layer weights (4D)
            - Any parameter with ndim >= 2
        AdamW (1D):
            - Biases (1D)
            - LayerNorm weights/biases (1D)
            - RMSNorm weights (1D)
            - Any parameter with ndim < 2

    Returns:
        (muon_params, adamw_params) - Two lists of parameters

    Example:
        >>> muon_params, adamw_params = build_param_groups(model)
        >>> print(f"Muon: {len(muon_params)}, AdamW: {len(adamw_params)}")
    """
    muon_params = []
    adamw_params = []

    for _name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if is_muon_compatible(p):
            muon_params.append(p)
        else:
            adamw_params.append(p)

    return muon_params, adamw_params


def build_param_groups_with_names(
    model: nn.Module,
) -> tuple[list[tuple[str, nn.Parameter]], list[tuple[str, nn.Parameter]]]:
    """
    Same as build_param_groups but includes parameter names for debugging.

    Returns:
        (muon_params_with_names, adamw_params_with_names)
        Each element is (name, parameter) tuple

    Example:
        >>> muon_params, adamw_params = build_param_groups_with_names(model)
        >>> for name, p in muon_params[:3]:
        ...     print(f"{name}: {p.shape}")
    """
    muon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if is_muon_compatible(p):
            muon_params.append((name, p))
        else:
            adamw_params.append((name, p))

    return muon_params, adamw_params


def print_param_split_summary(model: nn.Module) -> None:
    """
    Print summary of parameter split for debugging.

    Shows total parameters, counts, and percentage for each group.

    Example output:
        Parameter Split Summary:
        Muon (2D+): 1,234,567 params (95.2%)
        AdamW (1D): 62,345 params (4.8%)
        Total: 1,296,912 params
    """
    muon_params, adamw_params = build_param_groups(model)

    total_muon = sum(p.numel() for p in muon_params)
    total_adamw = sum(p.numel() for p in adamw_params)
    total_params = total_muon + total_adamw

    print("Parameter Split Summary:")
    print(f"  Muon (2D+): {total_muon:,} params ({100 * total_muon / total_params:.1f}%)")
    print(f"  AdamW (1D): {total_adamw:,} params ({100 * total_adamw / total_params:.1f}%)")
    print(f"  Total: {total_params:,} params")


def filter_muon_params_for_backend(muon_params: list[nn.Parameter], backend: str) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """
    Given a candidate Muon param list and backend name, return (kept, diverted).

    torch.optim.Muon only accepts 2D gradients; conv/embedding weights (3D/4D)
    must be diverted to AdamW to avoid runtime errors. flash-muon tolerates
    higher rank weights. If backend is unknown, assume strict torch behavior.
    """
    if backend != "flash-muon":
        kept = [p for p in muon_params if p.ndim == 2]
        diverted = [p for p in muon_params if p.ndim != 2]
        return kept, diverted
    return muon_params, []
