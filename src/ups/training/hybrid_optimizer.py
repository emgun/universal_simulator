"""Hybrid optimizer that combines multiple optimizers for different parameter groups."""
from __future__ import annotations

from typing import Any

from torch.optim import Optimizer


class HybridOptimizer(Optimizer):
    """
    Wrapper that manages multiple optimizers as a single unit.

    Designed for Muon+AdamW hybrid training where:
    - Muon optimizer handles 2D+ matrix parameters
    - AdamW optimizer handles 1D vector parameters (biases, norms)

    This wrapper provides a unified interface that looks like a single optimizer
    to the training loop, making it compatible with existing PyTorch training code,
    AMP GradScaler, LR schedulers, etc.

    Usage:
        >>> muon_opt = Muon(muon_params, lr=1e-3, ...)
        >>> adamw_opt = AdamW(adamw_params, lr=1e-3, ...)
        >>> hybrid = HybridOptimizer([muon_opt, adamw_opt])
        >>>
        >>> # Use like any optimizer
        >>> hybrid.zero_grad()
        >>> loss.backward()
        >>> hybrid.step()

    Attributes:
        optimizers: List of child optimizer instances
    """

    def __init__(self, optimizers: list[Optimizer]):
        """
        Initialize hybrid optimizer.

        Args:
            optimizers: List of optimizer instances to manage jointly
                       (e.g., [muon_opt, adamw_opt])
        """
        if not optimizers:
            raise ValueError("Must provide at least one optimizer")

        self.optimizers = optimizers
        self._param_groups = []  # Store internally

        # Initialize with a dummy parameter to satisfy PyTorch's Optimizer base class
        # The actual parameters are managed by child optimizers
        dummy_param = [{'params': []}]
        super().__init__(dummy_param, {})

    def step(self, closure: callable | None = None) -> float | None:
        """
        Perform optimization step for all child optimizers.

        Args:
            closure: Optional closure for re-evaluating the model

        Returns:
            Loss value if closure is provided, None otherwise
        """
        loss = None
        for opt in self.optimizers:
            if closure is not None:
                loss = opt.step(closure)
            else:
                opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Zero gradients for all child optimizers.

        Args:
            set_to_none: Set gradients to None instead of zero (more efficient)
        """
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """
        Get state dict with all child optimizer states.

        Returns:
            Dictionary mapping optimizer_i to child optimizer state dict
        """
        return {
            f"optimizer_{i}": opt.state_dict()
            for i, opt in enumerate(self.optimizers)
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load state for all child optimizers.

        Args:
            state_dict: Dictionary with optimizer_i keys
        """
        for i, opt in enumerate(self.optimizers):
            key = f"optimizer_{i}"
            if key in state_dict:
                opt.load_state_dict(state_dict[key])

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        """
        Get all param_groups from all child optimizers.

        This property is required for:
        - AMP GradScaler.unscale_() (iterates over param_groups)
        - LR schedulers (access param_groups to update learning rates)

        Returns:
            Flattened list of all param_groups from all child optimizers
        """
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    @param_groups.setter
    def param_groups(self, value: list[dict[str, Any]]) -> None:
        """
        Setter for param_groups (required by PyTorch Optimizer base class).

        We store the value but don't use it - the actual param_groups come from child optimizers.
        """
        self._param_groups = value

    def __repr__(self) -> str:
        """String representation showing child optimizers."""
        opt_reprs = [f"  [{i}] {opt.__class__.__name__}" for i, opt in enumerate(self.optimizers)]
        return "HybridOptimizer(\n" + "\n".join(opt_reprs) + "\n)"
