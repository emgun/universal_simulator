from __future__ import annotations

"""Small decoded-field transport heads for scoped rollout experiments."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ModelSideTransportHeadConfig:
    enabled: bool = False
    tasks: tuple[str, ...] = ("advection1d",)
    families: tuple[str, ...] = ()
    required_params: tuple[str, ...] = ("beta",)
    features: tuple[str, ...] = ("param:beta", "horizon_norm", "bias")
    mode: str = "periodic_roll"
    apply_at: str = "decoded_rollout"
    trainable: bool = True
    init: Mapping[str, float] | None = None
    min_shift: float = -64.0
    max_shift: float = 64.0
    missing_param_policy: str = "skip"


def _string_tuple(value: Any, *, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence):
        raise ValueError(f"model_side_transport_head.{name} must be a sequence")
    return tuple(str(item) for item in value)


def model_side_transport_head_config(raw: Any) -> ModelSideTransportHeadConfig:
    if raw is None:
        return ModelSideTransportHeadConfig()
    if not isinstance(raw, Mapping):
        raise ValueError("model_side_transport_head must be a mapping")
    enabled = bool(raw.get("enabled", False))
    mode = str(raw.get("mode", "periodic_roll"))
    if mode != "periodic_roll":
        raise ValueError("model_side_transport_head.mode must be 'periodic_roll'")
    apply_at = str(raw.get("apply_at", "decoded_rollout"))
    if apply_at != "decoded_rollout":
        raise ValueError("model_side_transport_head.apply_at must be 'decoded_rollout'")
    missing_param_policy = str(raw.get("missing_param_policy", "skip"))
    if missing_param_policy not in {"skip", "zero_shift"}:
        raise ValueError(
            "model_side_transport_head.missing_param_policy must be skip or zero_shift"
        )
    init = raw.get("init", {})
    if init is None:
        init = {}
    if not isinstance(init, Mapping):
        raise ValueError("model_side_transport_head.init must be a mapping")
    clamp = raw.get("clamp", {})
    if clamp is None:
        clamp = {}
    if not isinstance(clamp, Mapping):
        raise ValueError("model_side_transport_head.clamp must be a mapping")
    return ModelSideTransportHeadConfig(
        enabled=enabled,
        tasks=_string_tuple(raw.get("tasks", ("advection1d",)), name="tasks"),
        families=_string_tuple(raw.get("families", ()), name="families"),
        required_params=_string_tuple(
            raw.get("required_params", ("beta",)), name="required_params"
        ),
        features=_string_tuple(
            raw.get("features", ("param:beta", "horizon_norm", "bias")), name="features"
        ),
        mode=mode,
        apply_at=apply_at,
        trainable=bool(raw.get("trainable", True)),
        init={str(key): float(value) for key, value in init.items()},
        min_shift=float(clamp.get("min_shift", raw.get("min_shift", -64.0))),
        max_shift=float(clamp.get("max_shift", raw.get("max_shift", 64.0))),
        missing_param_policy=missing_param_policy,
    )


class ModelSideTransportHead(nn.Module):
    """Linear metadata-conditioned decoded transport displacement head.

    The first use is intentionally tiny: beta/horizon/bias -> periodic x-roll.
    The module is default-off at config level; this class only describes the
    enabled candidate once the evaluator or training loop opts in.
    """

    def __init__(self, cfg: ModelSideTransportHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        init = cfg.init or {}
        weights = [float(init.get(feature, 0.0)) for feature in cfg.features]
        self.weights = nn.Parameter(
            torch.tensor(weights, dtype=torch.float32), requires_grad=cfg.trainable
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def applies(self, *, task_name: str, task_family: str) -> bool:
        if not self.cfg.enabled:
            return False
        if self.cfg.tasks and task_name not in self.cfg.tasks:
            return False
        if self.cfg.families and task_family not in self.cfg.families:
            return False
        return True

    def _features(
        self,
        *,
        params: Mapping[str, torch.Tensor] | None,
        horizon: int,
        rollout_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, list[str]]:
        missing: list[str] = []
        values: list[torch.Tensor] = []
        param_map = params or {}
        for feature in self.cfg.features:
            if feature == "bias":
                values.append(torch.ones((), device=device, dtype=dtype))
                continue
            if feature == "horizon_norm":
                value = float(horizon) / max(float(rollout_steps), 1.0)
                values.append(torch.tensor(value, device=device, dtype=dtype))
                continue
            if feature.startswith("param:"):
                name = feature.split(":", 1)[1]
                value = param_map.get(name)
                if value is None:
                    missing.append(name)
                    values.append(torch.zeros((), device=device, dtype=dtype))
                    continue
                tensor = value.detach().to(device=device, dtype=dtype).reshape(-1)
                if tensor.numel() == 0:
                    missing.append(name)
                    values.append(torch.zeros((), device=device, dtype=dtype))
                    continue
                values.append(tensor[0])
                continue
            raise ValueError(f"Unsupported model-side transport feature '{feature}'")
        required_missing = sorted(
            {name for name in self.cfg.required_params if name in missing or name not in param_map}
        )
        if required_missing and self.cfg.missing_param_policy == "skip":
            return None, required_missing
        if not values:
            return torch.zeros(0, device=device, dtype=dtype), required_missing
        return torch.stack(values), required_missing

    def predict_shift(
        self,
        *,
        params: Mapping[str, torch.Tensor] | None,
        horizon: int,
        rollout_steps: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        device = device or self.weights.device
        feature_values, missing = self._features(
            params=params,
            horizon=horizon,
            rollout_steps=rollout_steps,
            device=device,
            dtype=dtype,
        )
        info: dict[str, Any] = {"missing_params": missing}
        if feature_values is None:
            info["skipped"] = True
            return None, info
        weights = self.weights.to(device=device, dtype=dtype)
        shift = torch.sum(feature_values * weights)
        shift = torch.clamp(shift, min=float(self.cfg.min_shift), max=float(self.cfg.max_shift))
        info["skipped"] = False
        return shift, info

    def resolved_config(self) -> dict[str, Any]:
        return {
            "enabled": self.cfg.enabled,
            "mode": self.cfg.mode,
            "apply_at": self.cfg.apply_at,
            "tasks": list(self.cfg.tasks),
            "families": list(self.cfg.families),
            "required_params": list(self.cfg.required_params),
            "features": list(self.cfg.features),
            "missing_param_policy": self.cfg.missing_param_policy,
            "trainable": self.cfg.trainable,
            "trainable_parameter_count": self.trainable_parameter_count,
            "coefficients": {
                feature: float(value.detach().cpu().item())
                for feature, value in zip(self.cfg.features, self.weights)
            },
            "clamp": {"min_shift": self.cfg.min_shift, "max_shift": self.cfg.max_shift},
        }
