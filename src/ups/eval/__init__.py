"""Evaluation utilities (metrics, calibration, reporting)."""

from .calibration import TemperatureScaler, expected_calibration_error, reliability_diagram
from .metrics import nrmse, spectral_energy_error, conservation_gap
from .gates import residual_gate, periodic_gate
from .promotion import PromotionResult, PromotionRule, evaluate_promotion_rules, parse_promotion_rule, promotion_rules_from_config
from .reports import MetricReport

__all__ = [
    "TemperatureScaler",
    "expected_calibration_error",
    "reliability_diagram",
    "nrmse",
    "spectral_energy_error",
    "conservation_gap",
    "residual_gate",
    "periodic_gate",
    "PromotionResult",
    "PromotionRule",
    "evaluate_promotion_rules",
    "parse_promotion_rule",
    "promotion_rules_from_config",
    "MetricReport",
]
