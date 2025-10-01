"""Evaluation utilities (metrics, calibration, reporting)."""

from .calibration import TemperatureScaler, expected_calibration_error, reliability_diagram
from .metrics import nrmse, spectral_energy_error, conservation_gap
from .gates import residual_gate, periodic_gate
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
    "MetricReport",
]
