#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

EXPERIMENT = "canonical_latent_e17_quadratic_closure"
ACTIVE_DIM = 49
LATENT_DIM = 52
FEATURE_NAMES = ("one", "sin1", "cos1", "sin2", "cos2", "sin3", "cos3")
REGIMES = ("x_only", "y_only", "mixed_same_sign", "mixed_opposite_sign")
CLASSIFICATIONS = (
    "preflight_failed",
    "incomplete",
    "latent_closure_insufficient",
    "quadratic_identification_failed",
    "constrained_quadratic_closure_qualified",
)

PREDECESSOR_HASHES = {
    "docs/research/artifacts/canonical_latent_e15_training_package/"
    "canonical_latent_e15_training_package_evidence_bundle.tar.gz": (
        "3347ec66843ed51e30a36996335915221407c979b64afa13b96f9ee0d76b618a"
    ),
    "docs/research/artifacts/canonical_latent_e15_training_package/"
    "canonical_latent_e15_training_package_result.json": (
        "e3b91ecc792085f45e6b80bd970cb6da15fb869a7a49e8fec4feb782b919768d"
    ),
    "docs/research/artifacts/canonical_latent_e15_training_package/"
    "canonical_latent_e15_training_package_manifest.json": (
        "1208b5e5158f9c2ff0ae0dd5ab310ec5967cfdc7bc5d0ab131e8c0387effd311"
    ),
    "docs/research/artifacts/canonical_latent_e16_multi_realization_robustness/"
    "canonical_latent_e16_multi_realization_robustness_evidence_bundle.tar.gz": (
        "71fc490c2bc361fbf0b26d5bfcccfc460bcf5af223b5000d1e6043672504a586"
    ),
    "docs/research/artifacts/canonical_latent_e16_multi_realization_robustness/"
    "canonical_latent_e16_multi_realization_robustness_result.json": (
        "6716273a3ea980f7d24462ec3e40eb37091d229d524aec5f9a0ad89bbb9d325a"
    ),
    "docs/research/artifacts/canonical_latent_e16_multi_realization_robustness/"
    "canonical_latent_e16_multi_realization_robustness_manifest.json": (
        "6af927037eebeebc3a9a95842d549c633279391d28715779b7fc04c05b59720f"
    ),
    "scripts/run_canonical_latent_e15_training_package.py": (
        "943558c42d2e8a13879fc3fe6f1301142efe7c7949f51e7e4ff509a6af6ae9ca"
    ),
    "scripts/run_canonical_latent_e12_structured_generator.py": (
        "8edb67652d53e101a63730b9ec4803a69067572a8bab6eee0fb98627785a926a"
    ),
    "scripts/run_canonical_latent_e10_geometry_universal_projection.py": (
        "a06486e5f6e77667fa06c65ee5dbff8c57cad6b505789b94d4596cc31515e404"
    ),
    "scripts/run_canonical_latent_e7_function_space.py": (
        "cf81597b3909e9693508b62e595eb006a8598d186de062eaf4a8f241d4b07488"
    ),
}

E15_RESULT_PATH = (
    "docs/research/artifacts/canonical_latent_e15_training_package/"
    "canonical_latent_e15_training_package_result.json"
)
E15_PACKAGE = "schedule_weighted_componentwise_lbfgs_neutral"
E15_CLASSIFICATION = "deterministic_objective_adamw_restart_repairs_e12_checkpoint_only"
E15_MODEL_SHA256 = "c11e1b311a6bbb12332732009e68b6efca5768943c23ca25b25cd4f28526e423"
E15_GENERATOR_SHA256 = {
    "A_x": "6d5be486068e5829d90dbac40a855a5242d5ebaca93aa744fb1cc1831b355cdd",
    "A_y": "947f713391f7f1a308e664094379dbc336034430c385406f9e423d9f5d485d55",
    "D": "57ea9e36fa5ae2746c848061c26748bb220cf2bb3d081e788315bd8c34c1b27f",
}
E15_RECOVERY_GATES = (
    "structure",
    "generator_identification",
    "high_frequency",
    "elementary_one_step_nonregression",
    "elementary_rollout_nonregression",
    "zero_shot_rollout",
    "zero_shot_to_persistence",
    "finite",
)
E17_SOURCE_PATHS = (
    "docs/research/2026-07-27-canonical-latent-e17-quadratic-closure-contract.md",
    "scripts/run_canonical_latent_e17_quadratic_closure.py",
    "tests/unit/test_run_canonical_latent_e17_quadratic_closure.py",
)


@dataclass(frozen=True)
class E17Config:
    truth_resolution: int = 216
    reference_resolution: int = 324
    comparison_resolution: int = 432
    truth_step: float = 0.001
    reference_step: float = 0.0005
    observation_step: float = 0.01
    observation_transitions: int = 16
    training_trajectories: int = 192
    validation_trajectories: int = 64
    validation_pairs: int = 32
    training_state_seed: int = 817001
    training_parameter_seed: int = 817002
    training_schedule_seed: int = 817003
    validation_state_seed: int = 917001
    validation_parameter_seed: int = 917002
    validation_schedule_seed: int = 917003
    triad_quadrature_resolution: int = 64
    triad_support_atol: float = 1e-12
    expected_axis_support: int = 1329
    maximum_design_condition: float = 1e8
    maximum_gram_condition: float = 1e16
    maximum_convergence_nrmse: float = 2e-4
    maximum_constant_drift: float = 1e-11
    maximum_energy_trajectory_mismatch: float = 5e-4
    maximum_nonlinear_energy_rate_residual: float = 1e-10

    def __post_init__(self) -> None:
        if self.truth_resolution != 216 or self.reference_resolution != 324:
            raise ValueError("E17 freezes 216/324 truth convergence resolutions")
        if self.comparison_resolution != 432:
            raise ValueError("E17 freezes the 432-square comparison grid")
        if self.training_trajectories != 192 or self.validation_trajectories != 64:
            raise ValueError("E17 freezes 192 training and 64 validation trajectories")
        if self.validation_pairs * 2 != self.validation_trajectories:
            raise ValueError("E17 validation must contain exactly two members per closure pair")
        if self.expected_axis_support != 1329:
            raise ValueError("E17 freezes 1,329 symmetric triad entries per axis")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    array = tensor.detach().cpu().contiguous().numpy()
    if array.dtype.kind == "f":
        canonical = array.astype("<f8", copy=False)
        dtype = "<f8"
    elif array.dtype.kind in ("i", "u"):
        canonical = array.astype("<i8", copy=False)
        dtype = "<i8"
    else:
        raise TypeError(f"unsupported canonical tensor dtype {array.dtype}")
    payload = canonical.tobytes(order="C")
    return {
        "shape": list(canonical.shape),
        "dtype": dtype,
        "order": "C",
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def centered_axis(resolution: int) -> torch.Tensor:
    return (torch.arange(resolution, dtype=torch.float64) + 0.5) / resolution


def fft_wavenumbers(resolution: int) -> torch.Tensor:
    return torch.fft.fftfreq(resolution, d=1.0 / resolution).to(torch.int64)


def retained_wavenumbers(resolution: int) -> tuple[int, ...]:
    values = fft_wavenumbers(resolution)
    retained = values[values.abs() < resolution / 3]
    return tuple(int(value) for value in retained.tolist())


def dealias_mask(resolution: int) -> torch.Tensor:
    frequencies = fft_wavenumbers(resolution)
    keep = frequencies.abs() < resolution / 3
    return keep[:, None] & keep[None, :]


def _one_dimensional(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    features = [torch.ones_like(values)]
    derivatives = [torch.zeros_like(values)]
    for frequency in range(1, 4):
        phase = 2.0 * math.pi * frequency * values
        scale = math.sqrt(2.0)
        omega = 2.0 * math.pi * frequency
        features.extend((scale * torch.sin(phase), scale * torch.cos(phase)))
        derivatives.extend((scale * omega * torch.cos(phase), -scale * omega * torch.sin(phase)))
    return torch.stack(features, dim=-1), torch.stack(derivatives, dim=-1)


def periodic_basis(
    resolution: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    axis = centered_axis(resolution)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    x_features, x_derivatives = _one_dimensional(x)
    y_features, y_derivatives = _one_dimensional(y)
    basis = (x_features.unsqueeze(-1) * y_features.unsqueeze(-2)).reshape(
        resolution, resolution, ACTIVE_DIM
    )
    derivative_x = (x_derivatives.unsqueeze(-1) * y_features.unsqueeze(-2)).reshape(
        resolution, resolution, ACTIVE_DIM
    )
    derivative_y = (x_features.unsqueeze(-1) * y_derivatives.unsqueeze(-2)).reshape(
        resolution, resolution, ACTIVE_DIM
    )
    return basis, derivative_x, derivative_y


def mode_names() -> tuple[str, ...]:
    return tuple(f"{x_name}*{y_name}" for x_name in FEATURE_NAMES for y_name in FEATURE_NAMES)


def project_periodic(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-2] != values.shape[-1]:
        raise ValueError("periodic projection requires a square grid")
    resolution = values.shape[-1]
    basis, _, _ = periodic_basis(resolution)
    leading = values.shape[:-2]
    flattened = values.reshape(-1, resolution * resolution)
    coefficients = flattened @ basis.reshape(-1, ACTIVE_DIM)
    coefficients = coefficients / (resolution * resolution)
    zeros = coefficients.new_zeros((coefficients.shape[0], LATENT_DIM - ACTIVE_DIM))
    return torch.cat((coefficients, zeros), dim=-1).reshape(*leading, LATENT_DIM)


def decode_periodic(coefficients: torch.Tensor, *, resolution: int) -> torch.Tensor:
    if coefficients.shape[-1] not in (ACTIVE_DIM, LATENT_DIM):
        raise ValueError("coefficients must have 49 or 52 entries")
    basis, _, _ = periodic_basis(resolution)
    active = coefficients[..., :ACTIVE_DIM]
    return torch.einsum("...i,xyi->...xy", active, basis)


def truth_vector_field(values: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3 or values.shape[-2] != values.shape[-1]:
        raise ValueError("truth states must have shape [batch, resolution, resolution]")
    if parameters.shape != (values.shape[0], 5):
        raise ValueError("parameters must have shape [batch, 5]")
    resolution = values.shape[-1]
    wavenumbers = fft_wavenumbers(resolution).to(torch.float64)
    kx = wavenumbers[:, None]
    ky = wavenumbers[None, :]
    omega_x = 2.0 * math.pi * kx
    omega_y = 2.0 * math.pi * ky
    mask = dealias_mask(resolution)
    spectrum = torch.fft.fft2(values.double()) * mask
    filtered = torch.fft.ifft2(spectrum).real
    derivative_x = torch.fft.ifft2(1j * omega_x * spectrum).real
    derivative_y = torch.fft.ifft2(1j * omega_y * spectrum).real
    laplacian = torch.fft.ifft2(-(omega_x.square() + omega_y.square()) * spectrum).real
    vx, vy, nu, gamma_x, gamma_y = (parameters[:, index, None, None] for index in range(5))
    nonlinear_unfiltered = gamma_x * filtered * derivative_x + gamma_y * filtered * derivative_y
    nonlinear = torch.fft.ifft2(torch.fft.fft2(nonlinear_unfiltered) * mask).real
    return -vx * derivative_x - vy * derivative_y - nonlinear + nu * laplacian


def rk4_step(
    values: torch.Tensor,
    parameters: torch.Tensor,
    *,
    step: float,
) -> torch.Tensor:
    k1 = truth_vector_field(values, parameters)
    k2 = truth_vector_field(values + 0.5 * step * k1, parameters)
    k3 = truth_vector_field(values + 0.5 * step * k2, parameters)
    k4 = truth_vector_field(values + step * k3, parameters)
    return values + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_truth(
    initial_values: torch.Tensor,
    parameters: torch.Tensor,
    *,
    internal_step: float,
    observation_step: float,
    transitions: int,
) -> torch.Tensor:
    internal_steps = round(observation_step / internal_step)
    if not math.isclose(
        internal_steps * internal_step,
        observation_step,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("observation step must be an integer multiple of the internal step")
    values = initial_values.double()
    trajectory = [values]
    for _ in range(transitions):
        for _ in range(internal_steps):
            values = rk4_step(values, parameters, step=internal_step)
        trajectory.append(values)
    return torch.stack(trajectory, dim=1)


def spectral_resample(values: torch.Tensor, *, target_resolution: int) -> torch.Tensor:
    if values.shape[-2] != values.shape[-1]:
        raise ValueError("spectral resampling requires square fields")
    source_resolution = values.shape[-1]
    if target_resolution < source_resolution:
        raise ValueError("E17 convergence resampling only supports refinement")
    source_frequencies = fft_wavenumbers(source_resolution)
    target_frequencies = fft_wavenumbers(target_resolution)
    target_lookup = {
        int(frequency): index for index, frequency in enumerate(target_frequencies.tolist())
    }
    source_spectrum = torch.fft.fft2(values.double())
    target_shape = (*values.shape[:-2], target_resolution, target_resolution)
    target_spectrum = torch.zeros(target_shape, dtype=torch.complex128)
    scale = (target_resolution / source_resolution) ** 2
    for source_x, frequency_x in enumerate(source_frequencies.tolist()):
        target_x = target_lookup[int(frequency_x)]
        for source_y, frequency_y in enumerate(source_frequencies.tolist()):
            target_y = target_lookup[int(frequency_y)]
            phase = (
                math.pi
                * (frequency_x + frequency_y)
                * (1.0 / target_resolution - 1.0 / source_resolution)
            )
            target_spectrum[..., target_x, target_y] = (
                source_spectrum[..., source_x, source_y]
                * scale
                * complex(math.cos(phase), math.sin(phase))
            )
    return torch.fft.ifft2(target_spectrum).real


def pooled_nrmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    numerator = (prediction.double() - target.double()).square().sum()
    denominator = target.double().square().sum().clamp_min(1e-24)
    return float(torch.sqrt(numerator / denominator).item())


def field_energy(values: torch.Tensor) -> torch.Tensor:
    return 0.5 * values.double().square().mean(dim=(-2, -1))


def calibration_cases() -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
    coefficients = torch.zeros(6, LATENT_DIM, dtype=torch.float64)
    parameters = torch.tensor(
        [
            [0.20, 0.0, 0.008, 0.0, 0.0],
            [0.0, -0.20, 0.008, 0.0, 0.0],
            [0.10, 0.0, 0.006, 0.80, 0.0],
            [0.0, -0.10, 0.006, 0.0, -0.80],
            [0.15, -0.10, 0.006, 0.75, -0.65],
            [0.30, -0.30, 0.004, 1.20, -1.20],
        ],
        dtype=torch.float64,
    )
    coefficients[0, 14] = 0.4
    coefficients[1, 2] = 0.4
    coefficients[2, 7] = 0.35
    coefficients[2, 14] = -0.25
    coefficients[3, 1] = 0.35
    coefficients[3, 2] = -0.25
    coefficients[4, 1] = 0.20
    coefficients[4, 7] = -0.25
    coefficients[4, 8] = 0.30
    coefficients[4, 16] = -0.20
    coefficients[5, 0] = 0.05
    coefficients[5, 1] = -0.30
    coefficients[5, 2] = 0.20
    coefficients[5, 7] = 0.35
    coefficients[5, 8] = 0.25
    coefficients[5, 14] = 0.25
    coefficients[5, 16] = -0.15
    return (
        coefficients,
        parameters,
        ("single_x", "single_y", "two_mode_x", "two_mode_y", "mixed", "stress"),
    )


def nonlinear_energy_rate_residual(
    trajectory: torch.Tensor,
    parameters: torch.Tensor,
) -> float:
    flattened = trajectory.reshape(-1, trajectory.shape[-2], trajectory.shape[-1])
    repeated = parameters[:, None, :].expand(-1, trajectory.shape[1], -1).reshape(-1, 5).clone()
    repeated[:, :3] = 0.0
    nonlinear = truth_vector_field(flattened, repeated)
    numerator = (flattened * nonlinear).mean(dim=(-2, -1)).abs()
    denominator = flattened.square().mean(dim=(-2, -1)).clamp_min(1e-24)
    return float((numerator / denominator).max().item())


def convergence_calibration(cfg: E17Config) -> dict[str, Any]:
    coefficients, parameters, names = calibration_cases()
    primary_initial = decode_periodic(coefficients, resolution=cfg.truth_resolution)
    reference_initial = decode_periodic(coefficients, resolution=cfg.reference_resolution)
    primary = integrate_truth(
        primary_initial,
        parameters,
        internal_step=cfg.truth_step,
        observation_step=cfg.observation_step,
        transitions=cfg.observation_transitions,
    )
    reference = integrate_truth(
        reference_initial,
        parameters,
        internal_step=cfg.reference_step,
        observation_step=cfg.observation_step,
        transitions=cfg.observation_transitions,
    )
    primary_coefficients = project_periodic(primary)
    reference_coefficients = project_periodic(reference)
    primary_comparison = spectral_resample(
        primary,
        target_resolution=cfg.comparison_resolution,
    )
    reference_comparison = spectral_resample(
        reference,
        target_resolution=cfg.comparison_resolution,
    )
    primary_energy = field_energy(primary_comparison)
    reference_energy = field_energy(reference_comparison)
    case_reports = []
    for index, name in enumerate(names):
        coefficient_nrmse = pooled_nrmse(
            primary_coefficients[index, :, :ACTIVE_DIM],
            reference_coefficients[index, :, :ACTIVE_DIM],
        )
        field_nrmse = pooled_nrmse(
            primary_comparison[index],
            reference_comparison[index],
        )
        energy_mismatch = pooled_nrmse(
            primary_energy[index],
            reference_energy[index],
        )
        constant_drift = max(
            float(
                (primary_coefficients[index, :, 0] - primary_coefficients[index, 0, 0])
                .abs()
                .max()
                .item()
            ),
            float(
                (reference_coefficients[index, :, 0] - reference_coefficients[index, 0, 0])
                .abs()
                .max()
                .item()
            ),
        )
        finite = all(
            torch.isfinite(value).all()
            for value in (
                primary[index],
                reference[index],
                primary_coefficients[index],
                reference_coefficients[index],
            )
        )
        report = {
            "name": name,
            "active_coefficient_trajectory_nrmse": coefficient_nrmse,
            "decoded_field_trajectory_nrmse": field_nrmse,
            "relative_energy_trajectory_mismatch": energy_mismatch,
            "maximum_constant_mode_drift": constant_drift,
            "finite": finite,
        }
        report["passed"] = (
            coefficient_nrmse <= cfg.maximum_convergence_nrmse
            and field_nrmse <= cfg.maximum_convergence_nrmse
            and energy_mismatch <= cfg.maximum_energy_trajectory_mismatch
            and constant_drift <= cfg.maximum_constant_drift
            and finite
        )
        case_reports.append(report)
    nonlinear_residual = max(
        nonlinear_energy_rate_residual(primary, parameters),
        nonlinear_energy_rate_residual(reference, parameters),
    )
    checks = {
        "all_cases": all(report["passed"] for report in case_reports),
        "nonlinear_energy_rate": nonlinear_residual <= cfg.maximum_nonlinear_energy_rate_residual,
        "truth_retained_set": sorted(retained_wavenumbers(cfg.truth_resolution))
        == list(range(-71, 72)),
        "reference_retained_set": sorted(retained_wavenumbers(cfg.reference_resolution))
        == list(range(-107, 108)),
        "finite": all(
            torch.isfinite(value).all()
            for value in (
                primary,
                reference,
                primary_comparison,
                reference_comparison,
            )
        ),
    }
    return {
        "cases": case_reports,
        "maximum_nonlinear_energy_rate_residual": nonlinear_residual,
        "checks": checks,
        "passed": all(checks.values()),
        "state_reads": {"training": 0, "validation": 0, "heldout": 0},
    }


def triad_coefficients(
    axis: Literal["x", "y"],
    *,
    resolution: int = 64,
    atol: float = 1e-12,
) -> list[tuple[int, int, int, float]]:
    basis, derivative_x, derivative_y = periodic_basis(resolution)
    flattened_basis = basis.reshape(-1, ACTIVE_DIM)
    derivative = (derivative_x if axis == "x" else derivative_y).reshape(-1, ACTIVE_DIM)
    raw = (
        torch.einsum(
            "ni,nj,nk->ijk",
            flattened_basis,
            flattened_basis,
            derivative,
        )
        / flattened_basis.shape[0]
    )
    support: list[tuple[int, int, int, float]] = []
    for output in range(ACTIVE_DIM):
        for left in range(ACTIVE_DIM):
            for right in range(left, ACTIVE_DIM):
                value = raw[output, left, right]
                if right != left:
                    value = value + raw[output, right, left]
                scalar = -float(value.item())
                if abs(scalar) > atol:
                    support.append((output, left, right, scalar))
    return support


def apply_sparse_quadratic(
    coefficients: torch.Tensor,
    support: list[tuple[int, int, int, float]],
) -> torch.Tensor:
    if coefficients.shape[-1] != ACTIVE_DIM:
        raise ValueError("quadratic action requires 49 active coefficients")
    result = torch.zeros_like(coefficients)
    for output, left, right, value in support:
        result[..., output] += value * coefficients[..., left] * coefficients[..., right]
    return result


def triad_preflight(cfg: E17Config) -> dict[str, Any]:
    reports = {}
    passed = True
    generator = torch.Generator().manual_seed(170017)
    probe = torch.randn(16, ACTIVE_DIM, generator=generator, dtype=torch.float64)
    probe[:, 0] = torch.linspace(-0.1, 0.1, probe.shape[0], dtype=torch.float64)
    for axis in ("x", "y"):
        support = triad_coefficients(
            axis,
            resolution=cfg.triad_quadrature_resolution,
            atol=cfg.triad_support_atol,
        )
        action = apply_sparse_quadratic(probe, support)
        energy_residual = torch.einsum("bi,bi->b", probe, action).abs()
        report = {
            "support_entries": len(support),
            "support_sha256": sha256_bytes(canonical_json_bytes(support)),
            "constant_output_entries": sum(entry[0] == 0 for entry in support),
            "maximum_energy_residual": float(energy_residual.max().item()),
        }
        report["passed"] = (
            report["support_entries"] == cfg.expected_axis_support
            and report["constant_output_entries"] == 0
            and report["maximum_energy_residual"] <= 1e-10
        )
        reports[axis] = report
        passed = passed and report["passed"]
    return {"axes": reports, "passed": passed}


def predecessor_report(repo_root: Path) -> dict[str, Any]:
    files = {}
    for relative, expected in PREDECESSOR_HASHES.items():
        path = repo_root / relative
        actual = sha256_file(path) if path.is_file() else None
        files[relative] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "passed": actual == expected,
        }
    return {"files": files, "passed": all(record["passed"] for record in files.values())}


def sealed_e15_report(repo_root: Path) -> dict[str, Any]:
    path = repo_root / E15_RESULT_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    evaluation = payload["evaluations"][E15_PACKAGE]
    gates = evaluation["gates"]
    checks = {
        "classification": payload["classification"] == E15_CLASSIFICATION,
        "model_sha256": evaluation["model_sha256"] == E15_MODEL_SHA256,
        "generator_sha256": evaluation["generator_sha256"] == E15_GENERATOR_SHA256,
        "gate_names": tuple(sorted(gates)) == tuple(sorted(E15_RECOVERY_GATES)),
        "all_eight_gates": all(gates[name] is True for name in E15_RECOVERY_GATES),
        "recovery_pass": evaluation["recovery_pass"] is True,
        "heldout_zero": payload["state_reads"]["heldout"] == 0,
    }
    return {
        "package": E15_PACKAGE,
        "checks": checks,
        "passed": all(checks.values()),
        "model_sha256": evaluation["model_sha256"],
        "generator_sha256": evaluation["generator_sha256"],
        "gates": gates,
    }


def configure_runtime() -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    record = {
        "intraop_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
    }
    record["passed"] = record == {
        "intraop_threads": 1,
        "interop_threads": 1,
    }
    return record


def source_state(repo_root: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    records = {}
    for relative in E17_SOURCE_PATHS:
        working = (repo_root / relative).read_bytes()
        committed = subprocess.run(
            ["git", "show", f"{head}:{relative}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
        records[relative] = {
            "working_sha256": sha256_bytes(working),
            "head_sha256": sha256_bytes(committed),
            "matches_head": working == committed,
        }
    return {
        "head": head,
        "clean": not bool(status.strip()),
        "sources": records,
        "sources_match_head": all(record["matches_head"] for record in records.values()),
    }


def classify(
    *,
    preflight_passed: bool,
    complete: bool,
    representation_passed: bool,
    candidate_passed: bool,
    coverage_passed: bool = True,
    finite: bool = True,
    boundary_passed: bool = True,
) -> str:
    if not preflight_passed:
        return "preflight_failed"
    if not (complete and coverage_passed and finite and boundary_passed):
        return "incomplete"
    if not representation_passed:
        return "latent_closure_insufficient"
    if not candidate_passed:
        return "quadratic_identification_failed"
    return "constrained_quadratic_closure_qualified"


def prestate_report(repo_root: Path, cfg: E17Config) -> dict[str, Any]:
    predecessors = predecessor_report(repo_root)
    e15 = sealed_e15_report(repo_root)
    triads = triad_preflight(cfg)
    masks = {
        str(resolution): {
            "retained_wavenumbers": list(retained_wavenumbers(resolution)),
            "retained_per_axis": len(retained_wavenumbers(resolution)),
            "retained_tensor_entries": int(dealias_mask(resolution).sum().item()),
        }
        for resolution in (cfg.truth_resolution, cfg.reference_resolution)
    }
    checks = {
        "predecessors": predecessors["passed"],
        "sealed_e15": e15["passed"],
        "triads": triads["passed"],
        "truth_mask": sorted(masks[str(cfg.truth_resolution)]["retained_wavenumbers"])
        == list(range(-71, 72)),
        "reference_mask": sorted(masks[str(cfg.reference_resolution)]["retained_wavenumbers"])
        == list(range(-107, 108)),
    }
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "phase": "source_only_prestate",
        "config": asdict(cfg),
        "predecessors": predecessors,
        "sealed_e15": e15,
        "triads": triads,
        "masks": masks,
        "checks": checks,
        "passed": all(checks.values()),
        "state_reads": {"e15_predecessor": 0, "training": 0, "validation": 0, "heldout": 0},
    }


def calibration_run_report(
    repo_root: Path,
    cfg: E17Config,
    *,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    sources = source_state(repo_root)
    prestate = prestate_report(repo_root, cfg)
    preflight_checks = {
        "runtime": runtime.get("passed") is True,
        "clean_head": sources["clean"],
        "sources_match_head": sources["sources_match_head"],
        "prestate": prestate["passed"],
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "phase": "analytic_convergence_calibration",
        "config": asdict(cfg),
        "runtime": runtime,
        "source_state": sources,
        "prestate": prestate,
        "preflight_checks": preflight_checks,
        "state_reads": {"training": 0, "validation": 0, "heldout": 0},
    }
    if not all(preflight_checks.values()):
        report["calibration"] = None
        report["passed"] = False
        return report
    calibration = convergence_calibration(cfg)
    report["calibration"] = calibration
    report["passed"] = calibration["passed"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the E17 quadratic closure protocol")
    parser.add_argument(
        "--prestate-only",
        action="store_true",
        help="run only zero-scientific-state source and structure preflights",
    )
    parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="run the literal analytic truth-convergence calibration only",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prestate_only == args.calibration_only:
        raise SystemExit(
            "select exactly one of --prestate-only or --calibration-only; "
            "E17 scientific execution is not implemented"
        )
    runtime = configure_runtime()
    repo_root = Path(__file__).resolve().parents[1]
    cfg = E17Config()
    report = (
        {
            **prestate_report(repo_root, cfg),
            "runtime": runtime,
        }
        if args.prestate_only
        else calibration_run_report(repo_root, cfg, runtime=runtime)
    )
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
