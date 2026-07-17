import math

import torch

from scripts import train as train_runtime
from scripts.run_universal_latent_codec_audit import (
    _checkpoint_alias_evidence,
    latent_geometry,
    linear_cka,
    summarize_errors,
    task_probe,
)


def test_error_summary_is_deterministic() -> None:
    summary = summarize_errors(
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.4, 0.6, 0.8],
        numerators=[1.0, 3.0],
        denominators=[4.0, 12.0],
    )

    assert summary["sample_count"] == 4
    assert math.isclose(summary["sample_mean_nrmse"], 0.25)
    assert math.isclose(summary["sample_median_nrmse"], 0.2)
    assert math.isclose(summary["sample_mean_spectral_nrmse"], 0.5)
    assert math.isclose(summary["global_nrmse"], 0.5)


def test_latent_geometry_detects_low_rank_representation() -> None:
    scalar = torch.linspace(-1.0, 1.0, 12).view(12, 1, 1)
    low_rank = scalar * torch.ones(1, 4, 5)

    geometry = latent_geometry(low_rank)

    assert geometry["effective_rank"] == 1.0
    assert geometry["stable_rank"] == 1.0
    assert geometry["fraction_dimensions_below_1e_6_max_variance"] == 0.0


def test_linear_cka_is_one_for_orthogonally_equivalent_latents() -> None:
    generator = torch.Generator().manual_seed(17)
    left = torch.randn(20, 3, 4, generator=generator)
    rotation, _ = torch.linalg.qr(torch.randn(12, 12, generator=generator))
    right = (left.reshape(20, 12) @ rotation).reshape(20, 3, 4)

    assert math.isclose(linear_cka(left, right), 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_task_probe_recovers_separated_classes() -> None:
    generator = torch.Generator().manual_seed(17)
    latents = {}
    for label, task in enumerate(("advection1d", "burgers1d", "darcy2d")):
        values = torch.randn(12, 2, 3, generator=generator) * 0.01
        values[:, :, label] += 10.0
        latents[task] = values

    result = task_probe(latents)

    assert result["accuracy"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["chance_accuracy"] == 1.0 / 3.0


def test_checkpoint_alias_does_not_masquerade_as_pre_joint_delta(tmp_path) -> None:
    base = tmp_path / "encoder.pt"
    joint = tmp_path / "encoder_joint.pt"
    state = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    torch.save(state, base)
    torch.save(state, joint)

    evidence = _checkpoint_alias_evidence(base, joint)

    assert evidence["tensor_values_equal"]
    assert not evidence["pre_joint_state_recoverable_from_base_checkpoint"]


def test_decoder_codec_supervision_includes_darcy_solution() -> None:
    coefficient = torch.zeros(2, 1, 4, 1)
    solution = torch.ones(2, 1, 4, 1)

    supervised = train_runtime._decoder_codec_supervision(
        coefficient,
        solution,
        task_name="darcy2d",
        canonical_steady=True,
    )

    assert supervised.shape == (2, 2, 4, 1)
    torch.testing.assert_close(supervised[:, 0], coefficient[:, 0])
    torch.testing.assert_close(supervised[:, 1], solution[:, 0])
