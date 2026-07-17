import math

import pytest
import torch

from ups.eval.latent_qualification import (
    cross_discretization_codec_report,
    paired_latent_report,
)


def test_paired_latent_report_recovers_exact_identity() -> None:
    generator = torch.Generator().manual_seed(17)
    latents = torch.randn(12, 4, 8, generator=generator)

    report = paired_latent_report(latents, latents.clone())

    assert report["standardized_pair_rmse"] == 0.0
    assert math.isclose(report["linear_cka"], 1.0, rel_tol=1e-12)
    assert report["retrieval"]["symmetric_top1"] == 1.0


def test_paired_report_exposes_mispaired_state_identity() -> None:
    latents = torch.eye(8).reshape(8, 2, 4)
    mispaired = latents.roll(1, dims=0)

    report = paired_latent_report(latents, mispaired)

    assert report["retrieval"]["symmetric_top1"] == 0.0
    assert report["standardized_pair_rmse"] > 0.0


def test_codec_report_separates_within_and_cross_decoding() -> None:
    target_grid = torch.tensor([[[1.0], [2.0]], [[2.0], [4.0]]])
    target_mesh = torch.tensor([[[1.5], [2.5], [3.5]], [[3.0], [5.0], [7.0]]])
    predictions = {
        "grid": {
            "grid": target_grid.clone(),
            "mesh": torch.zeros_like(target_mesh),
        },
        "mesh": {
            "grid": torch.zeros_like(target_grid),
            "mesh": target_mesh.clone(),
        },
    }

    report = cross_discretization_codec_report(
        predictions, {"grid": target_grid, "mesh": target_mesh}
    )

    assert report["mean_within_nrmse"] == 0.0
    assert report["mean_cross_nrmse"] == 1.0
    assert report["cross_to_within_ratio"] is None


def test_codec_report_fails_closed_without_paired_discretizations() -> None:
    target = torch.ones(2, 3, 1)
    with pytest.raises(ValueError, match="at least two discretizations"):
        cross_discretization_codec_report({"grid": {"grid": target}}, {"grid": target})
