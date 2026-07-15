from __future__ import annotations

import torch

from scripts import run_darcy_fno_regime_balanced_objective as d3


def _beta(samples_per_regime: int = 4) -> torch.Tensor:
    return torch.tensor(
        [value for value in d3.EXPECTED_BETAS for _ in range(samples_per_regime)],
        dtype=torch.float32,
    )


def test_regime_complete_batches_are_balanced_complete_and_deterministic():
    beta = _beta()
    first = list(d3.regime_complete_batches(beta, generator=torch.Generator().manual_seed(17)))
    second = list(d3.regime_complete_batches(beta, generator=torch.Generator().manual_seed(17)))

    assert len(first) == 2
    assert all(torch.equal(left, right) for left, right in zip(first, second, strict=True))
    assert sorted(torch.cat(first).tolist()) == list(range(len(beta)))
    for indices in first:
        assert len(indices) == 10
        values, counts = torch.unique(beta[indices], return_counts=True)
        assert torch.allclose(values, torch.tensor(d3.EXPECTED_BETAS), rtol=1e-6, atol=1e-8)
        assert counts.tolist() == [2, 2, 2, 2, 2]


def test_regime_complete_batches_reject_imbalanced_or_odd_regimes():
    beta = torch.cat((_beta(), torch.tensor([0.01])))
    try:
        list(d3.regime_complete_batches(beta, generator=torch.Generator().manual_seed(17)))
    except ValueError as error:
        assert "positive even sample count" in str(error)
    else:
        raise AssertionError("odd regime count must be rejected")


def test_registered_objectives_match_exact_formula_and_gradients():
    beta = _beta(samples_per_regime=2)
    target = torch.zeros(10, 1, 1, 1)
    prediction = (
        torch.tensor([value for value in (1.0, 2.0, 3.0, 4.0, 5.0) for _ in range(2)])
        .view_as(target)
        .requires_grad_()
    )

    mean_loss, per_regime = d3.regime_objective(prediction, target, beta, arm="R-mean")
    minimax_loss, repeated = d3.regime_objective(prediction, target, beta, arm="B-minimax")

    expected = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0])
    assert torch.equal(per_regime, expected)
    assert torch.equal(repeated, expected)
    assert mean_loss == expected.mean()
    assert minimax_loss == 0.5 * expected.mean() + 0.5 * expected.max()
    minimax_loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_objective_requires_regime_complete_batch_and_known_arm():
    beta = _beta(samples_per_regime=2)
    values = torch.zeros(10, 1, 1, 1)
    for bad_beta, arm in ((beta[:-1], "R-mean"), (beta, "unknown")):
        try:
            d3.regime_objective(values[: len(bad_beta)], values[: len(bad_beta)], bad_beta, arm=arm)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid objective contract must be rejected")


def test_d3_contract_constants_are_frozen():
    assert d3.ARMS == ("R-mean", "B-minimax")
    assert d3.RUNG_EPOCHS == (3, 6, 12, 24, 48, 96, 192, 384)
    assert d3.BATCH_SIZE == 10
    assert d3.SAMPLES_PER_REGIME_PER_BATCH == 2


def test_frozen_plan_batch_size_is_accepted_by_cli():
    args = d3.build_parser().parse_args(
        [
            "--data-root",
            "data",
            "--output-dir",
            "output",
            "--plan-sha256",
            "0" * 64,
            "--batch-size",
            "10",
        ]
    )
    assert args.batch_size == d3.BATCH_SIZE
