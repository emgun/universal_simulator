from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import torch
from torch import nn

from scripts import run_darcy_fno_conditioning_ablation as ablation


class TinyFNO(nn.Module):
    def __init__(
        self,
        *,
        n_modes: tuple[int, ...],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int,
    ) -> None:
        super().__init__()
        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.net = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def _synthetic():
    beta = torch.tensor([0.01, 0.1, 1.0, 10.0, 100.0])
    coefficients = torch.arange(5 * 4 * 4, dtype=torch.float32).reshape(5, 1, 4, 4) / 80
    targets = coefficients * 0.5 + torch.log10(beta).view(-1, 1, 1, 1) * 0.1
    return coefficients, targets, beta


def test_conditioning_arms_preserve_coefficient_and_add_train_normalized_channels():
    coefficients, _, beta = _synthetic()
    normalizer = ablation.BetaNormalizer.fit(beta)

    unconditioned = ablation.conditioned_inputs(coefficients, beta, arm="U", normalizer=normalizer)
    conditioned = ablation.conditioned_inputs(coefficients, beta, arm="K", normalizer=normalizer)

    assert unconditioned.data_ptr() == coefficients.data_ptr()
    assert conditioned.shape == (5, 3, 4, 4)
    assert torch.equal(conditioned[:, :1], coefficients)
    assert torch.allclose(conditioned[:, 1].mean(), torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(conditioned[:, 1].std(unbiased=False), torch.tensor(1.0))
    assert torch.equal(conditioned[:, 2], torch.ones_like(conditioned[:, 2]))


@pytest.mark.parametrize("bad_beta", [torch.tensor([0.0, 1.0]), torch.tensor([-1.0, 1.0])])
def test_beta_normalization_rejects_nonpositive_values(bad_beta):
    with pytest.raises(ValueError, match="finite, positive"):
        ablation.BetaNormalizer.fit(bad_beta)


def test_collect_requires_explicit_steady_targets_and_scalar_beta():
    coefficient = torch.zeros(1, 4, 4, 1)
    target = coefficient + 1
    dataset = [
        {
            "fields": coefficient,
            "targets": target,
            "params": {"beta": torch.tensor(1.0)},
        }
    ]

    coefficients, targets, beta = ablation._collect(dataset)

    assert coefficients.shape == targets.shape == (1, 1, 4, 4)
    assert beta.tolist() == [1.0]
    with pytest.raises(ValueError, match="explicit solution targets"):
        ablation._collect([{"fields": coefficient, "params": {"beta": torch.tensor(1.0)}}])


def test_train_arm_uses_one_continuous_fixed_budget_trajectory_for_each_arm():
    coefficients, targets, beta = _synthetic()
    normalizer = ablation.BetaNormalizer.fit(beta)

    u_models, u_fit = ablation.train_arm(
        coefficients,
        targets,
        beta,
        arm="U",
        normalizer=normalizer,
        hidden_channels=2,
        fourier_modes=2,
        n_layers=1,
        learning_rate=0.01,
        weight_decay=0.0,
        batch_size=2,
        device="cpu",
        fno_cls=TinyFNO,
    )
    k_models, k_fit = ablation.train_arm(
        coefficients,
        targets,
        beta,
        arm="K",
        normalizer=normalizer,
        hidden_channels=2,
        fourier_modes=2,
        n_layers=1,
        learning_rate=0.01,
        weight_decay=0.0,
        batch_size=2,
        device="cpu",
        fno_cls=TinyFNO,
    )

    assert tuple(u_models) == tuple(k_models) == ablation.RUNG_EPOCHS
    assert len(u_fit["epoch_train_mse"]) == len(k_fit["epoch_train_mse"]) == 24
    assert u_fit["optimizer_steps"] == k_fit["optimizer_steps"] == 72
    assert u_fit["examples_seen"] == k_fit["examples_seen"] == 120
    assert u_fit["sample_order"] == k_fit["sample_order"]
    assert u_fit["input_channels"] == 1
    assert k_fit["input_channels"] == 3


def test_evaluation_reports_primary_raw_and_corrected_metrics_per_beta():
    coefficients, targets, beta = _synthetic()
    normalizer = ablation.BetaNormalizer.fit(beta)
    model = TinyFNO(n_modes=(2, 2), in_channels=1, out_channels=1, hidden_channels=2, n_layers=1)
    with torch.no_grad():
        model.net.weight.fill_(0.5)
        model.net.bias.zero_()

    metrics = ablation.evaluate_arm(
        ablation.DarcyFNOAdapter(model),
        coefficients,
        targets,
        beta,
        arm="U",
        normalizer=normalizer,
    )

    assert metrics["primary_metric"] == "decoded_solution_nrmse"
    assert len(metrics["per_beta"]) == 5
    assert {item["beta"] for item in metrics["per_beta"]} == set(beta.tolist())
    assert all(item["element_count"] == 16 for item in metrics["per_beta"])
    assert metrics["maximum_corrected_spread_ratio"] >= 0


def test_counterfactual_and_shuffled_beta_diagnostics_are_deterministic():
    coefficients, targets, beta = _synthetic()
    normalizer = ablation.BetaNormalizer.fit(beta)
    models = {
        "U": ablation.build_model(
            input_channels=1,
            grid_shape=(4, 4),
            hidden_channels=2,
            fourier_modes=2,
            n_layers=1,
            fno_cls=TinyFNO,
        ),
        "K": ablation.build_model(
            input_channels=3,
            grid_shape=(4, 4),
            hidden_channels=2,
            fourier_modes=2,
            n_layers=1,
            fno_cls=TinyFNO,
        ),
    }
    first = ablation.beta_diagnostics(
        selected_models=models,
        coefficients=coefficients,
        targets=targets,
        beta=beta,
        normalizer=normalizer,
        device="cpu",
    )
    second = ablation.beta_diagnostics(
        selected_models=models,
        coefficients=coefficients,
        targets=targets,
        beta=beta,
        normalizer=normalizer,
        device="cpu",
    )

    assert first == second
    assert first["counterfactual_beta_sensitivity"]["U"]["prediction_rms_from_first_beta"] == 0.0
    assert len(first["deterministic_shuffled_beta"]["permutation_sha256"]) == 64
    assert len(first["deterministic_shuffled_beta"]["per_beta_true_regime"]) == 5


def test_checkpoint_records_content_hash(tmp_path):
    model = ablation.build_model(
        input_channels=1,
        grid_shape=(4, 4),
        hidden_channels=2,
        fourier_modes=2,
        n_layers=1,
        fno_cls=TinyFNO,
    )
    record = ablation._checkpoint(tmp_path / "rung.pt", model, arm="U", epoch=3, fit={})

    assert Path(record["path"]).is_file()
    assert len(record["sha256"]) == 64
    payload = torch.load(record["path"], weights_only=True)
    assert payload["arm"] == "U"
    assert payload["epoch"] == 3


def test_cli_has_no_split_or_task_surface_and_overwrite_refusal_precedes_data_access(tmp_path):
    parser = ablation.build_parser()
    destinations = {action.dest for action in parser._actions}
    assert not {"split", "train_split", "eval_split", "task", "tasks"} & destinations
    output = tmp_path / "existing"
    output.mkdir()
    args = Namespace(
        output_dir=str(output),
        training_lock="not-read.json",
        data_root=str(tmp_path),
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ablation.run(args, fno_cls=TinyFNO)
