from __future__ import annotations

import argparse

import pytest
import torch
from torch import nn

from scripts import run_darcy_conditioned_uno_ablation as d4


def test_uno_runtime_uses_warning_only_for_unsupported_cuda_kernels(monkeypatch):
    calls: list[tuple[bool, bool]] = []
    monkeypatch.setattr(d4.d2, "configure_deterministic_runtime", lambda: None)
    monkeypatch.setattr(
        d4.torch,
        "use_deterministic_algorithms",
        lambda enabled, *, warn_only=False: calls.append((enabled, warn_only)),
    )

    d4.configure_deterministic_runtime()

    assert calls == [(True, True)]


class TinyUNO(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: int, **_: object) -> None:
        super().__init__()
        self.net = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def _args(**changes: object) -> argparse.Namespace:
    values = {
        "hidden_channels": 16,
        "fourier_modes": 16,
        "n_layers": 4,
        "lifting_channels": 32,
        "projection_channels": 32,
        "channel_mlp_skip": "linear",
    }
    values.update(changes)
    return argparse.Namespace(**values)


def test_d4_model_is_direct_conditioned_three_to_one_uno() -> None:
    model = d4.build_model(grid_shape=(128, 128), uno_cls=TinyUNO)

    prediction = model(torch.randn(2, 3, 128, 128))

    assert prediction.shape == (2, 1, 128, 128)


def test_real_d4_uno_preserves_128_square_grid() -> None:
    pytest.importorskip("neuralop")
    model = d4.build_model(grid_shape=(128, 128))

    with torch.no_grad():
        prediction = model(torch.randn(1, 3, 128, 128))

    assert prediction.shape == (1, 1, 128, 128)


def test_d4_model_spec_freezes_audited_uno_recipe() -> None:
    spec = d4.model_spec(_args())

    assert spec == {
        "implementation": "neuralop.models.UNO",
        "in_channels": 3,
        "out_channels": 1,
        "hidden_channels": 16,
        "fourier_modes": 16,
        "n_layers": 4,
        "lifting_channels": 32,
        "projection_channels": 32,
        "channel_mlp_skip": "linear",
        "identity_scaling": False,
        "residual": False,
    }


def test_d4_parser_freezes_batch_and_requires_plan() -> None:
    parser = d4.build_parser()
    args = parser.parse_args(
        ["--data-root", "data", "--output-dir", "out", "--plan-sha256", "a" * 64]
    )
    assert args.batch_size == 10
    assert args.hidden_channels == 16
    assert args.lifting_channels == 32
    with pytest.raises(SystemExit):
        parser.parse_args(["--data-root", "data", "--output-dir", "out"])


def test_d4_completed_checkpoint_chain_resumes_without_new_updates(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(d4, "RUNG_EPOCHS", (1, 2))
    beta = torch.tensor(
        [value for value in d4.d3.EXPECTED_BETAS for _ in range(2)], dtype=torch.float32
    )
    coefficients = torch.randn(10, 1, 4, 4, generator=torch.Generator().manual_seed(5))
    targets = coefficients * 0.25
    normalizer = d4.BetaNormalizer.fit(beta)
    args = _args(
        device="cpu",
        learning_rate=1e-3,
        weight_decay=1e-4,
        resume=False,
    )
    bindings = d4.CheckpointBindings(
        model_spec=d4.model_spec(args),
        optimizer_spec={"kind": "AdamW"},
        normalizer_spec=normalizer.as_dict(),
        plan_fingerprint="plan",
        data_fingerprint="data",
        source_fingerprint="source",
        runtime_fingerprint="runtime",
    )
    checkpoint_dir = tmp_path / "checkpoints"

    first_models, first_fit = d4.train_arm(
        coefficients,
        targets,
        beta,
        normalizer=normalizer,
        args=args,
        checkpoint_dir=checkpoint_dir,
        checkpoint_bindings=bindings,
        uno_cls=TinyUNO,
    )
    args.resume = True
    resumed_models, resumed_fit = d4.train_arm(
        coefficients,
        targets,
        beta,
        normalizer=normalizer,
        args=args,
        checkpoint_dir=checkpoint_dir,
        checkpoint_bindings=bindings,
        uno_cls=TinyUNO,
    )

    assert set(first_models) == set(resumed_models) == {1, 2}
    assert resumed_fit["optimizer_steps"] == first_fit["optimizer_steps"]
    assert resumed_fit["epoch_train_objective"] == first_fit["epoch_train_objective"]
    for epoch in (1, 2):
        for expected, observed in zip(
            first_models[epoch].parameters(), resumed_models[epoch].parameters(), strict=True
        ):
            assert torch.equal(expected, observed)
