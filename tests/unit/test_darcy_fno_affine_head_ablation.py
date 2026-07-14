from __future__ import annotations

import os
from argparse import Namespace

import pytest
import torch
from torch import nn

from scripts import run_darcy_fno_affine_head_ablation as ablation
from scripts import run_darcy_fno_conditioning_ablation as d1


def test_deterministic_runtime_sets_cublas_workspace_before_cuda(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    ablation.configure_deterministic_runtime()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.are_deterministic_algorithms_enabled()


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


class BatchGuardFNO(TinyFNO):
    maximum_batch_seen = 0
    maximum_batch_allowed = 2

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        type(self).maximum_batch_seen = max(type(self).maximum_batch_seen, len(value))
        if len(value) > type(self).maximum_batch_allowed:
            raise RuntimeError("evaluation batch exceeded bounded contract")
        return super().forward(value)


def _synthetic():
    beta = torch.tensor([0.01, 0.1, 1.0, 10.0, 100.0])
    coefficients = torch.arange(5 * 4 * 4, dtype=torch.float32).reshape(5, 1, 4, 4) / 80
    targets = coefficients * 0.5 + beta.view(-1, 1, 1, 1) * 0.001
    return coefficients, targets, beta


def test_affine_head_reconstructs_h0_plus_train_normalized_raw_beta_h1():
    coefficients, _, beta = _synthetic()
    log_normalizer = ablation.BetaNormalizer.fit(beta)
    raw_normalizer = ablation.RawBetaNormalizer.fit(beta)
    model = ablation.build_model(
        arm="A-affine",
        grid_shape=(4, 4),
        hidden_channels=2,
        fourier_modes=2,
        n_layers=1,
        fno_cls=TinyFNO,
    )
    with torch.no_grad():
        model.fno.net.weight.zero_()
        model.fno.net.bias.copy_(torch.tensor([2.0, 3.0]))

    prediction = ablation.predict(
        model,
        coefficients,
        beta,
        arm="A-affine",
        log_normalizer=log_normalizer,
        raw_normalizer=raw_normalizer,
    )

    expected = 2.0 + raw_normalizer.transform(beta).view(-1, 1, 1, 1) * 3.0
    assert torch.allclose(prediction, expected.expand_as(prediction))
    assert ablation.model_inputs(
        coefficients, beta, arm="A-affine", log_normalizer=log_normalizer
    ).shape == (5, 2, 4, 4)


def test_k_long_matches_d1_channel_contract_and_affine_has_two_basis_outputs():
    coefficients, _, beta = _synthetic()
    log_normalizer = ablation.BetaNormalizer.fit(beta)

    k_inputs = ablation.model_inputs(
        coefficients, beta, arm="K-long", log_normalizer=log_normalizer
    )
    affine_inputs = ablation.model_inputs(
        coefficients, beta, arm="A-affine", log_normalizer=log_normalizer
    )
    k = ablation.build_model(
        arm="K-long",
        grid_shape=(4, 4),
        hidden_channels=2,
        fourier_modes=2,
        n_layers=1,
        fno_cls=TinyFNO,
    )
    affine = ablation.build_model(
        arm="A-affine",
        grid_shape=(4, 4),
        hidden_channels=2,
        fourier_modes=2,
        n_layers=1,
        fno_cls=TinyFNO,
    )

    assert k_inputs.shape == (5, 3, 4, 4)
    assert torch.equal(k_inputs[:, :1], coefficients)
    assert torch.equal(k_inputs[:, 2:3], torch.ones_like(coefficients))
    assert affine_inputs.shape == (5, 2, 4, 4)
    assert k(k_inputs).shape == (5, 1, 4, 4)
    assert affine(affine_inputs).shape == (5, 2, 4, 4)


def test_training_is_matched_through_192_epochs(monkeypatch):
    monkeypatch.setattr(ablation, "RUNG_EPOCHS", (1, 2))
    coefficients, targets, beta = _synthetic()
    log_normalizer = ablation.BetaNormalizer.fit(beta)
    raw_normalizer = ablation.RawBetaNormalizer.fit(beta)
    fits = {}
    for arm in ablation.ARMS:
        models, fit = ablation.train_arm(
            coefficients,
            targets,
            beta,
            arm=arm,
            log_normalizer=log_normalizer,
            raw_normalizer=raw_normalizer,
            hidden_channels=2,
            fourier_modes=2,
            n_layers=1,
            learning_rate=0.01,
            weight_decay=0.0,
            batch_size=2,
            device="cpu",
            fno_cls=TinyFNO,
        )
        assert tuple(models) == (1, 2)
        fits[arm] = fit
    assert fits["K-long"]["optimizer_steps"] == fits["A-affine"]["optimizer_steps"] == 6
    assert fits["K-long"]["examples_seen"] == fits["A-affine"]["examples_seen"] == 10
    assert fits["K-long"]["sample_order"] == fits["A-affine"]["sample_order"]


def test_k_long_is_exact_continuation_of_d1_conditioned_arm(monkeypatch):
    monkeypatch.setattr(ablation, "RUNG_EPOCHS", (1, 2))
    monkeypatch.setattr(d1, "RUNG_EPOCHS", (1, 2))
    coefficients, targets, beta = _synthetic()
    log_normalizer = ablation.BetaNormalizer.fit(beta)
    raw_normalizer = ablation.RawBetaNormalizer.fit(beta)
    common = dict(
        hidden_channels=2,
        fourier_modes=2,
        n_layers=1,
        learning_rate=0.01,
        weight_decay=0.0,
        batch_size=2,
        device="cpu",
        fno_cls=TinyFNO,
    )

    d1_models, d1_fit = d1.train_arm(
        coefficients,
        targets,
        beta,
        arm="K",
        normalizer=log_normalizer,
        **common,
    )
    d2_models, d2_fit = ablation.train_arm(
        coefficients,
        targets,
        beta,
        arm="K-long",
        log_normalizer=log_normalizer,
        raw_normalizer=raw_normalizer,
        **common,
    )

    assert d2_fit["epoch_train_mse"] == d1_fit["epoch_train_mse"]
    assert d2_fit["optimizer_steps"] == d1_fit["optimizer_steps"]
    for key, value in d1_models[2].state_dict().items():
        assert torch.equal(d2_models[2].state_dict()[key], value)


def test_interrupted_rung_checkpoint_resumes_exact_continuous_trajectory(tmp_path, monkeypatch):
    monkeypatch.setattr(ablation, "RUNG_EPOCHS", (1, 2))
    coefficients, targets, beta = _synthetic()
    log_normalizer = ablation.BetaNormalizer.fit(beta)
    raw_normalizer = ablation.RawBetaNormalizer.fit(beta)
    bindings = ablation.CheckpointBindings(
        model_spec={"arm": "K-long"},
        optimizer_spec={"name": "AdamW"},
        normalizer_spec=log_normalizer.as_dict(),
        plan_fingerprint="p",
        data_fingerprint="d",
        source_fingerprint="s",
        runtime_fingerprint="r",
    )
    arguments = dict(
        arm="K-long",
        log_normalizer=log_normalizer,
        raw_normalizer=raw_normalizer,
        hidden_channels=2,
        fourier_modes=2,
        n_layers=1,
        learning_rate=0.01,
        weight_decay=0.0,
        batch_size=2,
        device="cpu",
        fno_cls=TinyFNO,
    )
    uninterrupted, uninterrupted_fit = ablation.train_arm(coefficients, targets, beta, **arguments)
    original_save = ablation.save_training_checkpoint
    calls = 0

    def interrupt_after_first(*args, **kwargs):
        nonlocal calls
        record = original_save(*args, **kwargs)
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated preemption")
        return record

    monkeypatch.setattr(ablation, "save_training_checkpoint", interrupt_after_first)
    with pytest.raises(RuntimeError, match="simulated preemption"):
        ablation.train_arm(
            coefficients,
            targets,
            beta,
            **arguments,
            checkpoint_dir=tmp_path,
            checkpoint_bindings=bindings,
        )
    monkeypatch.setattr(ablation, "save_training_checkpoint", original_save)
    resumed, resumed_fit = ablation.train_arm(
        coefficients,
        targets,
        beta,
        **arguments,
        checkpoint_dir=tmp_path,
        checkpoint_bindings=bindings,
        resume=True,
    )

    assert resumed_fit["epoch_train_mse"] == uninterrupted_fit["epoch_train_mse"]
    assert resumed_fit["optimizer_steps"] == uninterrupted_fit["optimizer_steps"]
    assert resumed_fit["resume_provenance"]["resumed"] is True
    for key, value in uninterrupted[2].state_dict().items():
        assert torch.equal(resumed[2].state_dict()[key], value)
    assert resumed_fit["resumable_checkpoints"]["2"]["parent_checkpoint_sha256"] == (
        resumed_fit["resumable_checkpoints"]["1"]["sha256"]
    )


def test_evaluation_and_causal_diagnostics_cover_exact_beta_regimes():
    coefficients, targets, beta = _synthetic()
    log_normalizer = ablation.BetaNormalizer.fit(beta)
    raw_normalizer = ablation.RawBetaNormalizer.fit(beta)
    models = {
        arm: ablation.build_model(
            arm=arm,
            grid_shape=(4, 4),
            hidden_channels=2,
            fourier_modes=2,
            n_layers=1,
            fno_cls=TinyFNO,
        )
        for arm in ablation.ARMS
    }
    metrics = ablation.evaluate_arm(
        models["A-affine"],
        coefficients,
        targets,
        beta,
        arm="A-affine",
        log_normalizer=log_normalizer,
        raw_normalizer=raw_normalizer,
    )
    diagnostics = ablation.beta_diagnostics(
        selected_models=models,
        coefficients=coefficients,
        targets=targets,
        beta=beta,
        log_normalizer=log_normalizer,
        raw_normalizer=raw_normalizer,
        device="cpu",
    )

    assert [item["beta"] for item in metrics["per_beta"]] == sorted(beta.tolist())
    assert all(item["element_count"] == 16 for item in metrics["per_beta"])
    assert set(diagnostics["deterministic_shuffled_beta"]["arms"]) == set(ablation.ARMS)
    assert set(diagnostics["counterfactual_beta_sensitivity"]) == set(ablation.ARMS)
    assert len(diagnostics["deterministic_shuffled_beta"]["permutation_sha256"]) == 64


def test_evaluation_and_diagnostics_use_bounded_batches_and_release_models_to_cpu():
    coefficients, targets, beta = _synthetic()
    log_normalizer = ablation.BetaNormalizer.fit(beta)
    raw_normalizer = ablation.RawBetaNormalizer.fit(beta)
    BatchGuardFNO.maximum_batch_seen = 0
    models = {
        arm: ablation.build_model(
            arm=arm,
            grid_shape=(4, 4),
            hidden_channels=2,
            fourier_modes=2,
            n_layers=1,
            fno_cls=BatchGuardFNO,
        )
        for arm in ablation.ARMS
    }
    unbounded_reference_model = ablation.build_model(
        arm="A-affine",
        grid_shape=(4, 4),
        hidden_channels=2,
        fourier_modes=2,
        n_layers=1,
        fno_cls=TinyFNO,
    )
    unbounded_reference_model.load_state_dict(models["A-affine"].state_dict())
    reference = ablation.evaluate_arm(
        unbounded_reference_model,
        coefficients,
        targets,
        beta,
        arm="A-affine",
        log_normalizer=log_normalizer,
        raw_normalizer=raw_normalizer,
        batch_size=len(beta),
    )
    bounded = ablation.evaluate_arm(
        models["A-affine"],
        coefficients,
        targets,
        beta,
        arm="A-affine",
        log_normalizer=log_normalizer,
        raw_normalizer=raw_normalizer,
        batch_size=2,
    )
    diagnostics = ablation.beta_diagnostics(
        selected_models=models,
        coefficients=coefficients,
        targets=targets,
        beta=beta,
        log_normalizer=log_normalizer,
        raw_normalizer=raw_normalizer,
        device="cpu",
        batch_size=2,
    )

    # Batch partitioning can change the final floating-point bits of otherwise
    # equivalent CPU kernels, so assert numerical rather than bitwise identity.
    torch.testing.assert_close(
        reference["predictions"],
        bounded["predictions"],
        rtol=1e-6,
        atol=1e-6,
    )
    assert reference["primary_value"] == pytest.approx(bounded["primary_value"], rel=1e-6, abs=1e-7)
    assert bounded["predictions"].shape == targets.shape
    assert BatchGuardFNO.maximum_batch_seen == 2
    assert diagnostics["counterfactual_beta_sensitivity"]
    assert all(next(model.parameters()).device.type == "cpu" for model in models.values())


def test_exact_darcy_hashes_fail_closed():
    class Object:
        def __init__(self, object_id: str, sha256: str) -> None:
            self.object_id = object_id
            self.role = "train"
            self.path = "x.h5"
            self.checksums = {"sha256": sha256}
            self.size_bytes = 1

    runtime = Namespace(
        lock=Namespace(
            objects=[Object(key, value) for key, value in ablation.EXPECTED_DARCY_OBJECTS.items()]
        )
    )
    assert set(ablation._exact_darcy_objects(runtime)) == set(ablation.EXPECTED_DARCY_OBJECTS)
    runtime.lock.objects[0].checksums["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="do not match frozen D2 contract"):
        ablation._exact_darcy_objects(runtime)


def test_cli_has_no_split_task_or_test_surface_and_refuses_overwrite(tmp_path):
    parser = ablation.build_parser()
    destinations = {action.dest for action in parser._actions}
    assert not {"split", "train_split", "eval_split", "test", "task", "tasks"} & destinations
    output = tmp_path / "existing"
    output.mkdir()
    args = Namespace(output_dir=str(output), training_lock="not-read.json", data_root=str(tmp_path))

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ablation.run(args, fno_cls=TinyFNO)


def test_frozen_rungs_and_seed_are_explicit():
    assert ablation.RUNG_EPOCHS == (3, 6, 12, 24, 48, 96, 192)
    assert ablation.SEED == 17
    assert ablation.ARMS == ("K-long", "A-affine")
