from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from ups.training.resumable_checkpoint import (
    CheckpointBindings,
    CheckpointCompatibilityError,
    CheckpointIntegrityError,
    TrainingProgress,
    checkpoint_record_path,
    load_training_checkpoint,
    save_training_checkpoint,
    verify_checkpoint_record,
)


def _bindings(**changes: str) -> CheckpointBindings:
    values = {
        "model_spec": {"kind": "linear-dropout", "width": 4},
        "optimizer_spec": {"kind": "AdamW", "lr": 0.01},
        "normalizer_spec": {"kind": "none"},
        "plan_fingerprint": "plan-v1",
        "data_fingerprint": "data-v1",
        "source_fingerprint": "source-v1",
        "runtime_fingerprint": "torch-cpu-v1",
    }
    values.update(changes)
    return CheckpointBindings(**values)


def _make_training(seed: int = 11):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.Tanh(),
        torch.nn.Dropout(0.25),
        torch.nn.Linear(8, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    sampler = torch.Generator(device="cpu").manual_seed(37)
    return model, optimizer, sampler


def _train_epochs(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler: torch.Generator,
    *,
    first_epoch: int,
    last_epoch: int,
    history: list[dict[str, float | int]],
) -> tuple[int, int]:
    features = torch.arange(80, dtype=torch.float32).reshape(20, 4) / 80
    targets = features.square().sum(dim=1, keepdim=True)
    steps = 0
    examples = 0
    model.train()
    for epoch in range(first_epoch, last_epoch + 1):
        order = torch.randperm(len(features), generator=sampler)
        total = 0.0
        for indices in order.split(5):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features[indices])
            loss = torch.nn.functional.mse_loss(prediction, targets[indices])
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
            examples += len(indices)
        history.append({"epoch": epoch, "loss_sum": total})
    return steps, examples


def test_cpu_resume_is_exactly_equivalent_to_uninterrupted_training(tmp_path: Path) -> None:
    full_model, full_optimizer, full_sampler = _make_training()
    full_history: list[dict[str, float | int]] = []
    full_steps, full_examples = _train_epochs(
        full_model,
        full_optimizer,
        full_sampler,
        first_epoch=1,
        last_epoch=6,
        history=full_history,
    )

    interrupted_model, interrupted_optimizer, interrupted_sampler = _make_training()
    partial_history: list[dict[str, float | int]] = []
    partial_steps, partial_examples = _train_epochs(
        interrupted_model,
        interrupted_optimizer,
        interrupted_sampler,
        first_epoch=1,
        last_epoch=3,
        history=partial_history,
    )
    checkpoint = tmp_path / "epoch-3.pt"
    record = save_training_checkpoint(
        checkpoint,
        model=interrupted_model,
        optimizer=interrupted_optimizer,
        sampler_generator=interrupted_sampler,
        progress=TrainingProgress(3, partial_steps, partial_examples, partial_history),
        bindings=_bindings(),
    )

    resumed_model, resumed_optimizer, resumed_sampler = _make_training(seed=999)
    loaded = load_training_checkpoint(
        checkpoint,
        model=resumed_model,
        optimizer=resumed_optimizer,
        sampler_generator=resumed_sampler,
        expected_bindings=_bindings(),
    )
    resumed_history = list(loaded.progress.history)
    added_steps, added_examples = _train_epochs(
        resumed_model,
        resumed_optimizer,
        resumed_sampler,
        first_epoch=4,
        last_epoch=6,
        history=resumed_history,
    )

    assert record.checkpoint_sha256 == loaded.record.checkpoint_sha256
    assert loaded.progress.completed_epoch == 3
    assert loaded.progress.steps + added_steps == full_steps
    assert loaded.progress.examples + added_examples == full_examples
    assert resumed_history == full_history
    for full, resumed in zip(full_model.parameters(), resumed_model.parameters(), strict=True):
        assert torch.equal(full, resumed)
    assert (
        full_optimizer.state_dict()["state"].keys()
        == resumed_optimizer.state_dict()["state"].keys()
    )
    for parameter_id, full_state in full_optimizer.state_dict()["state"].items():
        for key, value in full_state.items():
            resumed_value = resumed_optimizer.state_dict()["state"][parameter_id][key]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, resumed_value)
            else:
                assert value == resumed_value
    assert torch.equal(full_sampler.get_state(), resumed_sampler.get_state())


def test_neuraloperator_checkpoint_is_tensor_only_and_weights_only_safe(tmp_path: Path) -> None:
    neuralop = pytest.importorskip("neuralop")
    model = neuralop.models.FNO(
        n_modes=(2, 2),
        in_channels=1,
        out_channels=1,
        hidden_channels=2,
        n_layers=1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    sampler = torch.Generator().manual_seed(37)
    checkpoint = tmp_path / "neuraloperator.pt"
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        sampler_generator=sampler,
        progress=TrainingProgress(0, 0, 0, []),
        bindings=_bindings(model_spec={"kind": "neuraloperator-fno"}),
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert "_metadata" not in payload["model_state"]
    assert all(isinstance(value, torch.Tensor) for value in payload["model_state"].values())

    target = neuralop.models.FNO(
        n_modes=(2, 2),
        in_channels=1,
        out_channels=1,
        hidden_channels=2,
        n_layers=1,
    )
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=0.01)
    load_training_checkpoint(
        checkpoint,
        model=target,
        optimizer=target_optimizer,
        sampler_generator=torch.Generator().manual_seed(99),
        expected_bindings=_bindings(model_spec={"kind": "neuraloperator-fno"}),
    )
    for expected, observed in zip(model.parameters(), target.parameters(), strict=True):
        assert torch.equal(expected, observed)


def test_load_fails_closed_before_mutation_on_binding_mismatch(tmp_path: Path) -> None:
    model, optimizer, sampler = _make_training()
    checkpoint = tmp_path / "checkpoint.pt"
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        sampler_generator=sampler,
        progress=TrainingProgress(0, 0, 0, []),
        bindings=_bindings(),
    )
    target_model, target_optimizer, target_sampler = _make_training(seed=91)
    before = copy.deepcopy(target_model.state_dict())

    with pytest.raises(CheckpointCompatibilityError, match="bindings"):
        load_training_checkpoint(
            checkpoint,
            model=target_model,
            optimizer=target_optimizer,
            sampler_generator=target_sampler,
            expected_bindings=_bindings(data_fingerprint="wrong-data"),
        )

    for key, value in before.items():
        assert torch.equal(value, target_model.state_dict()[key])


def test_tampered_binary_and_record_are_rejected(tmp_path: Path) -> None:
    model, optimizer, sampler = _make_training()
    checkpoint = tmp_path / "checkpoint.pt"
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        sampler_generator=sampler,
        progress=TrainingProgress(0, 0, 0, []),
        bindings=_bindings(),
    )
    with checkpoint.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(CheckpointIntegrityError, match="byte count"):
        load_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            sampler_generator=sampler,
            expected_bindings=_bindings(),
        )

    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        sampler_generator=sampler,
        progress=TrainingProgress(0, 0, 0, []),
        bindings=_bindings(),
    )
    record_path = checkpoint_record_path(checkpoint)
    record = json.loads(record_path.read_text())
    record["checkpoint_bytes"] += 1
    record_path.write_text(json.dumps(record))
    with pytest.raises(CheckpointIntegrityError, match="self-hash"):
        load_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            sampler_generator=sampler,
            expected_bindings=_bindings(),
        )


def test_normalizer_state_and_parent_lineage_are_restored_and_bound(tmp_path: Path) -> None:
    model, optimizer, sampler = _make_training()
    normalizer = torch.nn.BatchNorm1d(4)
    normalizer.running_mean.copy_(torch.arange(4, dtype=torch.float32))
    parent_sha = "a" * 64
    checkpoint = tmp_path / "child.pt"
    record = save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        normalizer=normalizer,
        sampler_generator=sampler,
        progress=TrainingProgress(2, 8, 40, [{"epoch": 2}]),
        bindings=_bindings(normalizer_spec={"kind": "BatchNorm1d", "features": 4}),
        parent_checkpoint_sha256=parent_sha,
    )
    assert (
        verify_checkpoint_record(
            checkpoint, expected_checkpoint_sha256=record.checkpoint_sha256
        ).parent_checkpoint_sha256
        == parent_sha
    )

    target_model, target_optimizer, target_sampler = _make_training(seed=99)
    target_normalizer = torch.nn.BatchNorm1d(4)
    load_training_checkpoint(
        checkpoint,
        model=target_model,
        optimizer=target_optimizer,
        normalizer=target_normalizer,
        sampler_generator=target_sampler,
        expected_bindings=_bindings(normalizer_spec={"kind": "BatchNorm1d", "features": 4}),
        expected_parent_checkpoint_sha256=parent_sha,
        expected_checkpoint_sha256=record.checkpoint_sha256,
    )
    assert torch.equal(target_normalizer.running_mean, normalizer.running_mean)

    with pytest.raises(CheckpointCompatibilityError, match="parent checkpoint"):
        load_training_checkpoint(
            checkpoint,
            model=target_model,
            optimizer=target_optimizer,
            normalizer=target_normalizer,
            sampler_generator=target_sampler,
            expected_bindings=_bindings(normalizer_spec={"kind": "BatchNorm1d", "features": 4}),
            expected_parent_checkpoint_sha256="b" * 64,
        )
