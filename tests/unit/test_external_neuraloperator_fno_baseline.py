from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from scripts import run_external_neuraloperator_fno_baseline as fno_runner
from scripts.run_external_neuraloperator_fno_baseline import (
    _external_test_measurement_key,
    bind_training_lock,
    build_neuraloperator_fno_model,
    build_parser,
    fno_modes_for_grid,
    run_baseline,
    train_fno_group_model,
    train_fno_groups_with_rungs,
    write_group_checkpoint,
)

ROOT = Path(__file__).resolve().parents[2]
TRAINING_LOCK = ROOT / (
    "docs/data/releases/strat-v1/universal/"
    "9d43d283f04f5b8d17cf6126ad189075c53307e715d7d4f61af440c2fed155c1/"
    "training.lock.json"
)
TRAINING_LOCK_SHA256 = "5666fa8bb646b270d23077d1936c8ee80fa1cfeaf8f5870c55103fc58dd80afd"


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
        if len(n_modes) == 1:
            self.net = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.net = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        return self.net(current)


class TinyMetadataFNO(TinyFNO):
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state["_metadata"] = {"n_modes": self.n_modes}
        return state


def _lock_args(*extra: str):
    return build_parser().parse_args(["--data-lock", str(TRAINING_LOCK), *extra])


def test_bind_training_lock_preserves_matching_config_pinned_identity():
    cfg = {"data": {"data_lock_sha256": TRAINING_LOCK_SHA256}}

    bind_training_lock(cfg, _lock_args("--strict-contract"))

    assert cfg["data"]["data_lock_sha256"] == TRAINING_LOCK_SHA256
    assert Path(cfg["data"]["data_lock_path"]) == TRAINING_LOCK.resolve()


def test_bind_training_lock_rejects_cli_and_config_identity_disagreement():
    cfg = {"data": {"data_lock_sha256": TRAINING_LOCK_SHA256}}

    with pytest.raises(ValueError, match="configured data lock identity disagrees"):
        bind_training_lock(
            cfg,
            _lock_args("--expected-data-lock-sha256", "f" * 64),
        )


def test_bind_training_lock_requires_expected_identity_in_strict_mode():
    with pytest.raises(ValueError, match="require an expected data lock identity"):
        bind_training_lock({}, _lock_args("--strict-contract"))


def test_bind_training_lock_rejects_config_identity_that_does_not_match_lock():
    cfg = {"data": {"data_lock_sha256": "f" * 64}}

    with pytest.raises(ValueError, match="does not match expected SHA-256"):
        bind_training_lock(cfg, _lock_args())


def test_fno_modes_for_grid_uses_1d_for_flat_tasks():
    assert fno_modes_for_grid((1, 64), 16) == (16,)
    assert fno_modes_for_grid((16, 32), 16) == (8, 16)


def test_build_neuraloperator_fno_model_adapts_1d_grid_to_repo_grid_shape():
    model = build_neuraloperator_fno_model(
        channels=1,
        grid_shape=(1, 8),
        hidden_channels=4,
        fourier_modes=4,
        n_layers=2,
        residual=False,
        fno_cls=TinyFNO,
    )

    pred = model(torch.randn(2, 1, 1, 8))

    assert pred.shape == (2, 1, 1, 8)
    assert model.fno.n_modes == (4,)
    assert model.fno.n_layers == 2


def test_train_fno_group_model_can_learn_simple_residual_with_fake_fno():
    generator = torch.Generator().manual_seed(13)
    currents = torch.randn(16, 1, 1, 8, generator=generator)
    targets = currents + 0.25

    model, fit = train_fno_group_model(
        currents,
        targets,
        hidden_channels=4,
        fourier_modes=4,
        n_layers=1,
        residual=True,
        epochs=80,
        learning_rate=0.05,
        batch_size=4,
        seed=7,
        fno_cls=TinyFNO,
    )

    with torch.no_grad():
        mse = torch.mean((model(currents) - targets) ** 2).item()
    assert fit["implementation"] == "neuralop.models.FNO"
    assert fit["n_modes"] == [4]
    assert len(fit["epoch_train_mse"]) == 80
    assert 1 <= fit["best_epoch"] <= 80
    assert mse < 0.01


def test_write_group_checkpoint_records_content_hash(tmp_path):
    model = nn.Conv2d(1, 1, 1)
    record = write_group_checkpoint(
        tmp_path / "models.pt",
        {("task", 1, 8, 1): model},
        model_family="fno",
        fit={"group_count": 1},
    )

    assert Path(record["path"]).is_file()
    assert len(record["sha256"]) == 64
    payload = torch.load(record["path"], weights_only=True)
    assert payload["model_family"] == "fno"


def test_train_fno_group_model_ignores_neuraloperator_state_metadata():
    generator = torch.Generator().manual_seed(23)
    currents = torch.randn(16, 1, 1, 8, generator=generator)
    targets = currents + 0.25

    model, fit = train_fno_group_model(
        currents,
        targets,
        hidden_channels=4,
        fourier_modes=4,
        n_layers=1,
        residual=True,
        epochs=80,
        learning_rate=0.05,
        batch_size=4,
        seed=9,
        fno_cls=TinyMetadataFNO,
    )

    with torch.no_grad():
        mse = torch.mean((model(currents) - targets) ** 2).item()
    assert fit["implementation"] == "neuralop.models.FNO"
    assert mse < 0.01


def test_train_fno_groups_with_rungs_retains_one_continuous_trajectory():
    currents = torch.zeros(4, 1, 1, 4)
    targets = torch.ones_like(currents)
    key = ("burgers1d", 1, 4, 1)

    _, fit, rung_models = train_fno_groups_with_rungs(
        {key: (currents, targets)},
        validation_rungs=[1, 2],
        hidden_channels=2,
        fourier_modes=2,
        n_layers=1,
        residual=False,
        epochs=2,
        learning_rate=0.1,
        weight_decay=0.0,
        batch_size=2,
        seed=3,
        device="cpu",
        fno_cls=TinyFNO,
    )

    epoch1 = rung_models[1][key].fno.net.bias.detach()
    epoch2 = rung_models[2][key].fno.net.bias.detach()
    assert not torch.equal(epoch1, epoch2)
    assert fit["optimizer_steps"] == 4
    assert fit["examples_seen"] == 8


def test_external_neuraloperator_fno_dry_run_writes_contract_summary(tmp_path):
    args = build_parser().parse_args(
        [
            "--dry-run",
            "--config",
            "configs/train_multitask_heterogeneous_light_best.yaml",
            "--name",
            "external_fno_dry_run",
            "--output-root",
            str(tmp_path),
            "--data-root",
            "data/pdebench",
            "--eval-split",
            "test",
            "--tasks",
            "advection1d",
            "burgers1d",
            "darcy2d",
        ]
    )

    summary_path = run_baseline(args)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["status"] == "dry_run"
    assert summary["metrics"] == {}
    assert summary["extra"]["baseline"] == "external_neuraloperator_fno"
    assert summary["extra"]["implementation"] == "neuralop.models.FNO"
    assert summary["extra"]["split"] == "test"
    assert summary["extra"]["epochs"] == 3
    assert summary["extra"]["batch_size"] == 8
    assert summary["extra"]["train_stride"] == 4
    assert "advection1d" in summary["extra"]["command"]
    assert "--dry-run" in summary["extra"]["command"]
    assert summary["details"]["contract"]["published_numbers_directly_comparable"] is False


def test_dry_run_command_records_strict_training_lock_identity(tmp_path):
    args = build_parser().parse_args(
        [
            "--dry-run",
            "--strict-contract",
            "--config",
            "configs/a4_strat_v1_baselines.yaml",
            "--name",
            "strict_lock_command",
            "--output-root",
            str(tmp_path),
            "--data-lock",
            str(TRAINING_LOCK),
            "--expected-data-lock-sha256",
            TRAINING_LOCK_SHA256,
            "--tasks",
            "burgers1d",
        ]
    )

    summary_path = run_baseline(args)
    command = json.loads(summary_path.read_text(encoding="utf-8"))["extra"]["command"]

    assert "--strict-contract" in command
    assert command[command.index("--data-lock") + 1] == str(TRAINING_LOCK)
    assert command[command.index("--expected-data-lock-sha256") + 1] == TRAINING_LOCK_SHA256


def test_validation_rungs_forbid_held_out_split_even_for_dry_run(tmp_path):
    args = build_parser().parse_args(
        [
            "--dry-run",
            "--eval-split",
            "test",
            "--epochs",
            "6",
            "--validation-rungs",
            "3",
            "6",
            "--output-root",
            str(tmp_path),
        ]
    )

    with pytest.raises(ValueError, match="forbid held-out test"):
        run_baseline(args)


def test_refuse_overwrite_blocks_existing_run_directory(tmp_path):
    (tmp_path / "existing").mkdir()
    args = build_parser().parse_args(
        [
            "--dry-run",
            "--refuse-overwrite",
            "--name",
            "existing",
            "--output-root",
            str(tmp_path),
        ]
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_baseline(args)


def test_rung_run_selects_finite_validation_best_and_hashes_every_checkpoint(tmp_path, monkeypatch):
    key = ("burgers1d", 1, 4, 1)

    def model_at(epoch):
        model = nn.Conv2d(1, 1, 1)
        with torch.no_grad():
            model.weight.fill_(epoch)
        return model

    def fake_train(*args, **kwargs):
        fit = {
            "groups": {str(key): {"optimizer_steps": 12, "examples_seen": 24}},
            "group_count": 1,
            "train_frames": 4,
            "optimizer_steps": 12,
            "examples_seen": 24,
        }
        return {key: model_at(6)}, fit, {3: {key: model_at(3)}, 6: {key: model_at(6)}}

    def fake_evaluate(cfg, models, **kwargs):
        epoch = int(next(iter(models.values())).weight.flatten()[0].item())
        return {"decoded_rollout_nrmse": {3: 0.4, 6: 0.6}[epoch]}

    monkeypatch.setattr(fno_runner, "load_neuraloperator_fno_class", lambda: TinyFNO)
    monkeypatch.setattr(
        fno_runner,
        "collect_training_pairs",
        lambda *args, **kwargs: {key: (torch.zeros(1), torch.zeros(1))},
    )
    monkeypatch.setattr(fno_runner, "train_fno_groups_with_rungs", fake_train)
    monkeypatch.setattr(fno_runner, "evaluate_external_fno_baseline", fake_evaluate)
    args = build_parser().parse_args(
        [
            "--name",
            "rung_run",
            "--output-root",
            str(tmp_path),
            "--tasks",
            "burgers1d",
            "--epochs",
            "6",
            "--validation-rungs",
            "3",
            "6",
        ]
    )

    summary = json.loads(run_baseline(args).read_text(encoding="utf-8"))

    assert summary["recipe_adequacy"]["selected_epoch"] == 3
    assert summary["metrics"]["decoded_rollout_nrmse"] == 0.4
    assert [item["epoch"] for item in summary["details"]["validation_history"]] == [3, 6]
    assert set(summary["checkpoints"]["rungs"]) == {"3", "6"}
    assert summary["checkpoints"]["selected"] == summary["checkpoints"]["rungs"]["3"]
    assert all(len(item["sha256"]) == 64 for item in summary["checkpoints"]["rungs"].values())
    assert summary["compute"]["optimizer_steps"] == 12
    assert summary["compute"]["examples_seen"] == 24


def test_live_test_split_requires_explicit_held_out_flag_before_import_or_data(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_external_neuraloperator_fno_baseline.py",
            "--config",
            "configs/train_multitask_heterogeneous_light_best.yaml",
            "--name",
            "blocked_live_test",
            "--output-root",
            str(tmp_path),
            "--data-root",
            "data/pdebench",
            "--eval-split",
            "test",
            "--max-train-samples",
            "1",
            "--max-eval-samples",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "--allow-held-out-test-eval" in proc.stderr


def test_live_test_split_blocks_repeat_external_ledger_before_measurement(tmp_path):
    ledger_path = tmp_path / "external-test-ledger.json"
    args = build_parser().parse_args(
        [
            "--config",
            "configs/train_multitask_heterogeneous_light_best.yaml",
            "--name",
            "repeat_blocked",
            "--output-root",
            str(tmp_path / "out"),
            "--data-root",
            "data/pdebench",
            "--eval-split",
            "test",
            "--tasks",
            "advection1d",
            "burgers1d",
            "darcy2d",
            "--max-train-samples",
            "32",
            "--max-eval-samples",
            "32",
            "--held-out-ledger-json",
            str(ledger_path),
            "--allow-held-out-test-eval",
        ]
    )
    measurement_key = _external_test_measurement_key(
        args=args,
        tasks=["advection1d", "burgers1d", "darcy2d"],
    )
    ledger_path.write_text(
        json.dumps({"measurements": [{"measurement_key": measurement_key}]}),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError, match="held-out external FNO test measurement already recorded"
    ):
        run_baseline(args)


def test_allow_repeat_test_is_explicit_in_command_record(tmp_path):
    args = build_parser().parse_args(
        [
            "--dry-run",
            "--config",
            "configs/train_multitask_heterogeneous_light_best.yaml",
            "--name",
            "external_fno_repeat_debug",
            "--output-root",
            str(tmp_path),
            "--eval-split",
            "test",
            "--allow-repeat-test",
        ]
    )

    summary_path = run_baseline(args)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert "--allow-repeat-test" in summary["extra"]["command"]
