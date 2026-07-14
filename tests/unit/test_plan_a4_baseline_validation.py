from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from torch import nn

from scripts import run_external_neuraloperator_fno_baseline as fno_runner
from scripts.plan_a4_baseline_validation import (
    RUNNERS,
    _verify_local_training_cache,
    build_plan,
)

RELEASE = Path(
    "docs/data/releases/strat-v1/universal/"
    "9d43d283f04f5b8d17cf6126ad189075c53307e715d7d4f61af440c2fed155c1"
)


def _args(tmp_path, **overrides):
    values = {
        "training_lock": str(RELEASE / "training.lock.json"),
        "source_manifest": str(RELEASE / "source.manifest.yaml"),
        "protocol_manifest": str(RELEASE / "protocol.manifest.yaml"),
        "config": "configs/a4_strat_v1_baselines.yaml",
        "data_root": str(tmp_path / "training-cache"),
        "output_root": str(tmp_path / "baselines"),
        "neuraloperator_version": "1.0.2",
        "max_train_samples": 288,
        "max_eval_samples": 72,
        "rollout_steps": 16,
        "device": "cuda",
    }
    values.update(overrides)
    return Namespace(**values)


def test_plan_pins_universal_training_lock_and_all_applicable_baselines(tmp_path):
    plan = build_plan(_args(tmp_path))

    assert plan["mode"] == "validation_only"
    assert plan["test_access"] == "forbidden"
    assert plan["training_lock"]["lock_sha256"] == (
        "5666fa8bb646b270d23077d1936c8ee80fa1cfeaf8f5870c55103fc58dd80afd"
    )
    assert {item["role"] for item in plan["training_lock"]["objects"]} == {
        "train",
        "valid",
    }
    assert len(plan["training_lock"]["objects"]) == 6
    assert len(plan["runs"]) == 12
    persistence = plan["runs"][0]
    assert persistence["run_id"] == "a4_strat_v1_universal_persistence_val"
    assert persistence["task"] == ["advection1d", "burgers1d", "darcy2d"]
    assert "--max-samples" not in persistence["command"]
    assert {run["model"] for run in plan["runs"] if run["task"] == "darcy2d"} == {
        "fno",
        "uno",
        "unet",
    }
    assert all(run["evaluation_role"] == "valid" for run in plan["runs"])
    assert all("test" not in run["command"] for run in plan["runs"])
    selection = plan["metric_contract"]["sample_selection"]
    assert selection["policy"] == "full_shards_no_first_n_truncation"
    assert selection["sample_counts"]["burgers1d"] == {"train": 288, "valid": 72}
    assert selection["pair_cap"] == 1152


def test_plan_is_deterministic_and_records_runner_and_external_source_identities(tmp_path):
    first = build_plan(_args(tmp_path))
    second = build_plan(_args(tmp_path))

    assert first == second
    assert first["plan_sha256"] == second["plan_sha256"]
    identities = {run["model"]: run["model_identity"] for run in first["runs"]}
    assert set(identities) == set(RUNNERS)
    assert all(len(identity["runner_sha256"]) == 64 for identity in identities.values())
    assert identities["fno"]["source_revision"] == "neuraloperator==1.0.2"
    assert identities["unet"]["source_revision"] == ("4ff3e3a4aa1561721b5571fa3a048a0a463e0568")
    assert identities["cno"]["source_revision"] == ("6e765198aa02b56352e0a3437104b9d9e337176e")


def test_scorecard_plan_has_fixed_rows_and_explains_cno_scope(tmp_path):
    plan = build_plan(_args(tmp_path))
    scorecard = plan["scorecard_plan"]

    assert len(scorecard["row_order"]) == 14
    assert scorecard["row_order"] == [row["row_id"] for row in scorecard["rows"]]
    assert scorecard["summary_inputs"] == [run["expected_summary"] for run in plan["runs"]]
    assert scorecard["primary_metric"] == "decoded_rollout_nrmse"
    assert "CNO1d-only" in scorecard["cno_exclusion"]


def test_plan_rejects_measurement_lock_even_though_it_is_valid(tmp_path):
    with pytest.raises(ValueError, match="requires one training lock"):
        build_plan(_args(tmp_path, training_lock=str(RELEASE / "measurement.lock.json")))


def test_plan_rejects_manifest_not_bound_to_training_lock(tmp_path):
    task_source = Path(
        "docs/data/releases/strat-v1/burgers1d/"
        "9120f76b0410aa1835821940d3d3b8461fbf8379e0bacaf127d704c8b5460115/"
        "source.manifest.yaml"
    )
    with pytest.raises(ValueError, match="source manifest hash does not match"):
        build_plan(_args(tmp_path, source_manifest=str(task_source)))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"max_train_samples": 287}, "first-N training truncation"),
        ({"max_eval_samples": 71}, "first-N validation truncation"),
    ],
)
def test_plan_rejects_first_n_caps_that_break_regime_balance(tmp_path, override, message):
    with pytest.raises(ValueError, match=message):
        build_plan(_args(tmp_path, **override))


def test_execution_preflight_refuses_cache_with_test_bytes(tmp_path):
    args = _args(tmp_path)
    plan = build_plan(args)
    data_root = Path(args.data_root)
    data_root.mkdir()
    (data_root / "burgers1d_test.h5").write_bytes(b"held-out")

    with pytest.raises(ValueError, match="refuses a data root containing test"):
        _verify_local_training_cache(plan, data_root)


class _MultiplyTwo(nn.Module):
    def forward(self, value):
        return value * 2


class _PlusOne(nn.Module):
    def forward(self, value):
        return value + 1


def test_shared_external_evaluator_is_autoregressive_and_reports_horizons_regimes(monkeypatch):
    frames = [torch.ones(8, 1), torch.full((8, 1), 3.0)]
    frames.extend(torch.full((8, 1), float(2**step)) for step in range(2, 17))
    sample = {"fields": torch.stack(frames), "params": {"nu": torch.tensor(0.1)}}
    monkeypatch.setattr(fno_runner, "_dataset", lambda *args, **kwargs: [sample])

    metrics = fno_runner.evaluate_external_fno_baseline(
        {},
        {("burgers1d", 1, 8, 1): _MultiplyTwo()},
        tasks=["burgers1d"],
        split="val",
        data_root=None,
        max_samples=None,
        rollout_steps=16,
        strict_contract=True,
    )

    assert metrics["task_burgers1d_decoded_h1_nrmse"] > 0
    assert metrics["task_burgers1d_decoded_h2_nrmse"] == 0
    assert metrics["task_burgers1d_decoded_h16_nrmse"] == 0
    regime_keys = [
        key
        for key in metrics
        if key.startswith("task_burgers1d_regime_") and key.endswith("_decoded_rollout_nrmse")
    ]
    assert regime_keys == ["task_burgers1d_regime_0p1_decoded_rollout_nrmse"]
    assert metrics["task_burgers1d_regime_0p1_decoded_rollout_global_scale_nrmse"] >= 0
    assert metrics["macro_primary_nrmse"] == metrics["decoded_rollout_nrmse"]


def test_shared_external_evaluator_scores_steady_input_to_explicit_target(monkeypatch):
    fields = torch.zeros(1, 4, 4, 1)
    sample = {
        "fields": fields,
        "targets": fields + 1,
        "params": {"beta": torch.tensor(1.0)},
    }
    monkeypatch.setattr(fno_runner, "_dataset", lambda *args, **kwargs: [sample])

    metrics = fno_runner.evaluate_external_fno_baseline(
        {},
        {("darcy2d", 4, 4, 1): _PlusOne()},
        tasks=["darcy2d"],
        split="val",
        data_root=None,
        max_samples=None,
        rollout_steps=16,
        strict_contract=True,
    )

    assert metrics["task_darcy2d_decoded_solution_nrmse"] == 0
    assert metrics["task_darcy2d_regime_1_decoded_solution_nrmse"] == 0
    assert metrics["task_darcy2d_regime_1_decoded_solution_global_scale_nrmse"] == 0


def test_cli_defaults_to_plan_only_and_emits_no_baseline_outputs(tmp_path):
    plan_path = tmp_path / "plan.json"
    output_root = tmp_path / "baseline-results"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/plan_a4_baseline_validation.py",
            "--neuraloperator-version",
            "1.0.2",
            "--output-plan",
            str(plan_path),
            "--output-root",
            str(output_root),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.count("DRY_RUN:") == 12
    assert plan_path.is_file()
    assert not output_root.exists()
    assert len(json.loads(plan_path.read_text())["runs"]) == 12


def test_cli_execute_requires_explicit_validation_only_acknowledgement(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/plan_a4_baseline_validation.py",
            "--neuraloperator-version",
            "1.0.2",
            "--output-plan",
            str(tmp_path / "plan.json"),
            "--execute",
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "--execute requires --confirm-validation-only" in proc.stderr
