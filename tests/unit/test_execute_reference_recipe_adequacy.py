from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import execute_reference_recipe_adequacy as executor
from scripts.plan_reference_recipe_adequacy import build_plan
from ups.data.manifests import canonical_sha256

RELEASE = Path(
    "docs/data/releases/strat-v1/universal/"
    "9d43d283f04f5b8d17cf6126ad189075c53307e715d7d4f61af440c2fed155c1"
)


def _args(tmp_path: Path, **overrides: object) -> Namespace:
    values: dict[str, object] = {
        "metric_addendum": "docs/data/protocols/strat_v1_1_metric_addendum.yaml",
        "training_lock": str(RELEASE / "training.lock.json"),
        "config": "configs/a4_strat_v1_baselines.yaml",
        "data_root": str(tmp_path / "training-cache"),
        "output_root": str(tmp_path / "adequacy"),
        "device": "cuda",
        "neuraloperator_version": "2.0.0",
        "selected_architecture": None,
        "selected_epochs": None,
    }
    values.update(overrides)
    return Namespace(**values)


def _resign(plan: dict[str, object]) -> None:
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    plan["plan_sha256"] = canonical_sha256(payload)


def _confirmation_plan(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    discovery = build_plan(_args(tmp_path))
    discovery_path = tmp_path / "discovery-plan.json"
    discovery_path.write_text(json.dumps(discovery, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint = {"path": "checkpoint.pt", "sha256": "c" * 64, "epoch": 24}
    selection_payload: dict[str, object] = {
        "schema_version": 1,
        "selection_id": "strat-v1.1-reference-recipe-adequacy-selection-v1",
        "status": "complete_validation_only",
        "plan_sha256": discovery["plan_sha256"],
        "architectures": {
            "fno": {"eligible": False},
            "uno": {
                "eligible": True,
                "chosen_epoch": 24,
                "chosen_macro_primary_nrmse": 0.25,
                "chosen_checkpoint": checkpoint,
            },
        },
        "selection": {
            "architecture": "uno",
            "epoch": 24,
            "macro_primary_nrmse": 0.25,
            "checkpoint": checkpoint,
        },
        "no_eligible_architecture": False,
        "held_out_measurements": 0,
    }
    selection = {
        **selection_payload,
        "selection_sha256": canonical_sha256(selection_payload),
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection, sort_keys=True) + "\n", encoding="utf-8")
    confirmation = build_plan(
        _args(
            tmp_path,
            discovery_plan=str(discovery_path),
            selection_artifact=str(selection_path),
        )
    )
    return confirmation, discovery_path, selection_path


def _permit_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(executor.importlib.metadata, "version", lambda _name: "2.0.0")
    monkeypatch.setattr(executor, "_verify_local_training_cache", lambda *_args: None)


def test_executes_only_exact_discovery_commands_after_full_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = build_plan(_args(tmp_path))
    _permit_environment(monkeypatch)
    calls: list[tuple[list[str], Path, bool]] = []

    def fake_run(command: list[str], *, cwd: Path, check: bool) -> None:
        calls.append((command, cwd, check))

    executor.execute_plan(plan, run_set="discovery", subprocess_run=fake_run)

    assert [call[0] for call in calls] == [run["command"] for run in plan["discovery"]["runs"]]
    assert all(cwd == executor.REPO_ROOT and check is True for _, cwd, check in calls)


def test_rejects_tampered_plan_before_environment_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = build_plan(_args(tmp_path))
    plan["mode"] = "heldout"
    monkeypatch.setattr(
        executor,
        "_verify_bound_files",
        lambda *_args: pytest.fail("environment checks must not run"),
    )

    with pytest.raises(ValueError, match="canonical payload"):
        executor.execute_plan(plan, run_set="discovery")


def test_rejects_heldout_command_even_when_plan_is_rehashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = build_plan(_args(tmp_path))
    command = plan["discovery"]["runs"][0]["command"]
    command.extend(["--eval-split", "test"])
    plan["discovery"]["runs"][0]["command_sha256"] = canonical_sha256(command)
    _resign(plan)
    _permit_environment(monkeypatch)

    with pytest.raises(ValueError, match="held-out role value|exactly once"):
        executor.execute_plan(plan, run_set="discovery")


def test_rejects_hyperparameter_tamper_even_when_plan_is_rehashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = build_plan(_args(tmp_path))
    command = plan["discovery"]["runs"][0]["command"]
    command[command.index("--learning-rate") + 1] = "0.01"
    plan["discovery"]["runs"][0]["command_sha256"] = canonical_sha256(command)
    _resign(plan)
    _permit_environment(monkeypatch)

    with pytest.raises(ValueError, match="changes frozen recipe option --learning-rate"):
        executor.execute_plan(plan, run_set="discovery")


def test_preflights_all_output_paths_before_starting_any_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = build_plan(_args(tmp_path))
    _permit_environment(monkeypatch)
    second = plan["discovery"]["runs"][1]
    Path(second["expected_summary"]).parent.mkdir(parents=True)
    calls: list[list[str]] = []

    with pytest.raises(FileExistsError, match="existing run output"):
        executor.execute_plan(
            plan,
            run_set="discovery",
            subprocess_run=lambda command, **_kwargs: calls.append(command),
        )

    assert calls == []


def test_confirmation_requires_exact_selected_two_seed_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _discovery_path, _selection_path = _confirmation_plan(tmp_path)
    plan["confirmation"]["runs"] = plan["confirmation"]["runs"][:1]
    _resign(plan)
    _permit_environment(monkeypatch)

    with pytest.raises(ValueError, match="exactly the selected confirmation run set"):
        executor.execute_plan(plan, run_set="confirmation")


def test_confirmation_rejects_tampered_bound_selection_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _discovery_path, selection_path = _confirmation_plan(tmp_path)
    _permit_environment(monkeypatch)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["selection"]["epoch"] = 12
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    with pytest.raises(ValueError, match="selection artifact file hash"):
        executor.execute_plan(plan, run_set="confirmation")


def test_confirmation_rejects_invalid_selection_self_hash_after_file_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _discovery_path, selection_path = _confirmation_plan(tmp_path)
    _permit_environment(monkeypatch)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["selection"]["epoch"] = 12
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    plan["confirmation"]["evidence_binding"]["selection_artifact"]["file_sha256"] = (
        executor._file_sha256(selection_path)
    )
    _resign(plan)

    with pytest.raises(ValueError, match="selection artifact SHA-256"):
        executor.execute_plan(plan, run_set="confirmation")


def test_confirmation_rejects_discovery_plan_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, discovery_path, _selection_path = _confirmation_plan(tmp_path)
    _permit_environment(monkeypatch)
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    discovery["discovery"]["seed"] = 29
    discovery_path.write_text(json.dumps(discovery), encoding="utf-8")

    with pytest.raises(ValueError, match="discovery plan file hash"):
        executor.execute_plan(plan, run_set="confirmation")


def test_rejects_runner_changed_since_plan_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = build_plan(_args(tmp_path))
    plan["runner_identity"]["fno"]["file_sha256"] = "0" * 64
    _resign(plan)
    monkeypatch.setattr(executor.importlib.metadata, "version", lambda _name: "2.0.0")

    with pytest.raises(ValueError, match="runner file changed"):
        executor.execute_plan(plan, run_set="discovery")


def test_rejects_wrong_neuraloperator_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = build_plan(_args(tmp_path))
    monkeypatch.setattr(executor.importlib.metadata, "version", lambda _name: "1.0.0")

    with pytest.raises(RuntimeError, match="version mismatch"):
        executor.execute_plan(plan, run_set="discovery")


def test_local_cache_verifier_binds_train_and_valid_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "cache"
    data_root.mkdir()
    train = data_root / "task_train.h5"
    valid = data_root / "task_val.h5"
    train.write_bytes(b"train")
    valid.write_bytes(b"valid")
    objects = [
        SimpleNamespace(
            path=path.name,
            size_bytes=path.stat().st_size,
            checksums={"sha256": executor._file_sha256(path)},
        )
        for path in (train, valid)
    ]
    monkeypatch.setattr(executor, "load_data_lock", lambda _path: SimpleNamespace(objects=objects))
    plan = {"bindings": {"training_lock": {"path": "ignored.lock.json"}}}

    executor._verify_local_training_cache(plan, data_root)

    valid.write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match frozen lock"):
        executor._verify_local_training_cache(plan, data_root)


def test_local_cache_verifier_rejects_test_object_presence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "cache"
    data_root.mkdir()
    (data_root / "task_test.h5").write_bytes(b"held out")
    monkeypatch.setattr(executor, "load_data_lock", lambda _path: SimpleNamespace(objects=[]))
    plan = {"bindings": {"training_lock": {"path": "ignored.lock.json"}}}

    with pytest.raises(ValueError, match="containing test HDF5"):
        executor._verify_local_training_cache(plan, data_root)


def test_cli_requires_explicit_validation_only_confirmation(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({}), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/execute_reference_recipe_adequacy.py",
            "--plan",
            str(plan_path),
            "--run-set",
            "discovery",
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    assert "--confirm-validation-only" in proc.stderr
