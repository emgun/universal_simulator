from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from scripts.finalize_reference_recipe_adequacy import build_confirmed_recipe
from scripts.materialize_reference_recipe_adequacy import build_selection
from tests.unit.test_materialize_reference_recipe_adequacy import _fixture, _metrics
from ups.data.manifests import canonical_sha256


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _confirmation_fixture(tmp_path: Path):
    discovery_plan, discovery_summaries = _fixture(tmp_path)
    selection = build_selection(
        discovery_plan, summary_paths=discovery_summaries, repo_root=tmp_path
    )
    discovery_plan_path = tmp_path / "artifacts/discovery-plan.json"
    discovery_plan_path.parent.mkdir(parents=True)
    discovery_plan_path.write_text(json.dumps(discovery_plan))
    selection_path = tmp_path / "artifacts/selection.json"
    selection_path.write_text(json.dumps(selection))
    architecture = selection["selection"]["architecture"]
    epoch = selection["selection"]["epoch"]
    discovery_path = next(path for path in discovery_summaries if architecture in path.parent.name)
    base_summary = json.loads(discovery_path.read_text())
    base_run = next(
        run for run in discovery_plan["discovery"]["runs"] if run["architecture"] == architecture
    )

    confirmation_paths = []
    runs = []
    for seed, value in ((29, 0.80), (43, 0.82)):
        run_id = f"r0_strat_v1_1_{architecture}_all_e{epoch}_s{seed}_confirmation_val"
        relative = Path("reports") / run_id / "summary.json"
        command = list(base_run["command"])
        command[command.index("--name") + 1] = run_id
        command[command.index("--seed") + 1] = str(seed)
        rung_index = command.index("--validation-rungs")
        command = command[: rung_index + 1] + [str(epoch)]
        run = {
            **base_run,
            "run_id": run_id,
            "phase": "confirmation",
            "seed": seed,
            "epochs": epoch,
            "expected_summary": str(relative),
            "command": command,
            "command_sha256": canonical_sha256(command),
        }
        runs.append(run)
        summary = copy.deepcopy(base_summary)
        summary["run_name"] = run_id
        summary["extra"]["seed"] = seed
        summary["extra"]["epochs"] = epoch
        summary["extra"]["command"] = command
        metrics = _metrics(value)
        summary["metrics"] = metrics
        summary["details"]["validation_history"] = [
            {
                "epoch": epoch,
                "metric_name": "decoded_rollout_nrmse",
                "metric_value": value,
                "metrics": metrics,
                "duration_sec": 1.0,
            }
        ]
        checkpoint_rel = Path("reports") / run_id / f"models_epoch_{epoch}.pt"
        checkpoint_path = tmp_path / checkpoint_rel
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(f"{architecture}-{epoch}-{seed}".encode())
        checkpoint = {
            "path": str(checkpoint_rel),
            "sha256": _sha(checkpoint_path),
            "epoch": epoch,
        }
        summary["checkpoints"] = {"rungs": {str(epoch): checkpoint}, "selected": checkpoint}
        summary["recipe_adequacy"] = {
            "validation_rungs": [epoch],
            "selection_metric": "decoded_rollout_nrmse",
            "selected_epoch": epoch,
            "selected_metric_value": value,
            "selection_rule": "minimum_finite_validation_metric_earliest_tie",
        }
        summary_path = tmp_path / relative
        summary_path.write_text(json.dumps(summary))
        confirmation_paths.append(summary_path)

    payload = copy.deepcopy(discovery_plan)
    payload["confirmation"] = {
        "selected_architecture": architecture,
        "selected_epochs": epoch,
        "required_seeds": [17, 29, 43],
        "reuse_discovery_seed_17": True,
        "additional_seeds": [29, 43],
        "evidence_binding": {
            "discovery_plan": {
                "path": str(discovery_plan_path.relative_to(tmp_path)),
                "file_sha256": _sha(discovery_plan_path),
                "plan_sha256": discovery_plan["plan_sha256"],
            },
            "selection_artifact": {
                "path": str(selection_path.resolve()),
                "file_sha256": _sha(selection_path),
                "selection_sha256": selection["selection_sha256"],
            },
        },
        "runs": runs,
    }
    payload.pop("plan_sha256")
    plan = {**payload, "plan_sha256": canonical_sha256(payload)}
    return plan, selection, selection_path, discovery_path, confirmation_paths


def test_confirms_three_seed_recipe_without_reselection(tmp_path: Path) -> None:
    plan, selection, selection_path, discovery, confirmations = _confirmation_fixture(tmp_path)

    artifact = build_confirmed_recipe(
        plan,
        selection=selection,
        selection_path=selection_path,
        discovery_summary_path=discovery,
        confirmation_summary_paths=confirmations,
        repo_root=tmp_path,
    )

    assert artifact["status"] == "confirmed_validation_only"
    assert artifact["architecture"] == selection["selection"]["architecture"]
    assert artifact["epoch"] == selection["selection"]["epoch"]
    assert [row["seed"] for row in artifact["seeds"]] == [17, 29, 43]
    assert artifact["held_out_measurements"] == 0
    payload = {key: value for key, value in artifact.items() if key != "recipe_sha256"}
    assert artifact["recipe_sha256"] == canonical_sha256(payload)


def test_confirmation_regime_failure_stops_without_reselection(tmp_path: Path) -> None:
    plan, selection, selection_path, discovery, confirmations = _confirmation_fixture(tmp_path)
    failed = json.loads(confirmations[0].read_text())
    failed_metrics = _metrics(0.8, bad_spread=True)
    failed["metrics"] = failed_metrics
    failed["details"]["validation_history"][0]["metrics"] = failed_metrics
    confirmations[0].write_text(json.dumps(failed))

    artifact = build_confirmed_recipe(
        plan,
        selection=selection,
        selection_path=selection_path,
        discovery_summary_path=discovery,
        confirmation_summary_paths=confirmations,
        repo_root=tmp_path,
    )

    assert artifact["status"] == "stopped_regime_ineligible"
    assert artifact["architecture"] == selection["selection"]["architecture"]
    assert artifact["all_seeds_regime_eligible"] is False


def test_rejects_confirmation_plan_that_changes_selected_epoch(tmp_path: Path) -> None:
    plan, selection, selection_path, discovery, confirmations = _confirmation_fixture(tmp_path)
    plan["confirmation"]["selected_epochs"] = 3
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    plan["plan_sha256"] = canonical_sha256(payload)

    try:
        build_confirmed_recipe(
            plan,
            selection=selection,
            selection_path=selection_path,
            discovery_summary_path=discovery,
            confirmation_summary_paths=confirmations,
            repo_root=tmp_path,
        )
    except ValueError as exc:
        assert "changed the selected architecture or epoch" in str(exc)
    else:
        raise AssertionError("changed selection was accepted")
