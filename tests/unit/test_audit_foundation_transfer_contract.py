from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from scripts.audit_foundation_transfer_contract import (
    build_foundation_transfer_contract,
    validate_foundation_transfer_contract,
)

ROOT = Path(__file__).resolve().parents[2]


def _write_h5(path: Path, shape: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=np.zeros(shape, dtype=np.float32))


def _write_fake_poseidon_repo(path: Path) -> None:
    (path / "scOT").mkdir(parents=True)
    (path / "scOT" / "problems").mkdir(parents=True)
    (path / "README.md").write_text(
        "\n".join(
            [
                'ScOT.from_pretrained("camlab-ethz/Poseidon-T")',
                "huggingface.co/collections/camlab-ethz/poseidon-664fa125729c53d8607e209a",
            ]
        ),
        encoding="utf-8",
    )
    (path / "pyproject.toml").write_text("[project]\nname='scOT'\n", encoding="utf-8")
    (path / "scOT" / "model.py").write_text("assumes square images\n", encoding="utf-8")
    (path / "scOT" / "train.py").write_text("ignore_mismatched_sizes=True\n", encoding="utf-8")
    (path / "scOT" / "inference.py").write_text("python -m scOT.inference\n", encoding="utf-8")
    (path / "scOT" / "problems" / "base.py").write_text(
        "def get_dataset(dataset, **kwargs): pass\n", encoding="utf-8"
    )


def _write_fake_cno_repo(path: Path) -> None:
    temporal = path / "CNO2d_temporal"
    (temporal / "DataLoaders").mkdir(parents=True)
    (temporal / "readme.md").write_text(
        "\n".join(
            [
                "CNO-Foundation Model weights are at zenodo.org/records/11401801.",
                "Input dimension of the CNO-FM is 5.",
                "Output dimension of the CNO-FM is 4.",
                "CNO_FineTune.py",
                "CNO_TimeLoaders.py",
            ]
        ),
        encoding="utf-8",
    )
    (temporal / "CNO_FineTune.py").write_text("in_dim = 5\nout_dim = 4\n", encoding="utf-8")
    (temporal / "CNO_timeModule_CIN.py").write_text("class CNO_time: pass\n", encoding="utf-8")
    (temporal / "DataLoaders" / "all_experiments.json").write_text("{}", encoding="utf-8")


def test_foundation_transfer_contract_marks_adapters_required_without_test_touch(tmp_path):
    data_root = tmp_path / "data"
    for split in ("train", "val"):
        _write_h5(data_root / f"advection1d_{split}.h5", (2, 4, 16, 1))
    poseidon_repo = tmp_path / "poseidon"
    cno_repo = tmp_path / "cno"
    _write_fake_poseidon_repo(poseidon_repo)
    _write_fake_cno_repo(cno_repo)

    contract = build_foundation_transfer_contract(
        cfg={"data": {"task": ["advection1d"], "root": str(data_root)}},
        config_path="fake_config.yaml",
        tasks=["advection1d"],
        data_root=str(data_root),
        inspected_splits=["train", "val"],
        max_samples=1,
        poseidon_repo=poseidon_repo,
        cno_repo=cno_repo,
        run_name="foundation_transfer_readiness_test",
    )

    checks = {check["key"]: check for check in contract["readiness_checks"]}
    assert contract["status"] == "contract_defined_measurement_pending"
    assert contract["held_out_test_used"] is False
    assert contract["held_out_test_data_read"] is False
    assert contract["inspected_splits"] == ["train", "val"]
    assert checks["held_out_budget_preserved"]["status"] == "pass"
    assert checks["poseidon_pretrained_entrypoint_declared"]["status"] == "pass"
    assert checks["poseidon_dataset_adapter_required"]["status"] == "blocker"
    assert checks["poseidon_shape_adapter_required"]["status"] == "blocker"
    assert checks["cno_fm_channel_adapter_required"]["status"] == "blocker"
    assert "foundation_measurement_ready" not in contract["measurement_blockers"]
    assert validate_foundation_transfer_contract(contract) == []


def test_foundation_transfer_contract_rejects_test_split_inspection(tmp_path):
    contract = {
        "schema_version": 1,
        "status": "contract_defined_measurement_pending",
        "held_out_test_used": False,
        "held_out_test_data_read": False,
        "claim_comparable": False,
        "published_numbers_directly_comparable": False,
        "claim_protocol_snapshot": {"inspected_splits": ["train", "test"]},
        "readiness_checks": [
            {"key": key, "status": "pass", "detail": key}
            for key in [
                "held_out_budget_preserved",
                "poseidon_source_available",
                "poseidon_pretrained_entrypoint_declared",
                "poseidon_dataset_adapter_required",
                "poseidon_shape_adapter_required",
                "cno_fm_source_available",
                "cno_fm_channel_adapter_required",
                "foundation_measurement_ready",
            ]
        ],
        "measurement_blockers": [],
    }

    errors = validate_foundation_transfer_contract(contract)

    assert "readiness contracts must not inspect split=test" in errors


def test_foundation_transfer_contract_cli_blocks_test_split_before_data(tmp_path):
    output_json = tmp_path / "contract.json"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/audit_foundation_transfer_contract.py",
            "--config",
            "configs/train_multitask_heterogeneous_light_best.yaml",
            "--output-json",
            str(output_json),
            "--inspect-splits",
            "test",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "must not inspect split=test" in proc.stderr
    assert not output_json.exists()
