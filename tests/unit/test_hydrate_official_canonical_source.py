from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from scripts.hydrate_official_canonical_source import (
    TASK_CONTRACTS,
    hydrate_canonical_source,
    load_frozen_schema,
    load_official_catalog,
    main,
)
from scripts.make_light_hdf5_shards import build_stratified_task_shard_records


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def _write_fixture(
    tmp_path: Path,
    task: str,
    *,
    paired_across_regimes: bool = False,
    cross_index_collision: bool = False,
    rows_per_source: int = 3,
    darcy_shape: tuple[int, int] = (2, 2),
    regime_stride: int = 100,
):
    contract = TASK_CONTRACTS[task]
    raw_root = tmp_path / "raw"
    rows = []
    for index, regime in enumerate(contract.expected_regimes):
        if task == "burgers1d":
            name = f"1D_Burgers_Sols_Nu{regime}.hdf5"
            logical_path = f"1D/Burgers/Train/{name}"
            data = (
                np.arange(rows_per_source * 2 * 4, dtype=np.float32).reshape(rows_per_source, 2, 4)
                + index * 100
            )
        else:
            name = f"2D_DarcyFlow_beta{regime}_Train.hdf5"
            logical_path = f"2D/DarcyFlow/{name}"
            data = (
                np.arange(rows_per_source * np.prod(darcy_shape), dtype=np.float32).reshape(
                    rows_per_source, *darcy_shape
                )
                + index * regime_stride
            )
            if paired_across_regimes:
                data = np.arange(rows_per_source * np.prod(darcy_shape), dtype=np.float32).reshape(
                    rows_per_source, *darcy_shape
                )
            if cross_index_collision and index == 1:
                data[1] = np.arange(4, dtype=np.float32).reshape(2, 2)
        path = raw_root / logical_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            if task == "burgers1d":
                handle.create_dataset("tensor", data=data)
            else:
                handle.create_dataset("nu", data=data)
                handle.create_dataset("tensor", data=(data + 0.25)[:, None, :, :])
        rows.append(
            {
                "path": logical_path,
                "file_id": 1000 + index,
                "size_bytes": path.stat().st_size,
                "checksum": _md5(path),
                "checksum_type": "MD5",
            }
        )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"files": rows}), encoding="utf-8")
    return raw_root, manifest


def _write_schema(tmp_path: Path, task: str, *, status: str = "frozen") -> Path:
    contract = TASK_CONTRACTS[task]
    schema = tmp_path / f"{task}-schema.yaml"
    schema.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "tasks": {
                    task: {
                        "status": status,
                        "semantic_role": (
                            "solution_trajectory"
                            if task == "burgers1d"
                            else "steady_elliptic_coefficient_to_solution_operator"
                        ),
                        "field_kind": contract.field_kind,
                        "parameter_name": contract.parameter_name,
                        **(
                            {
                                "dataset_key": "tensor",
                                "expected_sample_shape": [2, 4],
                            }
                            if task == "burgers1d"
                            else {
                                "input_dataset_key": "nu",
                                "target_dataset_key": "tensor",
                                "expected_input_sample_shape": [2, 2],
                                "expected_target_sample_shape": [1, 2, 2],
                                "input_raw_layout": "xy",
                                "target_raw_layout": "channel_xy",
                            }
                        ),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return schema


@pytest.mark.parametrize(
    ("task", "expected_regimes"),
    [("burgers1d", 12), ("darcy2d", 5)],
)
def test_load_official_catalog_requires_complete_ordered_regimes(tmp_path, task, expected_regimes):
    _, manifest = _write_fixture(tmp_path, task)

    catalog = load_official_catalog(manifest, task)

    assert len(catalog) == expected_regimes
    assert [row["regime"] for row in catalog] == sorted(row["regime"] for row in catalog)


@pytest.mark.parametrize(
    ("task", "counts", "field_kind", "time_axis"),
    [
        ("burgers1d", (12, 12, 12), "temporal", 1),
        ("darcy2d", (5, 5, 5), "steady", None),
    ],
)
def test_canonical_source_feeds_universal_builder(tmp_path, task, counts, field_kind, time_axis):
    raw_root, manifest = _write_fixture(tmp_path, task)
    schema = _write_schema(tmp_path, task)
    source_root = tmp_path / "source"
    source_path = source_root / f"{task}_train.h5"

    record = hydrate_canonical_source(
        manifest=manifest,
        raw_root=raw_root,
        out_path=source_path,
        task=task,
        samples_per_regime=3,
        schema_contract_path=schema,
    )
    shards = build_stratified_task_shard_records(
        root=source_root,
        out_root=tmp_path / "strat",
        task=task,
        source_split="train",
        train_count=counts[0],
        val_count=counts[1],
        test_count=counts[2],
        overwrite=False,
        provenance_datasets=["source_file_id", "source_sample_index"],
        regime_dataset=record["regime_dataset"],
        field_kind=field_kind,
        time_axis=time_axis,
    )

    assert record["status"] == "complete"
    assert record["sample_count"] == sum(counts)
    assert shards[0]["protocol_gate"]["status"] == "passed"
    with h5py.File(source_path, "r") as handle:
        assert bool(handle.attrs["conversion_complete"])
        assert "source_file_id" in handle
        assert record["regime_dataset"] in handle
        if task == "darcy2d":
            assert handle["data"].shape == (15, 1, 2, 2, 1)
            assert handle["targets"].shape == (15, 1, 2, 2, 1)
            assert handle.attrs["mapping_kind"] == "steady_operator"


def test_darcy_allows_paired_steady_fields_at_same_index_across_beta(tmp_path):
    raw_root, manifest = _write_fixture(tmp_path, "darcy2d", paired_across_regimes=True)
    schema = _write_schema(tmp_path, "darcy2d")

    record = hydrate_canonical_source(
        manifest=manifest,
        raw_root=raw_root,
        out_path=tmp_path / "darcy2d_train.h5",
        task="darcy2d",
        samples_per_regime=3,
        schema_contract_path=schema,
    )

    assert record["sample_count"] == 15


def test_darcy_rejects_cross_index_steady_field_collision(tmp_path):
    raw_root, manifest = _write_fixture(tmp_path, "darcy2d", cross_index_collision=True)
    schema = _write_schema(tmp_path, "darcy2d")

    with pytest.raises(ValueError, match="Duplicate steady field"):
        hydrate_canonical_source(
            manifest=manifest,
            raw_root=raw_root,
            out_path=tmp_path / "darcy2d_train.h5",
            task="darcy2d",
            samples_per_regime=3,
            schema_contract_path=schema,
        )


def test_canonical_source_rejects_checksum_mismatch_without_partial_output(tmp_path):
    raw_root, manifest = _write_fixture(tmp_path, "burgers1d")
    schema = _write_schema(tmp_path, "burgers1d")
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    payload["files"][0]["checksum"] = "0" * 32
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")
    output = tmp_path / "burgers1d_train.h5"

    with pytest.raises(ValueError, match="MD5 mismatch"):
        hydrate_canonical_source(
            manifest=manifest,
            raw_root=raw_root,
            out_path=output,
            task="burgers1d",
            samples_per_regime=3,
            schema_contract_path=schema,
        )

    assert not output.exists()
    assert not output.with_suffix(".h5.tmp").exists()


def test_canonical_source_uses_seeded_identity_ranking_and_persists_evidence(tmp_path):
    raw_root, manifest = _write_fixture(tmp_path, "darcy2d", rows_per_source=8)
    schema = _write_schema(tmp_path, "darcy2d")

    first = hydrate_canonical_source(
        manifest=manifest,
        raw_root=raw_root,
        out_path=tmp_path / "first.h5",
        task="darcy2d",
        samples_per_regime=3,
        schema_contract_path=schema,
        selection_seed=17,
        selection_protocol="strat-v1-test",
    )
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    payload["files"].reverse()
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")
    second = hydrate_canonical_source(
        manifest=manifest,
        raw_root=raw_root,
        out_path=tmp_path / "second.h5",
        task="darcy2d",
        samples_per_regime=3,
        schema_contract_path=schema,
        selection_seed=17,
        selection_protocol="strat-v1-test",
    )

    assert first["selection_algorithm"] == "sha256-protocol-seed-provenance-v1"
    assert first["selection_seed"] == 17
    assert first["selection_ranking_provenance_dataset"] == "source_sample_index"
    assert first["selected_identity_sha256"] == second["selected_identity_sha256"]
    assert any(entry["selected_sample_indices"] != [0, 1, 2] for entry in first["source_catalog"])
    assert len({tuple(entry["selected_sample_indices"]) for entry in first["source_catalog"]}) == 1
    with (
        h5py.File(tmp_path / "first.h5", "r") as left,
        h5py.File(tmp_path / "second.h5", "r") as right,
    ):
        assert left.attrs["selected_identity_sha256"] == first["selected_identity_sha256"]
        assert left.attrs["selection_ranking_provenance_dataset"] == "source_sample_index"
        np.testing.assert_array_equal(left["source_file_id"], right["source_file_id"])
        np.testing.assert_array_equal(left["source_sample_index"], right["source_sample_index"])
        np.testing.assert_array_equal(left["data"], right["data"])


def test_frozen_schema_selects_one_semantic_field_when_all_sources_have_alternatives(tmp_path):
    raw_root, manifest = _write_fixture(tmp_path, "darcy2d")
    for path in raw_root.rglob("*.hdf5"):
        with h5py.File(path, "r+") as handle:
            del handle["tensor"]
            handle.create_dataset(
                "tensor", data=(np.asarray(handle["nu"]) + np.float32(10000.0))[:, None]
            )
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    for row in payload["files"]:
        path = raw_root / row["path"]
        row["size_bytes"] = path.stat().st_size
        row["checksum"] = _md5(path)
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")
    schema = _write_schema(tmp_path, "darcy2d")

    output = tmp_path / "darcy2d_train.h5"
    hydrate_canonical_source(
        manifest=manifest,
        raw_root=raw_root,
        out_path=output,
        task="darcy2d",
        samples_per_regime=3,
        schema_contract_path=schema,
    )

    with h5py.File(output, "r") as handle:
        assert float(np.max(handle["data"])) < 10000.0
        assert float(np.min(handle["targets"])) >= 10000.0
        assert handle.attrs["raw_input_dataset_key"] == "nu"
        assert handle.attrs["raw_target_dataset_key"] == "tensor"


def test_checked_in_darcy_schema_freezes_official_operator_semantics():
    schema = load_frozen_schema("darcy2d")

    assert schema["input_dataset_key"] == "nu"
    assert schema["target_dataset_key"] == "tensor"
    assert schema["expected_input_sample_shape"] == [128, 128]
    assert schema["expected_target_sample_shape"] == [1, 128, 128]


def test_checked_in_darcy_schema_converts_exact_official_raw_shapes(tmp_path):
    raw_root, manifest = _write_fixture(
        tmp_path, "darcy2d", rows_per_source=1, darcy_shape=(128, 128)
    )
    output = tmp_path / "darcy2d_train.h5"

    hydrate_canonical_source(
        manifest=manifest,
        raw_root=raw_root,
        out_path=output,
        task="darcy2d",
        samples_per_regime=1,
    )

    with h5py.File(output, "r") as handle:
        assert handle["data"].shape == (5, 1, 128, 128, 1)
        assert handle["targets"].shape == (5, 1, 128, 128, 1)


def test_darcy_exact_260_65_65_protocol_is_balanced_and_disjoint(tmp_path):
    raw_root, manifest = _write_fixture(
        tmp_path, "darcy2d", rows_per_source=78, regime_stride=10_000
    )
    schema = _write_schema(tmp_path, "darcy2d")
    source_root = tmp_path / "source"
    hydrate_canonical_source(
        manifest=manifest,
        raw_root=raw_root,
        out_path=source_root / "darcy2d_train.h5",
        task="darcy2d",
        samples_per_regime=78,
        schema_contract_path=schema,
    )

    records = build_stratified_task_shard_records(
        root=source_root,
        out_root=tmp_path / "strat",
        task="darcy2d",
        source_split="train",
        train_count=260,
        val_count=65,
        test_count=65,
        overwrite=False,
        provenance_datasets=["source_file_id", "source_sample_index"],
        regime_dataset="beta",
        field_kind="steady",
        time_axis=None,
    )

    assert [record["sample_count"] for record in records] == [260, 65, 65]
    gate = records[0]["protocol_gate"]
    assert gate["status"] == "passed"
    assert gate["regime_counts"] == {
        "train": {"0.01": 52, "0.1": 52, "1.0": 52, "10.0": 52, "100.0": 52},
        "val": {"0.01": 13, "0.1": 13, "1.0": 13, "10.0": 13, "100.0": 13},
        "test": {"0.01": 13, "0.1": 13, "1.0": 13, "10.0": 13, "100.0": 13},
    }


def test_cli_refuses_json_sidecar_overwriting_hdf5(monkeypatch, tmp_path):
    output = tmp_path / "canonical.h5"
    monkeypatch.setattr(
        "sys.argv",
        [
            "hydrate_official_canonical_source",
            "--raw-root",
            str(tmp_path),
            "--out",
            str(output),
            "--output-json",
            str(output),
            "--task",
            "darcy2d",
            "--samples-per-regime",
            "3",
        ],
    )

    with pytest.raises(ValueError, match="must differ"):
        main()
