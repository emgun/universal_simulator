from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from scripts.make_light_hdf5_shards import build_stratified_task_shard_records
from scripts.protocol_split_gates import evaluate_protocol_splits
from ups.data.manifests import load_protocol_manifest, load_source_manifest, resolve_data_lock


def _write_protocol_source(
    path,
    *,
    regime_count: int = 3,
    rows_per_regime: int = 6,
    temporal: bool = True,
) -> None:
    rows = regime_count * rows_per_regime
    if temporal:
        data = np.arange(rows * 2 * 3, dtype=np.float32).reshape(rows, 2, 3)
    else:
        data = np.arange(rows * 2 * 3, dtype=np.float32).reshape(rows, 2, 3)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=data)
        handle.create_dataset(
            "source_file_index", data=np.repeat(np.arange(regime_count), rows_per_regime)
        )
        handle.create_dataset(
            "source_sample_index", data=np.tile(np.arange(rows_per_regime), regime_count)
        )
        handle.create_dataset("beta", data=np.repeat(np.arange(regime_count), rows_per_regime))


def _gate_arrays(provenance: np.ndarray, regime: np.ndarray | None = None):
    count = int(provenance.shape[0])
    return {
        split: {
            "data": np.arange(count * 2, dtype=np.float32).reshape(count, 2, 1) + split_index * 100,
            "source_sample_index": provenance.copy(),
            "beta": regime.copy() if regime is not None else np.arange(count, dtype=np.float64),
        }
        for split_index, split in enumerate(("train", "val", "test"))
    }


def test_build_universal_temporal_shards_records_passed_gate(tmp_path):
    root = tmp_path / "source"
    out_root = tmp_path / "universal"
    root.mkdir()
    _write_protocol_source(root / "advection1d_train.h5")

    records = build_stratified_task_shard_records(
        root=root,
        out_root=out_root,
        task="advection1d",
        source_split="train",
        train_count=6,
        val_count=6,
        test_count=6,
        overwrite=False,
        provenance_datasets=["source_file_index", "source_sample_index"],
        regime_dataset="beta",
        field_kind="temporal",
        time_axis=1,
    )

    gate = records[0]["protocol_gate"]
    assert gate["status"] == "passed"
    assert gate["regime_counts"] == {
        "train": {"0": 2, "1": 2, "2": 2},
        "val": {"0": 2, "1": 2, "2": 2},
        "test": {"0": 2, "1": 2, "2": 2},
    }
    assert all(
        check == {"provenance_overlap": 0, "field_overlap": 0}
        for check in gate["cross_split_overlap"].values()
    )


def test_stable_selection_is_invariant_to_source_file_and_row_order(tmp_path):
    def write_source(root, *, reverse_files: bool, reverse_rows: bool) -> None:
        root.mkdir()
        rows = []
        for regime in range(2):
            for sample in range(8):
                rows.append((100 + regime, sample, regime, regime * 1000 + sample))
        if reverse_rows:
            rows.reverse()
        halves = (rows[:8], rows[8:])
        if reverse_files:
            halves = tuple(reversed(halves))
        for suffix, file_rows in zip(("a", "b"), halves, strict=True):
            with h5py.File(root / f"advection1d_train_{suffix}.h5", "w") as handle:
                handle.create_dataset(
                    "data",
                    data=np.asarray(
                        [[[float(row[3])], [float(row[3]) + 0.5]] for row in file_rows],
                        dtype=np.float32,
                    ),
                )
                handle.create_dataset("source_file_id", data=[row[0] for row in file_rows])
                handle.create_dataset("source_sample_index", data=[row[1] for row in file_rows])
                handle.create_dataset("beta", data=[row[2] for row in file_rows])

    first_root = tmp_path / "first-source"
    second_root = tmp_path / "second-source"
    write_source(first_root, reverse_files=False, reverse_rows=False)
    write_source(second_root, reverse_files=True, reverse_rows=True)

    kwargs = {
        "task": "advection1d",
        "source_split": "train",
        "train_count": 4,
        "val_count": 4,
        "test_count": 4,
        "overwrite": False,
        "provenance_datasets": ["source_file_id", "source_sample_index"],
        "regime_dataset": "beta",
        "field_kind": "temporal",
        "time_axis": 1,
        "selection_seed": 23,
        "selection_protocol": "strat-v1-test",
    }
    first = build_stratified_task_shard_records(
        root=first_root, out_root=tmp_path / "first-out", **kwargs
    )
    second = build_stratified_task_shard_records(
        root=second_root, out_root=tmp_path / "second-out", **kwargs
    )

    assert [record["selected_identity_sha256"] for record in first] == [
        record["selected_identity_sha256"] for record in second
    ]
    for split in ("train", "val", "test"):
        with (
            h5py.File(tmp_path / "first-out" / f"advection1d_{split}.h5", "r") as left,
            h5py.File(tmp_path / "second-out" / f"advection1d_{split}.h5", "r") as right,
        ):
            np.testing.assert_array_equal(left["source_file_id"], right["source_file_id"])
            np.testing.assert_array_equal(left["source_sample_index"], right["source_sample_index"])
            np.testing.assert_array_equal(left["data"], right["data"])
            assert left.attrs["selection_algorithm"] == "sha256-protocol-seed-provenance-v1"
            assert int(left.attrs["selection_seed"]) == 23


def test_build_universal_allows_initial_field_reuse_across_regimes(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    path = root / "advection1d_train.h5"
    _write_protocol_source(path)
    with h5py.File(path, "r+") as handle:
        for regime in (1, 2):
            start = regime * 6
            handle["data"][start : start + 6, 0] = handle["data"][:6, 0]

    records = build_stratified_task_shard_records(
        root=root,
        out_root=tmp_path / "universal",
        task="advection1d",
        source_split="train",
        train_count=6,
        val_count=6,
        test_count=6,
        overwrite=False,
        provenance_datasets=["source_file_index", "source_sample_index"],
        regime_dataset="beta",
        field_kind="temporal",
        time_axis=1,
    )

    gate = records[0]["protocol_gate"]
    assert gate["unique_field_regime_pairs"] == {"train": 6, "val": 6, "test": 6}
    assert gate["unique_field_groups"] == {"train": 2, "val": 2, "test": 2}
    assert gate["within_split_field_regime_pairs_unique"] == {
        "train": True,
        "val": True,
        "test": True,
    }


def test_build_universal_rejects_initial_field_reuse_within_one_regime(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    path = root / "advection1d_train.h5"
    _write_protocol_source(path)
    with h5py.File(path, "r+") as handle:
        handle["data"][1, 0] = handle["data"][0, 0]

    try:
        build_stratified_task_shard_records(
            root=root,
            out_root=tmp_path / "universal",
            task="advection1d",
            source_split="train",
            train_count=6,
            val_count=6,
            test_count=6,
            overwrite=False,
            provenance_datasets=["source_file_index", "source_sample_index"],
            regime_dataset="beta",
            field_kind="temporal",
            time_axis=1,
        )
    except ValueError as exc:
        assert "initial-field overlap" in str(exc) or (
            "repeats an identical initial field within one regime" in str(exc)
        )
    else:
        raise AssertionError("Expected same-regime initial-field reuse to be rejected")


def test_build_universal_rejects_cross_split_initial_field_overlap(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    path = root / "advection1d_train.h5"
    _write_protocol_source(path)
    with h5py.File(path, "r+") as handle:
        # Regime 0 rows 0 and 2 are allocated to train and val respectively.
        handle["data"][2, 0] = handle["data"][0, 0]

    try:
        build_stratified_task_shard_records(
            root=root,
            out_root=tmp_path / "universal",
            task="advection1d",
            source_split="train",
            train_count=6,
            val_count=6,
            test_count=6,
            overwrite=False,
            provenance_datasets=["source_file_index", "source_sample_index"],
            regime_dataset="beta",
            field_kind="temporal",
            time_axis=1,
        )
    except ValueError as exc:
        assert "initial-field overlap" in str(exc)
    else:
        raise AssertionError("Expected initial-field overlap to be rejected")
    assert not (tmp_path / "universal").exists()


def test_build_universal_rejects_cross_split_provenance_overlap(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    path = root / "advection1d_train.h5"
    _write_protocol_source(path)
    with h5py.File(path, "r+") as handle:
        # Regime 0 rows 0 and 2 are allocated to train and val respectively.
        handle["source_sample_index"][2] = handle["source_sample_index"][0]

    try:
        build_stratified_task_shard_records(
            root=root,
            out_root=tmp_path / "universal",
            task="advection1d",
            source_split="train",
            train_count=6,
            val_count=6,
            test_count=6,
            overwrite=False,
            provenance_datasets=["source_file_index", "source_sample_index"],
            regime_dataset="beta",
            field_kind="temporal",
            time_axis=1,
        )
    except ValueError as exc:
        assert "provenance overlap" in str(exc)
    else:
        raise AssertionError("Expected provenance overlap to be rejected")


def test_build_universal_rejects_missing_provenance(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    path = root / "burgers1d_train.h5"
    _write_protocol_source(path)
    with h5py.File(path, "r+") as handle:
        del handle["source_sample_index"]

    try:
        build_stratified_task_shard_records(
            root=root,
            out_root=tmp_path / "universal",
            task="burgers1d",
            source_split="train",
            train_count=6,
            val_count=6,
            test_count=6,
            overwrite=False,
            provenance_datasets=["source_file_index", "source_sample_index"],
            regime_dataset="beta",
            field_kind="temporal",
            time_axis=1,
        )
    except ValueError as exc:
        assert "missing required dataset: source_sample_index" in str(exc)
    else:
        raise AssertionError("Expected missing provenance to be rejected")


def test_build_universal_rejects_unbalanced_or_missing_regime(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    path = root / "advection1d_train.h5"
    _write_protocol_source(path)

    try:
        build_stratified_task_shard_records(
            root=root,
            out_root=tmp_path / "unbalanced",
            task="advection1d",
            source_split="train",
            train_count=5,
            val_count=6,
            test_count=6,
            overwrite=False,
            provenance_datasets=["source_file_index", "source_sample_index"],
            regime_dataset="beta",
            field_kind="temporal",
            time_axis=1,
        )
    except ValueError as exc:
        assert "must be a positive multiple" in str(exc)
    else:
        raise AssertionError("Expected unbalanced regime allocation to be rejected")

    with h5py.File(path, "r+") as handle:
        del handle["beta"]
    try:
        build_stratified_task_shard_records(
            root=root,
            out_root=tmp_path / "missing",
            task="advection1d",
            source_split="train",
            train_count=6,
            val_count=6,
            test_count=6,
            overwrite=False,
            provenance_datasets=["source_file_index", "source_sample_index"],
            regime_dataset="beta",
            field_kind="temporal",
            time_axis=1,
        )
    except ValueError as exc:
        assert "missing required dataset: beta" in str(exc)
    else:
        raise AssertionError("Expected missing regime provenance to be rejected")


def test_build_universal_supports_steady_fields_without_time_axis(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    _write_protocol_source(root / "darcy2d_train.h5", temporal=False)

    records = build_stratified_task_shard_records(
        root=root,
        out_root=tmp_path / "universal",
        task="darcy2d",
        source_split="train",
        train_count=6,
        val_count=6,
        test_count=6,
        overwrite=False,
        provenance_datasets=["source_file_index", "source_sample_index"],
        regime_dataset="beta",
        field_kind="steady",
        time_axis=None,
    )

    assert records[0]["protocol_gate"]["field_kind"] == "steady"
    assert records[0]["protocol_gate"]["time_axis"] is None


def test_make_light_hdf5_universal_cli_writes_gate_to_manifest(tmp_path, monkeypatch):
    from scripts import make_light_hdf5_shards

    root = tmp_path / "source"
    root.mkdir()
    _write_protocol_source(root / "advection1d_train.h5")
    manifest = tmp_path / "manifest.yaml"
    monkeypatch.setattr(
        "sys.argv",
        [
            "make_light_hdf5_shards",
            "--root",
            str(root),
            "--out-root",
            str(tmp_path / "universal"),
            "--tasks",
            "advection1d",
            "--train-count",
            "6",
            "--val-count",
            "6",
            "--test-count",
            "6",
            "--provenance-dataset",
            "source_file_index",
            "--provenance-dataset",
            "source_sample_index",
            "--regime-dataset",
            "beta",
            "--field-kind",
            "temporal",
            "--time-axis",
            "1",
            "--manifest",
            str(manifest),
        ],
    )

    make_light_hdf5_shards.main()

    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert payload["protocol_mode"] == "strat-v1"
    assert payload["protocol_gates"]["advection1d"]["status"] == "passed"
    assert payload["records"][0]["protocol_gate"]["identical_regime_coverage"] is True
    source = load_source_manifest(tmp_path / "manifest.source.yaml")
    protocol = load_protocol_manifest(tmp_path / "manifest.protocol.yaml")
    training_lock = resolve_data_lock(
        source,
        protocol,
        requested_roles=("train", "valid"),
    )
    assert training_lock.requested_roles == ("train", "valid")
    assert all(item.role != "test" for item in training_lock.objects)


def test_make_light_hdf5_content_addressed_b2_mirror(tmp_path, monkeypatch):
    from scripts import make_light_hdf5_shards

    root = tmp_path / "source"
    root.mkdir()
    _write_protocol_source(root / "advection1d_train.h5")
    manifest = tmp_path / "manifest.yaml"
    monkeypatch.setattr(
        "sys.argv",
        [
            "make_light_hdf5_shards",
            "--root",
            str(root),
            "--out-root",
            str(tmp_path / "universal"),
            "--tasks",
            "advection1d",
            "--train-count",
            "6",
            "--val-count",
            "6",
            "--test-count",
            "6",
            "--provenance-dataset",
            "source_file_index",
            "--provenance-dataset",
            "source_sample_index",
            "--regime-dataset",
            "beta",
            "--field-kind",
            "temporal",
            "--time-axis",
            "1",
            "--manifest",
            str(manifest),
            "--mirror-uri-prefix",
            "b2://ups-datasets/runtime",
            "--content-addressed-mirror",
        ],
    )

    make_light_hdf5_shards.main()

    source = load_source_manifest(tmp_path / "manifest.source.yaml")
    construction = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    records = {Path(item["output_path"]).name: item for item in construction["records"]}
    for item in source.objects:
        key = f"runtime/immutable/sha256/{item.checksums['sha256']}/{item.path}"
        assert item.uris[0] == f"b2://ups-datasets/{key}"
        assert records[item.path]["remote_key"] == key


def test_make_light_hdf5_content_addressed_mirror_requires_prefix(monkeypatch):
    from scripts import make_light_hdf5_shards

    monkeypatch.setattr(
        "sys.argv",
        [
            "make_light_hdf5_shards",
            "--tasks",
            "advection1d",
            "--manifest",
            "manifest.yaml",
            "--provenance-dataset",
            "source_sample_index",
            "--regime-dataset",
            "beta",
            "--field-kind",
            "steady",
            "--content-addressed-mirror",
        ],
    )

    with pytest.raises(ValueError, match="requires --mirror-uri-prefix"):
        make_light_hdf5_shards.main()


@pytest.mark.parametrize(
    "removed_args",
    [
        ["--protocol-mode", "legacy"],
        ["--protocol-mode", "strat-v1"],
        ["--start-index", "0"],
        ["--split-source", "val=train"],
        ["--split-start-index", "train=0"],
        ["--split-block-size", "6"],
        ["--split-block-offset", "train=0"],
        ["--fallback-source-split", "train"],
    ],
)
def test_make_light_hdf5_cli_rejects_removed_legacy_options(monkeypatch, removed_args):
    from scripts import make_light_hdf5_shards

    monkeypatch.setattr(
        "sys.argv",
        [
            "make_light_hdf5_shards",
            "--tasks",
            "advection1d",
            "--manifest",
            "manifest.yaml",
            *removed_args,
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        make_light_hdf5_shards.main()


def test_make_light_hdf5_cli_requires_manifest(monkeypatch):
    from scripts import make_light_hdf5_shards

    monkeypatch.setattr("sys.argv", ["make_light_hdf5_shards", "--tasks", "advection1d"])

    with pytest.raises(SystemExit, match="2"):
        make_light_hdf5_shards.main()


@pytest.mark.parametrize("version", ["smoke-v1", "light-v1", "medium-v1"])
def test_make_light_hdf5_cli_reserves_legacy_version_labels(monkeypatch, version):
    from scripts import make_light_hdf5_shards

    monkeypatch.setattr(
        "sys.argv",
        [
            "make_light_hdf5_shards",
            "--tasks",
            "advection1d",
            "--manifest",
            "manifest.yaml",
            "--version",
            version,
            "--provenance-dataset",
            "source_sample_index",
            "--regime-dataset",
            "beta",
            "--field-kind",
            "steady",
        ],
    )

    with pytest.raises(ValueError, match="reserved for immutable legacy artifacts"):
        make_light_hdf5_shards.main()


@pytest.mark.parametrize("prefix", ["smoke-v1", "light-v1", "medium-v1"])
def test_make_light_hdf5_cli_reserves_legacy_remote_prefixes(monkeypatch, prefix):
    from scripts import make_light_hdf5_shards

    monkeypatch.setattr(
        "sys.argv",
        [
            "make_light_hdf5_shards",
            "--tasks",
            "advection1d",
            "--manifest",
            "manifest.yaml",
            "--remote-prefix",
            prefix,
            "--provenance-dataset",
            "source_sample_index",
            "--regime-dataset",
            "beta",
            "--field-kind",
            "steady",
        ],
    )

    with pytest.raises(ValueError, match="reserved for immutable legacy artifacts"):
        make_light_hdf5_shards.main()


@pytest.mark.parametrize(
    "omitted_option",
    ["--provenance-dataset", "--regime-dataset", "--field-kind"],
)
def test_make_light_hdf5_cli_requires_protocol_semantics(monkeypatch, omitted_option):
    from scripts import make_light_hdf5_shards

    option_pairs = [
        ("--provenance-dataset", "source_sample_index"),
        ("--regime-dataset", "beta"),
        ("--field-kind", "temporal"),
    ]
    required_args = [
        value
        for option, option_value in option_pairs
        if option != omitted_option
        for value in (option, option_value)
    ]
    monkeypatch.setattr(
        "sys.argv",
        [
            "make_light_hdf5_shards",
            "--tasks",
            "advection1d",
            "--manifest",
            "manifest.yaml",
            *required_args,
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        make_light_hdf5_shards.main()


def test_make_light_hdf5_cli_requires_time_axis_for_temporal_fields(monkeypatch):
    from scripts import make_light_hdf5_shards

    monkeypatch.setattr(
        "sys.argv",
        [
            "make_light_hdf5_shards",
            "--tasks",
            "advection1d",
            "--manifest",
            "manifest.yaml",
            "--provenance-dataset",
            "source_sample_index",
            "--regime-dataset",
            "beta",
            "--field-kind",
            "temporal",
        ],
    )

    with pytest.raises(ValueError, match="time-axis"):
        make_light_hdf5_shards.main()


@pytest.mark.parametrize("count_option", ["--train-count", "--val-count", "--test-count"])
def test_make_light_hdf5_cli_rejects_nonpositive_split_counts(monkeypatch, count_option):
    from scripts import make_light_hdf5_shards

    monkeypatch.setattr(
        "sys.argv",
        [
            "make_light_hdf5_shards",
            "--tasks",
            "advection1d",
            "--manifest",
            "manifest.yaml",
            "--provenance-dataset",
            "source_sample_index",
            "--regime-dataset",
            "beta",
            "--field-kind",
            "steady",
            count_option,
            "0",
        ],
    )

    with pytest.raises(ValueError, match="positive"):
        make_light_hdf5_shards.main()


def test_protocol_gate_rejects_all_nan_provenance_before_within_split_sets():
    arrays = _gate_arrays(np.full(2, np.nan))

    try:
        evaluate_protocol_splits(
            arrays,
            provenance_datasets=["source_sample_index"],
            regime_dataset="beta",
            field_kind="temporal",
            time_axis=1,
        )
    except ValueError as exc:
        assert "train identity dataset source_sample_index contains a non-finite value" in str(exc)
    else:
        raise AssertionError("Expected all-NaN provenance to be rejected")


def test_protocol_gate_rejects_nonfinite_provenance_across_splits():
    for value in (np.nan, np.inf, -np.inf):
        arrays = _gate_arrays(np.asarray([value]))
        try:
            evaluate_protocol_splits(
                arrays,
                provenance_datasets=["source_sample_index"],
                regime_dataset="beta",
                field_kind="temporal",
                time_axis=1,
            )
        except ValueError as exc:
            assert "identity dataset source_sample_index contains a non-finite value" in str(exc)
        else:
            raise AssertionError(f"Expected provenance value {value!r} to be rejected")


def test_protocol_gate_rejects_nonfinite_regime_identity():
    arrays = _gate_arrays(np.asarray([0.0]), regime=np.asarray([np.nan]))

    try:
        evaluate_protocol_splits(
            arrays,
            provenance_datasets=["source_sample_index"],
            regime_dataset="beta",
            field_kind="temporal",
            time_axis=1,
        )
    except ValueError as exc:
        assert "train identity dataset beta contains a non-finite value" in str(exc)
    else:
        raise AssertionError("Expected non-finite regime identity to be rejected")


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_protocol_gate_rejects_nonfinite_physical_fields(value):
    arrays = _gate_arrays(np.asarray([0.0]))
    arrays["val"]["data"][0, 0, 0] = value

    with pytest.raises(ValueError, match="val physical field data contains non-finite values"):
        evaluate_protocol_splits(
            arrays,
            provenance_datasets=["source_sample_index"],
            regime_dataset="beta",
            field_kind="temporal",
            time_axis=1,
        )
