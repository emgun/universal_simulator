from __future__ import annotations

import h5py
import numpy as np
import yaml

from scripts.make_light_hdf5_shards import build_task_shard_records, build_task_shards


def test_build_task_shards_slices_source_train_file(tmp_path):
    root = tmp_path / "source"
    out_root = tmp_path / "light"
    root.mkdir()
    with h5py.File(root / "burgers1d_train.h5", "w") as handle:
        data = np.arange(10 * 4, dtype=np.float32).reshape(10, 4)
        params = np.arange(10, dtype=np.float32).reshape(10, 1)
        handle.create_dataset("data", data=data)
        handle.create_dataset("nu", data=params)
        handle.create_dataset("not_sample_aligned", data=np.arange(3, dtype=np.float32))

    outputs = build_task_shards(
        root=root,
        out_root=out_root,
        task="burgers1d",
        source_split="train",
        train_count=3,
        val_count=2,
        test_count=1,
        start_index=1,
        overwrite=False,
    )

    assert [path.name for path in outputs] == ["burgers1d_train.h5", "burgers1d_val.h5", "burgers1d_test.h5"]
    with h5py.File(out_root / "burgers1d_train.h5", "r") as handle:
        assert handle["data"].shape == (3, 4)
        assert handle["data"][0].tolist() == [4.0, 5.0, 6.0, 7.0]
        assert handle["nu"][:, 0].tolist() == [1.0, 2.0, 3.0]
        assert "not_sample_aligned" not in handle
    with h5py.File(out_root / "burgers1d_val.h5", "r") as handle:
        assert handle["data"][:, 0].tolist() == [16.0, 20.0]
    with h5py.File(out_root / "burgers1d_test.h5", "r") as handle:
        assert handle["data"][:, 0].tolist() == [24.0]


def test_build_task_shard_records_prefers_native_splits_and_falls_back_to_train(tmp_path):
    root = tmp_path / "source"
    out_root = tmp_path / "light"
    root.mkdir()
    for split, offset in (("train", 0), ("val", 100), ("test", 200)):
        with h5py.File(root / f"burgers1d_{split}.h5", "w") as handle:
            data = (np.arange(10 * 2, dtype=np.float32).reshape(10, 2) + offset)
            handle.create_dataset("data", data=data)
    with h5py.File(root / "darcy2d_train.h5", "w") as handle:
        handle.create_dataset("data", data=np.arange(10 * 2, dtype=np.float32).reshape(10, 2))
    with h5py.File(root / "darcy2d_test.h5", "w") as handle:
        handle.create_dataset("data", data=np.arange(10 * 2, dtype=np.float32).reshape(10, 2) + 300)

    burgers = build_task_shard_records(
        root=root,
        out_root=out_root,
        task="burgers1d",
        source_split="train",
        train_count=2,
        val_count=2,
        test_count=2,
        start_index=1,
        overwrite=True,
        remote_prefix="light-v1",
    )
    darcy = build_task_shard_records(
        root=root,
        out_root=out_root,
        task="darcy2d",
        source_split="train",
        train_count=2,
        val_count=2,
        test_count=2,
        start_index=1,
        overwrite=True,
        remote_prefix="light-v1",
    )

    assert [record["source_split"] for record in burgers] == ["train", "val", "test"]
    assert [record["derived_from_source_split"] for record in burgers] == [False, False, False]
    assert [record["source_split"] for record in darcy] == ["train", "train", "test"]
    assert [record["derived_from_source_split"] for record in darcy] == [False, True, False]
    assert burgers[0]["remote_key"] == "light-v1/burgers1d/burgers1d_train.h5"
    assert len(str(burgers[0]["sha256"])) == 64

    with h5py.File(out_root / "burgers1d_val.h5", "r") as handle:
        assert handle["data"][0, 0] == 102.0
    with h5py.File(out_root / "darcy2d_val.h5", "r") as handle:
        assert handle["data"][0, 0] == 6.0


def test_build_task_shard_records_accepts_split_start_indices(tmp_path):
    root = tmp_path / "source"
    out_root = tmp_path / "light"
    root.mkdir()
    for split, offset in (("train", 0), ("val", 100), ("test", 200)):
        with h5py.File(root / f"advection1d_{split}.h5", "w") as handle:
            data = np.arange(10 * 2, dtype=np.float32).reshape(10, 2) + offset
            handle.create_dataset("data", data=data)

    records = build_task_shard_records(
        root=root,
        out_root=out_root,
        task="advection1d",
        source_split="train",
        train_count=2,
        val_count=2,
        test_count=2,
        start_index=0,
        overwrite=True,
        split_start_indices={"train": 4, "val": 1, "test": 2},
    )

    assert [record["start_index"] for record in records] == [4, 1, 2]
    with h5py.File(out_root / "advection1d_train.h5", "r") as handle:
        assert handle["data"][0, 0] == 8.0
    with h5py.File(out_root / "advection1d_val.h5", "r") as handle:
        assert handle["data"][0, 0] == 102.0
    with h5py.File(out_root / "advection1d_test.h5", "r") as handle:
        assert handle["data"][0, 0] == 204.0


def test_build_task_shard_records_accepts_stratified_block_offsets(tmp_path):
    root = tmp_path / "source"
    out_root = tmp_path / "light"
    root.mkdir()
    with h5py.File(root / "advection1d_train.h5", "w") as handle:
        data = np.arange(3 * 6, dtype=np.float32).reshape(18, 1)
        handle.create_dataset("data", data=data)

    records = build_task_shard_records(
        root=root,
        out_root=out_root,
        task="advection1d",
        source_split="train",
        train_count=9,
        val_count=6,
        test_count=3,
        start_index=0,
        overwrite=True,
        split_sources={"val": "train", "test": "train"},
        split_block_size=6,
        split_block_offsets={"train": 0, "val": 3, "test": 5},
    )

    assert [record["stratified_block_offset"] for record in records] == [0, 3, 5]
    with h5py.File(out_root / "advection1d_train.h5", "r") as handle:
        assert handle["data"][:, 0].tolist() == [0.0, 1.0, 2.0, 6.0, 7.0, 8.0, 12.0, 13.0, 14.0]
    with h5py.File(out_root / "advection1d_val.h5", "r") as handle:
        assert handle["data"][:, 0].tolist() == [3.0, 4.0, 9.0, 10.0, 15.0, 16.0]
    with h5py.File(out_root / "advection1d_test.h5", "r") as handle:
        assert handle["data"][:, 0].tolist() == [5.0, 11.0, 17.0]


def test_make_light_hdf5_manifest_cli(tmp_path, monkeypatch):
    from scripts import make_light_hdf5_shards

    root = tmp_path / "source"
    out_root = tmp_path / "light"
    manifest = tmp_path / "manifest.yaml"
    root.mkdir()
    with h5py.File(root / "burgers1d_train.h5", "w") as handle:
        handle.create_dataset("data", data=np.arange(10 * 2, dtype=np.float32).reshape(10, 2))

    monkeypatch.setattr(
        "sys.argv",
        [
            "make_light_hdf5_shards",
            "--root",
            str(root),
            "--out-root",
            str(out_root),
            "--tasks",
            "burgers1d",
            "--train-count",
            "2",
            "--val-count",
            "1",
            "--test-count",
            "1",
            "--manifest",
            str(manifest),
            "--version",
            "smoke-v1",
            "--remote-prefix",
            "smoke-v1",
        ],
    )

    make_light_hdf5_shards.main()

    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert payload["version"] == "smoke-v1"
    assert payload["remote_prefix"] == "smoke-v1"
    assert len(payload["records"]) == 3
    assert payload["records"][1]["split"] == "val"
    assert payload["records"][1]["derived_from_source_split"] is True
