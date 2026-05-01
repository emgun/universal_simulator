from __future__ import annotations

import h5py
import numpy as np

from scripts.make_light_hdf5_shards import build_task_shards


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
